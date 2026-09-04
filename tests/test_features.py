from __future__ import annotations

# Fold timestamps are intentionally timezone-naive by repository contract.
# ruff: noqa: DTZ001
import unittest
from datetime import datetime, timedelta
from math import isfinite

import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA
from features import (
    HistoryFeatureConfig,
    attach_als_factor_features,
    attach_history_features,
    attach_union_score_ranks,
    build_covisit_aggregate_features,
    build_item_features,
    build_user_features,
    validate_history_cutoff,
)
from item2item import COLLAPSED_HISTORY_SCHEMA, NEIGHBOR_TABLE_SCHEMA, Item2ItemConfig
from validation import ContractValidationError


def _history(cutoff: datetime) -> pl.DataFrame:
    timestamps = [
        cutoff - timedelta(hours=2),
        cutoff - timedelta(hours=10),
        cutoff - timedelta(hours=30),
    ]
    return pl.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "date": [value.date() for value in timestamps],
            "dt": timestamps,
            "views": [2, 1, 3],
            "watch_time": [61, 10, 30],
            "is_like": [0, 1, 0],
            "is_favorite": [0, 0, 1],
            "is_positive": [1, 1, 1],
        },
        schema=DAILY_INTERACTION_SCHEMA,
    )


class HistoryFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cutoff = datetime(2024, 1, 4, 12)
        self.config = HistoryFeatureConfig(
            windows_hours=(6.0, 24.0, 72.0),
            trend_windows_hours=(6.0, 24.0),
        )
        self.history = _history(self.cutoff)

    def test_cutoff_validator_rejects_validation_timestamp(self) -> None:
        diagnostics = validate_history_cutoff(self.history, cutoff=self.cutoff)
        self.assertEqual(diagnostics["rows"], 3)
        leaked = pl.concat(
            [
                self.history,
                pl.DataFrame(
                    {
                        "user_id": [3],
                        "item_id": [30],
                        "date": [self.cutoff.date()],
                        "dt": [self.cutoff],
                        "views": [1],
                        "watch_time": [0],
                        "is_like": [0],
                        "is_favorite": [0],
                        "is_positive": [0],
                    },
                    schema=DAILY_INTERACTION_SCHEMA,
                ),
            ]
        )
        with self.assertRaisesRegex(ContractValidationError, "at or after"):
            validate_history_cutoff(leaked, cutoff=self.cutoff)

    def test_user_and_item_features_use_explicit_windows(self) -> None:
        users = build_user_features(
            self.history, cutoff=self.cutoff, config=self.config
        )
        items = build_item_features(
            self.history, cutoff=self.cutoff, config=self.config
        )
        user = users.filter(pl.col("user_id") == 1).row(0, named=True)
        self.assertEqual(user["user_daily_rows_all"], 2)
        self.assertEqual(user["user_daily_rows_6h"], 1)
        self.assertEqual(user["user_daily_rows_24h"], 2)
        self.assertEqual(user["user_distinct_items_all"], 2)
        self.assertEqual(user["user_positive_daily_rate_all"], 1.0)
        self.assertAlmostEqual(user["user_hours_since_last_interaction"], 2.0)
        item = items.filter(pl.col("item_id") == 10).row(0, named=True)
        self.assertEqual(item["item_distinct_users_all"], 2)
        self.assertEqual(item["item_daily_rows_6h"], 1)
        self.assertEqual(item["item_daily_rows_prev_24h"], 1)
        self.assertTrue(isfinite(item["item_trend_views_log_ratio_24h"]))

    def test_missing_user_history_uses_flags_and_zero_sentinels(self) -> None:
        users = build_user_features(
            self.history, cutoff=self.cutoff, config=self.config
        )
        items = build_item_features(
            self.history, cutoff=self.cutoff, config=self.config
        )
        candidates = pl.DataFrame(
            {"user_id": [3], "item_id": [10]},
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        result = attach_history_features(
            candidates, user_features=users, item_features=items
        ).row(0, named=True)
        self.assertFalse(result["user_history_available"])
        self.assertEqual(result["user_daily_rows_all"], 0)
        self.assertTrue(result["item_history_available"])


class CandidateDerivedFeatureTests(unittest.TestCase):
    def test_union_score_rank_respects_availability_and_negative_scores(self) -> None:
        frame = pl.DataFrame(
            {
                "user_id": [1, 1, 1],
                "item_id": [10, 20, 30],
                "cross_score_item2item": [-2.0, 0.0, -1.0],
                "cross_score_available_item2item": [True, False, True],
            },
            schema={
                "user_id": pl.UInt64,
                "item_id": pl.Int32,
                "cross_score_item2item": pl.Float64,
                "cross_score_available_item2item": pl.Boolean,
            },
        )
        ranked = attach_union_score_ranks(frame, sources=("item2item",))
        self.assertEqual(
            ranked.get_column("union_rank_cross_score_item2item").to_list(),
            [2, 0, 1],
        )

    def test_covisit_aggregates_seed_evidence(self) -> None:
        cutoff = datetime(2024, 1, 2)
        candidates = pl.DataFrame(
            {"user_id": [1], "item_id": [30]},
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        seeds = pl.DataFrame(
            {
                "user_id": [1, 1],
                "item_id": [10, 20],
                "last_dt": [cutoff - timedelta(hours=1)] * 2,
                "views": [1, 1],
                "watch_time": [0, 0],
                "is_like": [0, 0],
                "is_favorite": [0, 0],
                "is_positive": [0, 0],
                "history_rank": [1, 2],
            },
            schema=COLLAPSED_HISTORY_SCHEMA,
        )
        neighbors = pl.DataFrame(
            {
                "item_id": [10, 20],
                "neighbor_item_id": [30, 30],
                "score": [0.5, 0.25],
                "co_user_count": [4, 2],
                "rank": [1, 2],
            },
            schema=NEIGHBOR_TABLE_SCHEMA,
        )
        config = Item2ItemConfig.from_dict(
            {
                "config_id": "test",
                "profile": "all",
                "direction": "undirected",
                "pair_weight": "uniform",
                "pair_half_life_hours": 24,
                "normalization": "raw",
                "min_pair_users": 1,
                "neighbor_k": 10,
                "history_cap": 10,
                "seed_k": 2,
                "seed_recency_half_life_hours": None,
                "seed_strength": "uniform",
            }
        )
        result = build_covisit_aggregate_features(
            candidates,
            seeds=seeds,
            neighbor_table=neighbors,
            config=config,
            cutoff=cutoff,
        ).row(0, named=True)
        self.assertTrue(result["covisit_available"])
        self.assertEqual(result["covisit_matched_seed_count"], 2)
        self.assertAlmostEqual(result["covisit_contribution_sum"], 0.75)
        self.assertAlmostEqual(result["covisit_contribution_max"], 0.5)

    def test_als_norms_produce_cosine_and_validate_availability(self) -> None:
        candidates = pl.DataFrame(
            {
                "user_id": [1],
                "item_id": [10],
                "cross_score_implicit_als": [4.0],
                "cross_score_available_implicit_als": [True],
            },
            schema={
                "user_id": pl.UInt64,
                "item_id": pl.Int32,
                "cross_score_implicit_als": pl.Float64,
                "cross_score_available_implicit_als": pl.Boolean,
            },
        )
        user_norms = pl.DataFrame(
            {"user_id": [1], "als_user_factor_norm": [2.0]},
            schema={"user_id": pl.UInt64, "als_user_factor_norm": pl.Float32},
        )
        item_norms = pl.DataFrame(
            {"item_id": [10], "als_item_factor_norm": [4.0]},
            schema={"item_id": pl.Int32, "als_item_factor_norm": pl.Float32},
        )
        result = attach_als_factor_features(
            candidates, user_norms=user_norms, item_norms=item_norms
        ).row(0, named=True)
        self.assertTrue(result["als_cosine_available"])
        self.assertAlmostEqual(result["als_factor_norm_product"], 8.0)
        self.assertAlmostEqual(result["als_cosine_similarity"], 0.5)
if __name__ == "__main__":
    unittest.main()

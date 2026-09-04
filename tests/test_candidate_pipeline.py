from __future__ import annotations

# Fold timestamps are intentionally timezone-naive by repository contract.
# ruff: noqa: DTZ001
import unittest
from datetime import date, datetime

import polars as pl

from candidate_pipeline import (
    CandidateUnionConfig,
    attach_implicit_als_cross_scores,
    attach_item2item_cross_scores,
    attach_ranking_cross_scores,
    build_candidate_union,
    cross_score_columns,
    generator_columns,
    iter_target_user_shards,
    validate_union_against_history,
    validate_union_features,
)
from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from implicit_model import ImplicitALSConfig, ImplicitALSDataLoader, ImplicitALSModel
from interfaces import CANDIDATE_SCHEMA
from item2item import (
    COLLAPSED_HISTORY_SCHEMA,
    NEIGHBOR_TABLE_SCHEMA,
    Item2ItemConfig,
)
from validation import ContractValidationError


def _candidates(source: str, rows: list[tuple[int, int, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [row[0] for row in rows],
            "item_id": [row[1] for row in rows],
            "score": [row[2] for row in rows],
            "rank": [row[3] for row in rows],
            "source": [source] * len(rows),
        },
        schema=CANDIDATE_SCHEMA,
    ).sort(("user_id", "source", "rank"))


def _history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 2, 9, 9, 9],
            "item_id": [99, 98, 10, 20, 30],
            "date": [date(2024, 1, 1)] * 5,
            "dt": [datetime(2024, 1, 1)] * 5,
            "views": [1] * 5,
            "watch_time": [61] * 5,
            "is_like": [0] * 5,
            "is_favorite": [0] * 5,
            "is_positive": [1] * 5,
        },
        schema=DAILY_INTERACTION_SCHEMA,
    )


class CandidateUnionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CandidateUnionConfig.from_mapping(
            {"global_popularity": 2, "implicit_als": 2}, total_cap=4
        )
        self.global_candidates = _candidates(
            "global_popularity",
            [
                (1, 10, 5.0, 1),
                (1, 20, 4.0, 2),
                (1, 30, 3.0, 3),
                (2, 10, 5.0, 1),
                (2, 30, 3.0, 2),
            ],
        )
        self.als_candidates = _candidates(
            "implicit_als",
            [
                (1, 20, 0.8, 1),
                (1, 30, 0.7, 2),
                (2, 20, 0.6, 1),
            ],
        )

    def union(self) -> pl.DataFrame:
        return build_candidate_union(
            {
                "global_popularity": self.global_candidates,
                "implicit_als": self.als_candidates,
            },
            self.config,
        )

    def test_union_preserves_provenance_and_deduplicates_pairs(self) -> None:
        union = self.union()
        self.assertEqual(union.height, 6)
        row = union.filter((pl.col("user_id") == 1) & (pl.col("item_id") == 20)).row(
            0, named=True
        )
        self.assertTrue(row["generated_by_global_popularity"])
        self.assertTrue(row["generated_by_implicit_als"])
        self.assertEqual(row["source_count"], 2)
        self.assertEqual(row["generator_rank_global_popularity"], 2)
        self.assertEqual(row["generator_rank_implicit_als"], 1)

    def test_missing_values_and_normalized_ranks_are_unambiguous(self) -> None:
        union = self.union()
        global_columns = generator_columns("global_popularity")
        als_columns = generator_columns("implicit_als")
        row = union.filter((pl.col("user_id") == 1) & (pl.col("item_id") == 10)).row(
            0, named=True
        )
        self.assertEqual(row[global_columns["rank_norm"]], 1.0)
        self.assertFalse(row[als_columns["generated"]])
        self.assertEqual(row[als_columns["score"]], 0.0)
        self.assertEqual(row[als_columns["rank"]], 0)
        self.assertEqual(row[als_columns["rank_norm"]], 0.0)

    def test_source_cap_is_applied_before_union(self) -> None:
        union = self.union()
        row = union.filter((pl.col("user_id") == 1) & (pl.col("item_id") == 30)).row(
            0, named=True
        )
        self.assertFalse(row["generated_by_global_popularity"])
        self.assertTrue(row["generated_by_implicit_als"])

    def test_sum_of_source_caps_must_fit_total_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum of source caps"):
            CandidateUnionConfig.from_mapping(
                {"global_popularity": 2, "implicit_als": 2}, total_cap=3
            )

    def test_cross_score_does_not_change_generator_membership(self) -> None:
        union = self.union()
        ranking = pl.DataFrame(
            {
                "item_id": [10, 20, 30],
                "score": [100.0, 90.0, 80.0],
                "global_rank": [1, 2, 3],
            },
            schema={
                "item_id": pl.Int32,
                "score": pl.Float64,
                "global_rank": pl.UInt32,
            },
        )
        scored = attach_ranking_cross_scores(
            union,
            source="global_popularity",
            item_ranking=ranking,
        )
        row = scored.filter((pl.col("user_id") == 1) & (pl.col("item_id") == 30)).row(
            0, named=True
        )
        columns = cross_score_columns("global_popularity")
        self.assertFalse(row["generated_by_global_popularity"])
        self.assertTrue(row[columns["available"]])
        self.assertEqual(row[columns["score"]], 80.0)
        self.assertEqual(row[columns["rank"]], 3)
        validate_union_features(
            scored,
            config=self.config,
            require_cross_scores=("global_popularity",),
        )

    def test_history_validator_rejects_seen_and_unknown_pairs(self) -> None:
        union = self.union()
        validate_union_against_history(union.lazy(), history_daily=_history().lazy())
        seen = (
            union.with_columns(
                pl.when((pl.col("user_id") == 1) & (pl.col("item_id") == 10))
                .then(pl.lit(99, dtype=pl.Int32))
                .otherwise(pl.col("item_id"))
                .alias("item_id")
            )
            .unique(("user_id", "item_id"), keep="first")
            .sort(("user_id", "item_id"))
        )
        with self.assertRaisesRegex(ContractValidationError, "seen=1"):
            validate_union_against_history(seen, history_daily=_history())

    def test_target_user_shards_are_sorted_and_complete(self) -> None:
        targets = pl.DataFrame({"user_id": [5, 1, 3, 2, 4]}, schema=TARGET_USER_SCHEMA)
        shards = list(iter_target_user_shards(targets, users_per_shard=2))
        self.assertEqual([index for index, _ in shards], [0, 1, 2])
        combined = pl.concat([frame for _, frame in shards])
        self.assertEqual(combined.get_column("user_id").to_list(), [1, 2, 3, 4, 5])


class Item2ItemCrossScoreTests(unittest.TestCase):
    def test_sparse_cross_score_is_available_outside_generator_top_k(self) -> None:
        config = Item2ItemConfig.from_dict(
            {
                "config_id": "test",
                "profile": "all",
                "direction": "undirected",
                "pair_weight": "uniform",
                "pair_half_life_hours": 24,
                "normalization": "raw",
                "min_pair_users": 1,
                "neighbor_k": 2,
                "history_cap": 2,
                "seed_k": 1,
                "seed_recency_half_life_hours": None,
                "seed_strength": "uniform",
            }
        )
        materialized = CandidateUnionConfig.from_mapping(
            {"global_popularity": 1}, total_cap=1
        )
        union = build_candidate_union(
            {"global_popularity": _candidates("global_popularity", [(1, 30, 1.0, 1)])},
            materialized,
        )
        seeds = pl.DataFrame(
            {
                "user_id": [1],
                "item_id": [10],
                "last_dt": [datetime(2024, 1, 1, 23)],
                "views": [1],
                "watch_time": [0],
                "is_like": [0],
                "is_favorite": [0],
                "is_positive": [0],
                "history_rank": [1],
            },
            schema=COLLAPSED_HISTORY_SCHEMA,
        )
        neighbors = pl.DataFrame(
            {
                "item_id": [10],
                "neighbor_item_id": [30],
                "score": [0.75],
                "co_user_count": [2],
                "rank": [1],
            },
            schema=NEIGHBOR_TABLE_SCHEMA,
        )
        scored = attach_item2item_cross_scores(
            union,
            seeds=seeds,
            neighbor_table=neighbors,
            config=config,
            reference_time=datetime(2024, 1, 2),
        )
        columns = cross_score_columns("item2item")
        self.assertTrue(scored.get_column(columns["available"]).item())
        self.assertAlmostEqual(scored.get_column(columns["score"]).item(), 0.75)


class ImplicitALSCrossScoreTests(unittest.TestCase):
    def test_factor_dot_product_and_unmapped_item_availability(self) -> None:
        als_config = ImplicitALSConfig(
            config_id="cross_score_test",
            factors=2,
            regularization=0.1,
            iterations=1,
            num_threads=1,
            seed=42,
        )
        loader = (
            ImplicitALSDataLoader(
                config=als_config,
                reference_time=datetime(2024, 1, 2),
                seed=42,
            )
            .load_fit_data(history=_history())
            .prepare_fit_data()
        )
        model = ImplicitALSModel(als_config).fit(loader, show_progress=False)
        materialized = CandidateUnionConfig.from_mapping(
            {"global_popularity": 2}, total_cap=2
        )
        union = build_candidate_union(
            {
                "global_popularity": _candidates(
                    "global_popularity",
                    [(1, 10, 1.0, 1), (1, 40, 0.5, 2)],
                )
            },
            materialized,
        )

        scored = attach_implicit_als_cross_scores(union, model=model, batch_size=1)
        columns = cross_score_columns("implicit_als")
        known = scored.filter(pl.col("item_id") == 10).row(0, named=True)
        unknown = scored.filter(pl.col("item_id") == 40).row(0, named=True)
        user_index = (
            model.user_mapping.filter(pl.col("user_id") == 1)
            .get_column("user_index")
            .item()
        )
        item_index = (
            model.item_mapping.filter(pl.col("item_id") == 10)
            .get_column("item_index")
            .item()
        )
        expected = float(
            model.backend.user_factors[user_index]
            @ model.backend.item_factors[item_index]
        )
        self.assertTrue(known[columns["available"]])
        self.assertAlmostEqual(known[columns["score"]], expected, places=6)
        self.assertFalse(unknown[columns["available"]])
        self.assertEqual(unknown[columns["score"]], 0.0)


if __name__ == "__main__":
    unittest.main()

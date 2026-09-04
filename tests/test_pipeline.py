from __future__ import annotations

# Synthetic timestamps are intentionally timezone-naive by contract.
# ruff: noqa: DTZ001
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from data_utils import (
    DAILY_INTERACTION_SCHEMA,
    RAW_INTERACTION_SCHEMA,
    TARGET_USER_SCHEMA,
)
from interfaces import RANKER_OUTPUT_SCHEMA
from pipeline import (
    derive_prediction_times,
    materialize_full_history,
    recommendations_with_fallback,
    sample_training_part,
)
from validation import (
    ContractValidationError,
    validate_recommendations_against_history,
)


class FullHistoryPipelineTests(unittest.TestCase):
    def test_full_history_daily_contract_and_prediction_horizon(self) -> None:
        high_user = 2**53 + 19
        raw = pl.DataFrame(
            [
                (high_user, 10, "watch_time", 60, datetime(2024, 1, 1, 8)),
                (high_user, 10, "like", 0, datetime(2024, 1, 1, 9)),
                (high_user, 10, "watch_time", 61, datetime(2024, 1, 2, 8)),
                (7, 11, "favorite", 0, datetime(2024, 1, 2, 10)),
            ],
            schema=RAW_INTERACTION_SCHEMA,
            orient="row",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.parquet"
            output = root / "history.parquet"
            raw.write_parquet(source)
            times = derive_prediction_times(source)
            diagnostics = materialize_full_history(source, output)
            history = pl.read_parquet(output)
            self.assertEqual(history.schema, DAILY_INTERACTION_SCHEMA)
            self.assertEqual(history.height, 3)
            self.assertEqual(diagnostics["duplicate_rows"], 0)
            self.assertEqual(diagnostics["out_of_order_rows"], 0)
            self.assertEqual(times["prediction_start"], raw.get_column("date").max() + timedelta(microseconds=1))
            self.assertEqual(times["prediction_end_exclusive"] - times["prediction_start"], timedelta(days=1))
            first = history.filter(
                (pl.col("user_id") == high_user) & (pl.col("date") == datetime(2024, 1, 1).date())
            ).row(0, named=True)
            self.assertEqual(first["views"], 2)
            self.assertEqual(first["watch_time"], 60)
            self.assertEqual(first["is_positive"], 1)

    def test_sampling_is_deterministic_keeps_positives_and_applies_ipw(self) -> None:
        frame = pl.DataFrame(
            {
                "user_id": pl.Series([1, 1, 2, 2, 3, 3], dtype=pl.UInt64),
                "item_id": pl.Series([10, 11, 20, 21, 30, 31], dtype=pl.Int32),
                "label": pl.Series([1, 0, 0, 0, 1, 0], dtype=pl.UInt8),
                "is_training_sample": [True] * 6,
                "sampling_probability": pl.Series([1.0, 1.0, 0.05, 1.0, 1.0, 0.05], dtype=pl.Float32),
                "source_count": pl.Series([1, 2, 1, 1, 1, 1], dtype=pl.UInt8),
                "generated_by_item2item": [False, True, False, True, False, False],
                "generator_rank_item2item": pl.Series([201, 10, 201, 10, 201, 201], dtype=pl.UInt32),
                "generated_by_implicit_als": [False] * 6,
                "generator_rank_implicit_als": pl.Series([201] * 6, dtype=pl.UInt32),
                "feature_a": pl.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=pl.Float32),
            }
        )
        first, diagnostics = sample_training_part(
            frame,
            feature_columns=["feature_a"],
            fold="rolling_1",
            fold_id=0,
            seed=42,
            secondary_negative_probability=1.0,
        )
        second, _ = sample_training_part(
            frame,
            feature_columns=["feature_a"],
            fold="rolling_1",
            fold_id=0,
            seed=42,
            secondary_negative_probability=1.0,
        )
        self.assertTrue(first.equals(second))
        self.assertEqual(diagnostics["positive_rows"], 2)
        self.assertEqual(first.filter(pl.col("label") == 1).height, 2)
        easy = first.filter(pl.col("full_sampling_probability") < 1)
        self.assertTrue((easy.get_column("sample_weight") == 20.0).all())
        self.assertEqual(first.get_column("fold_id").unique().to_list(), [0])

    def test_sampling_rejects_a_preexisting_positive_drop(self) -> None:
        frame = pl.DataFrame(
            {
                "user_id": pl.Series([1, 1], dtype=pl.UInt64),
                "item_id": pl.Series([10, 11], dtype=pl.Int32),
                "label": pl.Series([1, 0], dtype=pl.UInt8),
                "is_training_sample": [False, True],
                "sampling_probability": pl.Series([1.0, 1.0], dtype=pl.Float32),
                "source_count": pl.Series([1, 1], dtype=pl.UInt8),
                "generated_by_item2item": [False, False],
                "generator_rank_item2item": pl.Series([201, 201], dtype=pl.UInt32),
                "generated_by_implicit_als": [False, False],
                "generator_rank_implicit_als": pl.Series([201, 201], dtype=pl.UInt32),
                "feature_a": pl.Series([0.1, 0.2], dtype=pl.Float32),
            }
        )
        with self.assertRaisesRegex(ContractValidationError, "dropped positive"):
            sample_training_part(
                frame,
                feature_columns=["feature_a"],
                fold="canonical",
                fold_id=3,
                seed=42,
                secondary_negative_probability=1.0,
            )


class RecommendationFallbackTests(unittest.TestCase):
    def test_fallback_fills_exact_unique_target_universe(self) -> None:
        targets = pl.DataFrame(
            {"user_id": pl.Series([1, 2], dtype=pl.UInt64)}
        ).cast(TARGET_USER_SCHEMA)
        union = pl.DataFrame(
            {
                "user_id": pl.Series([1, 1, 1, 2, 2, 2], dtype=pl.UInt64),
                "item_id": pl.Series([10, 11, 12, 10, 11, 12], dtype=pl.Int32),
                "generated_by_global_popularity": [True] * 6,
                "generator_score_global_popularity": pl.Series([3.0, 2.0, 1.0] * 2, dtype=pl.Float64),
                "generator_rank_global_popularity": pl.Series([1, 2, 3] * 2, dtype=pl.UInt32),
            }
        )
        scores = pl.DataFrame(
            {
                "user_id": pl.Series([1], dtype=pl.UInt64),
                "item_id": pl.Series([12], dtype=pl.Int32),
                "ranker_score": pl.Series([9.0], dtype=pl.Float64),
            },
            schema=RANKER_OUTPUT_SCHEMA,
        )
        recommendations, long, diagnostics = recommendations_with_fallback(
            scores, union=union, target_users=targets, k=2
        )
        self.assertEqual(recommendations.height, 2)
        self.assertEqual(recommendations.row(0, named=True)["item_ids"], [12, 10])
        self.assertEqual(recommendations.row(1, named=True)["item_ids"], [10, 11])
        self.assertEqual(diagnostics["fallback_users"], 2)
        self.assertEqual(long.get_column("rank").max(), 2)

        history = pl.DataFrame(
            [
                (3, 10, datetime(2024, 1, 1).date(), datetime(2024, 1, 1), 1, 1, 0, 0, 0),
                (3, 11, datetime(2024, 1, 1).date(), datetime(2024, 1, 1), 1, 1, 0, 0, 0),
                (3, 12, datetime(2024, 1, 1).date(), datetime(2024, 1, 1), 1, 1, 0, 0, 0),
            ],
            schema=DAILY_INTERACTION_SCHEMA,
            orient="row",
        )
        validate_recommendations_against_history(
            recommendations, target_users=targets, history_daily=history, expected_k=2
        )


if __name__ == "__main__":
    unittest.main()

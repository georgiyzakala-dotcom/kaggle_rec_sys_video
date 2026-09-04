from __future__ import annotations

# Raw source timestamps are intentionally timezone-naive by contract.
# ruff: noqa: DTZ001
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import polars as pl

from data_utils import (
    DAILY_INTERACTION_SCHEMA,
    GROUND_TRUTH_SCHEMA,
    RAW_INTERACTION_SCHEMA,
    TARGET_USER_SCHEMA,
    DataPreparationError,
    aggregate_daily_interactions,
    build_ground_truth,
    infer_canonical_cutoff,
    prepare_temporal_fold,
    scan_raw_interactions,
    split_raw_interactions,
)


def raw_frame(rows: list[tuple[int, int, str, int, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=RAW_INTERACTION_SCHEMA,
        orient="row",
    )


class DailyAggregationTests(unittest.TestCase):
    def test_daily_aggregation_counts_rows_and_uses_maxima(self) -> None:
        high_user_id = 2**53 + 17
        raw = raw_frame(
            [
                (high_user_id, 10, "watch_time", 20, datetime(2024, 1, 1, 8)),
                (high_user_id, 10, "watch_time", 75, datetime(2024, 1, 1, 9)),
                (high_user_id, 10, "like", 0, datetime(2024, 1, 1, 10)),
                (high_user_id, 11, "watch_time", 60, datetime(2024, 1, 1, 11)),
                (high_user_id, 12, "favorite", 0, datetime(2024, 1, 2, 7)),
            ]
        )

        daily = aggregate_daily_interactions(raw.lazy()).collect()

        self.assertEqual(daily.schema, DAILY_INTERACTION_SCHEMA)
        self.assertEqual(daily.get_column("user_id").to_list()[0], high_user_id)
        item_10 = daily.filter(pl.col("item_id") == 10).row(0, named=True)
        self.assertEqual(item_10["views"], 3)
        self.assertEqual(item_10["watch_time"], 75)
        self.assertEqual(item_10["is_like"], 1)
        self.assertEqual(item_10["is_favorite"], 0)
        self.assertEqual(item_10["is_positive"], 1)
        self.assertEqual(item_10["dt"], datetime(2024, 1, 1, 8))
        self.assertEqual(
            daily.filter(pl.col("item_id") == 11).item(0, "is_positive"), 0
        )
        self.assertEqual(
            daily.filter(pl.col("item_id") == 12).item(0, "is_positive"), 1
        )

    def test_split_before_aggregation_keeps_same_daily_key_separate(self) -> None:
        cutoff = datetime(2024, 1, 1, 9)
        raw = raw_frame(
            [
                (1, 10, "watch_time", 30, datetime(2024, 1, 1, 8)),
                (1, 10, "watch_time", 90, datetime(2024, 1, 1, 10)),
            ]
        )

        history_raw, validation_raw = split_raw_interactions(
            raw.lazy(), cutoff=cutoff
        )
        history = aggregate_daily_interactions(history_raw).collect()
        validation = aggregate_daily_interactions(validation_raw).collect()

        self.assertEqual(history.height, 1)
        self.assertEqual(validation.height, 1)
        self.assertTrue(
            history.select("user_id", "item_id", "date").equals(
                validation.select("user_id", "item_id", "date")
            )
        )
        self.assertEqual(history.item(0, "views"), 1)
        self.assertEqual(validation.item(0, "views"), 1)
        self.assertEqual(history.item(0, "is_positive"), 0)
        self.assertEqual(validation.item(0, "is_positive"), 1)

    def test_optional_validation_end_is_half_open(self) -> None:
        raw = raw_frame(
            [
                (1, 1, "watch_time", 1, datetime(2024, 1, 1, 8)),
                (1, 2, "watch_time", 1, datetime(2024, 1, 2, 8)),
                (1, 3, "watch_time", 1, datetime(2024, 1, 3, 8)),
            ]
        )
        history, validation = split_raw_interactions(
            raw.lazy(),
            cutoff=datetime(2024, 1, 2, 8),
            validation_end_exclusive=datetime(2024, 1, 3, 8),
        )
        self.assertEqual(history.collect().get_column("item_id").to_list(), [1])
        self.assertEqual(validation.collect().get_column("item_id").to_list(), [2])


class GroundTruthTests(unittest.TestCase):
    def test_relevance_dedup_seen_and_cold_filtering(self) -> None:
        cutoff = datetime(2024, 1, 2, 9)
        raw = raw_frame(
            [
                (1, 10, "watch_time", 10, datetime(2024, 1, 1, 8)),
                (2, 13, "watch_time", 10, datetime(2024, 1, 1, 8)),
                (2, 14, "watch_time", 10, datetime(2024, 1, 1, 8)),
                (1, 10, "like", 0, datetime(2024, 1, 2, 10)),
                (1, 11, "watch_time", 60, datetime(2024, 1, 2, 10)),
                (1, 12, "watch_time", 61, datetime(2024, 1, 2, 10)),
                (1, 13, "watch_time", 70, datetime(2024, 1, 2, 10)),
                (1, 13, "like", 0, datetime(2024, 1, 2, 11)),
                (3, 14, "favorite", 0, datetime(2024, 1, 2, 10)),
            ]
        )
        history_raw, validation_raw = split_raw_interactions(
            raw.lazy(), cutoff=cutoff
        )
        history = aggregate_daily_interactions(history_raw)
        validation = aggregate_daily_interactions(validation_raw)
        targets = pl.DataFrame(
            {"user_id": pl.Series([1, 2], dtype=pl.UInt64)}
        ).lazy()

        stages = build_ground_truth(history, validation, targets)

        self.assertEqual(
            stages.positive_pairs.collect().select("user_id", "item_id").rows(),
            [(1, 10), (1, 12), (1, 13), (3, 14)],
        )
        self.assertEqual(
            stages.unseen_pairs.collect().rows(),
            [(1, 12), (1, 13), (3, 14)],
        )
        self.assertEqual(stages.cold_pairs.collect().rows(), [(1, 12)])
        self.assertEqual(
            stages.eligible_pairs.collect().rows(), [(1, 13), (3, 14)]
        )
        self.assertEqual(stages.target_pairs.collect().rows(), [(1, 13)])


class FoldMaterializationTests(unittest.TestCase):
    def test_tiny_fold_materializes_contract_and_refuses_overwrite(self) -> None:
        high_user_id = 2**53 + 99
        rows = [
            (high_user_id, 10, "watch_time", 20, datetime(2024, 1, 1, 8)),
            (2, 11, "watch_time", 20, datetime(2024, 1, 1, 8)),
            (high_user_id, 11, "like", 0, datetime(2024, 1, 2, 9)),
            (2, 12, "watch_time", 90, datetime(2024, 1, 2, 9)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.parquet"
            target_path = root / "targets.parquet"
            output = root / "fold"
            raw_frame(rows).write_parquet(train_path)
            pl.DataFrame(
                {
                    "user_id": pl.Series(
                        [high_user_id, 2], dtype=pl.UInt64
                    )
                }
            ).write_parquet(target_path)

            raw = scan_raw_interactions(train_path)
            self.assertEqual(
                infer_canonical_cutoff(raw), datetime(2024, 1, 1, 9)
            )
            result = prepare_temporal_fold(
                train_path=train_path,
                target_users_path=target_path,
                output_dir=output,
                run_id="tiny_test",
            )

            self.assertEqual(result.output_dir, output)
            self.assertEqual(
                pl.read_parquet(output / "history_daily.parquet").schema,
                DAILY_INTERACTION_SCHEMA,
            )
            self.assertEqual(
                pl.read_parquet(output / "ground_truth.parquet").schema,
                GROUND_TRUTH_SCHEMA,
            )
            self.assertEqual(
                pl.read_parquet(output / "target_users.parquet").schema,
                TARGET_USER_SCHEMA,
            )
            metrics = json.loads((output / "metrics.json").read_text())
            checksums = metrics["deterministic_diagnostics"]["output_sha256"]
            self.assertEqual(len(checksums), 5)
            self.assertTrue(all(len(value) == 64 for value in checksums.values()))
            with self.assertRaises(FileExistsError):
                prepare_temporal_fold(
                    train_path=train_path,
                    target_users_path=target_path,
                    output_dir=output,
                )

    def test_smoke_cannot_publish_to_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task01_canonical_data_v1"
            with self.assertRaises(DataPreparationError):
                prepare_temporal_fold(
                    train_path="data/train.parquet",
                    target_users_path="data/target_user_ids.parquet",
                    output_dir=output,
                    smoke_user_limit=1,
                )


if __name__ == "__main__":
    unittest.main()

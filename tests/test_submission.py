from __future__ import annotations

# Synthetic timestamps are intentionally timezone-naive by contract.
# ruff: noqa: DTZ001
import csv
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from interfaces import FINAL_RECOMMENDATION_SCHEMA
from submission import (
    read_submission,
    validate_submission_against_artifacts,
    write_submission,
)
from validation import ContractValidationError


def _recommendations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": pl.Series([2**64 - 1, 7], dtype=pl.UInt64),
            "item_ids": pl.Series(
                [[2**31 - 1, -2**31], [11, 12]], dtype=pl.List(pl.Int32)
            ),
        },
        schema=FINAL_RECOMMENDATION_SCHEMA,
    ).sort("user_id")


def _history() -> pl.DataFrame:
    rows = [
        (2**64 - 1, 99, date(2024, 1, 1), datetime(2024, 1, 1), 1, 10, 0, 0, 0),
        (7, 98, date(2024, 1, 1), datetime(2024, 1, 1), 1, 10, 0, 0, 0),
        (3, 2**31 - 1, date(2024, 1, 1), datetime(2024, 1, 1), 1, 10, 0, 0, 0),
        (3, -2**31, date(2024, 1, 1), datetime(2024, 1, 1), 1, 10, 0, 0, 0),
        (3, 11, date(2024, 1, 1), datetime(2024, 1, 1), 1, 10, 0, 0, 0),
        (3, 12, date(2024, 1, 1), datetime(2024, 1, 1), 1, 10, 0, 0, 0),
    ]
    return pl.DataFrame(rows, schema=DAILY_INTERACTION_SCHEMA, orient="row")


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


class SubmissionTests(unittest.TestCase):
    def test_uint_boundaries_json_round_trip_and_deterministic_bytes(self) -> None:
        recommendations = _recommendations()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a" / "submission.csv"
            second = root / "b" / "submission.csv"
            one = write_submission(recommendations, first, expected_k=2)
            two = write_submission(recommendations, second, expected_k=2)
            self.assertTrue(read_submission(first, expected_k=2).equals(recommendations))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(one["sha256"], two["sha256"])
            text = first.read_text(encoding="utf-8")
            self.assertEqual(text.splitlines()[0], "user_id,item_ids")
            self.assertIn(str(2**64 - 1), text)
            self.assertNotIn("e+", text.lower())

    def test_exact_target_known_unseen_validation(self) -> None:
        recommendations = _recommendations()
        targets = recommendations.select("user_id").cast(TARGET_USER_SCHEMA)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.csv"
            write_submission(recommendations, path, expected_k=2)
            diagnostics = validate_submission_against_artifacts(
                path,
                recommendations=recommendations,
                target_users=targets,
                history_daily=_history(),
                expected_k=2,
            )
            self.assertEqual(diagnostics["seen_pairs"], 0)
            wrong_targets = pl.DataFrame(
                {"user_id": pl.Series([7], dtype=pl.UInt64)}
            ).cast(TARGET_USER_SCHEMA)
            with self.assertRaises(ContractValidationError):
                validate_submission_against_artifacts(
                    path,
                    recommendations=recommendations,
                    target_users=wrong_targets,
                    history_daily=_history(),
                    expected_k=2,
                )

    def test_seen_pair_is_rejected(self) -> None:
        recommendations = _recommendations()
        history = pl.concat(
            [
                _history(),
                pl.DataFrame(
                    [
                        (
                            7,
                            11,
                            date(2024, 1, 2),
                            datetime(2024, 1, 2),
                            1,
                            10,
                            0,
                            0,
                            0,
                        )
                    ],
                    schema=DAILY_INTERACTION_SCHEMA,
                    orient="row",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.csv"
            write_submission(recommendations, path, expected_k=2)
            with self.assertRaisesRegex(ContractValidationError, "seen_pairs=1"):
                validate_submission_against_artifacts(
                    path,
                    recommendations=recommendations,
                    target_users=recommendations.select("user_id").cast(TARGET_USER_SCHEMA),
                    history_daily=history,
                    expected_k=2,
                )

    def test_malformed_csv_and_json_are_rejected(self) -> None:
        cases = {
            "missing_header": [["7", "[1, 2]"]],
            "wrong_column_order": [["item_ids", "user_id"], ["[1, 2]", "7"]],
            "extra_index": [["", "user_id", "item_ids"], ["0", "7", "[1, 2]"]],
            "malformed_json": [["user_id", "item_ids"], ["7", "[1,]"]],
            "duplicate_items": [["user_id", "item_ids"], ["7", "[1, 1]"]],
            "float_item": [["user_id", "item_ids"], ["7", "[1, 2.0]"]],
            "scientific_user": [["user_id", "item_ids"], ["7e0", "[1, 2]"]],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    _write_rows(path, rows)
                    with self.assertRaises(ContractValidationError):
                        read_submission(path, expected_k=2)

    def test_more_than_twenty_items_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "too_many.csv"
            _write_rows(
                path,
                [["user_id", "item_ids"], ["7", str(list(range(21)))]],
            )
            with self.assertRaisesRegex(ContractValidationError, "exactly 20"):
                read_submission(path, expected_k=20)


if __name__ == "__main__":
    unittest.main()

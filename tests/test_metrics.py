from __future__ import annotations

import unittest

import polars as pl

from data_utils import GROUND_TRUTH_SCHEMA, TARGET_USER_SCHEMA
from interfaces import FINAL_RECOMMENDATION_SCHEMA
from metrics import (
    candidate_count_statistics,
    candidate_oracle_precision_at_20,
    candidate_recall,
    candidate_user_hit_rate,
    evaluate_candidate_metrics,
    evaluate_candidate_metrics_lazy,
    evaluate_precision_at_20,
    precision_at_20_all_targets,
    precision_at_20_labeled_users,
)
from utils import prec_k


class PrecisionMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.high_user_id = 2**53 + 123
        self.targets = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [self.high_user_id, 2, 3], dtype=pl.UInt64
                )
            }
        ).cast(TARGET_USER_SCHEMA)
        self.ground_truth = pl.DataFrame(
            [(self.high_user_id, 10), (self.high_user_id, 11), (2, 12)],
            schema=GROUND_TRUTH_SCHEMA,
            orient="row",
        )
        self.recommendations = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [self.high_user_id, 2, 3], dtype=pl.UInt64
                ),
                "item_ids": pl.Series(
                    [[10, 99], [13], []], dtype=pl.List(pl.Int32)
                ),
            }
        ).cast(FINAL_RECOMMENDATION_SCHEMA)

    def test_both_precision_variants_use_fixed_denominator(self) -> None:
        values = evaluate_precision_at_20(
            self.recommendations, self.ground_truth, self.targets
        )
        self.assertAlmostEqual(values["precision_at_20_all_targets"], 1 / 60)
        self.assertAlmostEqual(
            values["precision_at_20_labeled_users"], 1 / 40
        )
        self.assertEqual(
            precision_at_20_all_targets(
                self.recommendations, self.ground_truth, self.targets
            ),
            values["precision_at_20_all_targets"],
        )
        self.assertEqual(
            precision_at_20_labeled_users(
                self.recommendations, self.ground_truth, self.targets
            ),
            values["precision_at_20_labeled_users"],
        )
        self.assertEqual(prec_k([10, 99], [10], 20), 0.05)

    def test_empty_ground_truth_returns_finite_zero(self) -> None:
        empty = pl.DataFrame(schema=GROUND_TRUTH_SCHEMA)
        values = evaluate_precision_at_20(
            self.recommendations, empty, self.targets
        )
        self.assertEqual(values["precision_at_20_all_targets"], 0.0)
        self.assertEqual(values["precision_at_20_labeled_users"], 0.0)


class CandidateMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.high_user_id = 2**53 + 123
        self.targets = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [self.high_user_id, 2, 3], dtype=pl.UInt64
                )
            }
        )
        self.ground_truth = pl.DataFrame(
            [(self.high_user_id, 10), (self.high_user_id, 11), (2, 12)],
            schema=GROUND_TRUTH_SCHEMA,
            orient="row",
        )
        self.candidates = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [
                        self.high_user_id,
                        self.high_user_id,
                        self.high_user_id,
                        2,
                        3,
                    ],
                    dtype=pl.UInt64,
                ),
                "item_id": pl.Series([10, 10, 11, 13, 77], dtype=pl.Int32),
                "source": ["a", "b", "a", "a", "a"],
            }
        )

    def test_candidate_union_metrics_deduplicate_across_sources(self) -> None:
        values = evaluate_candidate_metrics(
            self.candidates, self.ground_truth, self.targets
        )
        self.assertAlmostEqual(values["candidate_recall"], 2 / 3)
        self.assertAlmostEqual(values["candidate_user_hit_rate"], 1 / 2)
        self.assertAlmostEqual(
            values["candidate_oracle_p20_all_targets"], 1 / 30
        )
        self.assertAlmostEqual(
            values["candidate_oracle_p20_labeled_users"], 1 / 20
        )
        self.assertEqual(values["coverage"], 1.0)
        self.assertAlmostEqual(values["mean_candidate_count"], 4 / 3)
        self.assertEqual(values["p50_candidate_count"], 1.0)
        self.assertAlmostEqual(
            candidate_recall(self.candidates, self.ground_truth, self.targets),
            2 / 3,
        )
        self.assertAlmostEqual(
            candidate_user_hit_rate(
                self.candidates, self.ground_truth, self.targets
            ),
            1 / 2,
        )
        oracle = candidate_oracle_precision_at_20(
            self.candidates, self.ground_truth, self.targets
        )
        self.assertAlmostEqual(
            oracle["candidate_oracle_p20_all_targets"], 1 / 30
        )
        counts = candidate_count_statistics(
            self.candidates, self.ground_truth, self.targets
        )
        self.assertEqual(counts["coverage"], 1.0)

    def test_zero_candidate_users_are_in_coverage_and_counts(self) -> None:
        candidates = self.candidates.filter(
            pl.col("user_id") == self.high_user_id
        )
        values = evaluate_candidate_metrics(
            candidates, self.ground_truth, self.targets
        )
        self.assertAlmostEqual(values["coverage"], 1 / 3)
        self.assertAlmostEqual(values["mean_candidate_count"], 2 / 3)

    def test_lazy_candidate_metrics_match_eager_metrics(self) -> None:
        eager = evaluate_candidate_metrics(
            self.candidates, self.ground_truth, self.targets
        )
        lazy = evaluate_candidate_metrics_lazy(
            self.candidates.lazy(), self.ground_truth, self.targets
        )
        self.assertEqual(eager, lazy)


if __name__ == "__main__":
    unittest.main()

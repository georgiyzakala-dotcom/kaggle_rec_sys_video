from __future__ import annotations

import unittest

import polars as pl

from data_utils import GROUND_TRUTH_SCHEMA
from ranker_data import (
    NegativeSamplingConfig,
    apply_negative_sampling,
    assign_binary_labels,
    deterministic_sampling_uniform,
    feature_column_names,
    ranker_dataset_diagnostics,
    validate_candidate_pair_identity,
    validate_ranker_dataset,
)
from validation import ContractValidationError


def _candidates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "item_id": [10, 20, 30, 10],
            "source_count": [1, 2, 1, 1],
            "generated_by_item2item": [True, False, False, False],
            "generator_rank_item2item": [1, 0, 0, 0],
            "generated_by_implicit_als": [False, True, False, False],
            "generator_rank_implicit_als": [0, 40, 0, 0],
            "numeric_feature": [1.0, 2.0, 3.0, 4.0],
        },
        schema={
            "user_id": pl.UInt64,
            "item_id": pl.Int32,
            "source_count": pl.UInt8,
            "generated_by_item2item": pl.Boolean,
            "generator_rank_item2item": pl.UInt32,
            "generated_by_implicit_als": pl.Boolean,
            "generator_rank_implicit_als": pl.UInt32,
            "numeric_feature": pl.Float32,
        },
    )


class RankerLabelAndSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = _candidates()
        self.ground_truth = pl.DataFrame(
            {"user_id": [1], "item_id": [30]}, schema=GROUND_TRUTH_SCHEMA
        )
        self.config = NegativeSamplingConfig(
            seed=42,
            easy_negative_probability=0.25,
            hard_negative_rank=50,
        )

    def test_labels_are_only_candidate_ground_truth_intersection(self) -> None:
        labeled = assign_binary_labels(self.candidates, self.ground_truth)
        self.assertEqual(labeled.get_column("label").to_list(), [0, 0, 1, 0])
        changed_gt = pl.DataFrame(
            {"user_id": [1], "item_id": [10]}, schema=GROUND_TRUTH_SCHEMA
        )
        changed = assign_binary_labels(self.candidates, changed_gt)
        self.assertTrue(
            labeled.drop("label").equals(changed.drop("label")),
            "ground truth must not alter feature columns",
        )

    def test_sampling_is_deterministic_and_keeps_positive_and_hard_rows(self) -> None:
        labeled = assign_binary_labels(self.candidates, self.ground_truth)
        first = apply_negative_sampling(
            labeled, fold="rolling_1", config=self.config
        )
        second = apply_negative_sampling(
            labeled, fold="rolling_1", config=self.config
        )
        self.assertTrue(first.equals(second))
        self.assertTrue(
            first.filter(pl.col("label") == 1)
            .get_column("is_training_sample")
            .all()
        )
        self.assertEqual(first.get_column("is_hard_negative").to_list()[:2], [True, True])
        validate_ranker_dataset(first)
        diagnostics = ranker_dataset_diagnostics(first)
        self.assertEqual(diagnostics["rows"], 4)
        self.assertEqual(diagnostics["positive_rows"], 1)
        self.assertEqual(diagnostics["hard_negative_rows"], 2)

    def test_sampling_uniform_changes_with_fold_and_preserves_ids(self) -> None:
        first = deterministic_sampling_uniform(
            self.candidates, seed=42, fold="rolling_1"
        )
        second = deterministic_sampling_uniform(
            self.candidates, seed=42, fold="rolling_2"
        )
        self.assertFalse((first == second).all())
        self.assertTrue(((first >= 0) & (first < 1)).all())
        self.assertEqual(self.candidates.schema["user_id"], pl.UInt64)

    def test_feature_order_excludes_ids_label_and_sampling_metadata(self) -> None:
        dataset = apply_negative_sampling(
            assign_binary_labels(self.candidates, self.ground_truth),
            fold="rolling_1",
            config=self.config,
        )
        columns = feature_column_names(dataset)
        self.assertIn("numeric_feature", columns)
        self.assertNotIn("user_id", columns)
        self.assertNotIn("label", columns)
        self.assertNotIn("sampling_probability", columns)

    def test_pair_identity_validator_rejects_added_candidate(self) -> None:
        result = self.candidates.with_columns(pl.lit(0, dtype=pl.UInt8).alias("label"))
        validate_candidate_pair_identity(self.candidates, result)
        with self.assertRaisesRegex(ContractValidationError, "differ"):
            validate_candidate_pair_identity(self.candidates.head(3), result)


if __name__ == "__main__":
    unittest.main()

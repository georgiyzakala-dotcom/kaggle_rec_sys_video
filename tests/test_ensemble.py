from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from candidate_pipeline import CandidateUnionConfig, build_candidate_union
from data_utils import GROUND_TRUTH_SCHEMA
from ensemble import RRFConfig, RRFEnsembleModel, source_contribution_metrics
from interfaces import CANDIDATE_SCHEMA


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


class RRFEnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.materialized = CandidateUnionConfig.from_mapping(
            {"global_popularity": 2, "implicit_als": 2}, total_cap=4
        )
        self.features = build_candidate_union(
            {
                "global_popularity": _candidates(
                    "global_popularity",
                    [
                        (1, 10, 5.0, 1),
                        (1, 20, 4.0, 2),
                        (2, 10, 5.0, 1),
                    ],
                ),
                "implicit_als": _candidates(
                    "implicit_als",
                    [
                        (1, 20, 0.8, 1),
                        (1, 30, 0.7, 2),
                        (2, 20, 0.8, 1),
                    ],
                ),
            },
            self.materialized,
        )
        self.config = RRFConfig.from_dict(
            {
                "config_id": "test_rrf",
                "source_caps": {
                    "global_popularity": 1,
                    "implicit_als": 2,
                },
                "total_cap": 3,
                "weights": {
                    "global_popularity": 1.0,
                    "implicit_als": 2.0,
                },
                "rrf_constant": 10.0,
                "final_k": 2,
            }
        )

    def test_exact_rrf_scores_and_cap_filtering(self) -> None:
        scored = RRFEnsembleModel(self.config).score(
            self.features, materialized_config=self.materialized
        )
        user_one = scored.filter(pl.col("user_id") == 1)
        self.assertEqual(user_one.get_column("item_id").to_list(), [20, 30, 10])
        values = dict(
            zip(
                user_one.get_column("item_id").to_list(),
                user_one.get_column("ranker_score").to_list(),
                strict=True,
            )
        )
        self.assertAlmostEqual(values[10], 1.0 / 11.0)
        self.assertAlmostEqual(values[20], 2.0 / 11.0)
        self.assertAlmostEqual(values[30], 2.0 / 12.0)

    def test_equal_scores_use_item_id_tie_break(self) -> None:
        equal = RRFConfig.from_dict(
            {
                "config_id": "equal",
                "source_caps": {
                    "global_popularity": 1,
                    "implicit_als": 1,
                },
                "total_cap": 2,
                "weights": {
                    "global_popularity": 1.0,
                    "implicit_als": 1.0,
                },
                "rrf_constant": 10,
                "final_k": 2,
            }
        )
        scored = RRFEnsembleModel(equal).score(
            self.features, materialized_config=self.materialized
        )
        user_two = scored.filter(pl.col("user_id") == 2)
        self.assertEqual(user_two.get_column("item_id").to_list(), [10, 20])

    def test_rank_candidates_is_contiguous_and_deterministic(self) -> None:
        model = RRFEnsembleModel(self.config)
        first = model.rank_candidates(
            self.features, materialized_config=self.materialized
        )
        second = model.rank_candidates(
            self.features, materialized_config=self.materialized
        )
        self.assertTrue(first.equals(second))
        self.assertEqual(
            first.filter(pl.col("user_id") == 1).get_column("rank").to_list(),
            [1, 2, 3],
        )

    def test_source_contribution_respects_active_caps(self) -> None:
        ground_truth = pl.DataFrame(
            {"user_id": [1, 1, 2], "item_id": [10, 20, 20]},
            schema=GROUND_TRUTH_SCHEMA,
        )
        metrics = source_contribution_metrics(
            self.features,
            ground_truth=ground_truth,
            materialized_config=self.materialized,
            candidate_config=self.config.candidate_config,
        )
        self.assertEqual(metrics["union_relevant_hits"], 3)
        self.assertEqual(
            metrics["source_relevant_hits"],
            {"global_popularity": 1, "implicit_als": 2},
        )
        self.assertEqual(
            metrics["exclusive_relevant_hits"],
            {"global_popularity": 1, "implicit_als": 2},
        )
        self.assertEqual(
            metrics["pairwise_relevant_hit_overlap"]["global_popularity__implicit_als"],
            0,
        )

    def test_portable_restore_preserves_scores(self) -> None:
        model = RRFEnsembleModel(self.config)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "rrf"
            model.save(artifact)
            restored = RRFEnsembleModel.from_artifact(artifact)
            expected = model.score(self.features, materialized_config=self.materialized)
            actual = restored.score(
                self.features, materialized_config=self.materialized
            )
            self.assertTrue(expected.equals(actual))
            with self.assertRaises(FileExistsError):
                model.save(artifact)

    def test_json_round_trip_preserves_explicit_source_order(self) -> None:
        ordered = RRFConfig.from_dict(
            {
                "config_id": "ordered",
                "source_order": ["implicit_als", "global_popularity"],
                "source_caps": {
                    "implicit_als": 1,
                    "global_popularity": 1,
                },
                "total_cap": 2,
                "weights": {
                    "implicit_als": 1.0,
                    "global_popularity": 1.0,
                },
            }
        )
        encoded = json.loads(
            json.dumps(ordered.to_dict(), sort_keys=True, allow_nan=False)
        )
        restored = RRFConfig.from_dict(encoded)
        self.assertEqual(
            restored.candidate_config.source_names,
            ("implicit_als", "global_popularity"),
        )
        self.assertEqual(restored, ordered)

    def test_trusted_minimal_projection_matches_validated_scoring(self) -> None:
        model = RRFEnsembleModel(self.config)
        expected = model.rank_candidates(
            self.features, materialized_config=self.materialized
        )
        minimal = self.features.select(
            "user_id",
            "item_id",
            "generated_by_global_popularity",
            "generator_rank_global_popularity",
            "generated_by_implicit_als",
            "generator_rank_implicit_als",
        )
        actual = model.rank_candidates(
            minimal,
            materialized_config=self.materialized,
            validate_features=False,
        )
        self.assertTrue(expected.equals(actual))

    def test_invalid_weight_or_materialized_cap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one RRF weight"):
            RRFConfig.from_dict(
                {
                    "config_id": "zero",
                    "source_caps": {
                        "global_popularity": 1,
                        "implicit_als": 1,
                    },
                    "total_cap": 2,
                    "weights": {
                        "global_popularity": 0,
                        "implicit_als": 0,
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from catboost_selection import (
    CatBoostSelectionError,
    aggregate_fold_results,
    select_best_result,
    stage_challengers,
    validate_selection_config,
)
from experiment_utils import EventProgressReporter, sha256_file, write_json_atomic
from scripts.run_catboost_selection import (
    _discard_unregistered_file,
    cleanup_selection_state,
)

FEATURES = ("feature_a", "feature_b", "unstable_feature")


def _config() -> dict[str, object]:
    return {
        "task07_artifact": "artifacts/task07",
        "task06_rrf_artifact": "artifacts/task06",
        "task08_artifact": "artifacts/task08",
        "fold_pairs": [
            {
                "pair_id": "wf_r1_r2",
                "train_fold": "rolling_1",
                "eval_fold": "rolling_2",
                "pool_cache": "artifacts/pool_r1_r2",
            },
            {
                "pair_id": "wf_r2_r3",
                "train_fold": "rolling_2",
                "eval_fold": "rolling_3",
                "pool_cache": "artifacts/pool_r2_r3",
            },
        ],
        "negative_pool_caches": {
            "wf_r1_r2": "artifacts/negative_r1",
            "wf_r2_r3": "artifacts/negative_r2",
        },
        "row_sampling": {
            "eval_negative_keep_probability": 0.25,
            "weighting": "inverse_sampling_probability",
        },
        "pool": {
            "border_count": 32,
            "feature_border_type": "GreedyLogSum",
            "quantization_task_type": "CPU",
            "thread_count": 2,
        },
        "feature_sets": {
            "all": {"ignored_features": []},
            "stable": {"ignored_features": ["unstable_feature"]},
        },
        "baseline": {
            "config_id": "baseline",
            "train_negative_keep_probability": 1.0,
            "feature_set": "all",
            "catboost": {
                "config_id": "baseline",
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "iterations": 20,
                "depth": 4,
                "learning_rate": 0.1,
                "l2_leaf_reg": 3.0,
                "border_count": 32,
                "random_seed": 42,
                "task_type": "CPU",
                "devices": "0",
                "thread_count": 2,
                "early_stopping_rounds": 5,
                "metric_period": 1,
                "gpu_ram_part": 0.5,
                "boosting_type": "Plain",
                "bootstrap_type": "Bernoulli",
                "subsample": 0.8,
                "bagging_temperature": None,
                "random_strength": 1.0,
                "scale_pos_weight": 1.0,
                "ignored_features": [],
                "snapshot_interval_seconds": 1,
            },
        },
        "search": {
            "primary_metric": "precision_at_20_labeled_users",
            "secondary_metric": "precision_at_20_all_targets",
            "tie_epsilon": 1e-6,
            "max_unique_configs": 3,
            "stages": [
                {
                    "stage_id": "weight",
                    "challengers": [
                        {
                            "config_id": "weight_4",
                            "overrides": {"catboost": {"scale_pos_weight": 4.0}},
                        }
                    ],
                },
                {
                    "stage_id": "features",
                    "challengers": [
                        {
                            "config_id": "stable_features",
                            "overrides": {"feature_set": "stable"},
                        }
                    ],
                },
            ],
        },
        "canonical": {
            "fold": "canonical",
            "latest_pair_id": "wf_r2_r3",
            "evaluate_config_count": 1,
        },
        "inference": {"batch_size": 1024, "final_k": 20},
        "seed": 42,
    }


def _result(
    config_id: str,
    *,
    mean_labeled: float,
    minimum_labeled: float,
    spread: float,
    mean_all: float,
    trees: float,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "mean_precision_at_20_labeled_users": mean_labeled,
        "min_precision_at_20_labeled_users": minimum_labeled,
        "fold_spread_precision_at_20_labeled_users": spread,
        "mean_precision_at_20_all_targets": mean_all,
        "mean_tree_count": trees,
    }


class CatBoostSelectionTests(unittest.TestCase):
    def test_unregistered_file_is_discarded_but_checkpointed_file_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "part.parquet"
            reporter = EventProgressReporter(
                task_name="recovery_test",
                total_phases=1,
                log_file=None,
                show_progress=False,
            )
            path.write_bytes(b"partial")
            _discard_unregistered_file(
                path,
                checkpoint_record=None,
                reporter=reporter,
                stage="inference",
                config="candidate",
                fold="rolling_2",
                operation="part",
            )
            self.assertFalse(path.exists())
            path.write_bytes(b"complete")
            _discard_unregistered_file(
                path,
                checkpoint_record={"sha256": "recorded"},
                reporter=reporter,
                stage="inference",
                config="candidate",
                fold="rolling_2",
                operation="part",
            )
            reporter.close()
            self.assertEqual(path.read_bytes(), b"complete")

    def test_protocol_rejects_canonical_selection_and_broken_chain(self) -> None:
        canonical = _config()
        canonical["fold_pairs"][0]["eval_fold"] = "canonical"
        with self.assertRaisesRegex(CatBoostSelectionError, "canonical"):
            validate_selection_config(canonical, feature_columns=FEATURES)

        broken = _config()
        broken["fold_pairs"][1]["train_fold"] = "rolling_other"
        with self.assertRaisesRegex(CatBoostSelectionError, "chain"):
            validate_selection_config(broken, feature_columns=FEATURES)

    def test_stage_profiles_inherit_the_frozen_incumbent(self) -> None:
        source = validate_selection_config(_config(), feature_columns=FEATURES)
        weighted = stage_challengers(
            source["baseline_resolved"],
            source["search"]["stages"][0],
            feature_sets=source["feature_sets"],
        )[0]
        stable = stage_challengers(
            weighted,
            source["search"]["stages"][1],
            feature_sets=source["feature_sets"],
        )[0]
        self.assertEqual(stable["catboost"]["scale_pos_weight"], 4.0)
        self.assertEqual(stable["catboost"]["ignored_features"], ["unstable_feature"])
        self.assertEqual(stable["train_negative_keep_probability"], 1.0)

    def test_fold_aggregation_keeps_metric_denominators_separate(self) -> None:
        aggregate = aggregate_fold_results(
            "candidate",
            [
                {
                    "precision_at_20_labeled_users": 0.008,
                    "precision_at_20_all_targets": 0.006,
                    "tree_count": 100,
                    "final_hits": 12,
                    "runtime_seconds": 2.0,
                },
                {
                    "precision_at_20_labeled_users": 0.006,
                    "precision_at_20_all_targets": 0.004,
                    "tree_count": 120,
                    "final_hits": 10,
                    "runtime_seconds": 3.0,
                },
            ],
        )
        self.assertAlmostEqual(aggregate["mean_precision_at_20_labeled_users"], 0.007)
        self.assertAlmostEqual(aggregate["mean_precision_at_20_all_targets"], 0.005)
        self.assertAlmostEqual(
            aggregate["fold_spread_precision_at_20_labeled_users"], 0.002
        )
        self.assertEqual(aggregate["total_hits"], 22)

    def test_tie_break_prefers_fold_floor_then_stability_then_all_targets(self) -> None:
        incumbent = _result(
            "incumbent",
            mean_labeled=0.007,
            minimum_labeled=0.006,
            spread=0.002,
            mean_all=0.005,
            trees=100,
        )
        stronger_floor = _result(
            "stronger_floor",
            mean_labeled=0.0070005,
            minimum_labeled=0.0062,
            spread=0.0016,
            mean_all=0.004,
            trees=200,
        )
        winner = select_best_result(
            [incumbent, stronger_floor],
            tie_epsilon=1e-6,
            order=["incumbent", "stronger_floor"],
        )
        self.assertEqual(winner["config_id"], "stronger_floor")

        higher_all = copy.deepcopy(stronger_floor)
        higher_all["config_id"] = "higher_all"
        higher_all["mean_precision_at_20_all_targets"] = 0.0055
        winner = select_best_result(
            [stronger_floor, higher_all],
            tie_epsilon=1e-6,
            order=["stronger_floor", "higher_all"],
        )
        self.assertEqual(winner["config_id"], "higher_all")

    def test_cleanup_requires_complete_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts" / "run"
            checkpoint = root / "artifacts" / ".run.checkpoint"
            best = root / "artifacts" / ".run.best"
            pool = root / "artifacts" / "pool"
            source = root / "artifacts" / "source"
            for directory in (output / "model", checkpoint, best, pool, source):
                directory.mkdir(parents=True, exist_ok=True)
            model = output / "model" / "model.cbm"
            model.write_bytes(b"model")
            write_json_atomic(
                output / "config.json",
                {
                    "kind": "task09_catboost_selection",
                    "run_id": "run",
                },
            )
            write_json_atomic(
                output / "metrics.json",
                {"run_id": "run", "model_sha256": sha256_file(model)},
            )
            write_json_atomic(
                checkpoint / "checkpoint.json",
                {"artifact_version": 1, "run_id": "run"},
            )
            write_json_atomic(
                best / "best_model.json",
                {"artifact_version": 1, "run_id": "run"},
            )
            summary = cleanup_selection_state(
                checkpoint_dir=checkpoint,
                best_model_dir=best,
                output_dir=output,
                pool_dirs=[pool],
                input_dirs=[source],
                run_id="run",
            )
            self.assertFalse(checkpoint.exists())
            self.assertFalse(best.exists())
            self.assertTrue(output.is_dir())
            self.assertTrue(pool.is_dir())
            self.assertEqual(summary["retained_pools"], [pool.as_posix()])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from catboost import Pool
from catboost.utils import quantize

from data_utils import GROUND_TRUTH_SCHEMA, TARGET_USER_SCHEMA
from experiment_utils import config_sha256, sha256_file, write_json_atomic
from rankers import (
    CatBoostLTRConfig,
    CatBoostLTRDataLoader,
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
    CatBoostRankerModel,
    prepare_weighted_rows,
    ranker_scores_to_candidates,
    validate_feature_columns,
    write_catboost_dsv_part,
    write_column_description,
)
from scripts.run_catboost_ranker import (
    CatBoostExperimentError,
    _validate_task07,
    cleanup_recoverable_state,
)
from validation import ContractValidationError

FEATURES = ("feature_a", "feature_b")


def _ranker_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": pl.Series([2**53 + 7, 2**53 + 7, 9, 9], dtype=pl.UInt64),
            "item_id": pl.Series([10, 11, 20, 21], dtype=pl.Int32),
            "feature_a": pl.Series([0.0, 1.0, 2.0, 3.0], dtype=pl.Float32),
            "feature_b": pl.Series([1, 0, 1, 0], dtype=pl.UInt8),
            "label": pl.Series([0, 1, 0, 1], dtype=pl.UInt8),
            "is_training_sample": pl.Series([True, True, True, True]),
            "sampling_probability": pl.Series([0.05, 1.0, 0.25, 1.0], dtype=pl.Float32),
        }
    )


class CatBoostRankerContractTests(unittest.TestCase):
    def test_success_cleanup_removes_only_recoverable_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts" / "run"
            model = output / "model" / "model.cbm"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"portable-model")
            write_json_atomic(
                output / "config.json",
                {
                    "artifact_version": 1,
                    "kind": "task08_catboost_pointwise",
                    "run_id": "run",
                },
            )
            write_json_atomic(
                output / "metrics.json",
                {"run_id": "run", "model_sha256": sha256_file(model)},
            )
            checkpoint = root / "artifacts" / ".run.checkpoint"
            write_json_atomic(
                checkpoint / "checkpoint.json",
                {"artifact_version": 1, "run_id": "run"},
            )
            (checkpoint / "raw").mkdir()
            (checkpoint / "raw" / "train.tsv").write_bytes(b"temporary")
            best = root / "artifacts" / ".run.best-model"
            write_json_atomic(
                best / "best_model.json",
                {"artifact_version": 1, "run_id": "run"},
            )
            pool = root / "artifacts" / "pool"
            pool.mkdir()
            (pool / "train.quantized").write_bytes(b"keep")
            dataset = root / "artifacts" / "task07"
            dataset.mkdir()
            rrf = root / "artifacts" / "task06"
            rrf.mkdir()

            summary = cleanup_recoverable_state(
                checkpoint_dir=checkpoint,
                best_model_dir=best,
                output_dir=output,
                pool_cache_dir=pool,
                dataset_dir=dataset,
                rrf_artifact_dir=rrf,
                run_id="run",
            )

            self.assertGreater(summary["freed_bytes"], 0)
            self.assertFalse(checkpoint.exists())
            self.assertFalse(best.exists())
            self.assertTrue(output.is_dir())
            self.assertTrue(pool.is_dir())
            self.assertTrue(dataset.is_dir())
            self.assertTrue(rrf.is_dir())

    def test_success_cleanup_rejects_wrong_run_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts" / "run"
            model = output / "model" / "model.cbm"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"portable-model")
            write_json_atomic(
                output / "config.json",
                {"kind": "task08_catboost_pointwise", "run_id": "run"},
            )
            write_json_atomic(
                output / "metrics.json",
                {"run_id": "run", "model_sha256": sha256_file(model)},
            )
            checkpoint = root / "artifacts" / ".run.checkpoint"
            write_json_atomic(
                checkpoint / "checkpoint.json",
                {"artifact_version": 1, "run_id": "run"},
            )
            best = root / "artifacts" / ".other.best-model"
            write_json_atomic(
                best / "best_model.json",
                {"artifact_version": 1, "run_id": "other"},
            )
            pool = root / "artifacts" / "pool"
            pool.mkdir()
            dataset = root / "artifacts" / "task07"
            dataset.mkdir()
            rrf = root / "artifacts" / "task06"
            rrf.mkdir()

            with self.assertRaisesRegex(CatBoostExperimentError, "does not belong"):
                cleanup_recoverable_state(
                    checkpoint_dir=checkpoint,
                    best_model_dir=best,
                    output_dir=output,
                    pool_cache_dir=pool,
                    dataset_dir=dataset,
                    rrf_artifact_dir=rrf,
                    run_id="run",
                )
            self.assertTrue(checkpoint.is_dir())
            self.assertTrue(best.is_dir())
            self.assertTrue(output.is_dir())

    def test_feature_order_and_metadata_rejection(self) -> None:
        self.assertEqual(validate_feature_columns(FEATURES), FEATURES)
        with self.assertRaises(ValueError):
            validate_feature_columns(("feature_a", "feature_a"))
        with self.assertRaises(ValueError):
            validate_feature_columns(("feature_a", "label"))

    def test_pointwise_config_is_strict(self) -> None:
        config = CatBoostPointwiseConfig(task_type="CPU", iterations=5)
        self.assertEqual(config.loss_function, "Logloss")
        self.assertNotIn("devices", config.training_params())
        cross_entropy = CatBoostPointwiseConfig(
            loss_function="CrossEntropy",
            eval_metric="CrossEntropy",
        )
        self.assertEqual(cross_entropy.loss_function, "CrossEntropy")
        bayesian = CatBoostPointwiseConfig(
            bootstrap_type="Bayesian",
            subsample=None,
            bagging_temperature=0.5,
            random_strength=0.0,
        )
        self.assertNotIn("subsample", bayesian.training_params())
        self.assertEqual(bayesian.training_params()["bagging_temperature"], 0.5)
        with self.assertRaises(ValueError):
            CatBoostPointwiseConfig(loss_function="CrossEntropy", scale_pos_weight=4.0)
        with self.assertRaises(ValueError):
            CatBoostPointwiseConfig(bootstrap_type="Bayesian", subsample=0.8)
        with self.assertRaises(ValueError):
            CatBoostPointwiseConfig.from_mapping({"unknown": 1})

    def test_group_aware_ranker_is_explicitly_separate_from_pointwise(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            CatBoostLTRConfig.from_mapping(
                {
                    "config_id": "invalid",
                    "loss_function": "QuerySoftMax:beta=1",
                    "scale_pos_weight": 16,
                }
            )
        ltr = CatBoostRankerModel(
            CatBoostLTRConfig(
                config_id="cpu_ltr",
                loss_function="QuerySoftMax:beta=1",
                task_type="CPU",
                iterations=2,
            ),
            feature_columns=FEATURES,
        )
        self.assertEqual(ltr.get_config()["model"], "catboost_ltr")
        ungrouped = Pool(
            np.ones((4, 2), dtype=np.float32),
            label=[1, 0, 1, 0],
            feature_names=list(FEATURES),
        )
        loader = CatBoostLTRDataLoader(feature_columns=FEATURES).load_fit_data(
            train_pool=ungrouped,
            eval_pool=ungrouped,
        )
        with self.assertRaisesRegex(ContractValidationError, "group_id"):
            loader.prepare_fit_data()

    def test_feature_ablation_is_validated_by_model(self) -> None:
        config = CatBoostPointwiseConfig(
            task_type="CPU",
            iterations=5,
            ignored_features=("feature_b",),
        )
        model = CatBoostPointwiseModel(config, feature_columns=FEATURES)
        self.assertEqual(model.config.ignored_features, ("feature_b",))
        with self.assertRaisesRegex(ValueError, "absent"):
            CatBoostPointwiseModel(
                CatBoostPointwiseConfig(
                    task_type="CPU",
                    iterations=5,
                    ignored_features=("unknown",),
                ),
                feature_columns=FEATURES,
            )

    def test_inverse_sampling_weights_and_deterministic_eval_sampling(self) -> None:
        frame = _ranker_frame()
        weighted = prepare_weighted_rows(
            frame,
            feature_columns=FEATURES,
            fold="rolling_2:train",
            seed=42,
            negative_keep_probability=1.0,
        )
        self.assertEqual(weighted.columns, ["label", "sample_weight", *FEATURES])
        self.assertEqual(
            weighted.get_column("sample_weight").to_list(), [20.0, 1.0, 4.0, 1.0]
        )
        first = prepare_weighted_rows(
            frame,
            feature_columns=FEATURES,
            fold="rolling_3:eval",
            seed=42,
            negative_keep_probability=0.5,
        )
        second = prepare_weighted_rows(
            frame,
            feature_columns=FEATURES,
            fold="rolling_3:eval",
            seed=42,
            negative_keep_probability=0.5,
        )
        self.assertTrue(first.equals(second))
        self.assertEqual(first.filter(pl.col("label") == 1).height, 2)

    def test_score_ranking_uses_item_id_tie_break(self) -> None:
        scores = pl.DataFrame(
            {
                "user_id": pl.Series([1, 1, 1, 2, 2], dtype=pl.UInt64),
                "item_id": pl.Series([30, 10, 20, 8, 7], dtype=pl.Int32),
                "ranker_score": pl.Series([0.5, 0.5, 0.4, 1.0, 1.0], dtype=pl.Float64),
            }
        )
        ranked = ranker_scores_to_candidates(scores, k=2)
        self.assertEqual(
            ranked.filter(pl.col("user_id") == 1).get_column("item_id").to_list(),
            [10, 30],
        )
        self.assertEqual(
            ranked.filter(pl.col("user_id") == 2).get_column("item_id").to_list(),
            [7, 8],
        )

    def test_dsv_quantization_preserves_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dsv = root / "part.tsv"
            cd = root / "columns.cd"
            borders = root / "borders.tsv"
            quantized_path = root / "pool.quantized"
            diagnostics = write_catboost_dsv_part(
                _ranker_frame(),
                dsv,
                feature_columns=FEATURES,
                fold="test",
                seed=42,
                negative_keep_probability=1.0,
            )
            self.assertEqual(diagnostics["rows"], 4)
            write_column_description(cd, feature_columns=FEATURES)
            pool = quantize(
                data_path=dsv.as_posix(),
                column_description=cd.as_posix(),
                delimiter="\t",
                has_header=False,
                border_count=4,
                task_type="CPU",
                random_seed=42,
            )
            pool.save_quantization_borders(borders.as_posix())
            pool.save(quantized_path.as_posix())
            restored = Pool(f"quantized://{quantized_path.resolve()}")
            self.assertEqual(tuple(restored.get_feature_names()), FEATURES)
            self.assertEqual(restored.num_row(), 4)

    def test_task07_dataset_manifest_and_schema_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold_root = root / "folds" / "rolling"
            data_root = fold_root / "ranker_data"
            data_root.mkdir(parents=True)
            feature_schema = {
                "feature_count": 2,
                "feature_columns": list(FEATURES),
            }
            write_json_atomic(root / "feature_schema.json", feature_schema)
            part = data_root / "part-00000.parquet"
            _ranker_frame().write_parquet(part)
            target = fold_root / "target_users.parquet"
            ground_truth = fold_root / "target_ground_truth.parquet"
            history = root / "history.parquet"
            pl.DataFrame({"user_id": pl.Series([9, 2**53 + 7], dtype=pl.UInt64)}).cast(
                TARGET_USER_SCHEMA
            ).write_parquet(target)
            pl.DataFrame(
                {
                    "user_id": pl.Series([9], dtype=pl.UInt64),
                    "item_id": pl.Series([21], dtype=pl.Int32),
                }
            ).cast(GROUND_TRUTH_SCHEMA).write_parquet(ground_truth)
            pl.DataFrame({"placeholder": [1]}).write_parquet(history)
            fold_manifest = {
                "fold": "rolling",
                "part_count": 1,
                "parts": {part.name: sha256_file(part)},
                "target_users_sha256": sha256_file(target),
                "target_ground_truth_sha256": sha256_file(ground_truth),
                "history_path": history.as_posix(),
            }
            fold_manifest_path = fold_root / "fold_manifest.json"
            write_json_atomic(fold_manifest_path, fold_manifest)
            root_manifest = {
                "kind": "task07_ranker_datasets",
                "mode": "full",
                "schema_sha256": config_sha256(feature_schema),
                "folds": {
                    "rolling": {
                        "manifest": "folds/rolling/fold_manifest.json",
                        "manifest_sha256": sha256_file(fold_manifest_path),
                    }
                },
            }
            write_json_atomic(root / "dataset_manifest.json", root_manifest)
            contexts, columns, _, _ = _validate_task07(
                root,
                folds=["rolling"],
                part_limit=1,
                verify_checksums=True,
            )
            self.assertEqual(columns, FEATURES)
            self.assertEqual(contexts["rolling"].target_users.height, 2)
            root_manifest["schema_sha256"] = "0" * 64
            write_json_atomic(root / "dataset_manifest.json", root_manifest)
            with self.assertRaises(ContractValidationError):
                _validate_task07(
                    root,
                    folds=["rolling"],
                    part_limit=1,
                    verify_checksums=True,
                )

    def test_cpu_fit_persistence_and_deterministic_inference(self) -> None:
        rng = np.random.default_rng(42)
        train_x = rng.normal(size=(512, 2)).astype(np.float32)
        train_y = ((train_x[:, 0] + 0.5 * train_x[:, 1]) > 0).astype(np.uint8)
        eval_x = rng.normal(size=(128, 2)).astype(np.float32)
        eval_y = ((eval_x[:, 0] + 0.5 * eval_x[:, 1]) > 0).astype(np.uint8)
        train_pool = Pool(train_x, label=train_y, feature_names=list(FEATURES))
        eval_pool = Pool(eval_x, label=eval_y, feature_names=list(FEATURES))
        config = CatBoostPointwiseConfig(
            config_id="cpu_test",
            task_type="CPU",
            iterations=30,
            depth=3,
            learning_rate=0.2,
            early_stopping_rounds=5,
            metric_period=5,
            thread_count=2,
            ignored_features=("feature_b",),
            snapshot_interval_seconds=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_loader = (
                CatBoostRankerDataLoader(feature_columns=FEATURES)
                .load_fit_data(train_pool=train_pool, eval_pool=eval_pool)
                .prepare_fit_data()
            )
            model = CatBoostPointwiseModel(config, feature_columns=FEATURES)
            model.fit(
                fit_loader,
                train_dir=root / "train_dir",
                snapshot_file=root / "snapshot.cbsnapshot",
                log_cout=io.StringIO(),
                log_cerr=io.StringIO(),
            )
            artifact = root / "model"
            model.save(artifact)
            restored = CatBoostPointwiseModel.from_artifact(artifact)
            predict_frame = pl.DataFrame(
                {
                    "user_id": pl.Series(np.arange(128), dtype=pl.UInt64),
                    "item_id": pl.Series(np.arange(128), dtype=pl.Int32),
                    "feature_a": pl.Series(eval_x[:, 0], dtype=pl.Float32),
                    "feature_b": pl.Series(eval_x[:, 1], dtype=pl.Float32),
                }
            )
            first_loader = (
                CatBoostRankerDataLoader(feature_columns=FEATURES)
                .load_predict_data(frame=predict_frame)
                .prepare_predict_data()
            )
            second_loader = (
                CatBoostRankerDataLoader(feature_columns=FEATURES)
                .load_predict_data(frame=predict_frame)
                .prepare_predict_data()
            )
            first = model.predict(first_loader, batch_size=31)
            second = restored.predict(second_loader, batch_size=17)
            self.assertTrue(first.equals(second))
            self.assertEqual(model.tree_count, restored.tree_count)
            self.assertEqual(model.best_iteration, restored.best_iteration)
            self.assertEqual(restored.config.ignored_features, ("feature_b",))
            importance = restored.get_feature_importance()
            self.assertEqual(importance.height, len(FEATURES))

    def test_fixed_tree_budget_accepts_one_train_pool_without_eval(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.normal(size=(256, 2)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.uint8)
        pool = Pool(features, label=labels, feature_names=list(FEATURES))
        loader = (
            CatBoostRankerDataLoader(feature_columns=FEATURES)
            .load_fit_data(train_pool=pool)
            .prepare_fit_data()
        )
        model = CatBoostPointwiseModel(
            CatBoostPointwiseConfig(
                config_id="fixed_cpu_test",
                task_type="CPU",
                iterations=7,
                depth=3,
                metric_period=1,
            ),
            feature_columns=FEATURES,
        )
        model.fit(
            loader,
            fixed_tree_budget=True,
            log_cout=io.StringIO(),
            log_cerr=io.StringIO(),
        )
        self.assertEqual(model.tree_count, 7)
        self.assertEqual(model.best_iteration, 6)

        with self.assertRaisesRegex(ContractValidationError, "evaluation pool"):
            CatBoostPointwiseModel(
                CatBoostPointwiseConfig(task_type="CPU", iterations=2),
                feature_columns=FEATURES,
            ).fit(loader)

    def test_predict_rejects_feature_order_mismatch(self) -> None:
        model = CatBoostPointwiseModel(
            CatBoostPointwiseConfig(task_type="CPU", iterations=2),
            feature_columns=FEATURES,
        )
        loader = CatBoostRankerDataLoader(feature_columns=("feature_b", "feature_a"))
        with self.assertRaises(ContractValidationError):
            model.fit(loader)

    def test_prediction_schema_rejects_nulls(self) -> None:
        frame = pl.DataFrame(
            {
                "user_id": pl.Series([1], dtype=pl.UInt64),
                "item_id": pl.Series([2], dtype=pl.Int32),
                "feature_a": pl.Series([None], dtype=pl.Float32),
                "feature_b": pl.Series([1.0], dtype=pl.Float32),
            }
        )
        loader = CatBoostRankerDataLoader(feature_columns=FEATURES).load_predict_data(
            frame=frame
        )
        with self.assertRaises(ContractValidationError):
            loader.prepare_predict_data()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from catboost import Pool
from catboost.utils import quantize

from catboost_ltr import (
    CatBoostLTRError,
    build_dense_group_mapping,
    inherited_stage_challengers,
    prepare_grouped_rows,
    select_complete_eval_users,
    validate_ltr_selection_config,
    write_grouped_dsv_part,
    write_ltr_column_description,
)
from experiment_utils import read_json
from rankers import (
    CatBoostLTRConfig,
    CatBoostLTRDataLoader,
    CatBoostRankerDataLoader,
    CatBoostRankerModel,
    apply_inverse_positive_group_weights,
    validate_pool_groups,
)
from scripts.run_catboost_ltr import _is_gpu_oom_output
from validation import ContractValidationError

FEATURES = ("feature_a", "feature_b")


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": pl.Series(
                [7, 7, 7, 2**63 + 9, 2**63 + 9, 2**63 + 9],
                dtype=pl.UInt64,
            ),
            "item_id": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int32),
            "label": pl.Series([1, 0, 0, 0, 0, 0], dtype=pl.UInt8),
            "is_training_sample": [True, True, False, True, True, True],
            "sampling_probability": pl.Series(
                [1.0, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=pl.Float32
            ),
            "feature_a": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Float32),
            "feature_b": pl.Series([0, 1, 0, 1, 0, 1], dtype=pl.UInt8),
        }
    ).sort(("user_id", "item_id"))


class GroupConstructionTests(unittest.TestCase):
    def test_dense_mapping_preserves_large_uint64_without_float(self) -> None:
        users = _frame().select("user_id").unique()
        mapping = build_dense_group_mapping(users)
        self.assertEqual(mapping.schema["user_id"], pl.UInt64)
        self.assertEqual(mapping.schema["group_id"], pl.Int64)
        self.assertEqual(mapping.get_column("group_id").to_list(), [0, 1])
        self.assertEqual(mapping.get_column("user_id").to_list(), [7, 2**63 + 9])

    def test_grouped_rows_keep_identity_labels_and_training_flag_semantics(
        self,
    ) -> None:
        frame = _frame()
        mapping = build_dense_group_mapping(frame.select("user_id").unique())
        rows, diagnostics = prepare_grouped_rows(
            frame,
            group_mapping=mapping,
            feature_columns=FEATURES,
            training_only=True,
        )
        expected = frame.filter(pl.col("is_training_sample"))
        self.assertEqual(
            rows.get_column("label").to_list(), expected["label"].to_list()
        )
        self.assertEqual(rows.columns, ["label", "group_id", *FEATURES])
        self.assertEqual(diagnostics["rows"], 5)
        self.assertEqual(diagnostics["groups"], 2)
        self.assertEqual(diagnostics["zero_positive_groups"], 1)

    def test_complete_eval_sampling_is_deterministic_and_user_level(self) -> None:
        users = pl.DataFrame({"user_id": pl.Series(np.arange(100), dtype=pl.UInt64)})
        first = select_complete_eval_users(users, count=20, seed=42)
        second = select_complete_eval_users(users, count=20, seed=42)
        other = select_complete_eval_users(users, count=20, seed=43)
        self.assertTrue(first.equals(second))
        self.assertFalse(first.equals(other))
        self.assertEqual(first.height, 20)

    def test_grouped_quantized_pool_persists_contiguous_groups(self) -> None:
        frame = _frame()
        mapping = build_dense_group_mapping(frame.select("user_id").unique())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dsv = root / "pool.tsv"
            cd = root / "columns.cd"
            output = root / "pool.quantized"
            diagnostics = write_grouped_dsv_part(
                frame,
                dsv,
                group_mapping=mapping,
                feature_columns=FEATURES,
                training_only=False,
            )
            write_ltr_column_description(cd, feature_columns=FEATURES)
            pool = quantize(
                data_path=dsv.as_posix(),
                column_description=cd.as_posix(),
                delimiter="\t",
                has_header=False,
                border_count=4,
                task_type="CPU",
                random_seed=42,
            )
            pool.save(output.as_posix())
            restored = Pool(f"quantized://{output.resolve()}")
            group_diagnostics = validate_pool_groups(restored, name="test")
            self.assertEqual(group_diagnostics["groups"], 2)
            self.assertEqual(group_diagnostics["rows"], diagnostics["rows"])
            self.assertEqual(tuple(restored.get_feature_names()), FEATURES)
            self.assertIsNotNone(restored.get_group_id_hash())
            self.assertEqual(
                cd.read_text().splitlines()[:2], ["0\tLabel", "1\tGroupId"]
            )

    def test_inverse_positive_group_weights_are_finite(self) -> None:
        pool = Pool(
            np.arange(12, dtype=np.float32).reshape(6, 2),
            label=[1, 0, 0, 1, 1, 0],
            group_id=[0, 0, 1, 1, 1, 1],
            feature_names=list(FEATURES),
        )
        diagnostics = apply_inverse_positive_group_weights(pool)
        self.assertEqual(diagnostics["groups"], 2)
        self.assertEqual(diagnostics["positive_groups"], 2)
        self.assertAlmostEqual(diagnostics["mean_group_weight"], 1.0)
        self.assertGreater(diagnostics["min_group_weight"], 0)
        self.assertGreaterEqual(
            diagnostics["max_group_weight"], diagnostics["min_group_weight"]
        )

    def test_grouped_rows_reject_duplicates_and_reordering(self) -> None:
        frame = _frame()
        mapping = build_dense_group_mapping(frame.select("user_id").unique())
        with self.assertRaisesRegex(ContractValidationError, "ordered"):
            prepare_grouped_rows(
                frame.reverse(),
                group_mapping=mapping,
                feature_columns=FEATURES,
                training_only=False,
            )
        duplicate = pl.concat([frame, frame.head(1)]).sort(("user_id", "item_id"))
        with self.assertRaisesRegex(ContractValidationError, "duplicate"):
            prepare_grouped_rows(
                duplicate,
                group_mapping=mapping,
                feature_columns=FEATURES,
                training_only=False,
            )


class CatBoostLTRModelTests(unittest.TestCase):
    def test_config_rejects_unsupported_gpu_parameters(self) -> None:
        valid = CatBoostLTRConfig(
            config_id="qsm", loss_function="QuerySoftMax:beta=1", task_type="CPU"
        )
        self.assertEqual(valid.objective, "QuerySoftMax")
        self.assertNotIn("devices", valid.training_params())
        with self.assertRaisesRegex(ValueError, "unsupported"):
            CatBoostLTRConfig(
                config_id="bad",
                loss_function="QuerySoftMax:temperature=1",
            )
        with self.assertRaisesRegex(ValueError, "mode=Classic"):
            CatBoostLTRConfig(
                config_id="bad", loss_function="YetiRankPairwise:mode=NDCG"
            )
        with self.assertRaisesRegex(ValueError, "sampling_unit=Group"):
            CatBoostLTRConfig(
                config_id="bad",
                loss_function="QuerySoftMax:beta=1",
                sampling_unit="Group",
            )
        with self.assertRaisesRegex(ValueError, "random_strength"):
            CatBoostLTRConfig(
                config_id="bad",
                loss_function="QueryCrossEntropy:alpha=0.95",
                random_strength=1.0,
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            CatBoostLTRConfig.from_mapping(
                {
                    "config_id": "bad",
                    "loss_function": "QuerySoftMax:beta=1",
                    "scale_pos_weight": 16,
                }
            )

    def test_cpu_fit_restore_and_deterministic_inference(self) -> None:
        rng = np.random.default_rng(42)
        train_x = rng.normal(size=(120, 2)).astype(np.float32)
        train_group = np.repeat(np.arange(20), 6)
        train_y = np.tile([1, 0, 0, 0, 0, 0], 20)
        train_y[6:12] = 0
        eval_x = rng.normal(size=(60, 2)).astype(np.float32)
        eval_group = np.repeat(np.arange(10), 6)
        eval_y = np.tile([1, 0, 0, 0, 0, 0], 10)
        train = Pool(
            train_x,
            label=train_y,
            group_id=train_group,
            feature_names=list(FEATURES),
        )
        evaluation = Pool(
            eval_x,
            label=eval_y,
            group_id=eval_group,
            feature_names=list(FEATURES),
        )
        config = CatBoostLTRConfig(
            config_id="cpu_qsm",
            loss_function="QuerySoftMax:beta=1",
            task_type="CPU",
            iterations=12,
            depth=3,
            learning_rate=0.2,
            early_stopping_rounds=4,
            metric_period=2,
            thread_count=2,
            snapshot_interval_seconds=1,
            group_weighting="inverse_positive_count",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loader = (
                CatBoostLTRDataLoader(feature_columns=FEATURES)
                .load_fit_data(train_pool=train, eval_pool=evaluation)
                .prepare_fit_data()
            )
            model = CatBoostRankerModel(config, feature_columns=FEATURES)
            model.fit(
                loader,
                train_dir=root / "train_dir",
                snapshot_file=root / "snapshot.cbsnapshot",
                log_cout=io.StringIO(),
                log_cerr=io.StringIO(),
            )
            artifact = root / "model"
            model.save(artifact)
            restored = CatBoostRankerModel.from_artifact(artifact)
            frame = pl.DataFrame(
                {
                    "user_id": pl.Series(np.arange(60), dtype=pl.UInt64),
                    "item_id": pl.Series(np.arange(60), dtype=pl.Int32),
                    "feature_a": eval_x[:, 0],
                    "feature_b": eval_x[:, 1],
                }
            )
            first_loader = (
                CatBoostRankerDataLoader(feature_columns=FEATURES)
                .load_predict_data(frame=frame)
                .prepare_predict_data()
            )
            second_loader = (
                CatBoostRankerDataLoader(feature_columns=FEATURES)
                .load_predict_data(frame=frame)
                .prepare_predict_data()
            )
            first = model.predict(first_loader, batch_size=11)
            second = restored.predict(second_loader, batch_size=13)
            self.assertTrue(first.equals(second))
            self.assertEqual(model.tree_count, restored.tree_count)
            self.assertEqual(
                restored.group_weight_diagnostics["zero_positive_groups"], 1
            )
            self.assertEqual(restored.get_feature_importance().height, 2)


class CatBoostLTRSelectionTests(unittest.TestCase):
    def test_production_config_is_bounded_and_family_specific(self) -> None:
        feature_schema = read_json(
            "artifacts/task07_ranker_dataset_v1/feature_schema.json"
        )
        source = read_json("configs/task10_catboost_ltr_v1.json")
        checked = validate_ltr_selection_config(
            source, feature_columns=feature_schema["feature_columns"]
        )
        self.assertEqual(len(checked["objective_profiles"]), 2)
        self.assertEqual(checked["search"]["max_unique_configs"], 8)
        self.assertEqual(
            checked["objective_exclusions"][0]["reason"],
            "catboost_1_2_10_gpu_max_query_size_256",
        )
        qsm = checked["objective_profiles"][0]
        challengers = inherited_stage_challengers(qsm, checked["inherited_stages"][0])
        self.assertEqual(
            challengers[0]["catboost"]["loss_function"],
            "QuerySoftMax:beta=2",
        )

    def test_recovery_config_is_bounded_and_resource_explicit(self) -> None:
        feature_schema = read_json(
            "artifacts/task07_ranker_dataset_v1/feature_schema.json"
        )
        checked = validate_ltr_selection_config(
            read_json("configs/task10_catboost_ltr_v2.json"),
            feature_columns=feature_schema["feature_columns"],
        )
        recovery = checked["recovery"]
        self.assertEqual(recovery["source_run_id"], "task10_catboost_ltr_v1")
        self.assertEqual(recovery["import_completed_profiles"], ["s10_qsm_beta_1"])
        self.assertEqual(recovery["resource_probe"]["iterations"], 1)
        self.assertEqual(checked["resources"]["windows_host_drive"], "G")
        yeti = checked["objective_profiles"][1]["catboost"]
        self.assertEqual(yeti["gpu_ram_part"], 0.5)
        self.assertIn("permutations=2", yeti["loss_function"])
        self.assertNotIn("permutations=10", str(checked["inherited_stages"]))

    def test_native_gpu_oom_detection_is_specific(self) -> None:
        self.assertTrue(
            _is_gpu_oom_output(
                "NCudaLib::TOutOfMemoryError: cuda_lib/memory_pool: Out of memory"
            )
        )
        self.assertFalse(_is_gpu_oom_output("CatBoostError: invalid group_id"))

    def test_config_rejects_canonical_selection_and_feature_ablation(self) -> None:
        feature_schema = read_json(
            "artifacts/task07_ranker_dataset_v1/feature_schema.json"
        )
        source = read_json("configs/task10_catboost_ltr_v1.json")
        source["fold_pairs"][0]["eval_fold"] = "canonical"
        with self.assertRaisesRegex(CatBoostLTRError, "canonical"):
            validate_ltr_selection_config(
                source, feature_columns=feature_schema["feature_columns"]
            )
        with self.assertRaisesRegex(CatBoostLTRError, "201"):
            validate_ltr_selection_config(
                read_json("configs/task10_catboost_ltr_v1.json"),
                feature_columns=FEATURES,
            )


if __name__ == "__main__":
    unittest.main()

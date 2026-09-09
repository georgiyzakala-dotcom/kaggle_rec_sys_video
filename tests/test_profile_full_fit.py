"""Guard the production refit scope and recovery without reading competition data."""

import copy
import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from experiment_utils import config_sha256, read_json, write_json_atomic
from full_history_profiles import validate_recipe, validate_training_horizon
from history_profiles import PROFILE_FEATURES
from pipeline import TRAINING_FOLDS
from scripts.run_profile_full_fit import append_experiment, publish_only
from scripts.run_ranker_backtest import _file_manifest


class FinalRecipeTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(Path("configs/task14_full_fit_v1.json"))
        self.reference = {
            "quantization": copy.deepcopy(self.config["quantization"]),
            "training": {"target_rows": 30_000_000},
        }
        self.model = {
            "catboost": copy.deepcopy(self.config["catboost"]),
            "tree_count": 1030,
            "feature_columns": ["base_feature", *PROFILE_FEATURES],
        }

    def test_frozen_recipe_accepts_new_id_and_operational_limits(self):
        self.config["catboost"].update(
            config_id="new_full_fit", thread_count=4, gpu_ram_part=0.6
        )
        validate_recipe(self.config, self.reference, self.model)

    def test_rejects_missing_last_day_or_changed_statistics(self):
        for key, value in (
            ("depth", 8),
            ("iterations", 1000),
            ("scale_pos_weight", 16),
            ("ignored_features", [201]),
        ):
            c = copy.deepcopy(self.config)
            c["catboost"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(c, self.reference, self.model)
        c = copy.deepcopy(self.config)
        c["training"]["fold_order"].remove("canonical")
        with self.assertRaisesRegex(ValueError, "all four"):
            validate_recipe(c, self.reference, self.model)

    def test_rejects_frozen_borders_and_unbudgeted_training(self):
        c = copy.deepcopy(self.config)
        c["quantization"]["policies"] = ["frozen_task08"]
        with self.assertRaisesRegex(ValueError, "train-only"):
            validate_recipe(c, self.reference, self.model)
        c = copy.deepcopy(self.config)
        c["training"]["target_rows"] = 46_000_000
        with self.assertRaisesRegex(ValueError, "row budget"):
            validate_recipe(c, self.reference, self.model)

    def test_label_horizon_and_chronology(self):
        last = datetime.fromisoformat("2024-12-02T08:56:28")
        inputs = {
            fold: {"cutoff": (last - timedelta(days=3 - i)).isoformat()}
            for i, fold in enumerate(TRAINING_FOLDS)
        }
        validate_training_horizon(inputs, last + timedelta(days=1, microseconds=1))
        with self.assertRaisesRegex(ValueError, "prediction period"):
            validate_training_horizon(inputs, last + timedelta(hours=23))
        inputs["rolling_2"]["cutoff"] = inputs["rolling_1"]["cutoff"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_training_horizon(inputs, last + timedelta(days=1))


class PublicationRecoveryTests(unittest.TestCase):
    def test_full_fit_log_is_idempotent_and_does_not_invent_quality_metrics(self):
        source = read_json(Path("configs/task14_full_fit_v1.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            log = root / "experiments/results.csv"
            log.write_text(
                "run_id,p20_all_targets,p20_labeled_users,runtime,peak_memory,artifact_path\n"
            )
            cwd = Path.cwd()
            try:
                os.chdir(root)
                metrics = {
                    "coverage": 1.0,
                    "runtime_seconds": 123,
                    "peak_memory_mb": 456,
                }
                append_experiment(source, metrics, Path("artifacts/full_test"))
                append_experiment(source, metrics, Path("artifacts/full_test"))
                with log.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["p20_all_targets"], "")
                self.assertEqual(rows[0]["p20_labeled_users"], "")
                self.assertEqual(rows[0]["peak_memory"], "456")
                self.assertEqual(rows[0]["runtime"], "123")
            finally:
                os.chdir(cwd)

    def staged(self, root):
        source = {"run_id": "recovery_smoke", "mode": "smoke"}
        work = root / "artifacts/.recovery_smoke.work"
        artifact = work / "publication/artifact"
        artifact.mkdir(parents=True)
        config = {
            **source,
            "source_config_sha256": config_sha256(source),
            "implementation_sha256": {"original_training.py": "original-hash"},
        }
        for destination in (work, artifact):
            write_json_atomic(destination / "config.json", config)
        write_json_atomic(artifact / "metrics.json", {"mode": "smoke"})
        write_json_atomic(
            work / "publication/operation.json",
            {
                "config_sha256": config_sha256(config),
                "files": _file_manifest(work / "publication"),
            },
        )
        return source, work

    def test_publish_only_preserves_training_hashes_and_never_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, work = self.staged(root)
            original = (work / "config.json").read_bytes()
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch(
                        "scripts.run_profile_full_fit.load_config", return_value=source
                    ),
                    patch(
                        "scripts.run_profile_full_fit.verify_artifact",
                        return_value={"status": "verified"},
                    ),
                    patch(
                        "scripts.run_profile_full_fit.FullProfileRun.fit_final_ranker",
                        side_effect=AssertionError("must not fit"),
                    ),
                    patch(
                        "scripts.run_profile_full_fit.fit_candidate_source",
                        side_effect=AssertionError("must not refit"),
                    ),
                ):
                    result = publish_only(Path("unused.json"))
                    self.assertEqual(result["status"], "verified")
                    self.assertEqual(
                        Path("artifacts/recovery_smoke/config.json").read_bytes(),
                        original,
                    )
                    with self.assertRaises(FileExistsError):
                        publish_only(Path("unused.json"))
            finally:
                os.chdir(cwd)

    def test_incomplete_or_corrupt_staging_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, work = self.staged(root)
            (work / "publication/artifact/metrics.json").write_text("{}")
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch(
                        "scripts.run_profile_full_fit.load_config", return_value=source
                    ),
                    self.assertRaisesRegex(ValueError, "checksum mismatch"),
                ):
                    publish_only(Path("unused.json"))
                self.assertFalse(Path("artifacts/recovery_smoke").exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()

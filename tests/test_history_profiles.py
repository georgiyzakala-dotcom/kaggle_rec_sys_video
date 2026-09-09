import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from experiment_utils import config_sha256, sha256_file, write_json_atomic
from history_profiles import (
    PROFILE_FEATURES,
    HistoryProfiles,
    history_pairs,
    ignored_features,
    normalize_factors,
)
from metrics import evaluate_precision_at_20
from scripts.run_history_profiles import (
    load_config,
    publish_only,
    verify_precision_summary,
)
from scripts.run_ranker_backtest import _file_manifest
from scripts.task13_resources import supervise, violation


class HistoryProfileTests(unittest.TestCase):
    def setUp(self):
        self.uid = 2**63 + 123
        self.users = pl.DataFrame(
            {"user_id": [self.uid, self.uid + 1]}, schema={"user_id": pl.UInt64}
        )
        self.mapping = pl.DataFrame(
            {"item_id": [1, 2, 3], "item_index": [0, 1, 2]},
            schema={"item_id": pl.Int32, "item_index": pl.UInt32},
        )
        self.factors = normalize_factors(
            np.array([[2, 0], [0, 3], [0, 0]], dtype=np.float32)
        )
        self.daily = pl.DataFrame(
            {
                "user_id": [self.uid] * 4,
                "item_id": [1, 1, 2, 3],
                "is_positive": [1, 1, 1, 0],
                "is_like": [1, 1, 0, 0],
                "is_favorite": [0, 0, 1, 0],
                "watch_time": [60, 60, 61, 0],
            },
            schema={
                "user_id": pl.UInt64,
                "item_id": pl.Int32,
                "is_positive": pl.Int32,
                "is_like": pl.Int32,
                "is_favorite": pl.Int32,
                "watch_time": pl.Int64,
            },
        )

    def build(self):
        pairs = history_pairs(self.daily.lazy(), self.users)
        return HistoryProfiles.build(
            pairs, self.users, self.mapping, self.factors, batch_size=1
        )

    def test_relevance_boundary_and_unique_items(self):
        pairs = history_pairs(self.daily.lazy(), self.users)
        self.assertEqual(pairs.height, 2)
        model = self.build()
        np.testing.assert_array_equal(model.counts[:, 0], [2, 1, 1, 1])
        np.testing.assert_allclose(model.concentrations[0, 0], np.sqrt(0.5), rtol=1e-6)
        np.testing.assert_allclose(
            model.directions[0, 0], [np.sqrt(0.5)] * 2, rtol=1e-6
        )

    def test_missing_profiles_and_embeddings_are_explicit(self):
        model = self.build()
        candidates = pl.DataFrame(
            {
                "user_id": [self.uid, self.uid, self.uid + 1, self.uid + 2],
                "item_id": [1, 999, 1, 1],
            },
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        features = model.features(candidates, batch_size=2)
        self.assertEqual(features.schema["user_id"], pl.UInt64)
        self.assertEqual(features["user_id"].to_list(), candidates["user_id"].to_list())
        self.assertEqual(features["history_positive_available"].to_list(), [1, 0, 0, 0])
        self.assertEqual(features["history_positive_cosine"].to_list()[1:], [0, 0, 0])
        self.assertTrue(np.isfinite(features.select(PROFILE_FEATURES).to_numpy()).all())

    def test_orthogonal_rotation_invariance_and_roundtrip(self):
        base = self.build()
        rotation = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=np.float32)
        rotated = HistoryProfiles.build(
            history_pairs(self.daily.lazy(), self.users),
            self.users,
            self.mapping,
            self.factors @ rotation,
        )
        pairs = self.daily.select("user_id", "item_id").unique().sort("item_id")
        expected = base.features(pairs)
        np.testing.assert_allclose(
            expected.select(PROFILE_FEATURES).to_numpy(),
            rotated.features(pairs).select(PROFILE_FEATURES).to_numpy(),
            atol=1e-6,
        )
        with tempfile.TemporaryDirectory() as folder:
            base.save(Path(folder))
            restored = HistoryProfiles.load(Path(folder))
            self.assertTrue(expected.equals(restored.features(pairs)))

    def test_unmapped_positive_counts_without_false_direction(self):
        pairs = (
            history_pairs(self.daily.lazy(), self.users)
            .with_columns(pl.lit(999, dtype=pl.Int32).alias("item_id"))
            .head(1)
        )
        model = HistoryProfiles.build(pairs, self.users, self.mapping, self.factors)
        self.assertEqual(model.counts[0, 0], 1)
        self.assertEqual(model.matched[0, 0], 0)
        self.assertEqual(model.concentrations[0, 0], 0)

    def test_duplicate_membership_rejected(self):
        pairs = history_pairs(self.daily.lazy(), self.users)
        with self.assertRaisesRegex(ValueError, "unique"):
            HistoryProfiles.build(
                pl.concat([pairs, pairs]), self.users, self.mapping, self.factors
            )

    def test_zero_and_nonfinite_factors(self):
        self.assertTrue(np.isfinite(normalize_factors(np.zeros((2, 3)))).all())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_factors(np.array([[np.nan, 1]]))

    def test_feature_ablations_are_disjoint_and_exhaustive(self):
        self.assertEqual(len(ignored_features("A_base")), 20)
        self.assertEqual(len(ignored_features("B_positive")), 15)
        self.assertEqual(len(ignored_features("C_events")), 5)
        self.assertEqual(ignored_features("D_all"), [])
        self.assertEqual(
            set(ignored_features("B_positive")) & set(ignored_features("C_events")),
            set(),
        )

    def test_resource_limits(self):
        config = load_config(Path("configs/task13_history_profiles_v1.json"))
        limits = config["resources"]
        values = {
            "rss_gib": 10,
            "available_ram_gib": 42,
            "free_disk_gib": 400,
            "windows_free_gib": 180,
            "vram_free_mib": 14000,
        }
        self.assertIsNone(violation(values, limits, initial=True))
        self.assertIn("rss_gib", violation(dict(values, rss_gib=41), limits))
        self.assertIn(
            "available_ram", violation(dict(values, available_ram_gib=3), limits)
        )
        self.assertIn(
            "windows_free",
            violation(dict(values, windows_free_gib=109), limits, initial=True),
        )
        self.assertIn("vram", violation(dict(values, vram_free_mib=1000), limits))

    def test_smoke_cannot_silently_become_a_long_experiment(self):
        config = load_config(Path("configs/task13_history_profiles_smoke_v1.json"))
        config["selection"]["fixed_tree_count"] = 1030
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "config.json"
            p.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "bounded"):
                load_config(p)

    def test_supervisor_terminates_worker_on_memory_excess(self):
        config = load_config(Path("configs/task13_history_profiles_smoke_v1.json"))
        config["run_id"] = "test_resource_guard"
        normal = {
            "rss_gib": 0,
            "available_ram_gib": 42,
            "free_disk_gib": 400,
            "windows_free_gib": 180,
        }
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                with (
                    patch(
                        "scripts.task13_resources.resources",
                        side_effect=[normal, dict(normal, rss_gib=100)],
                    ),
                    self.assertRaisesRegex(RuntimeError, "rss_gib"),
                ):
                    supervise(
                        [sys.executable, "-c", "import time; time.sleep(30)"], config
                    )
                status = json.loads(
                    Path("logs/test_resource_guard.resources.json").read_text()
                )
                self.assertEqual(status["status"], "stopped")
                self.assertEqual(status["peak_rss_gib"], 100)
                self.assertNotIn(
                    "\x1b", Path("logs/test_resource_guard.resources.log").read_text()
                )
            finally:
                os.chdir(previous)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.users = pl.DataFrame({"user_id": [1, 2, 3]}, schema={"user_id": pl.UInt64})
        self.recs = self.users.with_columns(
            pl.Series(
                "item_ids",
                [list(range(1, 21)), list(range(21, 41)), list(range(41, 61))],
                dtype=pl.List(pl.Int32),
            )
        )
        self.truth = pl.DataFrame(
            {"user_id": [1, 1, 3], "item_id": [1, 2, 45]},
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        self.saved = {
            **evaluate_precision_at_20(self.recs, self.truth, self.users),
            "final_hits": 3,
            "target_users": 3,
            "labeled_users": 2,
        }

    def test_verifier_accepts_last_bit_rounding_with_exact_counts(self):
        drift = dict(self.saved)
        for key in ("precision_at_20_all_targets", "precision_at_20_labeled_users"):
            drift[key] = math.nextafter(drift[key], math.inf)
        result = verify_precision_summary(drift, self.recs, self.truth, self.users)
        self.assertEqual(result["final_hits"], 3)

    def test_verifier_rejects_changed_hit_count_or_denominator(self):
        for key in ("final_hits", "target_users", "labeled_users"):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "stored count differs"),
            ):
                verify_precision_summary(
                    dict(self.saved, **{key: self.saved[key] + 1}),
                    self.recs,
                    self.truth,
                    self.users,
                )

    def test_verifier_rejects_wrong_metric_and_nonfinite_values(self):
        for key in ("precision_at_20_all_targets", "precision_at_20_labeled_users"):
            for bad in (self.saved[key] + 1 / (20 * 200152), math.nan, math.inf):
                with (
                    self.subTest(key=key, bad=bad),
                    self.assertRaisesRegex(ValueError, "metric differs"),
                ):
                    verify_precision_summary(
                        dict(self.saved, **{key: bad}),
                        self.recs,
                        self.truth,
                        self.users,
                    )

    def test_verifier_handles_empty_ground_truth(self):
        truth = self.truth.clear()
        saved = {
            "final_hits": 0,
            "target_users": 3,
            "labeled_users": 0,
            "precision_at_20_all_targets": 0.0,
            "precision_at_20_labeled_users": 0.0,
        }
        self.assertEqual(
            verify_precision_summary(saved, self.recs, truth, self.users)["final_hits"],
            0,
        )

    def prepare_publication(self, source):
        source["run_id"] = "publication_test"
        config = dict(
            source, implementation_sha256={"old_runner": "original_training_hash"}
        )
        digest = config_sha256(config)
        staging = Path("artifacts/.publication_test.publish")
        work = Path("artifacts/.publication_test.work")
        staging.mkdir(parents=True)
        write_json_atomic(Path("source.json"), source)
        write_json_atomic(work / "config.json", config)
        write_json_atomic(staging / "config.json", config)
        comparisons = {"A_base": self.saved}
        metrics = dict(
            self.saved,
            winner="A_base",
            run_id="publication_test",
            selection_comparisons=comparisons,
            canonical_matched_control=self.saved,
        )
        write_json_atomic(staging / "metrics.json", metrics)
        (staging / "model").mkdir()
        (staging / "model/model.cbm").write_bytes(b"synthetic portable model")
        for name in ("recommendations.parquet", "control_recommendations.parquet"):
            self.recs.write_parquet(staging / name)
        records = {}

        def operation(key, relative, result):
            directory = work / relative
            manifest = {
                "config_sha256": digest,
                "files": _file_manifest(directory),
                "result": result,
            }
            write_json_atomic(directory / "operation.json", manifest)
            records[key] = {
                "path": relative,
                "manifest_sha256": sha256_file(directory / "operation.json"),
            }

        write_json_atomic(
            work / "frozen_selection/selection.json",
            {"winner": "A_base", "comparisons": comparisons},
        )
        operation(
            "selection::winner::rolling_3", "frozen_selection", {"winner": "A_base"}
        )
        model_path = work / "models/canonical/A_base/model/model.cbm"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes((staging / "model/model.cbm").read_bytes())
        operation("fit::A_base::canonical", "models/canonical/A_base", {})
        result_path = work / "evaluation/canonical/A_base/result"
        result_path.mkdir(parents=True)
        self.recs.write_parquet(result_path / "recommendations.parquet")
        write_json_atomic(result_path / "metrics.json", self.saved)
        operation(
            "validate_output::A_base::canonical",
            "evaluation/canonical/A_base/result",
            self.saved,
        )
        write_json_atomic(
            work / "checkpoint.json",
            {
                "config_sha256": digest,
                "run_id": "publication_test",
                "completed": records,
            },
        )
        write_json_atomic(
            staging / "artifact_manifest.json",
            {"kind": "task13_history_profiles", "files": _file_manifest(staging)},
        )
        return staging

    def test_publication_recovery_never_enters_training_and_preserves_provenance(self):
        source = load_config(Path("configs/task13_history_profiles_smoke_v1.json"))
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                staging = self.prepare_publication(source)
                original = (staging / "config.json").read_bytes()
                with (
                    patch(
                        "scripts.run_history_profiles.verify_artifact",
                        return_value=self.saved,
                    ),
                    patch(
                        "scripts.run_history_profiles.ProfileRun",
                        side_effect=AssertionError("training is forbidden"),
                    ),
                ):
                    result = publish_only(Path("source.json"), show_progress=False)
                output = Path(result["artifact"])
                self.assertFalse(staging.exists())
                self.assertEqual((output / "config.json").read_bytes(), original)
                recovery = json.loads(
                    (output / "publication_recovery.json").read_text()
                )
                self.assertFalse(recovery["training_started"])
                self.assertFalse(recovery["predictions_regenerated"])
                with self.assertRaises(FileExistsError):
                    publish_only(Path("source.json"), show_progress=False)
            finally:
                os.chdir(previous)

    def test_recovery_rejects_corrupted_staged_model(self):
        source = load_config(Path("configs/task13_history_profiles_smoke_v1.json"))
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                staging = self.prepare_publication(source)
                (staging / "model/model.cbm").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    publish_only(Path("source.json"), show_progress=False)
                self.assertFalse(Path("artifacts/publication_test").exists())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

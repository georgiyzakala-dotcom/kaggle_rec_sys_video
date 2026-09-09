from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch
from test_sasrec import synthetic_fold

from experiment_utils import read_json
from scripts.run_sasrec_benchmark import (
    BenchmarkPaused,
    load_config,
    resolve_config,
    run_worker,
    verify_artifact,
)
from scripts.task15_resources import violation

ROOT = Path(__file__).resolve().parents[1]


class SASRecBenchmarkTests(unittest.TestCase):
    def test_config_rejects_silent_cpu_fallback_and_unbounded_smoke(self):
        config = load_config(ROOT / "configs/task15_sasrec_benchmark_v1.json")
        config["device"] = "cpu"
        with tempfile.TemporaryDirectory() as directory:
            import json

            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)
            config = load_config(ROOT / "configs/task15_sasrec_cpu_smoke_v1.json")
            config["training"]["max_users"] = 999999
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_resource_limits_include_windows_disk_and_rss(self):
        limits = load_config(ROOT / "configs/task15_sasrec_benchmark_v1.json")[
            "resources"
        ]
        values = {
            "available_ram_gib": 30,
            "free_disk_gib": 100,
            "windows_free_gib": 80,
            "rss_gib": 10,
            "vram_free_mib": 12000,
            "run_disk_gib": 1,
        }
        self.assertIsNone(violation(values, limits, initial=True))
        for key, value in (
            ("windows_free_gib", 49),
            ("rss_gib", 41),
            ("run_disk_gib", 13),
            ("vram_free_mib", 1000),
        ):
            changed = dict(values, **{key: value})
            self.assertIsNotNone(violation(changed, limits))

    def test_end_to_end_resume_matches_independent_cpu_fit(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold, _ = synthetic_fold(root)
            config = load_config(ROOT / "configs/task15_sasrec_cpu_smoke_v1.json")
            config["run_id"] = "sasrec_test_resume"
            config["data"]["fold_artifact"] = str(fold)
            config["training"]["batch_size"] = 4
            config["training"]["max_users"] = 8
            config["retrieval_probe"]["users"] = 4
            try:
                os.chdir(root)
                resolved = resolve_config(config)
                with self.assertRaises(BenchmarkPaused):
                    run_worker(resolved, show_progress=False, stop_after_epoch=1)
                self.assertEqual(
                    read_json(
                        root / "artifacts/.sasrec_test_resume.work/checkpoint.json"
                    )["epoch"],
                    1,
                )
                with self.assertRaises(ValueError):
                    changed = copy.deepcopy(resolved)
                    changed["training"]["negative_count"] = 99
                    run_worker(changed, show_progress=False)
                output = run_worker(resolved, show_progress=False)
                self.assertTrue(verify_artifact(output)["verified"])
                repeat = copy.deepcopy(config)
                repeat["run_id"] = "sasrec_test_repeat"
                other = run_worker(resolve_config(repeat), show_progress=False)
                left = torch.load(output / "model/weights.pt", weights_only=True)
                right = torch.load(other / "model/weights.pt", weights_only=True)
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]), key)
                self.assertTrue(
                    pl.read_parquet(output / "probe_candidates.parquet").equals(
                        pl.read_parquet(other / "probe_candidates.parquet")
                    )
                )
                with self.assertRaises(FileExistsError):
                    run_worker(resolved, show_progress=False)
                self.assertNotIn(
                    "\x1b", (root / "logs/sasrec_test_resume.log").read_text()
                )
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

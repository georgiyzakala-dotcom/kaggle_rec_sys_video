from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl
import test_sasrec_optuna as optuna_fixtures
import torch

from experiment_utils import read_json, sha256_file, write_json_atomic
from implicit_model import ImplicitALSConfig, ImplicitALSDataLoader, ImplicitALSModel
from metrics import evaluate_candidate_metrics, evaluate_precision_at_20
from sasrec_evaluation import POLICY_CAPS, evaluate_shard, summarize
from sasrec_model import SASRecCandidateModel
from sasrec_selection import SOURCES, SelectionEvaluator
from scripts.run_sasrec_benchmark import BenchmarkPaused
from scripts.run_sasrec_folds import (
    FoldsRunner,
    load_config,
    metrics_equal,
    resolve_config,
    verify_artifact,
)
from scripts.run_sasrec_optuna import SearchRunner
from scripts.run_sasrec_optuna import resolve_config as resolve_selection

ROOT = Path(__file__).resolve().parents[1]


def fold_fixture(root):
    selection_config = optuna_fixtures.fixture(root)
    data = Path(selection_config["data"]["source_fold"])
    fold = Path(selection_config["data"]["fold_artifact"])
    from datetime import datetime

    cutoff = datetime.fromisoformat(read_json(data / "dataset_manifest.json")["cutoff"])
    als_dir = data / "sources" / "implicit_als"
    config = ImplicitALSConfig(
        config_id="tiny", factors=8, iterations=2, num_threads=1, seed=42
    )
    loader = ImplicitALSDataLoader(config=config, reference_time=cutoff)
    loader.load_fit_data(history=fold / "history_daily.parquet").prepare_fit_data()
    als = ImplicitALSModel(config).fit(loader, show_progress=False)
    als.save(als_dir / "model")
    users, truth = (
        pl.read_parquet(fold / "target_users.parquet"),
        pl.read_parquet(fold / "target_ground_truth.parquet"),
    )
    loader.load_predict_data(
        history=fold / "history_daily.parquet", target_users=users
    ).prepare_predict_data()
    frame = als.predict(loader, k=200, batch_size=4)
    frame.write_parquet(als_dir / "candidates.parquet")
    meta = read_json(als_dir / "metadata.json")
    meta.update(
        model_path="model",
        model_storage="embedded",
        candidates_sha256=sha256_file(als_dir / "candidates.parquet"),
        candidate_metrics=evaluate_candidate_metrics(frame, truth, users),
    )
    write_json_atomic(als_dir / "metadata.json", meta)
    manifest = read_json(data / "dataset_manifest.json")
    union = pl.concat(
        [
            pl.read_parquet(data / "sources" / s / "candidates.parquet").select(
                "user_id", "item_id"
            )
            for s in SOURCES
        ]
    ).unique()
    manifest["metrics"] = evaluate_candidate_metrics(union, truth, users)
    write_json_atomic(data / "dataset_manifest.json", manifest)
    selection_config["run_id"] = "tiny_selection"
    selection_config["search"]["trials"] = 1
    selection = SearchRunner(
        resolve_selection(selection_config), show_progress=False
    ).run()
    source_dataset = root / "sources_dataset"
    for name in ("rolling_1", "canonical"):
        d = source_dataset / "folds" / name
        shutil.copytree(data, d)
        m = read_json(d / "dataset_manifest.json")
        m["fold"] = name
        write_json_atomic(d / "dataset_manifest.json", m)
        for s in SOURCES:
            p = d / "sources" / s / "metadata.json"
            m = read_json(p)
            m["fold"] = name
            write_json_atomic(p, m)
    c = load_config(ROOT / "configs/task15_sasrec_folds_cpu_smoke_v1.json")
    c.update(
        run_id="tiny_folds",
        selection_artifact=str(selection),
        source_dataset=str(source_dataset),
        selection_recipe_sha256=sha256_file(selection / "best_recipe.json"),
        selection_manifest_sha256=sha256_file(selection / "manifest.json"),
        folds=[{"name": n, "artifact": str(fold)} for n in ("rolling_1", "canonical")],
    )
    c["smoke"].update(context_users=8, training_users=8, evaluation_users=8)
    c["evaluation"]["users_per_shard"] = 3
    return c


class FoldMetricTests(unittest.TestCase):
    def test_metric_verifier_tolerates_ulp_but_not_count_or_quality_changes(self):
        metric = {"nested": {"mean": 0.003125, "hits": 1775}}
        ulp = copy.deepcopy(metric)
        ulp["nested"]["mean"] = math.nextafter(metric["nested"]["mean"], 0.0)
        self.assertTrue(metrics_equal(metric, ulp))
        ulp["nested"]["hits"] += 1
        self.assertFalse(metrics_equal(metric, ulp))
        changed = copy.deepcopy(metric)
        changed["nested"]["mean"] += 1e-8
        self.assertFalse(metrics_equal(metric, changed))

    def setUp(self):
        fixture = optuna_fixtures.SelectionMetricTests()
        fixture.setUp()
        self.f = fixture

    def test_sharded_counts_equal_existing_metrics_with_empty_truth_shard(self):
        f = self.f
        als400 = f.sources["implicit_als"]
        stats, recs = evaluate_shard(
            f.users, f.truth, f.sources, f.native, als400, f.store
        )
        actual = summarize(stats, "blend_150_50")
        expected = SelectionEvaluator(f.users, f.truth, f.sources, f.store).evaluate(
            f.native
        )
        self.assertEqual(
            actual["candidate_recall"], expected["native_sasrec"]["candidate_recall"]
        )
        self.assertEqual(
            actual["frozen_policy_delta_recall"],
            expected["policies"]["blend_150_50"]["delta_recall"],
        )
        for key in expected["native_sasrec"]:
            self.assertEqual(
                actual["sources"]["sasrec"]["200"][key], expected["native_sasrec"][key]
            )
        for name in SOURCES:
            for key in expected["overlap_sasrec_vs_sources"][name]:
                self.assertEqual(
                    actual["overlap_sasrec_vs_sources"][name][key],
                    expected["overlap_sasrec_vs_sources"][name][key],
                )
        for key, value in evaluate_precision_at_20(
            recs["sasrec"], f.truth, f.users
        ).items():
            self.assertEqual(actual[key], value)
        shards = []
        for i in range(3):
            users = f.users.slice(i, 1)
            sub = lambda frame, users=users: frame.join(users, on="user_id", how="semi")
            stat, _ = evaluate_shard(
                users,
                sub(f.truth),
                {s: sub(v) for s, v in f.sources.items()},
                sub(f.native),
                sub(als400),
                f.store,
            )
            shards.append(stat)
        self.assertEqual(summarize(pl.concat(shards), "blend_150_50"), actual)
        self.assertFalse(actual["policy_selected_on_this_fold"])
        self.assertEqual(set(actual["policies"]), set(POLICY_CAPS))

    def test_changed_als_prefix_is_rejected(self):
        f = self.f
        changed = (
            f.sources["implicit_als"]
            .filter(pl.col("rank") > 1)
            .with_columns((pl.col("rank") - 1).alias("rank"))
        )
        with self.assertRaisesRegex(ValueError, "ALS400 top200"):
            evaluate_shard(f.users, f.truth, f.sources, f.native, changed, f.store)

    def test_duplicate_universe_and_seen_pairs_are_rejected(self):
        f = self.f
        with self.assertRaises(ValueError):
            evaluate_shard(
                pl.concat([f.users, f.users]),
                f.truth,
                f.sources,
                f.native,
                f.sources["implicit_als"],
                f.store,
            )
        bad = f.native.with_columns(pl.lit(1, dtype=pl.Int32).alias("item_id")).head(1)
        with self.assertRaisesRegex(ValueError, "seen"):
            evaluate_shard(
                f.users, f.truth, f.sources, bad, f.sources["implicit_als"], f.store
            )


class FoldRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cwd = Path.cwd()
        os.chdir(self.root)
        torch.set_num_threads(1)
        self.c = fold_fixture(self.root)

    def tearDown(self):
        os.chdir(self.cwd)
        self.temp.cleanup()

    def test_two_fold_full_path_and_deterministic_repeat(self):
        with patch.object(
            ImplicitALSModel, "fit", side_effect=AssertionError("ALS must not refit")
        ):
            result = FoldsRunner(resolve_config(self.c), show_progress=False).run()
        self.assertTrue(verify_artifact(result)["verified"])
        c = copy.deepcopy(self.c)
        c["run_id"] = "tiny_repeat"
        repeat = FoldsRunner(resolve_config(c), show_progress=False).run()
        for name in ("rolling_1", "canonical"):
            left, right = result / "folds" / name, repeat / "folds" / name
            a = torch.load(left / "trained/model/weights.pt", weights_only=True)
            b = torch.load(right / "trained/model/weights.pt", weights_only=True)
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            self.assertTrue(
                pl.read_parquet(left / "per_user.parquet").equals(
                    pl.read_parquet(right / "per_user.parquet")
                )
            )
            self.assertFalse(
                read_json(left / "metrics.json")["policy_selected_on_this_fold"]
            )
        self.assertFalse(Path("experiments/results.csv").exists())
        with self.assertRaises(FileExistsError):
            FoldsRunner(resolve_config(self.c), show_progress=False)

    def test_epoch_and_shard_resume_without_retraining(self):
        c = copy.deepcopy(self.c)
        c["folds"] = c["folds"][:1]
        with self.assertRaises(BenchmarkPaused):
            FoldsRunner(
                resolve_config(c),
                show_progress=False,
                stop_after_epoch=("rolling_1", 1),
            ).run()
        bad = copy.deepcopy(c)
        bad["evaluation"]["users_per_shard"] = 4
        with self.assertRaisesRegex(ValueError, "changed"):
            FoldsRunner(resolve_config(bad), show_progress=False)
        with self.assertRaises(BenchmarkPaused):
            FoldsRunner(
                resolve_config(c),
                show_progress=False,
                stop_after_shard=("rolling_1", 1),
            ).run()
        work = Path("artifacts/.tiny_folds.work/folds/rolling_1")
        before = sha256_file(work / "evaluation/part-00000/manifest.json")
        with patch.object(
            SASRecCandidateModel,
            "fit",
            side_effect=AssertionError("completed model must not refit"),
        ):
            result = FoldsRunner(resolve_config(c), show_progress=False).run()
        self.assertEqual(
            before,
            sha256_file(result / "folds/rolling_1/evaluation/part-00000/manifest.json"),
        )
        c["run_id"] = "uninterrupted"
        repeat = FoldsRunner(resolve_config(c), show_progress=False).run()
        self.assertTrue(
            pl.read_parquet(result / "folds/rolling_1/per_user.parquet").equals(
                pl.read_parquet(repeat / "folds/rolling_1/per_user.parquet")
            )
        )

    def test_frozen_selection_and_full_user_guards(self):
        bad = copy.deepcopy(self.c)
        bad["selection_recipe_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            resolve_config(bad)
        full = read_json(ROOT / "configs/task15_sasrec_folds_v1.json")
        full["folds"] = full["folds"][:1]
        path = self.root / "bad.json"
        write_json_atomic(path, full)
        with self.assertRaisesRegex(ValueError, "four ordered folds"):
            load_config(path)


if __name__ == "__main__":
    unittest.main()

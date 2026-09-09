from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl
import test_sasrec_folds as fold_fixtures
import test_sasrec_optuna as selection_fixtures

from experiment_utils import sha256_file
from implicit_model import ImplicitALSModel
from sasrec_evaluation import evaluate_shard
from sasrec_model import SASRecCandidateModel
from sasrec_top600 import PRIMARY, evaluate_deep_shard, summarize_deep
from scripts.run_sasrec_benchmark import BenchmarkPaused
from scripts.run_sasrec_folds import FoldsRunner
from scripts.run_sasrec_folds import resolve_config as resolve_folds
from scripts.run_sasrec_top600 import (
    Top600Runner,
    load_config,
    resolve_config,
    verify_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


class DeepMetricsTests(unittest.TestCase):
    def setUp(self):
        self.f = selection_fixtures.SelectionMetricTests()
        self.f.setUp()
        f = self.f
        user = f.users["user_id"][0]
        self.sas600 = selection_fixtures.candidates(
            "sasrec", {user: [*range(600, 1001), *range(401, 600)]}
        )
        self.als600 = selection_fixtures.candidates(
            "implicit_als",
            {user: [*range(2, 202), *range(801, 1001), *range(401, 601)]},
        )
        self.als400 = self.als600.filter(pl.col("rank") <= 400)
        self.truth = pl.concat(
            [
                f.truth,
                pl.DataFrame(
                    {"user_id": [user, user], "item_id": [850, 575]},
                    schema=f.truth.schema,
                ),
            ]
        )
        self.old, _ = evaluate_shard(
            f.users, self.truth, f.sources, f.native, self.als400, f.store
        )

    def evaluate(self, users=None, als=None, sas=None):
        f = self.f
        users = f.users if users is None else users
        sub = lambda x: x.join(users, on="user_id", how="semi")
        return evaluate_deep_shard(
            users,
            sub(self.truth),
            {s: sub(v) for s, v in f.sources.items()},
            sub(f.native),
            sub(self.als400),
            sub(self.sas600 if sas is None else sas),
            sub(self.als600 if als is None else als),
            f.store,
            sub(self.old),
        )

    def test_deep_unions_overlap_and_shard_aggregation(self):
        m = summarize_deep(self.evaluate(), "blend_150_50")
        p = m["policies"]
        self.assertEqual(p[PRIMARY]["candidate_recall"], 1.0)
        self.assertEqual(p[PRIMARY]["positive_hits"], 6)
        self.assertEqual(p[PRIMARY]["sum_source_caps"], 1800)
        self.assertEqual(p["als300_sasrec300_1200"]["candidate_recall"], 5 / 6)
        self.assertEqual(p["als600_only_1200"]["candidate_recall"], 1.0)
        self.assertAlmostEqual(m["equal_budget1200_delta_both300_minus_als600"], -1 / 6)
        self.assertEqual(m["overlap_at_depth"]["600"]["shared_positive_hits"], 3)
        self.assertIsNone(m["precision_at_20_all_targets"])
        self.assertIsNone(m["precision_at_20_labeled_users"])
        shards = [self.evaluate(self.f.users.slice(i, 1)) for i in range(3)]
        self.assertEqual(summarize_deep(pl.concat(shards), "blend_150_50"), m)
        self.assertEqual(m["target_users"], 3)
        self.assertEqual(m["labeled_users"], 2)

    def test_changed_prefix_and_seen_tail_fail(self):
        bad = self.als600.with_columns(
            pl.when(pl.col("rank") == 400)
            .then(777)
            .otherwise(pl.col("item_id"))
            .cast(pl.Int32)
            .alias("item_id")
        )
        with self.assertRaisesRegex(ValueError, "top400 differs"):
            self.evaluate(als=bad)
        bad = self.sas600.with_columns(
            pl.when(pl.col("rank") == 600)
            .then(1)
            .otherwise(pl.col("item_id"))
            .cast(pl.Int32)
            .alias("item_id")
        )
        with self.assertRaisesRegex(ValueError, "seen"):
            self.evaluate(sas=bad)


class DeepRunnerTests(unittest.TestCase):
    def test_end_to_end_resume_no_fit_and_exact_repeat(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                parent = FoldsRunner(
                    resolve_folds(fold_fixtures.fold_fixture(Path(temp))),
                    show_progress=False,
                ).run()
                c = load_config(ROOT / "configs/task15_sasrec_top600_gpu_smoke_v1.json")
                c.update(
                    parent_artifact=str(parent),
                    parent_manifest_sha256=sha256_file(parent / "manifest.json"),
                    run_id="top600_test",
                    device="cpu",
                    cpu_threads=1,
                )
                c["smoke"].update(shards_per_fold=2, users_per_shard=3)
                with (
                    patch.object(
                        SASRecCandidateModel,
                        "fit",
                        side_effect=AssertionError("SASRec must not fit"),
                    ),
                    patch.object(
                        ImplicitALSModel,
                        "fit",
                        side_effect=AssertionError("ALS must not fit"),
                    ),
                ):
                    with self.assertRaises(BenchmarkPaused):
                        Top600Runner(
                            resolve_config(c),
                            show_progress=False,
                            stop_after_shard=("rolling_1", 1),
                        ).run()
                    part = Path(
                        "artifacts/.top600_test.work/folds/rolling_1/parts/part-00000/manifest.json"
                    )
                    digest = sha256_file(part)
                    output = Top600Runner(resolve_config(c), show_progress=False).run()
                    self.assertEqual(
                        digest,
                        sha256_file(
                            output / "folds/rolling_1/parts/part-00000/manifest.json"
                        ),
                    )
                    self.assertTrue(verify_artifact(output)["verified"])
                    repeat = copy.deepcopy(c)
                    repeat["run_id"] = "top600_repeat"
                    second = Top600Runner(
                        resolve_config(repeat), show_progress=False
                    ).run()
                for name in ("rolling_1", "canonical"):
                    for file in (
                        "per_user.parquet",
                        "sasrec600.parquet",
                        "als600.parquet",
                    ):
                        rel = Path("folds") / name / "parts/part-00000" / file
                        self.assertTrue(
                            pl.read_parquet(output / rel).equals(
                                pl.read_parquet(second / rel)
                            )
                        )
                self.assertFalse(Path("experiments/results.csv").exists())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

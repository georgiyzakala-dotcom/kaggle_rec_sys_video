from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import test_sasrec_folds as fold_fixtures
import test_sasrec_optuna as selection_fixtures
import torch
from test_sasrec import tiny_store

from experiment_utils import sha256_file
from sasrec_data import SASRecDataLoader
from sasrec_evaluation import evaluate_shard
from sasrec_model import SASRecCandidateModel, SASRecConfig
from sasrec_top300 import (
    PRIMARY,
    extend_shard,
    predict_vectorized_seen,
    summarize_extension,
)
from scripts.run_sasrec_benchmark import BenchmarkPaused
from scripts.run_sasrec_folds import FoldsRunner
from scripts.run_sasrec_folds import resolve_config as resolve_folds
from scripts.run_sasrec_top300 import (
    Top300Runner,
    load_config,
    resolve_config,
    verify_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


class VectorizedSeenTests(unittest.TestCase):
    def test_identical_retrieval_with_ties_empty_queries_and_chunk_boundaries(self):
        store = tiny_store()
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = SASRecCandidateModel(
            store.item_ids, SASRecConfig(embedding_dim=8, max_length=5, num_heads=2)
        )
        loader = SASRecDataLoader(store, max_length=5)
        users = np.concatenate((store.user_ids, np.array([1], dtype=np.uint64)))
        for zero_weights in (False, True):
            if zero_weights:
                with torch.no_grad():
                    model.encoder.item_embedding.weight.zero_()
            loader.load_predict_data(user_ids=users).prepare_predict_data()
            reference = model.predict(loader, k=300, batch_size=2, item_chunk_size=7)
            actual = predict_vectorized_seen(
                model, loader, k=300, batch_size=2, item_chunk_size=7
            )
            self.assertTrue(actual.equals(reference))
        calls = []
        loader.load_predict_data(
            user_ids=np.array([1], dtype=np.uint64)
        ).prepare_predict_data()
        empty = predict_vectorized_seen(model, loader, callback=calls.append)
        self.assertTrue(empty.is_empty())
        self.assertEqual(calls, [1])


class Top300MetricsTests(unittest.TestCase):
    def setUp(self):
        f = selection_fixtures.SelectionMetricTests()
        f.setUp()
        self.f = f
        user = f.users["user_id"][0]
        self.truth = pl.concat(
            [
                f.truth,
                pl.DataFrame(
                    {"user_id": [user, user], "item_id": [850, 950]},
                    schema=f.truth.schema,
                ),
            ]
        )
        self.sas300 = selection_fixtures.candidates("sasrec", {user: range(600, 900)})
        self.als400 = selection_fixtures.candidates(
            "implicit_als",
            {user: [*range(2, 202), *range(901, 1001), *range(801, 901)]},
        )

    def test_unique_union_and_marginal_contributions_with_exact_old_audit(self):
        f = self.f
        old, _ = evaluate_shard(
            f.users, self.truth, f.sources, f.native, self.als400, f.store
        )
        stats = extend_shard(
            f.users,
            self.truth,
            f.sources,
            f.native,
            self.als400,
            self.sas300,
            f.store,
            old,
        )
        m = summarize_extension(stats, "blend_150_50")
        p = m["policies"]
        self.assertEqual(p[PRIMARY]["candidate_recall"], 1.0)
        self.assertEqual(p[PRIMARY]["positive_hits"], 6)
        self.assertEqual(p[PRIMARY]["delta_positive_hits_vs_both200"], 2)
        self.assertEqual(p[PRIMARY]["sum_source_caps"], 1200)
        self.assertEqual(p["als300_sasrec200_1100"]["candidate_recall"], 5 / 6)
        self.assertEqual(p["als200_sasrec300_1100"]["candidate_recall"], 5 / 6)
        self.assertIsNone(m["precision_at_20_all_targets"])
        self.assertIsNone(m["precision_at_20_labeled_users"])
        shards = []
        for i in range(3):
            users = f.users.slice(i, 1)
            sub = lambda frame, users=users: frame.join(users, on="user_id", how="semi")
            shards.append(
                extend_shard(
                    users,
                    sub(self.truth),
                    {s: sub(v) for s, v in f.sources.items()},
                    sub(f.native),
                    sub(self.als400),
                    sub(self.sas300),
                    f.store,
                    sub(old),
                )
            )
        self.assertEqual(summarize_extension(pl.concat(shards), "blend_150_50"), m)

    def test_mismatched_prefix_or_old_counts_fail(self):
        f = self.f
        bad = self.sas300.with_columns(
            pl.when(pl.col("rank") == 1)
            .then(550)
            .otherwise(pl.col("item_id"))
            .cast(pl.Int32)
            .alias("item_id")
        )
        with self.assertRaisesRegex(ValueError, "top200 differs"):
            extend_shard(
                f.users, self.truth, f.sources, f.native, self.als400, bad, f.store
            )
        old, _ = evaluate_shard(
            f.users, self.truth, f.sources, f.native, self.als400, f.store
        )
        old = old.with_columns(
            (pl.col("m__sasrec_at_200__hits") + 1).alias("m__sasrec_at_200__hits")
        )
        with self.assertRaisesRegex(ValueError, "recomputation differs"):
            extend_shard(
                f.users,
                self.truth,
                f.sources,
                f.native,
                self.als400,
                self.sas300,
                f.store,
                old,
            )


class Top300RunnerTests(unittest.TestCase):
    def test_end_to_end_resume_no_fit_and_repeat(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                root = Path(temp)
                parent_config = fold_fixtures.fold_fixture(root)
                parent = FoldsRunner(
                    resolve_folds(parent_config), show_progress=False
                ).run()
                c = load_config(ROOT / "configs/task15_sasrec_top300_gpu_smoke_v1.json")
                c.update(
                    parent_artifact=str(parent),
                    parent_manifest_sha256=sha256_file(parent / "manifest.json"),
                    run_id="top300_test",
                    device="cpu",
                    cpu_threads=1,
                )
                c["smoke"].update(shards_per_fold=2, users_per_shard=3)
                with patch.object(
                    SASRecCandidateModel,
                    "fit",
                    side_effect=AssertionError("extension must never fit"),
                ):
                    with self.assertRaises(BenchmarkPaused):
                        Top300Runner(
                            resolve_config(c),
                            show_progress=False,
                            stop_after_shard=("rolling_1", 1),
                        ).run()
                    part = Path(
                        "artifacts/.top300_test.work/folds/rolling_1/parts/part-00000/manifest.json"
                    )
                    digest = sha256_file(part)
                    output = Top300Runner(resolve_config(c), show_progress=False).run()
                    self.assertEqual(
                        digest,
                        sha256_file(
                            output / "folds/rolling_1/parts/part-00000/manifest.json"
                        ),
                    )
                    self.assertTrue(verify_artifact(output)["verified"])
                    repeat = copy.deepcopy(c)
                    repeat["run_id"] = "top300_repeat"
                    second = Top300Runner(
                        resolve_config(repeat), show_progress=False
                    ).run()
                for f in ("rolling_1", "canonical"):
                    self.assertTrue(
                        pl.read_parquet(
                            output / "folds" / f / "per_user.parquet"
                        ).equals(
                            pl.read_parquet(second / "folds" / f / "per_user.parquet")
                        )
                    )
                self.assertFalse(Path("experiments/results.csv").exists())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

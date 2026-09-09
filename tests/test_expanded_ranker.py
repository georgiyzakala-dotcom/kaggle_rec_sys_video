"""Expanded candidates: full-query ranks, unbiased sampling and temporal safety."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
import torch
from test_sasrec import synthetic_fold, tiny_store

from expanded_ranker import (
    CAPS,
    PHASE_FOLDS,
    attach_sasrec_scores,
    sample_union,
    source_union,
    validate_phase_horizon,
)
from experiment_utils import sha256_file
from features import attach_union_score_ranks
from interfaces import CANDIDATE_SCHEMA, FINAL_RECOMMENDATION_SCHEMA
from metrics import evaluate_precision_at_20
from sasrec_data import SASRecDataLoader, prepare_sequence_store
from sasrec_model import SASRecCandidateModel, SASRecConfig
from sasrec_selection import validate_unseen
from scripts.run_expanded_ranker import ExpandedRun, load_config
from submission import read_submission, write_submission


class ExpandedFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_cross_scores_for_non_native_pairs_and_empty_histories(self):
        store = tiny_store()
        model = SASRecCandidateModel(
            store.item_ids,
            SASRecConfig(
                embedding_dim=8, num_blocks=1, num_heads=2, max_length=5, dropout=0
            ),
            history_sha256="synthetic",
        )
        users = store.user_ids.tolist()
        pairs = pl.DataFrame(
            {
                "user_id": [users[0], users[0], users[1], users[2], 1],
                "item_id": [10, 999, 11, 10, 10],
            },
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        result = attach_sasrec_scores(pairs, model, store, batch_size=1)
        self.assertEqual(
            result["cross_score_available_sasrec"].to_list(),
            [True, False, True, False, False],
        )
        self.assertEqual(result["user_id"].to_list(), pairs["user_id"].to_list())
        loader = SASRecDataLoader(store, max_length=5)
        loader.load_predict_data(
            user_ids=np.array(users[:1], dtype=np.uint64)
        ).prepare_predict_data()
        batch = next(loader.iter_predict_batches(batch_size=1))
        with torch.inference_mode():
            q = model.encoder.encode_users(batch.inputs.long())
            expected = model.score_pairs(
                q, torch.tensor([np.searchsorted(store.item_ids, 10) + 1])
            ).item()
        self.assertAlmostEqual(result["cross_score_sasrec"][0], expected, places=6)
        self.assertTrue(
            result.equals(attach_sasrec_scores(pairs, model, store, batch_size=2))
        )
        unavailable = result.filter(~pl.col("cross_score_available_sasrec"))
        self.assertEqual(unavailable["sasrec_cosine"].sum(), 0.0)

    def test_sampling_keeps_positives_and_full_query_ranks(self):
        user = 2**64 - 7
        frame = pl.DataFrame(
            {
                "user_id": [user] * 1000,
                "item_id": range(1000),
                "cross_score_sasrec": np.arange(1000, dtype=np.float64),
                "cross_score_available_sasrec": [True] * 1000,
            },
            schema_overrides={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        full = attach_union_score_ranks(frame, sources=("sasrec",))
        truth = frame.filter(pl.col("item_id").is_in([2, 7, 50])).select(
            "user_id", "item_id"
        )
        sampled = sample_union(
            full, truth, fold="rolling_1", seed=42, negative_probability=0.1
        )
        self.assertEqual(sampled["label"].sum(), 3)
        self.assertGreater(sampled.height, 50)
        self.assertLess(sampled.height, 180)
        self.assertTrue(
            sampled.equals(
                sample_union(
                    full, truth, fold="rolling_1", seed=42, negative_probability=0.1
                )
            )
        )
        positive = sampled.filter(pl.col("label") == 1)
        self.assertEqual(positive["sample_weight"].to_list(), [1.0] * 3)
        self.assertEqual(
            sampled.filter(pl.col("label") == 0)["sample_weight"].unique().to_list(),
            [10.0],
        )
        joined = sampled.join(full, on=["user_id", "item_id"], suffix="_expected")
        self.assertTrue(
            (
                joined["union_rank_cross_score_sasrec"]
                == joined["union_rank_cross_score_sasrec_expected"]
            ).all()
        )
        # A sampled positive keeps its rank among all 1000 candidates.
        self.assertEqual(
            positive.filter(pl.col("item_id") == 2)[
                "union_rank_cross_score_sasrec"
            ].item(),
            998,
        )

    def test_deep_caps_deduplicate_across_sources(self):
        sources = {}
        for source, cap in CAPS.items():
            start = 0 if source != "sasrec" else 400
            sources[source] = pl.DataFrame(
                [
                    (2**63 + 7, i, -float(r), r, source)
                    for r, i in enumerate(range(start, start + cap), 1)
                ],
                schema=CANDIDATE_SCHEMA,
                orient="row",
            )
        union = source_union(sources)
        self.assertEqual(union.height, 1000)
        self.assertEqual(union["generated_by_sasrec"].sum(), 600)
        self.assertEqual(union["generated_by_implicit_als"].sum(), 600)
        self.assertEqual(
            union.filter(pl.col("item_id") == 500)["source_count"].item(), 2
        )

    def test_relevance_seen_dedup_precision_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold, cutoff = synthetic_fold(root)
            truth = pl.read_parquet(fold / "target_ground_truth.parquet")
            users = pl.read_parquet(fold / "target_users.parquet").sort("user_id")
            self.assertEqual(truth["item_id"].to_list(), [20])
            store = prepare_sequence_store(
                fold / "history_daily.parquet",
                root / "store",
                cutoff=cutoff,
                expected_sha256=sha256_file(fold / "history_daily.parquet"),
            )
            loader = SASRecDataLoader(store, max_length=5)
            loader.load_predict_data(
                user_ids=users["user_id"].to_numpy()
            ).prepare_predict_data()
            rows = []
            for batch in loader.iter_predict_batches(batch_size=8):
                for u, seen in zip(batch.user_ids, batch.seen_items, strict=True):
                    eligible = [
                        int(item)
                        for j, item in enumerate(store.item_ids, 1)
                        if j not in set(seen)
                    ]
                    # Relevant item is present in the first user's list.
                    if int(u) == int(users["user_id"][0]):
                        eligible = [20] + [i for i in eligible if i != 20]
                    rows.append((int(u), eligible[:20]))
            recs = pl.DataFrame(
                rows, schema=FINAL_RECOMMENDATION_SCHEMA, orient="row"
            ).sort("user_id")
            validate_unseen(
                recs.explode("item_ids", empty_as_null=True).rename(
                    {"item_ids": "item_id"}
                ),
                users,
                store,
            )
            metrics = evaluate_precision_at_20(recs, truth, users)
            self.assertEqual(metrics["precision_at_20_all_targets"], 1 / 160)
            self.assertEqual(metrics["precision_at_20_labeled_users"], 1 / 20)
            write_submission(recs, root / "submission.csv")
            self.assertTrue(recs.equals(read_submission(root / "submission.csv")))


class ExpandedRunSafetyTests(unittest.TestCase):
    def test_label_windows_precede_evaluation_and_future(self):
        start = datetime.fromisoformat("2024-11-29T08:56:28")
        specs = {
            f: {"cutoff": (start + timedelta(days=i)).isoformat()}
            for i, f in enumerate(PHASE_FOLDS["final"])
        }
        for phase in PHASE_FOLDS:
            validate_phase_horizon(
                phase, specs, start + timedelta(days=4, microseconds=1)
            )
        specs["rolling_2"]["cutoff"] = (
            start + timedelta(days=1, seconds=1)
        ).isoformat()
        with self.assertRaisesRegex(ValueError, "horizon"):
            validate_phase_horizon(
                "rolling_validation", specs, start + timedelta(days=4)
            )

    def test_completed_phase_does_not_rebuild_deleted_training_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = SimpleNamespace(
                work=Path(tmp),
                operation=lambda *args: {"phase": "rolling_validation"},
                cleanup_phase=lambda p: None,
                release=lambda: None,
            )
            with patch.object(
                ExpandedRun, "samples", side_effect=AssertionError("must not resample")
            ):
                self.assertEqual(
                    ExpandedRun.ranker_phase(run, "rolling_validation"),
                    {"phase": "rolling_validation"},
                )

    def test_full_config_is_gpu_and_smoke_is_bounded(self):
        c = load_config(Path("configs/task15_sasrec_ranker_v1.json"))
        self.assertEqual(c["catboost"]["task_type"], "GPU")
        self.assertEqual(c["training"]["target_rows"], 30_000_000)
        self.assertLessEqual(c["resources"]["maximum_rss_gib"], 40)


if __name__ == "__main__":
    unittest.main()

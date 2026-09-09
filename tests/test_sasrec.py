from __future__ import annotations

# The shared timeline is timezone-naive by repository contract.
# ruff: noqa: DTZ001
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from data_utils import RAW_INTERACTION_SCHEMA, TARGET_USER_SCHEMA, prepare_temporal_fold
from experiment_utils import read_json, sha256_file
from interfaces import FINAL_RECOMMENDATION_SCHEMA
from metrics import evaluate_precision_at_20
from sasrec_data import (
    SASRecDataLoader,
    SequenceStore,
    prepare_sequence_store,
    sample_unseen,
)
from sasrec_model import SASRecCandidateModel, SASRecConfig, SASRecEncoder, stable_topk


def tiny_store():
    # Daily positive item 1 repeats; weak item 4 must also be excluded.
    return SequenceStore(
        user_ids=np.array([2**53 + 17, 2**63 + 3, 2**64 - 5], dtype=np.uint64),
        item_ids=np.arange(-3, 57, dtype=np.int32),
        positive_items=np.array([1, 2, 1, 3, 2, 5], dtype=np.uint32),
        positive_offsets=np.array([0, 4, 6, 6], dtype=np.int64),
        seen_items=np.array([1, 2, 3, 4, 2, 5, 6], dtype=np.uint32),
        seen_offsets=np.array([0, 4, 6, 7], dtype=np.int64),
        metadata={"history_sha256": "synthetic"},
    )


def synthetic_fold(directory: Path):
    cutoff = datetime(2024, 1, 3, 8)
    users = [2**63 + 101 + i for i in range(8)]
    rows = []
    for u, user in enumerate(users):
        for j in range(8):
            rows.append(
                (
                    user,
                    (u * 7 + j) % 60,
                    "watch_time",
                    61 if j < 6 else 60,
                    cutoff - timedelta(days=1, minutes=8 - j),
                )
            )
    # Relevance, dedup, seen exclusion, and exact watch_time >60 boundary.
    rows.extend(
        [
            (users[0], 0, "like", 0, cutoff),  # seen
            (users[0], 20, "watch_time", 61, cutoff),
            (users[0], 20, "favorite", 0, cutoff + timedelta(seconds=1)),
            (users[0], 21, "watch_time", 60, cutoff),
            (users[0], 999, "like", 0, cutoff),  # cold
            (users[1], 20, "watch_time", 60, cutoff + timedelta(days=1)),
        ]
    )
    pl.DataFrame(rows, schema=RAW_INTERACTION_SCHEMA, orient="row").write_parquet(
        directory / "raw.parquet"
    )
    pl.DataFrame({"user_id": users}, schema=TARGET_USER_SCHEMA).write_parquet(
        directory / "targets.parquet"
    )
    fold = directory / "fold"
    prepare_temporal_fold(
        train_path=directory / "raw.parquet",
        target_users_path=directory / "targets.parquet",
        output_dir=fold,
        cutoff=cutoff,
        validation_end_exclusive=cutoff + timedelta(days=1),
        run_id="synthetic_sasrec_fold",
    )
    return fold, cutoff


class SASRecEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = SASRecConfig(
            embedding_dim=8, num_heads=2, num_blocks=2, max_length=5, dropout=0
        )
        self.model = SASRecEncoder(60, self.config)

    def test_padding_states_and_last_valid(self):
        self.model.eval()
        seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        with torch.inference_mode():
            outputs = self.model(seq)
            last = self.model.encode_users(seq)
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(
            torch.equal(outputs[seq == 0], torch.zeros_like(outputs[seq == 0]))
        )
        torch.testing.assert_close(last, torch.stack([outputs[0, 2], outputs[1, 1]]))

    def test_future_tokens_cannot_change_prefix_train_or_eval(self):
        for training in (True, False):
            self.model.train(training)
            with torch.no_grad():
                left = self.model(torch.tensor([[1, 2, 3, 4, 5]]))
                right = self.model(torch.tensor([[1, 2, 9, 8, 7]]))
            torch.testing.assert_close(left[:, :2], right[:, :2], rtol=0, atol=0)

    def test_right_padding_is_equivalent_to_short_input(self):
        self.model.eval()
        with torch.no_grad():
            full = self.model.encode_users(torch.tensor([[1, 2, 0, 0, 0]]))
            short = self.model.encode_users(torch.tensor([[1, 2]]))
        torch.testing.assert_close(full, short)

    def test_invalid_sequences_are_rejected(self):
        for seq in ([[0, 0]], [[0, 1]], [[1, 0, 2]], [[61]], [[-1]], [[1] * 6]):
            with self.assertRaises(ValueError):
                self.model(torch.tensor(seq))
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 2))

    def test_sampled_loss_backward_and_padding_gradient(self):
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]])
        targets = torch.tensor([[2, 3, 0], [4, 5, 6]])
        negatives = torch.tensor([[10, 11]] * 5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        loss = self.model.sampled_loss(inputs, targets, negatives, query_chunk_size=2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in self.model.parameters()
                if p.grad is not None
            )
        )
        self.assertTrue(
            torch.equal(self.model.item_embedding.weight.grad[0], torch.zeros(8))
        )
        optimizer.step()
        self.assertTrue(
            torch.equal(self.model.item_embedding.weight[0], torch.zeros(8))
        )

    def test_loss_matches_direct_sampled_formula(self):
        inputs = torch.tensor([[1, 2, 0]])
        targets = torch.tensor([[2, 3, 0]])
        negatives = torch.tensor([[8, 9], [9, 10]])
        states = self.model(inputs)[targets > 0]
        positive = (states * self.model.item_embedding(targets[targets > 0])).sum(-1)
        negative = (states[:, None] * self.model.item_embedding(negatives)).sum(-1)
        expected = (
            torch.nn.functional.softplus(-positive)
            + torch.nn.functional.softplus(negative).mean(-1)
        ).mean()
        torch.testing.assert_close(
            self.model.sampled_loss(inputs, targets, negatives, query_chunk_size=1),
            expected,
        )

    def test_topk_resolves_boundary_ties_by_item_id(self):
        scores = torch.tensor(
            [[2.0, 1.0, 2.0, 2.0, -torch.inf], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        ids = torch.tensor([[9, 3, 7, 1, 5], [9, 3, 7, 1, 5]])
        values, selected = stable_topk(scores, ids, 2)
        self.assertEqual(selected.tolist(), [[1, 7], [1, 3]])
        self.assertEqual(values.tolist(), [[2.0, 2.0], [0.0, 0.0]])

    def test_config_rejects_incompatible_heads(self):
        with self.assertRaises(ValueError):
            SASRecConfig(embedding_dim=7, num_heads=2)


class SASRecDataTests(unittest.TestCase):
    def test_unseen_sampler_matches_complement_for_dense_and_empty_histories(self):
        for seen in ([], [1, 3], [2, 4, 5], [1, 2, 3, 4]):
            seen_array = np.array(seen, dtype=np.uint32)
            actual = sample_unseen(np.random.default_rng(42), seen_array, 5, 20000)
            expected = set(range(1, 6)) - set(seen)
            self.assertEqual(set(actual.tolist()), expected)
            counts = np.array([(actual == value).sum() for value in expected])
            self.assertLess(float(counts.max() / counts.min()), 1.1)
        with self.assertRaises(ValueError):
            sample_unseen(np.random.default_rng(42), np.arange(1, 6), 5, 1)

    def test_epoch_windows_negatives_repeat_and_large_ids(self):
        store = tiny_store()
        loader = SASRecDataLoader(store, max_length=3)
        loader.load_fit_data().prepare_fit_data(epoch=1, negative_count=4)
        batch = next(loader.iter_fit_batches(batch_size=10))
        self.assertEqual(batch.negatives.shape[0], int((batch.targets > 0).sum()))
        before = [v.clone() for v in (batch.inputs, batch.targets, batch.negatives)]
        loader.load_fit_data().prepare_fit_data(epoch=1, negative_count=4)
        repeat = next(loader.iter_fit_batches(batch_size=10))
        for left, right in zip(
            before, (repeat.inputs, repeat.targets, repeat.negatives), strict=True
        ):
            self.assertTrue(torch.equal(left, right))
        loader.load_predict_data(
            user_ids=np.array([2**64 - 5, 2**63 + 3, 2**53 + 17, 7], dtype=np.uint64)
        ).prepare_predict_data()
        batch = next(loader.iter_predict_batches(batch_size=10))
        self.assertEqual(batch.user_ids.dtype, np.uint64)
        self.assertEqual(
            batch.inputs.tolist(), [[0, 0, 0], [2, 1, 3], [2, 5, 0], [0, 0, 0]]
        )
        self.assertIn(4, batch.seen_items[1])

    def test_epoch_allocation_limit(self):
        loader = SASRecDataLoader(tiny_store(), max_length=3)
        with self.assertRaises(MemoryError):
            loader.load_fit_data().prepare_fit_data(
                epoch=1, negative_count=64, memory_budget_bytes=1
            )

    def test_shared_fold_store_and_relevance_metric_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold, cutoff = synthetic_fold(root)
            path = fold / "history_daily.parquet"
            digest = sha256_file(path)
            store = prepare_sequence_store(
                path, root / "sequences", cutoff=cutoff, expected_sha256=digest
            )
            self.assertEqual(store.user_ids.dtype, np.uint64)
            self.assertEqual(store.item_ids.dtype, np.int32)
            self.assertNotIn(999, store.item_ids)
            gt = pl.read_parquet(fold / "target_ground_truth.parquet")
            self.assertEqual(gt["item_id"].to_list(), [20])
            users = pl.read_parquet(fold / "target_users.parquet")
            output = pl.DataFrame(
                [(int(u), [20]) for u in users["user_id"]],
                schema=FINAL_RECOMMENDATION_SCHEMA,
                orient="row",
            )
            metrics = evaluate_precision_at_20(output, gt, users)
            self.assertEqual(metrics["precision_at_20_all_targets"], 1 / (20 * 8))
            self.assertEqual(metrics["precision_at_20_labeled_users"], 1 / 20)
            self.assertEqual(
                read_json(root / "sequences/manifest.json")["history_sha256"], digest
            )
            with self.assertRaises(FileExistsError):
                prepare_sequence_store(
                    path, root / "sequences", cutoff=cutoff, expected_sha256=digest
                )
            with self.assertRaises(ValueError):
                prepare_sequence_store(
                    path,
                    root / "bad",
                    cutoff=cutoff - timedelta(days=2),
                    expected_sha256=digest,
                )


class SASRecRetrievalTests(unittest.TestCase):
    def test_model_rejects_other_fold_or_item_mapping(self):
        store = tiny_store()
        model = SASRecCandidateModel(
            store.item_ids, SASRecConfig(embedding_dim=8, num_heads=2, max_length=3)
        )
        loader = SASRecDataLoader(store, max_length=3)
        loader.load_predict_data(
            user_ids=store.user_ids[:1].copy()
        ).prepare_predict_data()
        model.predict(loader, k=3)
        store.metadata = {"history_sha256": "different_fold"}
        with self.assertRaises(ValueError):
            model.predict(loader, k=3)
        store.metadata = {"history_sha256": "synthetic"}
        store.item_ids = store.item_ids + 1
        with self.assertRaises(ValueError):
            model.predict(loader, k=3)

    def test_chunked_retrieval_matches_bruteforce_and_portable_scores(self):
        torch.manual_seed(42)
        store = tiny_store()
        config = SASRecConfig(embedding_dim=8, num_heads=2, max_length=3, dropout=0)
        model = SASRecCandidateModel(store.item_ids, config)
        loader = SASRecDataLoader(store, max_length=3)
        loader.load_predict_data(user_ids=store.user_ids.copy()).prepare_predict_data()
        candidates = model.predict(loader, k=20, item_chunk_size=7, batch_size=2)
        self.assertEqual(candidates.height, 40)
        batch = next(loader.iter_predict_batches(batch_size=2))
        with torch.inference_mode():
            queries = model.encoder.encode_users(batch.inputs.long())
            scores = (queries @ model.encoder.item_embedding.weight[1:].T).numpy()
        for row in range(2):
            scores[row, store.seen_for(row) - 1] = -np.inf
            expected = store.item_ids[np.lexsort((store.item_ids, -scores[row]))[:20]]
            actual = candidates.filter(pl.col("user_id") == int(store.user_ids[row]))[
                "item_id"
            ].to_numpy()
            np.testing.assert_array_equal(actual, expected)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model"
            model.save(artifact)
            restored = SASRecCandidateModel.from_artifact(artifact)
            self.assertTrue(
                candidates.equals(
                    restored.predict(loader, k=20, item_chunk_size=13, batch_size=2)
                )
            )
            with torch.inference_mode():
                item_indices = torch.tensor([8, 9])
                expected = torch.tensor([scores[0, 7], scores[1, 8]])
                torch.testing.assert_close(
                    restored.score_pairs(queries, item_indices), expected
                )

    def test_tied_scores_and_insufficient_unseen_items(self):
        store = tiny_store()
        model = SASRecCandidateModel(
            store.item_ids,
            SASRecConfig(embedding_dim=8, num_heads=2, max_length=3, dropout=0),
        )
        with torch.no_grad():
            model.encoder.item_embedding.weight.zero_()
        loader = SASRecDataLoader(store, max_length=3)
        loader.load_predict_data(
            user_ids=store.user_ids[:1].copy()
        ).prepare_predict_data()
        result = model.predict(loader, k=100, item_chunk_size=7)
        self.assertEqual(result.height, len(store.item_ids) - 4)
        self.assertEqual(result["item_id"].to_list(), store.item_ids[4:].tolist())


if __name__ == "__main__":
    unittest.main()

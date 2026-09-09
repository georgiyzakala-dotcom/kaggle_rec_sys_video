"""History-only daily sequences and deterministic, bounded SASRec batches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from data_utils import DAILY_INTERACTION_SCHEMA
from experiment_utils import read_json, sha256_file, write_json_atomic
from interfaces import CandidateDataLoader

Progress = Callable[[int, int], None]
ARRAY_NAMES = (
    "user_ids",
    "item_ids",
    "positive_items",
    "positive_offsets",
    "seen_items",
    "seen_offsets",
)


@dataclass
class SequenceStore:
    user_ids: np.ndarray
    item_ids: np.ndarray
    positive_items: np.ndarray
    positive_offsets: np.ndarray
    seen_items: np.ndarray
    seen_offsets: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, directory: Path, *, verify: bool = True) -> SequenceStore:
        metadata = read_json(directory / "manifest.json")
        arrays = {}
        for name in ARRAY_NAMES:
            path = directory / f"{name}.npy"
            if verify and sha256_file(path) != metadata["sha256"][path.name]:
                raise ValueError(f"sequence cache checksum mismatch: {path}")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        return cls(**arrays, metadata=metadata)

    @property
    def eligible_users(self) -> np.ndarray:
        lengths = np.diff(self.positive_offsets)
        has_negative = np.diff(self.seen_offsets) < len(self.item_ids)
        return np.flatnonzero((lengths >= 2) & has_negative)

    def seen_for(self, user_index: int) -> np.ndarray:
        start, end = self.seen_offsets[user_index : user_index + 2]
        return self.seen_items[int(start) : int(end)]


def _offsets(indices: np.ndarray, user_count: int) -> np.ndarray:
    counts = np.bincount(indices.astype(np.int64), minlength=user_count)
    return np.concatenate((np.zeros(1, dtype=np.int64), counts.cumsum()))


def prepare_sequence_store(
    history_path: Path,
    destination: Path,
    *,
    cutoff: datetime,
    expected_sha256: str,
    context_user_limit: int | None = None,
    preserve_full_catalog: bool = True,
    seed: int = 42,
    event: Callable[..., None] | None = None,
) -> SequenceStore:
    """Create a new store; caller publishes the enclosing run atomically.

    Smoke can restrict context users while retaining the true history vocabulary.
    Validation/ground truth is never a preparation input.
    """
    if destination.exists():
        raise FileExistsError(destination)
    if sha256_file(history_path) != expected_sha256:
        raise ValueError("history checksum differs from immutable fold manifest")
    history = pl.scan_parquet(history_path)
    if history.collect_schema() != DAILY_INTERACTION_SCHEMA:
        raise ValueError("SASRec requires the shared daily interaction schema")
    maximum = history.select(pl.col("dt").max()).collect().item()
    if maximum is None or maximum >= cutoff:
        raise ValueError("history is empty or crosses the fold cutoff")
    if event:
        event("sequence_history_verified", operation="history_hash_and_cutoff")
    users = history.select("user_id").unique().sort("user_id").collect()
    if context_user_limit is not None:
        if context_user_limit <= 0:
            raise ValueError("context_user_limit must be positive")
        users = users.with_columns(pl.col("user_id").hash(seed=seed).alias("sample"))
        users = (
            users.sort("sample", "user_id")
            .head(context_user_limit)
            .drop("sample")
            .sort("user_id")
        )
    selected = history.join(users.lazy(), on="user_id", how="semi")
    catalog = history if preserve_full_catalog else selected
    items = catalog.select("item_id").unique().sort("item_id").collect()
    if items.height < 2 or items.height >= np.iinfo(np.int32).max:
        raise ValueError("invalid history catalog size")
    user_mapping = users.with_row_index("user_index")
    item_mapping = items.with_row_index("item_index", offset=1)
    if event:
        event("sequence_mappings", users=users.height, items=items.height)
    indexed = (
        selected.select("user_id", "item_id", "dt", "is_positive")
        .join(user_mapping.lazy(), on="user_id", how="inner")
        .join(item_mapping.lazy(), on="item_id", how="inner")
    )
    positive = (
        indexed.filter(pl.col("is_positive") == 1)
        .sort("user_index", "dt", "item_id")
        .select("user_index", "item_index")
        .collect(engine="streaming")
    )
    positive_offsets = _offsets(positive["user_index"].to_numpy(), users.height)
    positive_items = positive["item_index"].to_numpy().copy()
    del positive
    if event:
        event("sequence_positives", positive_daily_rows=len(positive_items))
    seen = (
        indexed.select("user_index", "item_index")
        .unique()
        .sort("user_index", "item_index")
        .collect(engine="streaming")
    )
    arrays = {
        "user_ids": users["user_id"].to_numpy(),
        "item_ids": items["item_id"].to_numpy(),
        "positive_items": positive_items,
        "positive_offsets": positive_offsets,
        "seen_items": seen["item_index"].to_numpy(),
        "seen_offsets": _offsets(seen["user_index"].to_numpy(), users.height),
    }
    destination.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(destination / f"{name}.npy", array, allow_pickle=False)
    metadata = {
        "kind": "sasrec_daily_positive_sequences_v1",
        "seed": seed,
        "history_path": str(history_path),
        "history_sha256": expected_sha256,
        "cutoff": cutoff.isoformat(),
        "max_history_dt": maximum.isoformat(),
        "sequence_order": ["user_id", "dt", "item_id"],
        "timestamp_semantics": "daily_row_first_timestamp",
        "context_user_limit": context_user_limit,
        "full_catalog": preserve_full_catalog,
        "users": users.height,
        "items": items.height,
        "positive_daily_rows": len(positive_items),
        "seen_pairs": seen.height,
        "sha256": {
            f"{n}.npy": sha256_file(destination / f"{n}.npy") for n in ARRAY_NAMES
        },
    }
    lengths = np.diff(positive_offsets)
    metadata["positive_length_quantiles"] = {
        str(q): float(np.quantile(lengths, q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)
    }
    metadata["eligible_users"] = int(
        ((lengths >= 2) & (np.diff(arrays["seen_offsets"]) < items.height)).sum()
    )
    metadata["array_bytes"] = sum(a.nbytes for a in arrays.values())
    write_json_atomic(destination / "manifest.json", metadata)
    return SequenceStore.load(destination, verify=False)


def sample_unseen(
    rng: np.random.Generator,
    seen: np.ndarray,
    item_count: int,
    count: int,
) -> np.ndarray:
    """Uniform with replacement from the history catalog minus all user-seen IDs."""
    if len(seen) >= item_count:
        raise ValueError("user has no unseen negative items")
    # Complement-rank mapping avoids rejection loops for dense user histories.
    gaps = seen.astype(np.int64) - np.arange(1, len(seen) + 1, dtype=np.int64)
    ranks = rng.integers(0, item_count - len(seen), size=count, dtype=np.int64)
    return (ranks + np.searchsorted(gaps, ranks, side="right") + 1).astype(np.int32)


@dataclass
class FitBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    negatives: torch.Tensor


@dataclass
class PredictBatch:
    user_ids: np.ndarray
    inputs: torch.Tensor
    seen_items: list[np.ndarray]


class SASRecDataLoader(CandidateDataLoader[FitBatch, PredictBatch]):
    """Epoch preparation owns RNG/windows/negatives; iteration only slices tensors."""

    def __init__(self, store: SequenceStore, *, max_length: int = 50, seed: int = 42):
        super().__init__(seed=seed)
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.store = store
        self.max_length = max_length
        self.fit_arrays = None
        self.predict_arrays = None

    def _load_fit_data(self, **kwargs):
        self.fit_arrays = None

    def _prepare_fit_data(
        self,
        *,
        epoch: int,
        negative_count: int,
        max_users: int | None = None,
        memory_budget_bytes: int = 8 * 2**30,
        progress: Progress | None = None,
    ):
        if epoch < 1 or negative_count < 1:
            raise ValueError("epoch and negative_count must be positive")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        users = rng.permutation(self.store.eligible_users)
        if max_users is not None:
            users = users[:max_users]
        if not len(users):
            raise ValueError("no users with >=2 positives and an unseen negative")
        full_lengths = np.diff(self.store.positive_offsets)[users]
        ends = rng.integers(2, full_lengths + 1)
        lengths = np.minimum(ends - 1, self.max_length)
        offsets = np.concatenate(([0], lengths.cumsum()))
        estimated_bytes = (
            len(users) * self.max_length * 8 + int(offsets[-1]) * negative_count * 4
        )
        if estimated_bytes > memory_budget_bytes:
            raise MemoryError(
                f"epoch arrays need {estimated_bytes / 2**30:.2f} GiB; lower batching/window/negative budget"
            )
        inputs = np.zeros((len(users), self.max_length), dtype=np.int32)
        targets = np.zeros_like(inputs)
        negatives = np.empty((int(offsets[-1]), negative_count), dtype=np.int32)
        for row, (user, end, length) in enumerate(
            zip(users, ends, lengths, strict=True)
        ):
            base = int(self.store.positive_offsets[user])
            window = self.store.positive_items[base + end - length - 1 : base + end]
            inputs[row, :length] = window[:-1]
            targets[row, :length] = window[1:]
            negatives[offsets[row] : offsets[row + 1]] = sample_unseen(
                rng,
                self.store.seen_for(int(user)),
                len(self.store.item_ids),
                int(length) * negative_count,
            ).reshape(int(length), negative_count)
            if progress and ((row + 1) % 2048 == 0 or row + 1 == len(users)):
                progress(row + 1, len(users))
        self.fit_arrays = (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(negatives),
            offsets,
        )
        self.epoch_metadata = {
            "epoch": epoch,
            "users": len(users),
            "valid_targets": int(offsets[-1]),
            "negative_count": negative_count,
            "array_bytes": estimated_bytes,
            "max_sequence_length": self.max_length,
            "sampling": "one_uniform_endpoint_window_per_eligible_user_per_epoch",
            "negative_sampling": "uniform_with_replacement_excluding_all_fit_history_items",
        }

    def _iter_fit_batches(self, *, batch_size):
        inputs, targets, negatives, offsets = self.fit_arrays
        size = batch_size or 128
        for start in range(0, len(inputs), size):
            end = min(start + size, len(inputs))
            yield FitBatch(
                inputs[start:end],
                targets[start:end],
                negatives[offsets[start] : offsets[end]],
            )

    def _load_predict_data(self, *, user_ids: np.ndarray):
        if user_ids.dtype != np.uint64:
            raise ValueError("prediction user IDs must remain UInt64")
        if len(np.unique(user_ids)) != len(user_ids):
            raise ValueError("duplicate prediction user IDs")
        self.requested_users = np.sort(user_ids)

    def _prepare_predict_data(self, **kwargs):
        indices = np.searchsorted(self.store.user_ids, self.requested_users)
        inputs = np.zeros((len(indices), self.max_length), dtype=np.int32)
        seen = []
        for row, index in enumerate(indices):
            if (
                index >= len(self.store.user_ids)
                or self.store.user_ids[index] != self.requested_users[row]
            ):
                seen.append(np.empty(0, dtype=np.uint32))
                continue
            start, end = self.store.positive_offsets[index : index + 2]
            values = self.store.positive_items[
                max(int(start), int(end) - self.max_length) : int(end)
            ]
            inputs[row, : len(values)] = values
            seen.append(self.store.seen_for(int(index)))
        self.predict_arrays = (torch.from_numpy(inputs), seen)

    def _iter_predict_batches(self, *, batch_size):
        inputs, seen = self.predict_arrays
        size = batch_size or 128
        for start in range(0, len(inputs), size):
            end = start + size
            yield PredictBatch(
                self.requested_users[start:end], inputs[start:end], seen[start:end]
            )

    def get_config(self):
        return {
            "loader": "sasrec_daily_positive",
            "seed": self.seed,
            "max_length": self.max_length,
            "history_sha256": self.store.metadata["history_sha256"],
        }

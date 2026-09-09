"""Small pre-norm SASRec with tied embeddings and sampled training scores."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from experiment_utils import read_json, sha256_file, write_json_atomic
from interfaces import CANDIDATE_SCHEMA, CandidateModel
from sasrec_data import SASRecDataLoader


@dataclass(frozen=True)
class SASRecConfig:
    embedding_dim: int = 64
    num_heads: int = 2
    num_blocks: int = 2
    dropout: float = 0.1
    max_length: int = 50
    initialization_std: float = 0.02
    use_input_layer_norm: bool = False

    def __post_init__(self):
        for name in ("embedding_dim", "num_heads", "num_blocks", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1 or not 0 < self.initialization_std < 1:
            raise ValueError("invalid dropout or initialization_std")


class TransformerBlock(nn.Module):
    def __init__(self, config: SASRecConfig):
        super().__init__()
        dim = config.embedding_dim
        self.attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, valid, causal_mask):
        normalized = self.norm1(x)
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=~valid,
            need_weights=False,
        )
        x = x + self.dropout(attention)
        x = x + self.feed_forward(self.norm2(x))
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class SASRecEncoder(nn.Module):
    def __init__(self, num_items: int, config: SASRecConfig):
        super().__init__()
        if num_items < 1:
            raise ValueError("num_items must be positive")
        self.config = config
        self.num_items = num_items
        self.item_embedding = nn.Embedding(
            num_items + 1, config.embedding_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(config.max_length, config.embedding_dim)
        self.input_norm = (
            nn.LayerNorm(config.embedding_dim)
            if config.use_input_layer_norm
            else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_blocks)]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.max_length, config.max_length, dtype=torch.bool).triu(1),
            persistent=False,
        )
        nn.init.normal_(self.item_embedding.weight, std=config.initialization_std)
        nn.init.normal_(self.position_embedding.weight, std=config.initialization_std)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.dtype not in (torch.int32, torch.int64):
            raise ValueError("inputs must be an integer [batch,length] tensor")
        length = inputs.shape[1]
        if not 1 <= length <= self.config.max_length:
            raise ValueError("invalid sequence length")
        valid = inputs != 0
        if bool((inputs < 0).any()) or bool((inputs > self.num_items).any()):
            raise ValueError("item index outside history catalog")
        if bool((valid.sum(1) == 0).any()):
            raise ValueError("empty histories must bypass the encoder")
        if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("SASRec requires contiguous right padding")
        positions = torch.arange(length, device=inputs.device)
        x = self.input_norm(
            self.item_embedding(inputs) + self.position_embedding(positions)
        )
        x = self.dropout(x).masked_fill(~valid.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, valid, self.causal_mask[:length, :length])
        return self.final_norm(x).masked_fill(~valid.unsqueeze(-1), 0.0)

    def encode_users(self, inputs):
        states = self(inputs)
        last = (inputs != 0).sum(1) - 1
        return states[torch.arange(len(inputs), device=inputs.device), last]

    def sampled_loss(self, inputs, targets, negatives, *, query_chunk_size=512):
        """Binary sampled objective; never allocates batch x length x catalog."""
        states = self(inputs)
        valid = targets > 0
        queries = states[valid]
        positive_ids = targets[valid]
        if not len(queries) or len(queries) != len(negatives):
            raise ValueError("negative rows must match valid shifted targets")
        if (
            negatives.ndim != 2
            or negatives.shape[1] < 1
            or bool((negatives <= 0).any())
        ):
            raise ValueError("sampled negatives must be non-padding item IDs")
        # One lookup prevents one full dense embedding-gradient allocation per
        # scoring chunk. The lookup is only for sampled IDs, never the catalog.
        sampled_ids = torch.cat((positive_ids, negatives.reshape(-1)))
        sampled_vectors = self.item_embedding(sampled_ids)
        positive_vectors = sampled_vectors[: len(queries)]
        negative_vectors = sampled_vectors[len(queries) :].reshape(
            len(queries), negatives.shape[1], self.config.embedding_dim
        )
        loss = torch.zeros((), device=states.device, dtype=torch.float32)
        for start in range(0, len(queries), query_chunk_size):
            stop = start + query_chunk_size
            q = queries[start:stop]
            positive = (q * positive_vectors[start:stop]).sum(-1).float()
            negative = (q[:, None, :] * negative_vectors[start:stop]).sum(-1).float()
            loss = loss + (F.softplus(-positive) + F.softplus(negative).mean(-1)).sum()
        return loss / len(queries)


def save_torch_atomic(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_topk(scores: torch.Tensor, ids: torch.Tensor, k: int, *, ids_sorted=False):
    """Top-k with exact item-ID tie resolution, including ties at the boundary."""
    k = min(k, scores.shape[1])
    # Put IDs in ascending order first; merged chunks need not have that order.
    if not ids_sorted:
        order = torch.argsort(ids, dim=1, stable=True)
        ids = ids.gather(1, order)
        scores = scores.gather(1, order)
    values, indices = torch.topk(scores, k, dim=1, sorted=False)
    boundary = values.min(1).values
    ambiguous = (scores == boundary[:, None]).sum(1) > (
        values == boundary[:, None]
    ).sum(1)
    if bool(ambiguous.any()):
        rows = torch.where(ambiguous)[0]
        indices[rows] = torch.argsort(
            scores[rows], dim=1, descending=True, stable=True
        )[:, :k]
        values = scores.gather(1, indices)
    selected_ids = ids.gather(1, indices)
    item_order = torch.argsort(selected_ids, dim=1, stable=True)
    values = values.gather(1, item_order)
    selected_ids = selected_ids.gather(1, item_order)
    score_order = torch.argsort(values, dim=1, descending=True, stable=True)
    return values.gather(1, score_order), selected_ids.gather(1, score_order)


class SASRecCandidateModel(CandidateModel):
    def __init__(
        self,
        item_ids: np.ndarray,
        config: SASRecConfig,
        *,
        device="cpu",
        history_sha256=None,
    ):
        if item_ids.dtype != np.int32 or not np.array_equal(
            item_ids, np.unique(item_ids)
        ):
            raise ValueError("item_ids must be sorted unique Int32")
        self.item_ids = item_ids.copy()
        self.config = config
        self.history_sha256 = history_sha256
        self.device = torch.device(device)
        self.encoder = SASRecEncoder(len(item_ids), config).to(self.device)

    @property
    def source_name(self):
        return "sasrec"

    def _check_loader(self, loader):
        if not isinstance(loader, SASRecDataLoader):
            raise TypeError("SASRecCandidateModel requires SASRecDataLoader")
        if loader.max_length != self.config.max_length or not np.array_equal(
            self.item_ids, loader.store.item_ids
        ):
            raise ValueError(
                "SASRec model/loader sequence length or item mapping differs"
            )
        history = loader.store.metadata["history_sha256"]
        if self.history_sha256 is not None and self.history_sha256 != history:
            raise ValueError("SASRec model/loader history snapshots differ")
        self.history_sha256 = history

    def _fit(
        self,
        loader: SASRecDataLoader,
        *,
        optimizer,
        batch_size=128,
        precision="float32",
        gradient_clip_norm=1.0,
        query_chunk_size=512,
        callback=None,
        check_stop=None,
        warmup_steps=5,
    ):
        self._check_loader(loader)
        if precision not in ("float32", "bfloat16"):
            raise ValueError("supported precision: float32 or bfloat16")
        self.encoder.train()
        total_loss = 0.0
        total_targets = 0
        durations = []
        epoch_start = time.perf_counter()
        for index, batch in enumerate(loader.iter_fit_batches(batch_size=batch_size)):
            if check_stop:
                check_stop()
            started = time.perf_counter()
            inputs, targets, negatives = (
                value.to(self.device, dtype=torch.int64)
                for value in (batch.inputs, batch.targets, batch.negatives)
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
            ):
                loss = self.encoder.sampled_loss(
                    inputs, targets, negatives, query_chunk_size=query_chunk_size
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite SASRec loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                gradient_clip_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            optimizer.step()
            with torch.no_grad():
                self.encoder.item_embedding.weight[0].zero_()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - started
            durations.append(seconds)
            count = len(negatives)
            value = float(loss.detach())
            total_loss += value * count
            total_targets += count
            if callback:
                callback(index + 1, value, seconds, count, float(norm))
        steady = durations[min(warmup_steps, max(0, len(durations) - 1)) :]
        self.last_fit_metrics = {
            "training_loss": total_loss / total_targets,
            "batches": len(durations),
            "valid_targets": total_targets,
            "train_seconds": time.perf_counter() - epoch_start,
            "step_seconds_median": float(np.median(durations)),
            "step_seconds_p90": float(np.quantile(durations, 0.9)),
            "steady_step_seconds_mean": float(np.mean(steady)),
            "warmup_steps_excluded": len(durations) - len(steady),
        }

    @torch.inference_mode()
    def _predict(
        self, loader, *, k, batch_size=128, item_chunk_size=32768, callback=None
    ):
        self._check_loader(loader)
        self.encoder.eval()
        outputs = []
        for batch_index, batch in enumerate(
            loader.iter_predict_batches(batch_size=batch_size)
        ):
            active = torch.where((batch.inputs != 0).any(1))[0]
            if not len(active):
                continue
            query = self.encoder.encode_users(
                batch.inputs[active].to(self.device, dtype=torch.int64)
            ).float()
            batch_users = batch.user_ids[active.numpy()]
            seen = [batch.seen_items[i] for i in active.tolist()]
            best_values = torch.empty((len(active), 0), device=self.device)
            best_ids = torch.empty(
                (len(active), 0), device=self.device, dtype=torch.int64
            )
            for start in range(1, len(self.item_ids) + 1, item_chunk_size):
                end = min(start + item_chunk_size, len(self.item_ids) + 1)
                scores = query @ self.encoder.item_embedding.weight[start:end].float().T
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("non-finite SASRec retrieval score")
                for row, observed in enumerate(seen):
                    left, right = np.searchsorted(observed, [start, end])
                    positions = torch.as_tensor(
                        observed[left:right].astype(np.int64) - start,
                        device=self.device,
                    )
                    scores[row, positions] = -torch.inf
                ids = torch.arange(start, end, device=self.device).expand(
                    len(active), -1
                )
                values, indices = stable_topk(scores, ids, k, ids_sorted=True)
                best_values, best_ids = stable_topk(
                    torch.cat((best_values, values), 1),
                    torch.cat((best_ids, indices), 1),
                    k,
                )
            values = best_values.cpu().numpy()
            indices = best_ids.cpu().numpy()
            for row, user_id in enumerate(batch_users):
                for rank, (value, index) in enumerate(
                    zip(values[row], indices[row], strict=True), 1
                ):
                    if math.isfinite(float(value)):
                        outputs.append(
                            (
                                int(user_id),
                                int(self.item_ids[index - 1]),
                                float(value),
                                rank,
                                self.source_name,
                            )
                        )
            if callback:
                callback(batch_index + 1)
        return pl.DataFrame(outputs, schema=CANDIDATE_SCHEMA, orient="row")

    @torch.inference_mode()
    def score_pairs(self, queries: torch.Tensor, item_indices: torch.Tensor):
        self.encoder.eval()
        if bool((item_indices <= 0).any()) or bool(
            (item_indices > len(self.item_ids)).any()
        ):
            raise ValueError("score_pairs requires non-padding history item indices")
        return (
            queries.float() * self.encoder.item_embedding(item_indices).float()
        ).sum(-1)

    def get_config(self):
        return {
            "model": "sasrec",
            "history_sha256": self.history_sha256,
            "architecture": asdict(self.config),
            "num_items": len(self.item_ids),
        }

    def save(self, directory: Path):
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        save_torch_atomic(directory / "weights.pt", self.encoder.state_dict())
        np.save(directory / "item_ids.npy", self.item_ids, allow_pickle=False)
        write_json_atomic(
            directory / "model_config.json",
            {
                **self.get_config(),
                "artifact_version": 1,
                "sha256": {
                    name: sha256_file(directory / name)
                    for name in ("weights.pt", "item_ids.npy")
                },
            },
        )

    @classmethod
    def from_artifact(cls, directory: Path, *, device="cpu"):
        config = read_json(directory / "model_config.json")
        if config["artifact_version"] != 1:
            raise ValueError("unsupported SASRec artifact version")
        for name, digest in config["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"SASRec model checksum mismatch: {name}")
        result = cls(
            np.load(directory / "item_ids.npy", allow_pickle=False),
            SASRecConfig(**config["architecture"]),
            device=device,
            history_sha256=config.get("history_sha256"),
        )
        result.encoder.load_state_dict(
            torch.load(directory / "weights.pt", map_location=device, weights_only=True)
        )
        result.encoder.eval()
        return result

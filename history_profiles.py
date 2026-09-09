"""Fold-local ALS history profiles; only rotation-invariant scalars reach the ranker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

PROFILE_KINDS = ("positive", "like", "favorite", "long_watch")
PROFILE_STATS = (
    "cosine",
    "concentration",
    "log_count",
    "embedding_coverage",
    "available",
)
PROFILE_FEATURES = tuple(
    f"history_{kind}_{stat}" for kind in PROFILE_KINDS for stat in PROFILE_STATS
)
VARIANTS = {
    "A_base": (),
    "B_positive": ("positive",),
    "C_events": ("like", "favorite", "long_watch"),
    "D_all": PROFILE_KINDS,
}


def ignored_features(variant: str) -> list[str]:
    kinds = VARIANTS[variant]
    return [
        f"history_{kind}_{stat}"
        for kind in PROFILE_KINDS
        if kind not in kinds
        for stat in PROFILE_STATS
    ]


def normalize_factors(factors: np.ndarray) -> np.ndarray:
    """Normalize in bounded blocks, retaining zero rows as explicitly unavailable."""
    factors = np.array(factors, dtype=np.float32, copy=True, order="C")
    for start in range(0, len(factors), 65536):
        block = factors[start : start + 65536]
        if not np.isfinite(block).all():
            raise ValueError("ALS factors contain non-finite values")
        norms = np.linalg.norm(block, axis=1)
        np.divide(block, norms[:, None], out=block, where=norms[:, None] > 1e-12)
        block[norms <= 1e-12] = 0
    return factors


def history_pairs(history: pl.LazyFrame, users: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate items across days, with event-specific membership independent of dt."""
    return (
        history.select(
            "user_id", "item_id", "is_positive", "is_like", "is_favorite", "watch_time"
        )
        .join(users.lazy(), on="user_id", how="semi")
        .group_by("user_id", "item_id")
        .agg(
            pl.col("is_positive").max().alias("positive"),
            pl.col("is_like").max().alias("like"),
            pl.col("is_favorite").max().alias("favorite"),
            (pl.col("watch_time").max() > 60).cast(pl.Int32).alias("long_watch"),
        )
        .filter(pl.any_horizontal(pl.col(k) > 0 for k in PROFILE_KINDS))
        .collect(engine="streaming")
        .sort("user_id", "item_id")
    )


@dataclass
class HistoryProfiles:
    users: pl.DataFrame
    mapping: pl.DataFrame
    factors: np.ndarray
    # The following arrays are [kind, user, dimension] or [kind, user].
    directions: np.ndarray
    concentrations: np.ndarray
    counts: np.ndarray
    matched: np.ndarray

    @classmethod
    def build(
        cls,
        pairs: pl.DataFrame,
        users: pl.DataFrame,
        mapping: pl.DataFrame,
        factors: np.ndarray,
        *,
        batch_size: int = 16384,
        progress: Callable[[int], None] | None = None,
    ) -> HistoryProfiles:
        users = users.select("user_id").unique().sort("user_id")
        if (
            users.schema["user_id"] != pl.UInt64
            or mapping.schema["item_id"] != pl.Int32
        ):
            raise ValueError("source ID dtypes must be preserved")
        if pairs.select("user_id", "item_id").n_unique() != pairs.height:
            raise ValueError("profile input must contain unique user-item pairs")
        if (
            mapping["item_id"].n_unique() != mapping.height
            or mapping["item_index"].n_unique() != mapping.height
        ):
            raise ValueError("ALS mapping must be one-to-one")
        if mapping.height != len(factors) or (
            mapping.height and int(mapping["item_index"].max()) != len(factors) - 1
        ):
            raise ValueError("ALS mapping and factor rows differ")
        joined = (
            pairs.join(
                users.with_row_index("user_index"),
                on="user_id",
                how="inner",
                validate="m:1",
            )
            .join(mapping, on="item_id", how="left", validate="m:1")
            .sort("user_id", "item_id")
        )
        sums = np.zeros(
            (len(PROFILE_KINDS), users.height, factors.shape[1]), dtype=np.float32
        )
        counts = np.zeros(sums.shape[:2], dtype=np.int64)
        matched = np.zeros_like(counts)
        for index, start in enumerate(range(0, joined.height, batch_size)):
            batch = joined.slice(start, batch_size)
            u = batch["user_index"].to_numpy()
            item = batch["item_index"].fill_null(0).to_numpy()
            valid = batch["item_index"].is_not_null().to_numpy()
            vectors = factors[item]
            valid &= np.linalg.norm(vectors, axis=1) > 1e-12
            for k, kind in enumerate(PROFILE_KINDS):
                member = batch[kind].to_numpy() > 0
                np.add.at(counts[k], u[member], 1)
                active = member & valid
                np.add.at(matched[k], u[active], 1)
                np.add.at(sums[k], u[active], vectors[active])
            if progress is not None:
                progress(index)
        means = sums / np.maximum(matched, 1)[:, :, None]
        concentration = np.linalg.norm(means, axis=2).astype(np.float32)
        directions = np.divide(
            means, np.maximum(concentration, 1e-12)[:, :, None]
        ).astype(np.float32)
        return cls(users, mapping, factors, directions, concentration, counts, matched)

    def save(self, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        self.users.write_parquet(destination / "users.parquet")
        self.mapping.write_parquet(destination / "item_mapping.parquet")
        for name in ("factors", "directions", "concentrations", "counts", "matched"):
            np.save(
                destination / f"{name}.npy", getattr(self, name), allow_pickle=False
            )

    @classmethod
    def load(cls, directory: Path) -> HistoryProfiles:
        return cls(
            pl.read_parquet(directory / "users.parquet"),
            pl.read_parquet(directory / "item_mapping.parquet"),
            *(
                np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in (
                    "factors",
                    "directions",
                    "concentrations",
                    "counts",
                    "matched",
                )
            ),
        )

    def features(
        self, candidates: pl.DataFrame, *, batch_size: int = 16384
    ) -> pl.DataFrame:
        pairs = candidates.select("user_id", "item_id")
        joined = (
            pairs.with_row_index("row_index")
            .join(
                self.users.with_row_index("user_index"),
                on="user_id",
                how="left",
                validate="m:1",
            )
            .join(self.mapping, on="item_id", how="left", validate="m:1")
            .sort("row_index")
        )
        output = np.zeros((pairs.height, len(PROFILE_FEATURES)), dtype=np.float32)
        for start in range(0, pairs.height, batch_size):
            batch = joined.slice(start, batch_size)
            u = batch["user_index"].fill_null(0).to_numpy()
            i = batch["item_index"].fill_null(0).to_numpy()
            known_user = batch["user_index"].is_not_null().to_numpy()
            item_vectors = self.factors[i]
            known_item = batch["item_index"].is_not_null().to_numpy() & (
                np.linalg.norm(item_vectors, axis=1) > 1e-12
            )
            for k in range(len(PROFILE_KINDS)):
                valid = known_user & known_item & (self.concentrations[k, u] > 1e-12)
                cos = np.einsum("ij,ij->i", item_vectors, self.directions[k, u])
                columns = [
                    np.where(valid, np.clip(cos, -1, 1), 0),
                    np.where(known_user, self.concentrations[k, u], 0),
                    np.where(known_user, np.log1p(self.counts[k, u]), 0),
                    np.where(
                        known_user,
                        self.matched[k, u] / np.maximum(self.counts[k, u], 1),
                        0,
                    ),
                    valid.astype(np.float32),
                ]
                output[
                    start : start + batch.height,
                    k * len(PROFILE_STATS) : (k + 1) * len(PROFILE_STATS),
                ] = np.column_stack(columns)
        return pairs.hstack(pl.DataFrame(output, schema=list(PROFILE_FEATURES)))

    def diagnostics(self) -> dict:
        return {
            kind: {
                "users_with_events": int((self.counts[k] > 0).sum()),
                "users_with_direction": int((self.concentrations[k] > 1e-12).sum()),
                "unique_user_item_memberships": int(self.counts[k].sum()),
                "matched_memberships": int(self.matched[k].sum()),
                "mean_concentration": float(self.concentrations[k].mean()),
            }
            for k, kind in enumerate(PROFILE_KINDS)
        }

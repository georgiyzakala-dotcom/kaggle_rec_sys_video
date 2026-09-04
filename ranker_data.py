"""Labels, deterministic sampling, and shard assembly for ranker datasets."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from data_utils import GROUND_TRUTH_SCHEMA
from features import (
    attach_als_factor_features,
    attach_history_features,
    attach_union_score_ranks,
    build_covisit_aggregate_features,
)
from item2item import Item2ItemConfig
from validation import (
    ContractValidationError,
    validate_feature_table,
    validate_id_columns,
    validate_no_nulls,
    validate_unique_keys,
)

LABEL_COLUMN: Final = "label"
TRAINING_METADATA_COLUMNS: Final = (
    "is_hard_negative",
    "is_training_sample",
    "sampling_probability",
)


def _checked_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("easy_negative_probability must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or result > 1:
        raise ValueError("easy_negative_probability must be in (0, 1]")
    return result


def _checked_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Label-safe hard-negative retention and reproducible easy sampling."""

    seed: int = 42
    easy_negative_probability: float = 0.05
    hard_negative_rank: int = 50
    collaborative_sources: tuple[str, ...] = ("item2item", "implicit_als")
    keep_multisource: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        probability = _checked_probability(self.easy_negative_probability)
        rank = _checked_positive_int(
            self.hard_negative_rank, name="hard_negative_rank"
        )
        sources = tuple(self.collaborative_sources)
        if not sources or any(not isinstance(value, str) or not value for value in sources):
            raise ValueError("collaborative_sources must contain non-empty names")
        if len(set(sources)) != len(sources):
            raise ValueError("collaborative_sources must be unique")
        if not isinstance(self.keep_multisource, bool):
            raise TypeError("keep_multisource must be boolean")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "easy_negative_probability", probability)
        object.__setattr__(self, "hard_negative_rank", rank)
        object.__setattr__(self, "collaborative_sources", sources)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> NegativeSamplingConfig:
        if not isinstance(source, Mapping):
            raise TypeError("negative_sampling must be an object")
        return cls(
            seed=source.get("seed", 42),
            easy_negative_probability=source.get(
                "easy_negative_probability", 0.05
            ),
            hard_negative_rank=source.get("hard_negative_rank", 50),
            collaborative_sources=tuple(
                source.get(
                    "collaborative_sources", ("item2item", "implicit_als")
                )
            ),
            keep_multisource=source.get("keep_multisource", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "easy_negative_probability": self.easy_negative_probability,
            "hard_negative_rank": self.hard_negative_rank,
            "collaborative_sources": list(self.collaborative_sources),
            "keep_multisource": self.keep_multisource,
            "algorithm": "splitmix64_user_item_fold",
        }


def assign_binary_labels(
    candidates: pl.DataFrame, ground_truth: pl.DataFrame
) -> pl.DataFrame:
    """Mark candidate pairs that occur in the fold's eligible target GT."""

    if not isinstance(candidates, pl.DataFrame):
        raise TypeError("candidates must be a Polars DataFrame")
    if not isinstance(ground_truth, pl.DataFrame):
        raise TypeError("ground_truth must be a Polars DataFrame")
    validate_id_columns(candidates)
    validate_unique_keys(candidates, keys=("user_id", "item_id"), name="candidates")
    if ground_truth.schema != GROUND_TRUTH_SCHEMA:
        raise ContractValidationError("ground truth has invalid schema")
    validate_no_nulls(ground_truth, name="ground truth")
    validate_unique_keys(
        ground_truth, keys=("user_id", "item_id"), name="ground truth"
    )
    labeled = candidates.join(
        ground_truth.with_columns(pl.lit(1, dtype=pl.UInt8).alias(LABEL_COLUMN)),
        on=("user_id", "item_id"),
        how="left",
    ).with_columns(pl.col(LABEL_COLUMN).fill_null(0).cast(pl.UInt8))
    return labeled.sort(("user_id", "item_id"))


def _fold_seed(seed: int, fold: str) -> np.uint64:
    digest = hashlib.blake2b(
        f"{seed}:{fold}".encode(), digest_size=8
    ).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def deterministic_sampling_uniform(
    frame: pl.DataFrame, *, seed: int, fold: str
) -> np.ndarray:
    """Return stable [0, 1) values without casting source IDs to floats."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if not isinstance(fold, str) or not fold:
        raise ValueError("fold must be a non-empty string")
    validate_id_columns(frame)
    users = frame.get_column("user_id").to_numpy().astype(np.uint64, copy=False)
    items = (
        frame.get_column("item_id")
        .to_numpy()
        .astype(np.int64, copy=False)
        .astype(np.uint64, copy=False)
    )
    with np.errstate(over="ignore"):
        value = (
            users * np.uint64(0xD6E8FEB86659FD93)
            ^ items * np.uint64(0xA5A3564E27F8862F)
            ^ _fold_seed(int(seed), fold)
        )
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        value = (value ^ (value >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        value = value ^ (value >> np.uint64(31))
    return (value >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def apply_negative_sampling(
    labeled: pl.DataFrame,
    *,
    fold: str,
    config: NegativeSamplingConfig,
) -> pl.DataFrame:
    """Add a training mask while retaining the complete scoring universe."""

    if not isinstance(config, NegativeSamplingConfig):
        raise TypeError("config must be NegativeSamplingConfig")
    if labeled.schema.get(LABEL_COLUMN) != pl.UInt8:
        raise ContractValidationError("labeled candidates require UInt8 label")
    hard = pl.lit(False)
    if config.keep_multisource:
        if labeled.schema.get("source_count") != pl.UInt8:
            raise ContractValidationError("source_count must be UInt8")
        hard = hard | (pl.col("source_count") >= 2)
    for source in config.collaborative_sources:
        generated = f"generated_by_{source}"
        rank = f"generator_rank_{source}"
        if labeled.schema.get(generated) != pl.Boolean or labeled.schema.get(
            rank
        ) != pl.UInt32:
            raise ContractValidationError(
                f"missing generator provenance for hard negatives: {source}"
            )
        hard = hard | (
            pl.col(generated) & (pl.col(rank) <= config.hard_negative_rank)
        )
    random_values = deterministic_sampling_uniform(
        labeled, seed=config.seed, fold=fold
    )
    result = (
        labeled.with_columns(
            pl.Series("__sampling_uniform", random_values, dtype=pl.Float64)
        )
        .with_columns(
            ((pl.col(LABEL_COLUMN) == 0) & hard).alias("is_hard_negative")
        )
        .with_columns(
            (
                (pl.col(LABEL_COLUMN) == 1)
                | pl.col("is_hard_negative")
                | (pl.col("__sampling_uniform") < config.easy_negative_probability)
            ).alias("is_training_sample"),
            pl.when(
                (pl.col(LABEL_COLUMN) == 1) | pl.col("is_hard_negative")
            )
            .then(1.0)
            .otherwise(config.easy_negative_probability)
            .cast(pl.Float32)
            .alias("sampling_probability"),
        )
        .drop("__sampling_uniform")
        .sort(("user_id", "item_id"))
    )
    return result


def feature_column_names(frame: pl.DataFrame) -> list[str]:
    """Return the stable ordered model feature list, excluding IDs and metadata."""

    excluded = {
        "user_id",
        "item_id",
        LABEL_COLUMN,
        *TRAINING_METADATA_COLUMNS,
    }
    return [name for name in frame.columns if name not in excluded]


def validate_candidate_pair_identity(
    source: pl.DataFrame, result: pl.DataFrame
) -> None:
    expected = source.select("user_id", "item_id").sort(("user_id", "item_id"))
    actual = result.select("user_id", "item_id").sort(("user_id", "item_id"))
    if not expected.equals(actual):
        raise ContractValidationError(
            "ranker dataset candidate pairs differ from task06 input"
        )


def validate_ranker_dataset(frame: pl.DataFrame) -> None:
    """Validate labels, sampling metadata, feature finiteness, and ordering."""

    validate_feature_table(frame)
    required = {
        LABEL_COLUMN: pl.UInt8,
        "is_hard_negative": pl.Boolean,
        "is_training_sample": pl.Boolean,
        "sampling_probability": pl.Float32,
    }
    invalid = {
        name: (dtype, frame.schema.get(name))
        for name, dtype in required.items()
        if frame.schema.get(name) != dtype
    }
    if invalid:
        raise ContractValidationError(f"ranker dataset metadata is invalid: {invalid}")
    if frame.filter(~pl.col(LABEL_COLUMN).is_in([0, 1])).height:
        raise ContractValidationError("labels must be binary")
    if frame.filter((pl.col(LABEL_COLUMN) == 1) & ~pl.col("is_training_sample")).height:
        raise ContractValidationError("all positive rows must be retained for training")
    if frame.filter(pl.col("is_hard_negative") & ~pl.col("is_training_sample")).height:
        raise ContractValidationError("all hard negatives must be retained")
    if frame.filter(
        (pl.col("sampling_probability") <= 0)
        | (pl.col("sampling_probability") > 1)
    ).height:
        raise ContractValidationError("sampling probabilities must be in (0, 1]")
    for name, dtype in frame.schema.items():
        if dtype.is_float() and not frame.get_column(name).is_finite().all():
            raise ContractValidationError(f"feature {name} contains non-finite values")
    if not frame.equals(frame.sort(("user_id", "item_id"))):
        raise ContractValidationError(
            "ranker dataset must be ordered by user_id ASC, item_id ASC"
        )


def ranker_dataset_diagnostics(frame: pl.DataFrame) -> dict[str, Any]:
    """Return JSON-safe class balance and sampling diagnostics for one shard."""

    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col(LABEL_COLUMN).sum().alias("positive_rows"),
        pl.col("is_hard_negative").sum().alias("hard_negative_rows"),
        pl.col("is_training_sample").sum().alias("training_rows"),
        (pl.col("is_training_sample") & (pl.col(LABEL_COLUMN) == 1))
        .sum()
        .alias("training_positive_rows"),
    ).row(0, named=True)
    rows = int(summary["rows"])
    positives = int(summary["positive_rows"])
    training = int(summary["training_rows"])
    training_positives = int(summary["training_positive_rows"])
    return {
        "rows": rows,
        "users": int(summary["users"]),
        "positive_rows": positives,
        "negative_rows": rows - positives,
        "positive_rate": positives / rows if rows else 0.0,
        "hard_negative_rows": int(summary["hard_negative_rows"]),
        "training_rows": training,
        "training_positive_rows": training_positives,
        "training_negative_rows": training - training_positives,
        "training_positive_rate": training_positives / training if training else 0.0,
    }


def build_ranker_shard(
    union: pl.DataFrame,
    *,
    ground_truth: pl.DataFrame,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
    item2item_seeds: pl.DataFrame,
    item2item_neighbors: pl.DataFrame | pl.LazyFrame | str | Path,
    item2item_config: Item2ItemConfig,
    als_user_norms: pl.DataFrame,
    als_item_norms: pl.DataFrame,
    cutoff: datetime,
    fold: str,
    sampling_config: NegativeSamplingConfig,
    rank_sources: Sequence[str] = ("item2item", "implicit_als"),
) -> pl.DataFrame:
    """Assemble one complete, inference-available ranker feature shard."""

    result = build_inference_ranker_shard(
        union,
        user_features=user_features,
        item_features=item_features,
        item2item_seeds=item2item_seeds,
        item2item_neighbors=item2item_neighbors,
        item2item_config=item2item_config,
        als_user_norms=als_user_norms,
        als_item_norms=als_item_norms,
        cutoff=cutoff,
        rank_sources=rank_sources,
    )
    result = assign_binary_labels(result, ground_truth)
    result = apply_negative_sampling(
        result, fold=fold, config=sampling_config
    )
    validate_ranker_dataset(result)
    return result


def build_inference_ranker_shard(
    union: pl.DataFrame,
    *,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
    item2item_seeds: pl.DataFrame,
    item2item_neighbors: pl.DataFrame | pl.LazyFrame | str | Path,
    item2item_config: Item2ItemConfig,
    als_user_norms: pl.DataFrame,
    als_item_norms: pl.DataFrame,
    cutoff: datetime,
    rank_sources: Sequence[str] = ("item2item", "implicit_als"),
) -> pl.DataFrame:
    """Attach the frozen ranker feature set without future labels."""

    result = attach_history_features(
        union, user_features=user_features, item_features=item_features
    )
    result = attach_union_score_ranks(result, sources=rank_sources)
    result = build_covisit_aggregate_features(
        result,
        seeds=item2item_seeds,
        neighbor_table=item2item_neighbors,
        config=item2item_config,
        cutoff=cutoff,
    )
    availability_mismatch = result.filter(
        pl.col("covisit_available")
        != pl.col("cross_score_available_item2item")
    ).height
    score_mismatch = result.filter(
        pl.col("covisit_available")
        & (
            (pl.col("covisit_contribution_max") - pl.col("cross_score_item2item"))
            .abs()
            > 1e-9
        )
    ).height
    if availability_mismatch or score_mismatch:
        raise ContractValidationError(
            "co-vis aggregates disagree with task06 cross-score: "
            f"availability={availability_mismatch}, score={score_mismatch}"
        )
    result = attach_als_factor_features(
        result, user_norms=als_user_norms, item_norms=als_item_norms
    )
    validate_candidate_pair_identity(union, result)
    validate_feature_table(result)
    return result


__all__ = [
    "LABEL_COLUMN",
    "TRAINING_METADATA_COLUMNS",
    "NegativeSamplingConfig",
    "apply_negative_sampling",
    "assign_binary_labels",
    "build_inference_ranker_shard",
    "build_ranker_shard",
    "deterministic_sampling_uniform",
    "feature_column_names",
    "ranker_dataset_diagnostics",
    "validate_candidate_pair_identity",
    "validate_ranker_dataset",
]

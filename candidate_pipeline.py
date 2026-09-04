"""Offline candidate union, provenance, and cross-model scoring utilities.

The functions in this module operate on already prepared fold-history models
and typed candidate tables.  They never read validation events and deliberately
keep generator provenance separate from scores assigned after the union is
built.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from interfaces import CANDIDATE_SCHEMA, CandidateDataLoader, CandidateModel
from item2item import Item2ItemConfig, SeedStrength
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_id_columns,
    validate_no_nulls,
    validate_unique_keys,
)

SOURCE_NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_source_name(value: object) -> str:
    if not isinstance(value, str) or SOURCE_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("source names must match ^[a-z][a-z0-9_]*$")
    return value


@dataclass(frozen=True)
class CandidateSourceSpec:
    """One stably named source and its contribution cap to the union."""

    source: str
    cap: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _checked_source_name(self.source))
        object.__setattr__(self, "cap", _positive_int(self.cap, name="cap"))


@dataclass(frozen=True)
class CandidateUnionConfig:
    """Deterministic source order and a hard per-user union bound."""

    sources: tuple[CandidateSourceSpec, ...]
    total_cap: int

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if not sources:
            raise ValueError("at least one candidate source is required")
        if not all(isinstance(value, CandidateSourceSpec) for value in sources):
            raise TypeError("sources must contain CandidateSourceSpec values")
        names = [value.source for value in sources]
        if len(set(names)) != len(names):
            raise ValueError("candidate source names must be unique")
        total_cap = _positive_int(self.total_cap, name="total_cap")
        if sum(value.cap for value in sources) > total_cap:
            raise ValueError(
                "sum of source caps must not exceed total_cap; post-union "
                "heuristic pruning would change the candidate oracle"
            )
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "total_cap", total_cap)

    @classmethod
    def from_mapping(
        cls, source_caps: Mapping[str, int], *, total_cap: int
    ) -> CandidateUnionConfig:
        if not isinstance(source_caps, Mapping):
            raise TypeError("source_caps must be a mapping")
        return cls(
            sources=tuple(
                CandidateSourceSpec(source=source, cap=cap)
                for source, cap in source_caps.items()
            ),
            total_cap=total_cap,
        )

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(value.source for value in self.sources)

    def cap_for(self, source: str) -> int:
        for value in self.sources:
            if value.source == source:
                return value.cap
        raise KeyError(source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_order": list(self.source_names),
            "source_caps": {value.source: value.cap for value in self.sources},
            "total_cap": self.total_cap,
        }


def generator_columns(source: str) -> dict[str, str]:
    """Return the stable provenance column names for one source."""

    checked = _checked_source_name(source)
    return {
        "generated": f"generated_by_{checked}",
        "score": f"generator_score_{checked}",
        "rank": f"generator_rank_{checked}",
        "rank_norm": f"generator_rank_norm_{checked}",
    }


def cross_score_columns(source: str) -> dict[str, str]:
    """Return stable post-union score column names for one source."""

    checked = _checked_source_name(source)
    return {
        "score": f"cross_score_{checked}",
        "available": f"cross_score_available_{checked}",
        "rank": f"cross_rank_{checked}",
    }


def run_candidate_model(
    model: CandidateModel,
    loader: CandidateDataLoader[Any, Any],
    *,
    k: int,
    batch_size: int | None = None,
) -> pl.DataFrame:
    """Run and validate one already fitted/restored candidate model."""

    checked_k = _positive_int(k, name="k")
    result = model.predict(loader, k=checked_k, batch_size=batch_size)
    validate_candidate_output(result, k=checked_k, source_name=model.source_name)
    return result


def _validate_source_frame(frame: pl.DataFrame, *, source: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"candidate source {source!r} must be a Polars DataFrame")
    if frame.schema != CANDIDATE_SCHEMA:
        raise ContractValidationError(f"candidate source {source!r} has invalid schema")
    inferred_k = int(frame.get_column("rank").max() or 1)
    validate_candidate_output(frame, k=inferred_k, source_name=source)


def build_candidate_union(
    source_candidates: Mapping[str, pl.DataFrame],
    config: CandidateUnionConfig,
) -> pl.DataFrame:
    """Deduplicate source candidates and emit a wide provenance table.

    Source scores and ranks are meaningful only when the corresponding
    ``generated_by_*`` flag is true.  Zero is an unambiguous missing-rank
    sentinel because valid source ranks start at one.
    """

    if not isinstance(config, CandidateUnionConfig):
        raise TypeError("config must be CandidateUnionConfig")
    if not isinstance(source_candidates, Mapping):
        raise TypeError("source_candidates must be a mapping")
    expected = set(config.source_names)
    actual = set(source_candidates)
    if actual != expected:
        raise ValueError(
            "source candidate names differ from config: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    prepared: list[pl.DataFrame] = []
    for spec in config.sources:
        frame = source_candidates[spec.source]
        _validate_source_frame(frame, source=spec.source)
        columns = generator_columns(spec.source)
        prepared.append(
            frame.filter(pl.col("rank") <= spec.cap)
            .select(
                "user_id",
                "item_id",
                pl.col("score").alias(columns["score"]),
                pl.col("rank").alias(columns["rank"]),
                (
                    (spec.cap - pl.col("rank") + 1)
                    .cast(pl.Float64)
                    .truediv(float(spec.cap))
                ).alias(columns["rank_norm"]),
                pl.lit(True).alias(columns["generated"]),
            )
            .sort(("user_id", "item_id"))
        )

    union = prepared[0]
    for frame in prepared[1:]:
        union = union.join(
            frame,
            on=["user_id", "item_id"],
            how="full",
            coalesce=True,
        )

    fill_expressions: list[pl.Expr] = []
    generated_columns: list[str] = []
    for spec in config.sources:
        columns = generator_columns(spec.source)
        generated_columns.append(columns["generated"])
        fill_expressions.extend(
            (
                pl.col(columns["score"]).fill_null(0.0).cast(pl.Float64),
                pl.col(columns["rank"]).fill_null(0).cast(pl.UInt32),
                pl.col(columns["rank_norm"]).fill_null(0.0).cast(pl.Float64),
                pl.col(columns["generated"]).fill_null(False).cast(pl.Boolean),
            )
        )
    union = (
        union.with_columns(fill_expressions)
        .with_columns(
            pl.sum_horizontal(
                pl.col(column).cast(pl.UInt8) for column in generated_columns
            )
            .cast(pl.UInt8)
            .alias("source_count")
        )
        .sort(("user_id", "item_id"))
    )
    validate_union_features(union, config=config)
    return union


def validate_union_features(
    frame: pl.DataFrame,
    *,
    config: CandidateUnionConfig,
    require_cross_scores: Sequence[str] = (),
) -> None:
    """Validate IDs, provenance invariants, caps, and optional cross-scores."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("candidate union must be a Polars DataFrame")
    validate_id_columns(frame)
    validate_no_nulls(frame, name="candidate union")
    validate_unique_keys(frame, keys=("user_id", "item_id"), name="candidate union")
    if frame.schema.get("source_count") != pl.UInt8:
        raise ContractValidationError("source_count must have UInt8 dtype")

    generated_names: list[str] = []
    for spec in config.sources:
        columns = generator_columns(spec.source)
        expected = {
            columns["score"]: pl.Float64,
            columns["rank"]: pl.UInt32,
            columns["rank_norm"]: pl.Float64,
            columns["generated"]: pl.Boolean,
        }
        invalid = {
            name: (dtype, frame.schema.get(name))
            for name, dtype in expected.items()
            if frame.schema.get(name) != dtype
        }
        if invalid:
            raise ContractValidationError(
                f"invalid provenance columns for {spec.source}: {invalid}"
            )
        if not frame.get_column(columns["score"]).is_finite().all():
            raise ContractValidationError(
                f"generator scores for {spec.source} must be finite"
            )
        generated_names.append(columns["generated"])
        invalid_missing = frame.filter(
            (~pl.col(columns["generated"]))
            & (
                (pl.col(columns["score"]) != 0.0)
                | (pl.col(columns["rank"]) != 0)
                | (pl.col(columns["rank_norm"]) != 0.0)
            )
        ).height
        invalid_present = frame.filter(
            pl.col(columns["generated"])
            & (
                (pl.col(columns["rank"]) < 1)
                | (pl.col(columns["rank"]) > spec.cap)
                | (
                    (
                        pl.col(columns["rank_norm"])
                        - (
                            (spec.cap - pl.col(columns["rank"]) + 1)
                            .cast(pl.Float64)
                            .truediv(float(spec.cap))
                        )
                    ).abs()
                    > 1e-12
                )
            )
        ).height
        if invalid_missing or invalid_present:
            raise ContractValidationError(
                f"invalid provenance values for {spec.source}: "
                f"missing={invalid_missing}, present={invalid_present}"
            )

    expected_count = pl.sum_horizontal(
        pl.col(column).cast(pl.UInt8) for column in generated_names
    ).cast(pl.UInt8)
    if frame.filter(pl.col("source_count") != expected_count).height:
        raise ContractValidationError("source_count differs from provenance flags")
    if frame.filter(pl.col("source_count") == 0).height:
        raise ContractValidationError("candidate union contains zero-source rows")
    if frame.group_by("user_id").len().filter(pl.col("len") > config.total_cap).height:
        raise ContractValidationError(
            f"candidate union exceeds total_cap={config.total_cap}"
        )

    physical = frame.sort(("user_id", "item_id"))
    if not frame.equals(physical):
        raise ContractValidationError(
            "candidate union must be ordered by user_id ASC, item_id ASC"
        )

    for source in require_cross_scores:
        columns = cross_score_columns(source)
        expected = {
            columns["score"]: pl.Float64,
            columns["available"]: pl.Boolean,
        }
        invalid = {
            name: (dtype, frame.schema.get(name))
            for name, dtype in expected.items()
            if frame.schema.get(name) != dtype
        }
        if invalid:
            raise ContractValidationError(
                f"invalid cross-score columns for {source}: {invalid}"
            )
        if not frame.get_column(columns["score"]).is_finite().all():
            raise ContractValidationError(f"cross scores for {source} must be finite")
        invalid_missing = frame.filter(
            (~pl.col(columns["available"])) & (pl.col(columns["score"]) != 0.0)
        ).height
        if invalid_missing:
            raise ContractValidationError(
                f"unavailable cross scores for {source} must use zero sentinel"
            )


def attach_ranking_cross_scores(
    union: pl.DataFrame,
    *,
    source: str,
    item_ranking: pl.DataFrame,
) -> pl.DataFrame:
    """Attach a history-only global item score/rank lookup to every union row."""

    columns = cross_score_columns(source)
    required = {"item_id": pl.Int32, "score": pl.Float64, "global_rank": pl.UInt32}
    if not isinstance(item_ranking, pl.DataFrame):
        raise TypeError("item_ranking must be a Polars DataFrame")
    if any(item_ranking.schema.get(name) != dtype for name, dtype in required.items()):
        raise ContractValidationError("item ranking has invalid schema")
    validate_no_nulls(item_ranking, name=f"{source} item ranking")
    validate_unique_keys(item_ranking, keys=("item_id",), name=f"{source} item ranking")
    if not item_ranking.get_column("score").is_finite().all():
        raise ContractValidationError("item ranking scores must be finite")

    result = (
        union.join(
            item_ranking.select(
                "item_id",
                pl.col("score").alias(columns["score"]),
                pl.col("global_rank").alias(columns["rank"]),
            ),
            on="item_id",
            how="left",
        )
        .with_columns(
            pl.col(columns["score"]).is_not_null().alias(columns["available"]),
            pl.col(columns["score"]).fill_null(0.0).cast(pl.Float64),
            pl.col(columns["rank"]).fill_null(0).cast(pl.UInt32),
        )
        .sort(("user_id", "item_id"))
    )
    return result


def attach_implicit_als_cross_scores(
    union: pl.DataFrame,
    *,
    model: Any,
    batch_size: int = 65_536,
) -> pl.DataFrame:
    """Attach ALS factor dot products for arbitrary union pairs."""

    from implicit_model import ImplicitALSModel

    if not isinstance(model, ImplicitALSModel):
        raise TypeError("model must be ImplicitALSModel")
    checked_batch = _positive_int(batch_size, name="batch_size")
    columns = cross_score_columns(model.source_name)
    indexed = (
        union.with_row_index("__row_index")
        .join(model.user_mapping, on="user_id", how="left")
        .join(model.item_mapping, on="item_id", how="left")
        .sort("__row_index")
    )
    available = (
        indexed.get_column("user_index").is_not_null()
        & indexed.get_column("item_index").is_not_null()
    ).to_numpy()
    scores = np.zeros(indexed.height, dtype=np.float64)
    valid_rows = np.flatnonzero(available)
    user_indices = indexed.get_column("user_index").fill_null(0).to_numpy()
    item_indices = indexed.get_column("item_index").fill_null(0).to_numpy()
    for offset in range(0, len(valid_rows), checked_batch):
        rows = valid_rows[offset : offset + checked_batch]
        user_factors = model.backend.user_factors[user_indices[rows]]
        item_factors = model.backend.item_factors[item_indices[rows]]
        scores[rows] = np.einsum(
            "ij,ij->i", user_factors, item_factors, optimize=True
        ).astype(np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ContractValidationError("ALS cross scores must be finite")
    return (
        indexed.drop("__row_index", "user_index", "item_index")
        .with_columns(
            pl.Series(columns["score"], scores, dtype=pl.Float64),
            pl.Series(columns["available"], available, dtype=pl.Boolean),
        )
        .sort(("user_id", "item_id"))
    )


def _lazy_frame(source: pl.DataFrame | pl.LazyFrame | str | Path) -> pl.LazyFrame:
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, (str, Path)):
        return pl.scan_parquet(source)
    raise TypeError("source must be a Polars frame or Parquet path")


def attach_item2item_cross_scores(
    union: pl.DataFrame,
    *,
    seeds: pl.DataFrame,
    neighbor_table: pl.DataFrame | pl.LazyFrame | str | Path,
    config: Item2ItemConfig,
    reference_time: datetime,
) -> pl.DataFrame:
    """Attach sparse co-vis scores to arbitrary union pairs.

    The calculation matches ``Item2ItemModel`` inference: recent seed
    contributions are multiplied by optional time decay and event strength,
    then aggregated with ``max`` over seeds.
    """

    if not isinstance(config, Item2ItemConfig):
        raise TypeError("config must be Item2ItemConfig")
    if not isinstance(reference_time, datetime) or reference_time.tzinfo is not None:
        raise ValueError("reference_time must be a timezone-naive datetime")
    required_seed_columns = {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "last_dt": pl.Datetime("us"),
        "views": pl.UInt64,
        "watch_time": pl.Int64,
        "is_like": pl.Int32,
        "is_favorite": pl.Int32,
        "history_rank": pl.UInt32,
    }
    invalid = {
        name: (dtype, seeds.schema.get(name))
        for name, dtype in required_seed_columns.items()
        if seeds.schema.get(name) != dtype
    }
    if invalid:
        raise ContractValidationError(f"item2item seeds have invalid schema: {invalid}")
    validate_no_nulls(seeds, name="item2item cross-score seeds")
    validate_unique_keys(
        seeds, keys=("user_id", "item_id"), name="item2item cross-score seeds"
    )
    neighbor = _lazy_frame(neighbor_table)
    required_neighbors = {
        "item_id": pl.Int32,
        "neighbor_item_id": pl.Int32,
        "score": pl.Float64,
    }
    neighbor_schema = neighbor.collect_schema()
    if any(
        neighbor_schema.get(name) != dtype for name, dtype in required_neighbors.items()
    ):
        raise ContractValidationError("item2item neighbor table has invalid schema")

    columns = cross_score_columns("item2item")
    reference = pl.lit(reference_time, dtype=pl.Datetime("us"))
    age_hours = (reference - pl.col("last_dt")).dt.total_microseconds().cast(
        pl.Float64
    ) / 3_600_000_000.0
    if config.seed_recency_half_life_hours is None:
        recency = pl.lit(1.0)
    else:
        recency = (
            -math.log(2.0) * age_hours / config.seed_recency_half_life_hours
        ).exp()
    if config.seed_strength is SeedStrength.UNIFORM:
        strength = pl.lit(1.0)
    else:
        strength = (
            pl.lit(1.0)
            + pl.col("views").cast(pl.Float64).log1p()
            + (pl.col("watch_time") > 60).cast(pl.Float64)
            + pl.col("is_like").cast(pl.Float64)
            + pl.col("is_favorite").cast(pl.Float64)
        )
    pairs = union.select("user_id", "item_id").lazy()
    scored = (
        seeds.lazy()
        .filter(pl.col("history_rank") <= config.seed_k)
        .rename({"item_id": "seed_item_id"})
        .join(
            neighbor.select(
                pl.col("item_id").alias("seed_item_id"),
                pl.col("neighbor_item_id").alias("item_id"),
                pl.col("score").alias("neighbor_score"),
            ),
            on="seed_item_id",
            how="inner",
        )
        .join(pairs, on=["user_id", "item_id"], how="semi")
        .with_columns(
            (pl.col("neighbor_score") * recency * strength)
            .cast(pl.Float64)
            .alias(columns["score"])
        )
        .group_by("user_id", "item_id")
        .agg(pl.col(columns["score"]).max())
        .collect(engine="streaming")
    )
    result = (
        union.join(scored, on=["user_id", "item_id"], how="left")
        .with_columns(
            pl.col(columns["score"]).is_not_null().alias(columns["available"]),
            pl.col(columns["score"]).fill_null(0.0).cast(pl.Float64),
        )
        .sort(("user_id", "item_id"))
    )
    if not result.get_column(columns["score"]).is_finite().all():
        raise ContractValidationError("item2item cross scores must be finite")
    return result


def validate_union_against_history(
    frame: pl.DataFrame | pl.LazyFrame | str | Path,
    *,
    history_daily: pl.DataFrame | pl.LazyFrame | str | Path,
) -> None:
    """Require all materialized union pairs to be known and user-unseen.

    Both inputs may be lazy/file-backed so a sharded union can be checked once
    per fold instead of rescanning the complete history for every shard.
    """

    union = _lazy_frame(frame)
    union_schema = union.collect_schema()
    invalid_ids = {
        column: (dtype, union_schema.get(column))
        for column, dtype in {"user_id": pl.UInt64, "item_id": pl.Int32}.items()
        if union_schema.get(column) != dtype
    }
    if invalid_ids:
        raise ContractValidationError(f"invalid ID columns: {invalid_ids}")
    history = _lazy_frame(history_daily)
    if history.collect_schema() != DAILY_INTERACTION_SCHEMA:
        raise ContractValidationError("history has invalid daily schema")
    pairs = union.select("user_id", "item_id")
    known_items = history.select("item_id").unique()
    seen_pairs = history.select("user_id", "item_id").unique()
    unknown, seen = pl.collect_all(
        (
            pairs.join(known_items, on="item_id", how="anti").select(pl.len()),
            pairs.join(seen_pairs, on=["user_id", "item_id"], how="semi").select(
                pl.len()
            ),
        )
    )
    unknown_count = int(unknown.item())
    seen_count = int(seen.item())
    if unknown_count or seen_count:
        raise ContractValidationError(
            "candidate union violates history rules: "
            f"unknown={unknown_count}, seen={seen_count}"
        )


def iter_target_user_shards(
    target_users: pl.DataFrame, *, users_per_shard: int
) -> Iterator[tuple[int, pl.DataFrame]]:
    """Yield deterministic contiguous user shards for bounded materialization."""

    checked_size = _positive_int(users_per_shard, name="users_per_shard")
    if target_users.schema != TARGET_USER_SCHEMA:
        raise ContractValidationError("target users have invalid schema")
    validate_no_nulls(target_users, name="target users")
    validate_unique_keys(target_users, keys=("user_id",), name="target users")
    ordered = target_users.sort("user_id")
    for shard_index, offset in enumerate(range(0, ordered.height, checked_size)):
        yield shard_index, ordered.slice(offset, checked_size)


__all__ = [
    "CandidateSourceSpec",
    "CandidateUnionConfig",
    "attach_implicit_als_cross_scores",
    "attach_item2item_cross_scores",
    "attach_ranking_cross_scores",
    "build_candidate_union",
    "cross_score_columns",
    "generator_columns",
    "iter_target_user_shards",
    "run_candidate_model",
    "validate_union_against_history",
    "validate_union_features",
]

"""Fold-specific, history-only features for supervised candidate ranking.

Every aggregate in this module is computed from one immutable prepared history
snapshot and a timestamp cutoff.  Ground truth is deliberately absent from all
feature-building APIs; labels are attached later by :mod:`ranker_data`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA
from item2item import (
    COLLAPSED_HISTORY_SCHEMA,
    NEIGHBOR_TABLE_SCHEMA,
    Item2ItemConfig,
    SeedStrength,
)
from validation import (
    ContractValidationError,
    validate_id_columns,
    validate_no_nulls,
    validate_unique_keys,
)

HOURS_IN_MICROSECOND: Final = 3_600_000_000.0


def _positive_hours(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be positive numbers")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} values must be positive and finite")
    return result


def _hours_tuple(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_positive_hours(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be unique and sorted ascending")
    return result


def _window_token(hours: float) -> str:
    return f"{int(hours)}h" if hours.is_integer() else f"{hours:g}h".replace(".", "p")


@dataclass(frozen=True)
class HistoryFeatureConfig:
    """Timestamp windows and trend comparisons shared by every fold."""

    windows_hours: tuple[float, ...] = (6.0, 24.0, 72.0)
    trend_windows_hours: tuple[float, ...] = (6.0, 24.0)

    def __post_init__(self) -> None:
        windows = _hours_tuple(self.windows_hours, name="windows_hours")
        trends = _hours_tuple(
            self.trend_windows_hours, name="trend_windows_hours"
        )
        missing = sorted(set(trends) - set(windows))
        if missing:
            raise ValueError(
                "trend_windows_hours must also be present in windows_hours: "
                f"{missing}"
            )
        object.__setattr__(self, "windows_hours", windows)
        object.__setattr__(self, "trend_windows_hours", trends)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> HistoryFeatureConfig:
        if not isinstance(source, Mapping):
            raise TypeError("history feature config must be an object")
        return cls(
            windows_hours=tuple(source.get("windows_hours", (6, 24, 72))),
            trend_windows_hours=tuple(
                source.get("trend_windows_hours", (6, 24))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows_hours": list(self.windows_hours),
            "trend_windows_hours": list(self.trend_windows_hours),
            "timestamp_semantics": "prepared_daily_row_first_timestamp",
        }


def _lazy_daily(
    source: pl.DataFrame | pl.LazyFrame | str | Path,
) -> pl.LazyFrame:
    if isinstance(source, pl.DataFrame):
        frame = source.lazy()
    elif isinstance(source, pl.LazyFrame):
        frame = source
    elif isinstance(source, (str, Path)):
        frame = pl.scan_parquet(source)
    else:
        raise TypeError("history must be a Polars frame or Parquet path")
    if frame.collect_schema() != DAILY_INTERACTION_SCHEMA:
        raise ContractValidationError("history has invalid daily interaction schema")
    return frame


def validate_history_cutoff(
    history: pl.DataFrame | pl.LazyFrame | str | Path,
    *,
    cutoff: datetime,
) -> dict[str, Any]:
    """Require a non-empty history whose timestamps are strictly before cutoff."""

    if not isinstance(cutoff, datetime) or cutoff.tzinfo is not None:
        raise ValueError("cutoff must be a timezone-naive datetime")
    frame = _lazy_daily(history)
    diagnostics = frame.select(
        pl.len().alias("rows"),
        pl.col("dt").min().alias("min_dt"),
        pl.col("dt").max().alias("max_dt"),
        pl.col("dt").is_null().sum().alias("null_timestamps"),
        (pl.col("dt") >= pl.lit(cutoff, dtype=pl.Datetime("us")))
        .sum()
        .alias("timestamps_at_or_after_cutoff"),
    ).collect(engine="streaming").row(0, named=True)
    if int(diagnostics["rows"]) == 0:
        raise ContractValidationError("history must not be empty")
    if int(diagnostics["null_timestamps"]):
        raise ContractValidationError("history timestamps contain nulls")
    if int(diagnostics["timestamps_at_or_after_cutoff"]):
        raise ContractValidationError(
            "history contains timestamps at or after the feature cutoff"
        )
    return {
        "rows": int(diagnostics["rows"]),
        "min_dt": diagnostics["min_dt"].isoformat(),
        "max_dt": diagnostics["max_dt"].isoformat(),
        "cutoff": cutoff.isoformat(),
    }


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(
            pl.col(numerator).cast(pl.Float64)
            / pl.col(denominator).cast(pl.Float64)
        )
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias(alias)
    )


def _window_aggregates(
    *,
    prefix: str,
    counterpart: str,
    cutoff: datetime,
    hours: float,
    suffix: str,
    start_offset: float = 0.0,
) -> list[pl.Expr]:
    upper = cutoff - timedelta(hours=start_offset)
    lower = upper - timedelta(hours=hours)
    mask = (pl.col("dt") >= pl.lit(lower, dtype=pl.Datetime("us"))) & (
        pl.col("dt") < pl.lit(upper, dtype=pl.Datetime("us"))
    )
    return [
        pl.col("dt").filter(mask).count().cast(pl.UInt32).alias(
            f"{prefix}_daily_rows_{suffix}"
        ),
        pl.col("views").filter(mask).sum().fill_null(0).cast(pl.UInt64).alias(
            f"{prefix}_views_{suffix}"
        ),
        pl.col(counterpart).filter(mask).n_unique().cast(pl.UInt32).alias(
            f"{prefix}_distinct_{counterpart.replace('_id', 's')}_{suffix}"
        ),
        pl.col("is_positive").filter(mask).sum().fill_null(0).cast(pl.UInt32).alias(
            f"{prefix}_positive_rows_{suffix}"
        ),
        pl.col("is_like").filter(mask).sum().fill_null(0).cast(pl.UInt32).alias(
            f"{prefix}_like_rows_{suffix}"
        ),
        pl.col("is_favorite").filter(mask).sum().fill_null(0).cast(pl.UInt32).alias(
            f"{prefix}_favorite_rows_{suffix}"
        ),
        pl.col("watch_time").filter(mask).mean().fill_null(0.0).cast(pl.Float32).alias(
            f"{prefix}_watch_time_mean_{suffix}"
        ),
        pl.col("watch_time").filter(mask).max().fill_null(0).cast(pl.Int64).alias(
            f"{prefix}_watch_time_max_{suffix}"
        ),
    ]


def _entity_feature_query(
    history: pl.LazyFrame,
    *,
    entity: str,
    counterpart: str,
    prefix: str,
    cutoff: datetime,
    config: HistoryFeatureConfig,
    include_trends: bool,
) -> pl.LazyFrame:
    counterpart_plural = counterpart.replace("_id", "s")
    aggregates: list[pl.Expr] = [
        pl.len().cast(pl.UInt32).alias(f"{prefix}_daily_rows_all"),
        pl.col("views").sum().cast(pl.UInt64).alias(f"{prefix}_views_all"),
        pl.col(counterpart).n_unique().cast(pl.UInt32).alias(
            f"{prefix}_distinct_{counterpart_plural}_all"
        ),
        pl.col("date").n_unique().cast(pl.UInt16).alias(
            f"{prefix}_active_days_all"
        ),
        pl.col("is_positive").sum().cast(pl.UInt32).alias(
            f"{prefix}_positive_rows_all"
        ),
        pl.col("is_like").sum().cast(pl.UInt32).alias(f"{prefix}_like_rows_all"),
        pl.col("is_favorite").sum().cast(pl.UInt32).alias(
            f"{prefix}_favorite_rows_all"
        ),
        pl.col(counterpart)
        .filter(pl.col("is_positive") > 0)
        .n_unique()
        .cast(pl.UInt32)
        .alias(f"{prefix}_positive_{counterpart_plural}_all"),
        pl.col(counterpart)
        .filter(pl.col("is_like") > 0)
        .n_unique()
        .cast(pl.UInt32)
        .alias(f"{prefix}_liked_{counterpart_plural}_all"),
        pl.col(counterpart)
        .filter(pl.col("is_favorite") > 0)
        .n_unique()
        .cast(pl.UInt32)
        .alias(f"{prefix}_favorited_{counterpart_plural}_all"),
        pl.col("watch_time").mean().cast(pl.Float32).alias(
            f"{prefix}_watch_time_mean_all"
        ),
        pl.col("watch_time").max().cast(pl.Int64).alias(
            f"{prefix}_watch_time_max_all"
        ),
        pl.col("dt").max().alias("__last_dt"),
        pl.col("dt").filter(pl.col("is_positive") > 0).max().alias(
            "__last_positive_dt"
        ),
        pl.col("dt").filter(pl.col("is_like") > 0).max().alias("__last_like_dt"),
        pl.col("dt").filter(pl.col("is_favorite") > 0).max().alias(
            "__last_favorite_dt"
        ),
    ]
    for hours in config.windows_hours:
        token = _window_token(hours)
        aggregates.extend(
            _window_aggregates(
                prefix=prefix,
                counterpart=counterpart,
                cutoff=cutoff,
                hours=hours,
                suffix=token,
            )
        )
    if include_trends:
        for hours in config.trend_windows_hours:
            token = _window_token(hours)
            aggregates.extend(
                _window_aggregates(
                    prefix=prefix,
                    counterpart=counterpart,
                    cutoff=cutoff,
                    hours=hours,
                    suffix=f"prev_{token}",
                    start_offset=hours,
                )
            )

    result = history.group_by(entity).agg(aggregates)
    derived: list[pl.Expr] = [
        pl.lit(True).alias(f"{prefix}_history_available"),
        _safe_ratio(
            f"{prefix}_positive_rows_all",
            f"{prefix}_daily_rows_all",
            f"{prefix}_positive_daily_rate_all",
        ),
        _safe_ratio(
            f"{prefix}_like_rows_all",
            f"{prefix}_daily_rows_all",
            f"{prefix}_like_daily_rate_all",
        ),
        _safe_ratio(
            f"{prefix}_favorite_rows_all",
            f"{prefix}_daily_rows_all",
            f"{prefix}_favorite_daily_rate_all",
        ),
        _safe_ratio(
            f"{prefix}_positive_{counterpart_plural}_all",
            f"{prefix}_distinct_{counterpart_plural}_all",
            f"{prefix}_positive_counterpart_rate_all",
        ),
        _safe_ratio(
            f"{prefix}_liked_{counterpart_plural}_all",
            f"{prefix}_distinct_{counterpart_plural}_all",
            f"{prefix}_like_counterpart_rate_all",
        ),
        _safe_ratio(
            f"{prefix}_favorited_{counterpart_plural}_all",
            f"{prefix}_distinct_{counterpart_plural}_all",
            f"{prefix}_favorite_counterpart_rate_all",
        ),
    ]
    for hours in config.windows_hours:
        token = _window_token(hours)
        for signal in ("positive", "like", "favorite"):
            derived.append(
                _safe_ratio(
                    f"{prefix}_{signal}_rows_{token}",
                    f"{prefix}_daily_rows_{token}",
                    f"{prefix}_{signal}_daily_rate_{token}",
                )
            )
        derived.extend(
            (
                _safe_ratio(
                    f"{prefix}_daily_rows_{token}",
                    f"{prefix}_daily_rows_all",
                    f"{prefix}_daily_row_share_{token}",
                ),
                _safe_ratio(
                    f"{prefix}_views_{token}",
                    f"{prefix}_views_all",
                    f"{prefix}_view_share_{token}",
                ),
            )
        )

    reference = pl.lit(cutoff, dtype=pl.Datetime("us"))
    for event, temporary in (
        ("interaction", "__last_dt"),
        ("positive", "__last_positive_dt"),
        ("like", "__last_like_dt"),
        ("favorite", "__last_favorite_dt"),
    ):
        derived.extend(
            (
                pl.col(temporary).is_not_null().alias(
                    f"{prefix}_last_{event}_available"
                ),
                pl.when(pl.col(temporary).is_not_null())
                .then(
                    (reference - pl.col(temporary))
                    .dt.total_microseconds()
                    .cast(pl.Float64)
                    / HOURS_IN_MICROSECOND
                )
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias(f"{prefix}_hours_since_last_{event}"),
            )
        )

    if include_trends:
        for hours in config.trend_windows_hours:
            token = _window_token(hours)
            for signal in ("daily_rows", "views", "positive_rows"):
                current = f"{prefix}_{signal}_{token}"
                previous = f"{prefix}_{signal}_prev_{token}"
                derived.append(
                    (
                        (pl.col(current).cast(pl.Float64) + 1.0).log()
                        - (pl.col(previous).cast(pl.Float64) + 1.0).log()
                    )
                    .cast(pl.Float32)
                    .alias(f"{prefix}_trend_{signal}_log_ratio_{token}")
                )

    temporary_columns = [
        "__last_dt",
        "__last_positive_dt",
        "__last_like_dt",
        "__last_favorite_dt",
    ]
    return result.with_columns(derived).drop(temporary_columns).sort(entity)


def build_user_features(
    history: pl.DataFrame | pl.LazyFrame | str | Path,
    *,
    cutoff: datetime,
    config: HistoryFeatureConfig,
) -> pl.DataFrame:
    """Aggregate deterministic user activity features from fold history."""

    if not isinstance(config, HistoryFeatureConfig):
        raise TypeError("config must be HistoryFeatureConfig")
    result = _entity_feature_query(
        _lazy_daily(history),
        entity="user_id",
        counterpart="item_id",
        prefix="user",
        cutoff=cutoff,
        config=config,
        include_trends=False,
    ).collect(engine="streaming")
    validate_no_nulls(result, name="user feature lookup")
    validate_unique_keys(result, keys=("user_id",), name="user feature lookup")
    return result


def build_item_features(
    history: pl.DataFrame | pl.LazyFrame | str | Path,
    *,
    cutoff: datetime,
    config: HistoryFeatureConfig,
) -> pl.DataFrame:
    """Aggregate deterministic item activity and trend features."""

    if not isinstance(config, HistoryFeatureConfig):
        raise TypeError("config must be HistoryFeatureConfig")
    result = _entity_feature_query(
        _lazy_daily(history),
        entity="item_id",
        counterpart="user_id",
        prefix="item",
        cutoff=cutoff,
        config=config,
        include_trends=True,
    ).collect(engine="streaming")
    validate_no_nulls(result, name="item feature lookup")
    validate_unique_keys(result, keys=("item_id",), name="item feature lookup")
    return result


def _fill_lookup_nulls(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for name in columns:
        dtype = frame.schema[name]
        if dtype == pl.Boolean:
            expressions.append(pl.col(name).fill_null(False))
        elif dtype.is_numeric():
            expressions.append(pl.col(name).fill_null(0))
        else:
            raise ContractValidationError(
                f"unsupported lookup feature dtype for {name}: {dtype}"
            )
    return frame.with_columns(expressions)


def attach_history_features(
    candidates: pl.DataFrame,
    *,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
) -> pl.DataFrame:
    """Join fold-specific lookups while preserving the candidate pair set."""

    if not isinstance(candidates, pl.DataFrame):
        raise TypeError("candidates must be a Polars DataFrame")
    validate_id_columns(candidates)
    validate_unique_keys(candidates, keys=("user_id", "item_id"), name="candidates")
    if user_features.schema.get("user_id") != pl.UInt64:
        raise ContractValidationError("user feature lookup has invalid user_id")
    if item_features.schema.get("item_id") != pl.Int32:
        raise ContractValidationError("item feature lookup has invalid item_id")
    validate_unique_keys(
        user_features, keys=("user_id",), name="user feature lookup"
    )
    validate_unique_keys(
        item_features, keys=("item_id",), name="item feature lookup"
    )
    user_columns = [name for name in user_features.columns if name != "user_id"]
    item_columns = [name for name in item_features.columns if name != "item_id"]
    result = candidates.join(user_features, on="user_id", how="left")
    result = _fill_lookup_nulls(result, user_columns)
    result = result.join(item_features, on="item_id", how="left")
    missing_items = result.filter(pl.col("item_history_available").is_null()).height
    if missing_items:
        raise ContractValidationError(
            f"candidate items missing from fold history features: {missing_items}"
        )
    result = _fill_lookup_nulls(result, item_columns)
    return result.sort(("user_id", "item_id"))


def attach_union_score_ranks(
    frame: pl.DataFrame,
    *,
    sources: Sequence[str] = ("item2item", "implicit_als"),
) -> pl.DataFrame:
    """Rank arbitrary cross-scores inside each user's materialized union."""

    result = frame
    for source in sources:
        score = f"cross_score_{source}"
        available = f"cross_score_available_{source}"
        if result.schema.get(score) != pl.Float64 or result.schema.get(
            available
        ) != pl.Boolean:
            raise ContractValidationError(
                f"missing or invalid cross-score columns for {source}"
            )
        rank = f"union_rank_cross_score_{source}"
        count = f"__available_count_{source}"
        ranked_value = f"__ranked_value_{source}"
        result = (
            result.with_columns(
                pl.when(pl.col(available))
                .then(pl.col(score))
                .otherwise(float("-inf"))
                .alias(ranked_value),
                pl.col(available).sum().over("user_id").cast(pl.UInt32).alias(count),
            )
            .with_columns(
                pl.when(pl.col(available))
                .then(
                    pl.col(ranked_value)
                    .rank(method="ordinal", descending=True)
                    .over("user_id")
                )
                .otherwise(0)
                .cast(pl.UInt32)
                .alias(rank)
            )
            .with_columns(
                pl.when(pl.col(available) & (pl.col(count) > 0))
                .then(
                    (pl.col(count) - pl.col(rank) + 1).cast(pl.Float64)
                    / pl.col(count).cast(pl.Float64)
                )
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias(f"union_rank_norm_cross_score_{source}")
            )
            .drop(count, ranked_value)
        )
    return result.sort(("user_id", "item_id"))


def _lazy_neighbors(
    source: pl.DataFrame | pl.LazyFrame | str | Path,
) -> pl.LazyFrame:
    if isinstance(source, pl.DataFrame):
        frame = source.lazy()
    elif isinstance(source, pl.LazyFrame):
        frame = source
    elif isinstance(source, (str, Path)):
        frame = pl.scan_parquet(source)
    else:
        raise TypeError("neighbor_table must be a Polars frame or Parquet path")
    schema = frame.collect_schema()
    if any(schema.get(name) != dtype for name, dtype in NEIGHBOR_TABLE_SCHEMA.items()):
        raise ContractValidationError("item2item neighbor table has invalid schema")
    return frame


def build_covisit_aggregate_features(
    candidates: pl.DataFrame,
    *,
    seeds: pl.DataFrame,
    neighbor_table: pl.DataFrame | pl.LazyFrame | str | Path,
    config: Item2ItemConfig,
    cutoff: datetime,
) -> pl.DataFrame:
    """Build aggregate seed-neighbor evidence for already existing candidates."""

    if not isinstance(config, Item2ItemConfig):
        raise TypeError("config must be Item2ItemConfig")
    if seeds.schema != COLLAPSED_HISTORY_SCHEMA:
        raise ContractValidationError("item2item seeds have invalid schema")
    validate_no_nulls(seeds, name="item2item seeds")
    validate_unique_keys(seeds, keys=("user_id", "item_id"), name="item2item seeds")
    neighbors = _lazy_neighbors(neighbor_table)
    reference = pl.lit(cutoff, dtype=pl.Datetime("us"))
    age_hours = (
        (reference - pl.col("last_dt"))
        .dt.total_microseconds()
        .cast(pl.Float64)
        / HOURS_IN_MICROSECOND
    )
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
    candidate_pairs = candidates.select("user_id", "item_id").lazy()
    matched = (
        seeds.lazy()
        .filter(pl.col("history_rank") <= config.seed_k)
        .rename({"item_id": "seed_item_id"})
        .join(
            neighbors.select(
                pl.col("item_id").alias("seed_item_id"),
                pl.col("neighbor_item_id").alias("item_id"),
                pl.col("score").alias("neighbor_score"),
                pl.col("co_user_count"),
                pl.col("rank").alias("neighbor_rank"),
            ),
            on="seed_item_id",
            how="inner",
        )
        .join(candidate_pairs, on=("user_id", "item_id"), how="semi")
        .with_columns(
            (pl.col("neighbor_score") * recency * strength).alias("contribution")
        )
        .group_by("user_id", "item_id")
        .agg(
            pl.col("seed_item_id").n_unique().cast(pl.UInt8).alias(
                "covisit_matched_seed_count"
            ),
            pl.col("contribution").sum().cast(pl.Float32).alias(
                "covisit_contribution_sum"
            ),
            pl.col("contribution").mean().cast(pl.Float32).alias(
                "covisit_contribution_mean"
            ),
            pl.col("contribution").max().cast(pl.Float64).alias(
                "covisit_contribution_max"
            ),
            pl.col("neighbor_score").sum().cast(pl.Float32).alias(
                "covisit_neighbor_score_sum"
            ),
            pl.col("neighbor_score").mean().cast(pl.Float32).alias(
                "covisit_neighbor_score_mean"
            ),
            pl.col("neighbor_score").max().cast(pl.Float32).alias(
                "covisit_neighbor_score_max"
            ),
            pl.col("co_user_count").max().cast(pl.UInt32).alias(
                "covisit_co_user_count_max"
            ),
            pl.col("co_user_count").mean().cast(pl.Float32).alias(
                "covisit_co_user_count_mean"
            ),
            pl.col("neighbor_rank").min().cast(pl.UInt32).alias(
                "covisit_best_neighbor_rank"
            ),
        )
        .collect(engine="streaming")
    )
    feature_columns = [
        name for name in matched.columns if name not in {"user_id", "item_id"}
    ]
    result = candidates.join(matched, on=("user_id", "item_id"), how="left")
    result = _fill_lookup_nulls(result, feature_columns).with_columns(
        (pl.col("covisit_matched_seed_count") > 0).alias("covisit_available")
    )
    return result.sort(("user_id", "item_id"))


def build_als_factor_norm_lookups(model: object) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract compact factor norms from a restored portable ALS model."""

    from implicit_model import ImplicitALSModel

    if not isinstance(model, ImplicitALSModel):
        raise TypeError("model must be ImplicitALSModel")
    user_mapping = model.user_mapping.sort("user_index")
    item_mapping = model.item_mapping.sort("item_index")
    user_indices = user_mapping.get_column("user_index").to_numpy()
    item_indices = item_mapping.get_column("item_index").to_numpy()
    if not np.array_equal(user_indices, np.arange(len(user_indices))):
        raise ContractValidationError("ALS user mapping is not contiguous")
    if not np.array_equal(item_indices, np.arange(len(item_indices))):
        raise ContractValidationError("ALS item mapping is not contiguous")
    user_factors = model.backend.user_factors
    item_factors = model.backend.item_factors
    if user_factors.shape[0] != user_mapping.height:
        raise ContractValidationError("ALS user factors differ from mapping")
    if item_factors.shape[0] != item_mapping.height:
        raise ContractValidationError("ALS item factors differ from mapping")
    user_norms = np.linalg.norm(user_factors, axis=1).astype(np.float32)
    item_norms = np.linalg.norm(item_factors, axis=1).astype(np.float32)
    if not np.isfinite(user_norms).all() or not np.isfinite(item_norms).all():
        raise ContractValidationError("ALS factor norms must be finite")
    return (
        user_mapping.select("user_id").with_columns(
            pl.Series("als_user_factor_norm", user_norms, dtype=pl.Float32)
        ),
        item_mapping.select("item_id").with_columns(
            pl.Series("als_item_factor_norm", item_norms, dtype=pl.Float32)
        ),
    )


def attach_als_factor_features(
    candidates: pl.DataFrame,
    *,
    user_norms: pl.DataFrame,
    item_norms: pl.DataFrame,
) -> pl.DataFrame:
    """Attach ALS norms and cosine similarity without invoking model inference."""

    if user_norms.schema != {
        "user_id": pl.UInt64,
        "als_user_factor_norm": pl.Float32,
    }:
        raise ContractValidationError("ALS user norm lookup has invalid schema")
    if item_norms.schema != {
        "item_id": pl.Int32,
        "als_item_factor_norm": pl.Float32,
    }:
        raise ContractValidationError("ALS item norm lookup has invalid schema")
    result = (
        candidates.join(user_norms, on="user_id", how="left")
        .join(item_norms, on="item_id", how="left")
        .with_columns(
            (
                pl.col("als_user_factor_norm").is_not_null()
                & pl.col("als_item_factor_norm").is_not_null()
            ).alias("__als_mapping_available"),
            pl.col("als_user_factor_norm").fill_null(0.0),
            pl.col("als_item_factor_norm").fill_null(0.0),
        )
        .with_columns(
            (
                pl.col("als_user_factor_norm").cast(pl.Float64)
                * pl.col("als_item_factor_norm").cast(pl.Float64)
            )
            .cast(pl.Float32)
            .alias("als_factor_norm_product")
        )
        .with_columns(
            (
                pl.col("__als_mapping_available")
                & (pl.col("als_factor_norm_product") > 0)
                & pl.col("cross_score_available_implicit_als")
            ).alias("als_cosine_available")
        )
        .with_columns(
            pl.when(pl.col("als_cosine_available"))
            .then(
                pl.col("cross_score_implicit_als")
                / pl.col("als_factor_norm_product").cast(pl.Float64)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("als_cosine_similarity")
        )
    )
    mismatch = result.filter(
        pl.col("__als_mapping_available")
        != pl.col("cross_score_available_implicit_als")
    ).height
    if mismatch:
        raise ContractValidationError(
            f"ALS norm availability differs from task06 cross-score: {mismatch}"
        )
    return result.drop("__als_mapping_available").sort(("user_id", "item_id"))


__all__ = [
    "HistoryFeatureConfig",
    "attach_als_factor_features",
    "attach_history_features",
    "attach_union_score_ranks",
    "build_als_factor_norm_lookups",
    "build_covisit_aggregate_features",
    "build_item_features",
    "build_user_features",
    "validate_history_cutoff",
]

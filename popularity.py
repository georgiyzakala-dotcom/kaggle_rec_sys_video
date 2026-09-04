"""History-only global-popularity candidates and deterministic fallback.

The loader consumes the shared daily fold artifact.  It prepares the four
statistics required by task 02 once per fold and exposes target users together
with their exact seen-item sets.  The model never reads raw or validation data.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from interfaces import (
    CANDIDATE_SCHEMA,
    FINAL_RECOMMENDATION_SCHEMA,
    CandidateDataLoader,
    CandidateModel,
)
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_final_recommendations,
    validate_id_columns,
    validate_no_nulls,
    validate_unique_keys,
)


class PopularityScore(str, Enum):
    """Supported fold-history item popularity definitions."""

    RAW_INTERACTION_COUNT = "raw_interaction_count"
    DISTINCT_INTERACTING_USERS = "distinct_interacting_users"
    RELEVANT_INTERACTION_COUNT = "relevant_interaction_count"
    DISTINCT_RELEVANT_USERS = "distinct_relevant_users"


POPULARITY_SCORE_ORDER: Final = tuple(score.value for score in PopularityScore)


class RecencySignal(str, Enum):
    """Signals available from the shared daily interaction snapshot."""

    RAW_VIEWS = "raw_views"
    POSITIVE_DAILY_ROWS = "positive_daily_rows"


class RecencyScoreKind(str, Enum):
    """Supported temporal popularity score families."""

    WINDOW = "window"
    DECAY = "decay"
    TRENDING = "trending"
    WINDOW_BLEND = "window_blend"


RECENCY_SIGNAL_ORDER: Final = tuple(signal.value for signal in RecencySignal)
RECENCY_SCORE_KIND_ORDER: Final = tuple(kind.value for kind in RecencyScoreKind)


def _positive_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _window_value(value: object, *, name: str) -> float | None:
    if value is None or value == "full":
        return None
    return _positive_finite_number(value, name=name)


@dataclass(frozen=True)
class RecencyPopularityConfig:
    """Portable configuration for one temporal-popularity candidate source."""

    config_id: str
    score_kind: RecencyScoreKind | str
    signal: RecencySignal | str
    window_hours: float | None = None
    half_life_hours: float | None = None
    short_window_hours: float | None = None
    long_window_hours: float | None = None
    smoothing: float | None = None
    window_weights: tuple[tuple[float | None, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise ValueError("config_id must be a non-empty string")
        try:
            score_kind = RecencyScoreKind(self.score_kind)
        except ValueError as error:
            raise ValueError(
                f"score_kind must be one of {list(RECENCY_SCORE_KIND_ORDER)}"
            ) from error
        try:
            signal = RecencySignal(self.signal)
        except ValueError as error:
            raise ValueError(
                f"signal must be one of {list(RECENCY_SIGNAL_ORDER)}"
            ) from error
        object.__setattr__(self, "score_kind", score_kind)
        object.__setattr__(self, "signal", signal)

        optional_values = {
            "window_hours": self.window_hours,
            "half_life_hours": self.half_life_hours,
            "short_window_hours": self.short_window_hours,
            "long_window_hours": self.long_window_hours,
            "smoothing": self.smoothing,
        }
        checked: dict[str, float | None] = {}
        for name, value in optional_values.items():
            checked[name] = (
                None
                if value is None
                else _positive_finite_number(value, name=name)
            )
            object.__setattr__(self, name, checked[name])

        normalized_weights: list[tuple[float | None, float]] = []
        seen_windows: set[float | None] = set()
        for index, pair in enumerate(self.window_weights):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TypeError(
                    "window_weights entries must be (window_hours, weight) pairs"
                )
            window = _window_value(
                pair[0], name=f"window_weights[{index}].window_hours"
            )
            weight = _positive_finite_number(
                pair[1], name=f"window_weights[{index}].weight"
            )
            if window in seen_windows:
                raise ValueError("window_weights must not repeat a window")
            seen_windows.add(window)
            normalized_weights.append((window, weight))
        object.__setattr__(self, "window_weights", tuple(normalized_weights))

        used = {
            "window_hours": checked["window_hours"] is not None,
            "half_life_hours": checked["half_life_hours"] is not None,
            "short_window_hours": checked["short_window_hours"] is not None,
            "long_window_hours": checked["long_window_hours"] is not None,
            "smoothing": checked["smoothing"] is not None,
            "window_weights": bool(normalized_weights),
        }
        allowed_by_kind = {
            RecencyScoreKind.WINDOW: {"window_hours"},
            RecencyScoreKind.DECAY: {"half_life_hours"},
            RecencyScoreKind.TRENDING: {
                "short_window_hours",
                "long_window_hours",
                "smoothing",
            },
            RecencyScoreKind.WINDOW_BLEND: {"window_weights"},
        }
        required = allowed_by_kind[score_kind].difference({"window_hours"})
        missing = sorted(name for name in required if not used[name])
        unexpected = sorted(
            name
            for name, present in used.items()
            if present and name not in allowed_by_kind[score_kind]
        )
        if missing or unexpected:
            raise ValueError(
                f"invalid {score_kind.value} parameters: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if score_kind is RecencyScoreKind.TRENDING:
            assert self.short_window_hours is not None
            assert self.long_window_hours is not None
            if self.short_window_hours >= self.long_window_hours:
                raise ValueError(
                    "short_window_hours must be less than long_window_hours"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RecencyPopularityConfig:
        """Validate and construct a config read from JSON."""

        if not isinstance(value, Mapping):
            raise TypeError("recency model config must be a mapping")
        allowed = {
            "config_id",
            "score_kind",
            "signal",
            "window_hours",
            "half_life_hours",
            "short_window_hours",
            "long_window_hours",
            "smoothing",
            "window_weights",
        }
        unexpected = sorted(set(value).difference(allowed))
        if unexpected:
            raise ValueError(f"unexpected recency config fields: {unexpected}")
        if value.get("score_kind") == RecencyScoreKind.WINDOW.value and (
            "window_hours" not in value
        ):
            raise ValueError("window config requires window_hours (null means full)")
        weights_value = value.get("window_weights", ())
        if not isinstance(weights_value, (list, tuple)):
            raise TypeError("window_weights must be a list")
        weights: list[tuple[float | None, float]] = []
        for index, entry in enumerate(weights_value):
            if not isinstance(entry, Mapping):
                raise TypeError(f"window_weights[{index}] must be an object")
            if set(entry) != {"window_hours", "weight"}:
                raise ValueError(
                    "window weight objects require window_hours and weight"
                )
            weights.append(
                (
                    _window_value(
                        entry["window_hours"],
                        name=f"window_weights[{index}].window_hours",
                    ),
                    entry["weight"],
                )
            )
        return cls(
            config_id=value.get("config_id", ""),
            score_kind=value.get("score_kind", ""),
            signal=value.get("signal", ""),
            window_hours=_window_value(
                value.get("window_hours"), name="window_hours"
            ),
            half_life_hours=value.get("half_life_hours"),
            short_window_hours=value.get("short_window_hours"),
            long_window_hours=value.get("long_window_hours"),
            smoothing=value.get("smoothing"),
            window_weights=tuple(weights),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a minimal JSON-serializable representation."""

        result: dict[str, Any] = {
            "config_id": self.config_id,
            "score_kind": self.score_kind.value,
            "signal": self.signal.value,
        }
        if self.score_kind is RecencyScoreKind.WINDOW:
            result["window_hours"] = self.window_hours
        elif self.score_kind is RecencyScoreKind.DECAY:
            result["half_life_hours"] = self.half_life_hours
        elif self.score_kind is RecencyScoreKind.TRENDING:
            result.update(
                {
                    "short_window_hours": self.short_window_hours,
                    "long_window_hours": self.long_window_hours,
                    "smoothing": self.smoothing,
                }
            )
        else:
            result["window_weights"] = [
                {
                    "window_hours": window,
                    "weight": weight,
                }
                for window, weight in self.window_weights
            ]
        return result
ITEM_POPULARITY_STATS_SCHEMA: Final = pl.Schema(
    {
        "item_id": pl.Int32,
        "raw_interaction_count": pl.UInt64,
        "distinct_interacting_users": pl.UInt64,
        "relevant_interaction_count": pl.UInt64,
        "distinct_relevant_users": pl.UInt64,
    }
)
ITEM_RANKING_SCHEMA: Final = pl.Schema(
    {
        "item_id": pl.Int32,
        "score": pl.Float64,
        "global_rank": pl.UInt32,
    }
)
PREDICTION_USER_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "seen_item_ids": pl.List(pl.Int32),
    }
)


@dataclass(frozen=True)
class PopularityPredictBatch:
    """Sorted target users and their history-seen item lists."""

    users: pl.DataFrame


def _daily_source(source: object, *, name: str) -> tuple[pl.LazyFrame, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        frame = pl.scan_parquet(path)
        description = path.as_posix()
    elif isinstance(source, pl.DataFrame):
        frame = source.lazy()
        description = "<in_memory_dataframe>"
    elif isinstance(source, pl.LazyFrame):
        frame = source
        description = "<in_memory_lazyframe>"
    else:
        raise TypeError(
            f"{name} must be a path, polars.DataFrame, or polars.LazyFrame"
        )
    schema = frame.collect_schema()
    if schema != DAILY_INTERACTION_SCHEMA:
        raise ContractValidationError(
            f"{name} schema must be {DAILY_INTERACTION_SCHEMA}, got {schema}"
        )
    return frame, description


def _target_source(source: object) -> tuple[pl.LazyFrame, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"target users do not exist: {path}")
        frame = pl.scan_parquet(path)
        description = path.as_posix()
    elif isinstance(source, pl.DataFrame):
        frame = source.lazy()
        description = "<in_memory_dataframe>"
    elif isinstance(source, pl.LazyFrame):
        frame = source
        description = "<in_memory_lazyframe>"
    else:
        raise TypeError(
            "target users must be a path, polars.DataFrame, or polars.LazyFrame"
        )
    schema = frame.collect_schema()
    if schema != TARGET_USER_SCHEMA:
        raise ContractValidationError(
            f"target users schema must be {TARGET_USER_SCHEMA}, got {schema}"
        )
    return frame, description


def _hours_token(hours: float) -> str:
    return format(hours, ".12g").replace(".", "p")


def _window_column(signal: RecencySignal, hours: float | None) -> str:
    suffix = "full" if hours is None else f"{_hours_token(hours)}h"
    return f"{signal.value}__window__{suffix}"


def _decay_column(signal: RecencySignal, half_life_hours: float) -> str:
    return f"{signal.value}__decay__{_hours_token(half_life_hours)}h"


def _normalize_grid(
    values: Sequence[float], *, name: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive numbers")
    checked = tuple(
        _positive_finite_number(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(checked))


def _prepare_prediction_users(
    history: pl.LazyFrame, targets_source: pl.LazyFrame
) -> pl.DataFrame:
    targets = targets_source.collect(engine="streaming")
    validate_id_columns(targets, require_item=False)
    validate_no_nulls(targets, name="target users")
    validate_unique_keys(targets, keys=("user_id",), name="target users")
    targets = targets.sort("user_id")

    seen = (
        history.select("user_id", "item_id")
        .join(targets.lazy(), on="user_id", how="semi")
        .unique()
        .group_by("user_id")
        .agg(pl.col("item_id").sort().alias("seen_item_ids"))
    )
    users = (
        targets.lazy()
        .join(seen, on="user_id", how="left")
        .with_columns(
            pl.col("seen_item_ids").fill_null(
                pl.lit([], dtype=pl.List(pl.Int32))
            )
        )
        .select(PREDICTION_USER_SCHEMA.names())
        .cast(PREDICTION_USER_SCHEMA)
        .sort("user_id")
        .collect(engine="streaming")
    )
    validate_no_nulls(users, name="popularity prediction users")
    validate_unique_keys(
        users, keys=("user_id",), name="popularity prediction users"
    )
    return users


class PopularityDataLoader(
    CandidateDataLoader[pl.DataFrame, PopularityPredictBatch]
):
    """Prepare global item statistics and target-user seen sets."""

    def __init__(self, *, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._fit_history: pl.LazyFrame | None = None
        self._predict_history: pl.LazyFrame | None = None
        self._predict_targets: pl.LazyFrame | None = None
        self._item_stats: pl.DataFrame | None = None
        self._prediction_users: pl.DataFrame | None = None
        self._fit_source = "<not_loaded>"
        self._predict_history_source = "<not_loaded>"
        self._target_source = "<not_loaded>"

    @property
    def item_stats(self) -> pl.DataFrame:
        if self._item_stats is None:
            raise RuntimeError("fit data has not been prepared")
        return self._item_stats

    @property
    def prediction_users(self) -> pl.DataFrame:
        if self._prediction_users is None:
            raise RuntimeError("prediction data has not been prepared")
        return self._prediction_users

    def _load_fit_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history"}:
            raise TypeError("load_fit_data requires only history=")
        self._fit_history, self._fit_source = _daily_source(
            kwargs["history"], name="fit history"
        )
        self._item_stats = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_fit_data does not accept arguments")
        assert self._fit_history is not None
        stats = (
            self._fit_history.group_by("item_id")
            .agg(
                pl.col("views")
                .sum()
                .cast(pl.UInt64)
                .alias(PopularityScore.RAW_INTERACTION_COUNT.value),
                pl.col("user_id")
                .n_unique()
                .cast(pl.UInt64)
                .alias(PopularityScore.DISTINCT_INTERACTING_USERS.value),
                pl.col("is_positive")
                .sum()
                .cast(pl.UInt64)
                .alias(PopularityScore.RELEVANT_INTERACTION_COUNT.value),
                pl.col("user_id")
                .filter(pl.col("is_positive") == 1)
                .n_unique()
                .cast(pl.UInt64)
                .alias(PopularityScore.DISTINCT_RELEVANT_USERS.value),
            )
            .select(ITEM_POPULARITY_STATS_SCHEMA.names())
            .cast(ITEM_POPULARITY_STATS_SCHEMA)
            .sort("item_id")
            .collect(engine="streaming")
        )
        if stats.schema != ITEM_POPULARITY_STATS_SCHEMA:
            raise ContractValidationError("invalid prepared item-stat schema")
        validate_no_nulls(stats, name="item popularity statistics")
        validate_unique_keys(
            stats, keys=("item_id",), name="item popularity statistics"
        )
        self._item_stats = stats

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[pl.DataFrame]:
        assert self._item_stats is not None
        size = batch_size or max(self._item_stats.height, 1)
        for offset in range(0, self._item_stats.height, size):
            yield self._item_stats.slice(offset, size)

    def _load_predict_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history", "target_users"}:
            raise TypeError(
                "load_predict_data requires history= and target_users="
            )
        self._predict_history, self._predict_history_source = _daily_source(
            kwargs["history"], name="prediction history"
        )
        self._predict_targets, self._target_source = _target_source(
            kwargs["target_users"]
        )
        self._prediction_users = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_predict_data does not accept arguments")
        assert self._predict_history is not None
        assert self._predict_targets is not None
        self._prediction_users = _prepare_prediction_users(
            self._predict_history, self._predict_targets
        )

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[PopularityPredictBatch]:
        assert self._prediction_users is not None
        size = batch_size or max(self._prediction_users.height, 1)
        for offset in range(0, self._prediction_users.height, size):
            yield PopularityPredictBatch(
                users=self._prediction_users.slice(offset, size)
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "popularity",
            "seed": self.seed,
            "fit_history": self._fit_source,
            "predict_history": self._predict_history_source,
            "target_users": self._target_source,
            "statistics": list(POPULARITY_SCORE_ORDER),
            "positive_unit": "daily_user_item_row",
        }


class RecencyPopularityDataLoader(
    CandidateDataLoader[pl.DataFrame, PopularityPredictBatch]
):
    """Prepare reusable temporal item statistics from fold history only."""

    def __init__(
        self,
        *,
        reference_time: datetime,
        windows_hours: Sequence[float],
        half_lives_hours: Sequence[float],
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        if not isinstance(reference_time, datetime):
            raise TypeError("reference_time must be a datetime")
        if reference_time.tzinfo is not None:
            raise ValueError("reference_time must be timezone-naive")
        self._reference_time = reference_time
        self._windows_hours = _normalize_grid(
            windows_hours, name="windows_hours"
        )
        self._half_lives_hours = _normalize_grid(
            half_lives_hours, name="half_lives_hours"
        )
        self._fit_history: pl.LazyFrame | None = None
        self._predict_history: pl.LazyFrame | None = None
        self._predict_targets: pl.LazyFrame | None = None
        self._item_stats: pl.DataFrame | None = None
        self._prediction_users: pl.DataFrame | None = None
        self._history_span_hours: float | None = None
        self._fit_source = "<not_loaded>"
        self._predict_history_source = "<not_loaded>"
        self._target_source = "<not_loaded>"

    @property
    def reference_time(self) -> datetime:
        return self._reference_time

    @property
    def windows_hours(self) -> tuple[float, ...]:
        return self._windows_hours

    @property
    def half_lives_hours(self) -> tuple[float, ...]:
        return self._half_lives_hours

    @property
    def history_span_hours(self) -> float:
        if self._history_span_hours is None:
            raise RuntimeError("fit data has not been prepared")
        return self._history_span_hours

    @property
    def item_stats(self) -> pl.DataFrame:
        if self._item_stats is None:
            raise RuntimeError("fit data has not been prepared")
        return self._item_stats.clone()

    @property
    def prediction_users(self) -> pl.DataFrame:
        if self._prediction_users is None:
            raise RuntimeError("prediction data has not been prepared")
        return self._prediction_users.clone()

    def _load_fit_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history"}:
            raise TypeError("load_fit_data requires only history=")
        self._fit_history, self._fit_source = _daily_source(
            kwargs["history"], name="recency fit history"
        )
        self._item_stats = None
        self._history_span_hours = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_fit_data does not accept arguments")
        assert self._fit_history is not None
        time_summary = self._fit_history.select(
            pl.len().alias("rows"),
            pl.col("dt").min().alias("min_dt"),
            pl.col("dt").max().alias("max_dt"),
        ).collect(engine="streaming")
        row = time_summary.row(0, named=True)
        if row["rows"] == 0:
            raise ValueError("recency fit history must not be empty")
        min_dt = row["min_dt"]
        max_dt = row["max_dt"]
        if not isinstance(min_dt, datetime) or not isinstance(max_dt, datetime):
            raise ContractValidationError("history timestamps are invalid")
        if max_dt >= self._reference_time:
            raise ContractValidationError(
                "recency fit history contains dt at or after reference_time"
            )
        history_span_hours = (
            self._reference_time - min_dt
        ).total_seconds() / 3600.0
        if history_span_hours <= 0:
            raise ContractValidationError("history span must be positive")

        age_hours = (
            (
                pl.lit(self._reference_time, dtype=pl.Datetime("us"))
                - pl.col("dt")
            )
            .dt.total_microseconds()
            .cast(pl.Float64)
            / 3_600_000_000.0
        )
        expressions: list[pl.Expr] = []
        for signal in RecencySignal:
            signal_expression = (
                pl.col("views").cast(pl.Float64)
                if signal is RecencySignal.RAW_VIEWS
                else pl.col("is_positive").cast(pl.Float64)
            )
            expressions.append(
                signal_expression.sum().alias(_window_column(signal, None))
            )
            for hours in self._windows_hours:
                boundary = self._reference_time - timedelta(hours=hours)
                expressions.append(
                    pl.when(pl.col("dt") >= boundary)
                    .then(signal_expression)
                    .otherwise(0.0)
                    .sum()
                    .alias(_window_column(signal, hours))
                )
            for half_life in self._half_lives_hours:
                decay = (-math.log(2.0) * age_hours / half_life).exp()
                expressions.append(
                    (signal_expression * decay)
                    .sum()
                    .alias(_decay_column(signal, half_life))
                )

        stats = (
            self._fit_history.group_by("item_id")
            .agg(expressions)
            .sort("item_id")
            .collect(engine="streaming")
        )
        expected_schema = pl.Schema(
            {"item_id": pl.Int32}
            | {column: pl.Float64 for column in stats.columns[1:]}
        )
        stats = stats.cast(expected_schema)
        validate_no_nulls(stats, name="recency popularity statistics")
        validate_unique_keys(
            stats, keys=("item_id",), name="recency popularity statistics"
        )
        score_columns = stats.columns[1:]
        if not score_columns:
            raise ContractValidationError("recency statistics have no score columns")
        finite = stats.select(
            pl.col(column).is_finite().all().alias(column)
            for column in score_columns
        ).row(0)
        if not all(finite):
            raise ContractValidationError("recency statistics must be finite")
        self._item_stats = stats
        self._history_span_hours = history_span_hours

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[pl.DataFrame]:
        assert self._item_stats is not None
        size = batch_size or max(self._item_stats.height, 1)
        for offset in range(0, self._item_stats.height, size):
            yield self._item_stats.slice(offset, size)

    def _load_predict_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history", "target_users"}:
            raise TypeError(
                "load_predict_data requires history= and target_users="
            )
        self._predict_history, self._predict_history_source = _daily_source(
            kwargs["history"], name="recency prediction history"
        )
        self._predict_targets, self._target_source = _target_source(
            kwargs["target_users"]
        )
        self._prediction_users = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_predict_data does not accept arguments")
        assert self._predict_history is not None
        assert self._predict_targets is not None
        self._prediction_users = _prepare_prediction_users(
            self._predict_history, self._predict_targets
        )

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[PopularityPredictBatch]:
        assert self._prediction_users is not None
        size = batch_size or max(self._prediction_users.height, 1)
        for offset in range(0, self._prediction_users.height, size):
            yield PopularityPredictBatch(
                users=self._prediction_users.slice(offset, size)
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "recency_popularity",
            "seed": self.seed,
            "fit_history": self._fit_source,
            "predict_history": self._predict_history_source,
            "target_users": self._target_source,
            "reference_time": self._reference_time.isoformat(),
            "windows_hours": list(self._windows_hours),
            "half_lives_hours": list(self._half_lives_hours),
            "history_span_hours": self._history_span_hours,
            "timestamp_semantics": "daily_row_first_timestamp",
            "raw_signal": "daily_row_views_raw_event_count",
            "positive_signal": "daily_user_item_positive_row",
        }


def _predict_ranked_items(
    loader: CandidateDataLoader[Any, Any],
    *,
    ranked_item_ids: np.ndarray[Any, np.dtype[np.int32]],
    ranked_scores: np.ndarray[Any, np.dtype[np.float64]],
    source_name: str,
    k: int,
    batch_size: int | None,
    require_exact: bool,
) -> pl.DataFrame:
    outputs: list[pl.DataFrame] = []
    for batch in loader.iter_predict_batches(batch_size=batch_size):
        if not isinstance(batch, PopularityPredictBatch):
            raise TypeError("popularity models require PopularityPredictBatch")
        user_count = batch.users.height
        if not user_count:
            continue
        capacity = user_count * k
        user_values = np.empty(capacity, dtype=np.uint64)
        item_values = np.empty(capacity, dtype=np.int32)
        score_values = np.empty(capacity, dtype=np.float64)
        rank_values = np.empty(capacity, dtype=np.uint32)

        position = 0
        for user_id, seen_items in batch.users.iter_rows():
            seen = set(seen_items)
            selected_indices: list[int] = []
            for index, item_id in enumerate(ranked_item_ids):
                if int(item_id) not in seen:
                    selected_indices.append(index)
                    if len(selected_indices) == k:
                        break
            selected_count = len(selected_indices)
            if require_exact and selected_count != k:
                available = len(ranked_item_ids) - len(seen)
                raise ValueError(
                    f"user_id={user_id} has only {available} known unseen "
                    f"items, fewer than requested k={k}"
                )
            if not selected_count:
                continue
            end = position + selected_count
            indices = np.asarray(selected_indices, dtype=np.int64)
            user_values[position:end] = user_id
            item_values[position:end] = ranked_item_ids[indices]
            score_values[position:end] = ranked_scores[indices]
            rank_values[position:end] = np.arange(
                1, selected_count + 1, dtype=np.uint32
            )
            position = end

        if position:
            output = pl.DataFrame(
                {
                    "user_id": pl.Series(
                        user_values[:position], dtype=pl.UInt64
                    ),
                    "item_id": pl.Series(
                        item_values[:position], dtype=pl.Int32
                    ),
                    "score": pl.Series(
                        score_values[:position], dtype=pl.Float64
                    ),
                    "rank": pl.Series(rank_values[:position], dtype=pl.UInt32),
                }
            ).with_columns(pl.lit(source_name).alias("source"))
            outputs.append(output.select(CANDIDATE_SCHEMA.names()))

    if not outputs:
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)
    return pl.concat(outputs, rechunk=True).cast(CANDIDATE_SCHEMA)


class RecencyPopularityModel(CandidateModel):
    """History-only temporal popularity candidate generator."""

    def __init__(self, config: RecencyPopularityConfig | Mapping[str, Any]) -> None:
        self._config = (
            config
            if isinstance(config, RecencyPopularityConfig)
            else RecencyPopularityConfig.from_dict(config)
        )
        self._item_ranking: pl.DataFrame | None = None
        self._ranked_item_ids: np.ndarray[Any, np.dtype[np.int32]] | None = None
        self._ranked_scores: np.ndarray[Any, np.dtype[np.float64]] | None = None

    @property
    def source_name(self) -> str:
        return "recency_popularity"

    @property
    def config(self) -> RecencyPopularityConfig:
        return self._config

    @property
    def item_ranking(self) -> pl.DataFrame:
        if self._item_ranking is None:
            raise RuntimeError("model has not been fitted")
        return self._item_ranking.clone()

    @classmethod
    def from_fitted_ranking(
        cls,
        config: RecencyPopularityConfig | Mapping[str, Any],
        item_ranking: pl.DataFrame,
    ) -> RecencyPopularityModel:
        """Restore fitted state supplied by orchestration after artifact load."""

        model = cls(config)
        model._set_item_ranking(item_ranking)
        return model

    def _score_expression(
        self, loader: RecencyPopularityDataLoader
    ) -> pl.Expr:
        config = self._config
        signal = RecencySignal(config.signal)
        if config.score_kind is RecencyScoreKind.WINDOW:
            return pl.col(_window_column(signal, config.window_hours))
        if config.score_kind is RecencyScoreKind.DECAY:
            assert config.half_life_hours is not None
            return pl.col(_decay_column(signal, config.half_life_hours))
        if config.score_kind is RecencyScoreKind.WINDOW_BLEND:
            expressions = [
                pl.col(_window_column(signal, window)) * weight
                for window, weight in config.window_weights
            ]
            return pl.sum_horizontal(expressions)

        assert config.short_window_hours is not None
        assert config.long_window_hours is not None
        assert config.smoothing is not None
        short = pl.col(_window_column(signal, config.short_window_hours))
        long = pl.col(_window_column(signal, config.long_window_hours))
        effective_short = min(
            config.short_window_hours, loader.history_span_hours
        )
        effective_long = min(
            config.long_window_hours, loader.history_span_hours
        )
        prior_fraction = min(1.0, effective_short / effective_long)
        smoothed_lift = (
            (short + config.smoothing * prior_fraction)
            / (long + config.smoothing)
            / prior_fraction
        )
        return short.log1p() * smoothed_lift

    def _set_item_ranking(self, ranking: pl.DataFrame) -> None:
        if not isinstance(ranking, pl.DataFrame):
            raise TypeError("item_ranking must be a polars.DataFrame")
        if ranking.schema != ITEM_RANKING_SCHEMA:
            raise ContractValidationError(
                f"item ranking schema must be {ITEM_RANKING_SCHEMA}"
            )
        validate_no_nulls(ranking, name="recency item ranking")
        validate_unique_keys(
            ranking, keys=("item_id",), name="recency item ranking"
        )
        if ranking.height == 0:
            raise ContractValidationError("recency item ranking must not be empty")
        if not ranking.get_column("score").is_finite().all():
            raise ContractValidationError("recency scores must be finite")
        if (ranking.get_column("score") <= 0).any():
            raise ContractValidationError("recency scores must be positive")
        expected = (
            ranking.sort(("score", "item_id"), descending=(True, False))
            .with_row_index("expected_rank", offset=1)
            .select("item_id", "score", "expected_rank")
        )
        if (
            not ranking.select("item_id", "score").equals(
                expected.select("item_id", "score")
            )
            or not ranking.get_column("global_rank").equals(
                expected.get_column("expected_rank").rename("global_rank")
            )
        ):
            raise ContractValidationError(
                "recency item ranking must follow score DESC, item_id ASC "
                "with contiguous ranks"
            )
        self._item_ranking = ranking.clone()
        self._ranked_item_ids = ranking.get_column("item_id").to_numpy()
        self._ranked_scores = ranking.get_column("score").to_numpy()

    def _fit(
        self, loader: CandidateDataLoader[Any, Any], **kwargs: Any
    ) -> None:
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unexpected fit arguments: {sorted(kwargs)}")
        if not isinstance(loader, RecencyPopularityDataLoader):
            raise TypeError(
                "RecencyPopularityModel requires RecencyPopularityDataLoader"
            )
        batches = list(loader.iter_fit_batches(batch_size=batch_size))
        if not batches:
            raise ValueError("recency fit history must contain known items")
        stats = pl.concat(batches, rechunk=True)
        try:
            score_expression = self._score_expression(loader)
            ranking = (
                stats.select(
                    "item_id", score_expression.cast(pl.Float64).alias("score")
                )
                .filter(pl.col("score") > 0)
                .sort(("score", "item_id"), descending=(True, False))
                .with_row_index("global_rank", offset=1)
                .select(ITEM_RANKING_SCHEMA.names())
                .cast(ITEM_RANKING_SCHEMA)
            )
        except pl.exceptions.ColumnNotFoundError as error:
            raise ValueError(
                "loader temporal grid does not cover the model configuration"
            ) from error
        self._set_item_ranking(ranking)

    def _predict(
        self,
        loader: CandidateDataLoader[Any, Any],
        *,
        k: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unexpected predict arguments: {sorted(kwargs)}")
        if not isinstance(loader, RecencyPopularityDataLoader):
            raise TypeError(
                "RecencyPopularityModel requires RecencyPopularityDataLoader"
            )
        if self._ranked_item_ids is None or self._ranked_scores is None:
            raise RuntimeError("fit must be called before predict")
        return _predict_ranked_items(
            loader,
            ranked_item_ids=self._ranked_item_ids,
            ranked_scores=self._ranked_scores,
            source_name=self.source_name,
            k=k,
            batch_size=batch_size,
            require_exact=False,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "model": "recency_popularity",
            "source_name": self.source_name,
            "recency_config": self._config.to_dict(),
            "tie_break": ["score DESC", "item_id ASC"],
            "timestamp_semantics": "daily_row_first_timestamp",
        }


class GlobalPopularityModel(CandidateModel):
    """Rank history-known items globally and emit unseen top-k per user."""

    def __init__(self, score_type: PopularityScore | str) -> None:
        try:
            self._score_type = PopularityScore(score_type)
        except ValueError as error:
            raise ValueError(
                f"score_type must be one of {list(POPULARITY_SCORE_ORDER)}"
            ) from error
        self._item_ranking: pl.DataFrame | None = None
        self._ranked_item_ids: np.ndarray[Any, np.dtype[np.int32]] | None = None
        self._ranked_scores: np.ndarray[Any, np.dtype[np.float64]] | None = None

    @property
    def source_name(self) -> str:
        return "global_popularity"

    @property
    def score_type(self) -> PopularityScore:
        return self._score_type

    @property
    def item_ranking(self) -> pl.DataFrame:
        if self._item_ranking is None:
            raise RuntimeError("model has not been fitted")
        return self._item_ranking.clone()

    @classmethod
    def from_fitted_ranking(
        cls,
        score_type: PopularityScore | str,
        item_ranking: pl.DataFrame,
    ) -> GlobalPopularityModel:
        """Restore fitted global ranking without repeating history fit."""

        model = cls(score_type)
        model._set_item_ranking(item_ranking)
        return model

    def _set_item_ranking(self, ranking: pl.DataFrame) -> None:
        if not isinstance(ranking, pl.DataFrame):
            raise TypeError("item_ranking must be a polars.DataFrame")
        if ranking.schema != ITEM_RANKING_SCHEMA:
            raise ContractValidationError(
                f"item ranking schema must be {ITEM_RANKING_SCHEMA}"
            )
        validate_no_nulls(ranking, name="global item ranking")
        validate_unique_keys(
            ranking, keys=("item_id",), name="global item ranking"
        )
        if ranking.is_empty():
            raise ContractValidationError("global item ranking must not be empty")
        if not ranking.get_column("score").is_finite().all():
            raise ContractValidationError("global scores must be finite")
        expected = (
            ranking.sort(("score", "item_id"), descending=(True, False))
            .with_row_index("expected_rank", offset=1)
            .select("item_id", "score", "expected_rank")
        )
        if (
            not ranking.select("item_id", "score").equals(
                expected.select("item_id", "score")
            )
            or not ranking.get_column("global_rank").equals(
                expected.get_column("expected_rank").rename("global_rank")
            )
        ):
            raise ContractValidationError(
                "global item ranking must follow score DESC, item_id ASC "
                "with contiguous ranks"
            )
        self._item_ranking = ranking.clone()
        self._ranked_item_ids = ranking.get_column("item_id").to_numpy()
        self._ranked_scores = ranking.get_column("score").to_numpy()

    def _fit(
        self, loader: CandidateDataLoader[Any, Any], **kwargs: Any
    ) -> None:
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unexpected fit arguments: {sorted(kwargs)}")
        batches = list(loader.iter_fit_batches(batch_size=batch_size))
        if not batches:
            raise ValueError("popularity fit history must contain known items")
        stats = pl.concat(batches, rechunk=True)
        score_column = self._score_type.value
        ranking = (
            stats.select(
                "item_id",
                pl.col(score_column).cast(pl.Float64).alias("score"),
            )
            .sort(("score", "item_id"), descending=(True, False))
            .with_row_index("global_rank", offset=1)
            .select(ITEM_RANKING_SCHEMA.names())
            .cast(ITEM_RANKING_SCHEMA)
        )
        self._set_item_ranking(ranking)

    def _predict(
        self,
        loader: CandidateDataLoader[Any, Any],
        *,
        k: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unexpected predict arguments: {sorted(kwargs)}")
        if self._ranked_item_ids is None or self._ranked_scores is None:
            raise RuntimeError("fit must be called before predict")
        if k > len(self._ranked_item_ids):
            raise ValueError(
                f"cannot emit k={k} from a catalog of "
                f"{len(self._ranked_item_ids)} items"
            )
        return _predict_ranked_items(
            loader,
            ranked_item_ids=self._ranked_item_ids,
            ranked_scores=self._ranked_scores,
            source_name=self.source_name,
            k=k,
            batch_size=batch_size,
            require_exact=True,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "model": "global_popularity",
            "source_name": self.source_name,
            "score_type": self._score_type.value,
            "tie_break": ["score DESC", "item_id ASC"],
        }


def candidates_to_recommendations(
    candidates: pl.DataFrame,
    target_users: pl.DataFrame,
    *,
    k: int = 20,
) -> pl.DataFrame:
    """Take the first ranked candidates and return one typed list per target."""

    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    validate_id_columns(target_users, require_item=False)
    validate_no_nulls(target_users, name="target users")
    validate_unique_keys(target_users, keys=("user_id",), name="target users")
    validate_candidate_output(
        candidates,
        k=max(k, int(candidates.get_column("rank").max() or k)),
        source_name=(
            candidates.get_column("source").item(0)
            if candidates.height
            else "global_popularity"
        ),
    )
    selected = candidates.filter(pl.col("rank") <= k)
    counts = selected.group_by("user_id").len(name="count")
    invalid = (
        target_users.join(counts, on="user_id", how="left")
        .with_columns(pl.col("count").fill_null(0))
        .filter(pl.col("count") != k)
    )
    if invalid.height:
        raise ContractValidationError(
            f"{invalid.height} target user(s) do not have exactly {k} candidates"
        )
    recommendations = (
        selected.sort(("user_id", "rank"))
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").alias("item_ids"))
        .select(FINAL_RECOMMENDATION_SCHEMA.names())
        .cast(FINAL_RECOMMENDATION_SCHEMA)
    )
    validate_final_recommendations(recommendations, expected_k=k)
    return recommendations


def fill_with_global_popularity(
    primary_candidates: pl.DataFrame,
    popularity_candidates: pl.DataFrame,
    target_users: pl.DataFrame,
    history_daily: pl.DataFrame | pl.LazyFrame,
    *,
    k: int = 20,
) -> pl.DataFrame:
    """Filter a ranked source and fill missing slots with global popularity.

    Primary rows take precedence.  Within each input, rank then item ID define
    deterministic order.  Unknown items, history-seen pairs, and duplicates are
    removed before the first ``k`` rows per target are materialized.
    """

    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    if primary_candidates.schema != CANDIDATE_SCHEMA:
        raise ContractValidationError("primary candidates have invalid schema")
    if popularity_candidates.schema != CANDIDATE_SCHEMA:
        raise ContractValidationError("popularity candidates have invalid schema")
    history = history_daily.lazy() if isinstance(history_daily, pl.DataFrame) else history_daily
    if history.collect_schema() != DAILY_INTERACTION_SCHEMA:
        raise ContractValidationError("history has invalid daily schema")
    validate_id_columns(target_users, require_item=False)
    validate_no_nulls(target_users, name="target users")
    validate_unique_keys(target_users, keys=("user_id",), name="target users")

    known_items = history.select("item_id").unique()
    seen_pairs = history.select("user_id", "item_id").unique()
    combined = pl.concat(
        [
            primary_candidates.with_columns(pl.lit(0, dtype=pl.UInt8).alias("priority")),
            popularity_candidates.with_columns(pl.lit(1, dtype=pl.UInt8).alias("priority")),
        ],
        how="vertical",
    )
    filtered = (
        combined.lazy()
        .join(target_users.lazy(), on="user_id", how="semi")
        .join(known_items, on="item_id", how="semi")
        .join(seen_pairs, on=["user_id", "item_id"], how="anti")
        .sort(
            ("user_id", "priority", "rank", "score", "item_id"),
            descending=(False, False, False, True, False),
        )
        .unique(subset=("user_id", "item_id"), keep="first", maintain_order=True)
        .with_columns(
            pl.col("item_id").cum_count().over("user_id").alias("final_rank")
        )
        .filter(pl.col("final_rank") <= k)
        .collect(engine="streaming")
    )
    counts = filtered.group_by("user_id").len(name="count")
    missing = (
        target_users.join(counts, on="user_id", how="left")
        .with_columns(pl.col("count").fill_null(0))
        .filter(pl.col("count") != k)
    )
    if missing.height:
        raise ContractValidationError(
            f"global fallback could not fill {missing.height} target user(s) to {k}"
        )
    recommendations = (
        filtered.sort(("user_id", "final_rank"))
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").alias("item_ids"))
        .select(FINAL_RECOMMENDATION_SCHEMA.names())
        .cast(FINAL_RECOMMENDATION_SCHEMA)
    )
    validate_final_recommendations(recommendations, expected_k=k)
    return recommendations


__all__ = [
    "ITEM_POPULARITY_STATS_SCHEMA",
    "ITEM_RANKING_SCHEMA",
    "POPULARITY_SCORE_ORDER",
    "RECENCY_SCORE_KIND_ORDER",
    "RECENCY_SIGNAL_ORDER",
    "GlobalPopularityModel",
    "PopularityDataLoader",
    "PopularityPredictBatch",
    "PopularityScore",
    "RecencyPopularityConfig",
    "RecencyPopularityDataLoader",
    "RecencyPopularityModel",
    "RecencyScoreKind",
    "RecencySignal",
    "candidates_to_recommendations",
    "fill_with_global_popularity",
]

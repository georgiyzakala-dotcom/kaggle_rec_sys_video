"""History-only CPU implicit-ALS candidate source.

The loader owns fold-specific aggregation, stable ID mappings, confidence
construction, CSR materialization, and prediction batches.  The model only
consumes those prepared batches and can be restored from a portable artifact
without fitting again.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, Self

import implicit
import numpy as np
import polars as pl
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from interfaces import CANDIDATE_SCHEMA, CandidateDataLoader, CandidateModel
from popularity import _daily_source, _target_source
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_id_columns,
    validate_no_nulls,
    validate_unique_keys,
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    checked = int(value)
    if checked <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return checked


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-negative finite number")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return checked


def _positive_float(value: object, *, name: str) -> float:
    checked = _nonnegative_float(value, name=name)
    if checked <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return checked


@dataclass(frozen=True)
class ImplicitALSConfig:
    """Portable ALS and confidence configuration."""

    config_id: str
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 10
    num_threads: int = 8
    seed: int = 42
    interaction_weight: float = 1.0
    view_weight: float = 0.0
    long_watch_weight: float = 0.0
    like_weight: float = 0.0
    favorite_weight: float = 0.0
    half_life_hours: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise ValueError("config_id must be a non-empty string")
        for name in ("factors", "iterations", "num_threads"):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "regularization",
            _positive_float(self.regularization, name="regularization"),
        )
        weight_names = (
            "interaction_weight",
            "view_weight",
            "long_watch_weight",
            "like_weight",
            "favorite_weight",
        )
        for name in weight_names:
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name=name),
            )
        if not any(getattr(self, name) > 0 for name in weight_names):
            raise ValueError("at least one confidence signal weight must be positive")
        if self.half_life_hours is not None:
            object.__setattr__(
                self,
                "half_life_hours",
                _positive_float(self.half_life_hours, name="half_life_hours"),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Validate and construct a config loaded from JSON."""

        if not isinstance(value, Mapping):
            raise TypeError("implicit ALS config must be a mapping")
        allowed = {
            "config_id",
            "factors",
            "regularization",
            "iterations",
            "num_threads",
            "seed",
            "interaction_weight",
            "view_weight",
            "long_watch_weight",
            "like_weight",
            "favorite_weight",
            "half_life_hours",
        }
        unexpected = sorted(set(value).difference(allowed))
        if unexpected:
            raise ValueError(f"unexpected implicit ALS config fields: {unexpected}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-serializable configuration."""

        return {
            "config_id": self.config_id,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "interaction_weight": self.interaction_weight,
            "view_weight": self.view_weight,
            "long_watch_weight": self.long_watch_weight,
            "like_weight": self.like_weight,
            "favorite_weight": self.favorite_weight,
            "half_life_hours": self.half_life_hours,
        }

    def confidence_key(self) -> tuple[float | None, ...]:
        """Return fields that uniquely determine the confidence matrix."""

        return (
            self.interaction_weight,
            self.view_weight,
            self.long_watch_weight,
            self.like_weight,
            self.favorite_weight,
            self.half_life_hours,
        )


USER_MAPPING_SCHEMA: Final = pl.Schema(
    {"user_id": pl.UInt64, "user_index": pl.UInt32}
)
ITEM_MAPPING_SCHEMA: Final = pl.Schema(
    {"item_id": pl.Int32, "item_index": pl.UInt32}
)
COLLAPSED_ALS_HISTORY_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "views": pl.UInt64,
        "watch_time": pl.Int64,
        "is_like": pl.Int32,
        "is_favorite": pl.Int32,
        "last_dt": pl.Datetime("us"),
        "confidence": pl.Float32,
    }
)


@dataclass(frozen=True)
class ImplicitALSFitBatch:
    """One complete confidence matrix and its stable mappings."""

    user_items: csr_matrix
    user_mapping: pl.DataFrame
    item_mapping: pl.DataFrame


@dataclass(frozen=True)
class ImplicitALSPredictBatch:
    """Known target users with matching CSR rows for official recommend()."""

    user_ids: np.ndarray[Any, np.dtype[np.uint64]]
    user_indices: np.ndarray[Any, np.dtype[np.int32]]
    user_items: csr_matrix


def _collapse_with_confidence(
    history: pl.LazyFrame,
    *,
    reference_time: datetime,
    config: ImplicitALSConfig,
) -> pl.DataFrame:
    """Collapse daily history to unique pairs and calculate explicit confidence."""

    duplicate_daily = (
        history.group_by("user_id", "item_id", "date")
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .collect(engine="streaming")
    )
    if duplicate_daily.height:
        raise ContractValidationError(
            "implicit ALS history contains duplicate daily user-item keys"
        )
    collapsed = (
        history.group_by("user_id", "item_id")
        .agg(
            pl.col("views").sum().cast(pl.UInt64).alias("views"),
            pl.col("watch_time").max().alias("watch_time"),
            pl.col("is_like").max().alias("is_like"),
            pl.col("is_favorite").max().alias("is_favorite"),
            pl.col("dt").max().alias("last_dt"),
        )
        .sort(("user_id", "item_id"))
        .collect(engine="streaming")
    )
    if collapsed.is_empty():
        raise ValueError("implicit ALS fit history must not be empty")
    validate_no_nulls(collapsed, name="implicit ALS collapsed history")
    validate_unique_keys(
        collapsed,
        keys=("user_id", "item_id"),
        name="implicit ALS collapsed history",
    )
    if (collapsed.get_column("last_dt") >= reference_time).any():
        raise ContractValidationError(
            "implicit ALS history contains last_dt at or after reference_time"
        )
    if (collapsed.get_column("views") <= 0).any():
        raise ContractValidationError("implicit ALS views must be positive")
    for column in ("is_like", "is_favorite"):
        if (~collapsed.get_column(column).is_in([0, 1])).any():
            raise ContractValidationError(f"{column} must be binary")

    age_hours = (
        (
            pl.lit(reference_time, dtype=pl.Datetime("us"))
            - pl.col("last_dt")
        )
        .dt.total_microseconds()
        .cast(pl.Float64)
        / 3_600_000_000.0
    )
    repeated_views = (
        pl.col("views").cast(pl.Float64).sub(1.0).clip(lower_bound=0.0).log1p()
    )
    signal = (
        pl.lit(config.interaction_weight)
        + config.view_weight * repeated_views
        + config.long_watch_weight
        * (pl.col("watch_time") > 60).cast(pl.Float64)
        + config.like_weight * pl.col("is_like").cast(pl.Float64)
        + config.favorite_weight * pl.col("is_favorite").cast(pl.Float64)
    )
    decay = (
        pl.lit(1.0)
        if config.half_life_hours is None
        else (-math.log(2.0) * age_hours / config.half_life_hours).exp()
    )
    result = (
        collapsed.lazy()
        .with_columns((1.0 + signal * decay).cast(pl.Float32).alias("confidence"))
        .select(COLLAPSED_ALS_HISTORY_SCHEMA.names())
        .cast(COLLAPSED_ALS_HISTORY_SCHEMA)
        .collect(engine="streaming")
    )
    confidence = result.get_column("confidence")
    if not confidence.is_finite().all() or (confidence <= 0).any():
        raise ContractValidationError(
            "implicit ALS confidence must be finite and positive"
        )
    return result


def _mapping_from_ids(
    values: pl.Series, *, id_column: str, index_column: str, schema: pl.Schema
) -> pl.DataFrame:
    unique = values.unique().sort()
    if unique.len() > np.iinfo(np.uint32).max:
        raise ContractValidationError(f"too many {id_column} values for UInt32 mapping")
    return pl.DataFrame(
        {
            id_column: unique,
            index_column: pl.Series(
                np.arange(unique.len(), dtype=np.uint32), dtype=pl.UInt32
            ),
        },
        schema=schema,
    )


def _validate_mapping(
    mapping: pl.DataFrame,
    *,
    id_column: str,
    index_column: str,
    schema: pl.Schema,
) -> None:
    if not isinstance(mapping, pl.DataFrame) or mapping.schema != schema:
        raise ContractValidationError(f"{id_column} mapping schema must be {schema}")
    if mapping.is_empty():
        raise ContractValidationError(f"{id_column} mapping must not be empty")
    validate_no_nulls(mapping, name=f"{id_column} mapping")
    validate_unique_keys(mapping, keys=(id_column,), name=f"{id_column} mapping")
    validate_unique_keys(
        mapping, keys=(index_column,), name=f"{id_column} mapping"
    )
    expected = pl.DataFrame(
        {
            id_column: mapping.get_column(id_column).sort(),
            index_column: pl.Series(
                np.arange(mapping.height, dtype=np.uint32), dtype=pl.UInt32
            ),
        },
        schema=schema,
    )
    if not mapping.equals(expected):
        raise ContractValidationError(
            f"{id_column} mapping must be ID-sorted with contiguous indices"
        )


def _build_confidence_csr(
    collapsed: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> csr_matrix:
    """Build a sorted float32 CSR without passing source IDs through floats."""

    if collapsed.schema != COLLAPSED_ALS_HISTORY_SCHEMA:
        raise ContractValidationError("invalid collapsed implicit ALS schema")
    validate_unique_keys(
        collapsed,
        keys=("user_id", "item_id"),
        name="implicit ALS CSR source",
    )
    _validate_mapping(
        user_mapping,
        id_column="user_id",
        index_column="user_index",
        schema=USER_MAPPING_SCHEMA,
    )
    _validate_mapping(
        item_mapping,
        id_column="item_id",
        index_column="item_index",
        schema=ITEM_MAPPING_SCHEMA,
    )
    sorted_pairs = collapsed.sort(("user_id", "item_id"))
    user_ids = user_mapping.get_column("user_id").to_numpy()
    item_ids = item_mapping.get_column("item_id").to_numpy()
    source_users = sorted_pairs.get_column("user_id").to_numpy()
    source_items = sorted_pairs.get_column("item_id").to_numpy()
    rows = np.searchsorted(user_ids, source_users).astype(np.int64, copy=False)
    columns = np.searchsorted(item_ids, source_items).astype(
        np.int32, copy=False
    )
    if (
        np.any(rows >= len(user_ids))
        or np.any(columns >= len(item_ids))
        or not np.array_equal(user_ids[rows], source_users)
        or not np.array_equal(item_ids[columns], source_items)
    ):
        raise ContractValidationError("collapsed IDs are absent from ALS mappings")
    counts = np.bincount(rows, minlength=len(user_ids)).astype(np.int64)
    indptr = np.empty(len(user_ids) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    data = sorted_pairs.get_column("confidence").to_numpy().astype(
        np.float32, copy=False
    )
    matrix = csr_matrix(
        (data, columns, indptr),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    if matrix.dtype != np.float32 or matrix.nnz != collapsed.height:
        raise ContractValidationError("invalid implicit ALS CSR dtype or nnz")
    if not matrix.has_sorted_indices:
        raise ContractValidationError("implicit ALS CSR indices must be sorted")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data <= 0):
        raise ContractValidationError("implicit ALS CSR data must be finite and positive")
    return matrix


class ImplicitALSDataLoader(
    CandidateDataLoader[ImplicitALSFitBatch, ImplicitALSPredictBatch]
):
    """Build fold-specific confidence CSR and mapped target prediction rows."""

    def __init__(
        self,
        *,
        config: ImplicitALSConfig | Mapping[str, Any],
        reference_time: datetime,
        user_mapping: pl.DataFrame | None = None,
        item_mapping: pl.DataFrame | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self._config = (
            config
            if isinstance(config, ImplicitALSConfig)
            else ImplicitALSConfig.from_dict(config)
        )
        if not isinstance(reference_time, datetime):
            raise TypeError("reference_time must be a datetime")
        if reference_time.tzinfo is not None:
            raise ValueError("reference_time must be timezone-naive")
        if self._config.seed != self.seed:
            raise ValueError("loader seed must equal implicit ALS config seed")
        self._reference_time = reference_time
        self._fit_history: pl.LazyFrame | None = None
        self._predict_history: pl.LazyFrame | None = None
        self._predict_targets: pl.LazyFrame | None = None
        self._user_mapping = user_mapping.clone() if user_mapping is not None else None
        self._item_mapping = item_mapping.clone() if item_mapping is not None else None
        if self._user_mapping is not None:
            _validate_mapping(
                self._user_mapping,
                id_column="user_id",
                index_column="user_index",
                schema=USER_MAPPING_SCHEMA,
            )
        if self._item_mapping is not None:
            _validate_mapping(
                self._item_mapping,
                id_column="item_id",
                index_column="item_index",
                schema=ITEM_MAPPING_SCHEMA,
            )
        if (self._user_mapping is None) != (self._item_mapping is None):
            raise ValueError("user_mapping and item_mapping must be supplied together")
        self._fit_matrix: csr_matrix | None = None
        self._prediction_user_ids = np.empty(0, dtype=np.uint64)
        self._prediction_user_indices = np.empty(0, dtype=np.int32)
        self._prediction_seen: csr_matrix | None = None
        self._fit_source = "<not_loaded>"
        self._predict_history_source = "<not_loaded>"
        self._target_source = "<not_loaded>"

    @property
    def config(self) -> ImplicitALSConfig:
        return self._config

    @property
    def reference_time(self) -> datetime:
        return self._reference_time

    @property
    def user_mapping(self) -> pl.DataFrame:
        if self._user_mapping is None:
            raise RuntimeError("ALS user mapping has not been prepared")
        return self._user_mapping.clone()

    @property
    def item_mapping(self) -> pl.DataFrame:
        if self._item_mapping is None:
            raise RuntimeError("ALS item mapping has not been prepared")
        return self._item_mapping.clone()

    @property
    def fit_matrix(self) -> csr_matrix:
        if self._fit_matrix is None:
            raise RuntimeError("ALS fit matrix has not been prepared")
        return self._fit_matrix

    def _load_fit_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history"}:
            raise TypeError("load_fit_data requires only history=")
        self._fit_history, self._fit_source = _daily_source(
            kwargs["history"], name="implicit ALS fit history"
        )
        self._fit_matrix = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_fit_data does not accept arguments")
        assert self._fit_history is not None
        collapsed = _collapse_with_confidence(
            self._fit_history,
            reference_time=self._reference_time,
            config=self._config,
        )
        generated_users = _mapping_from_ids(
            collapsed.get_column("user_id"),
            id_column="user_id",
            index_column="user_index",
            schema=USER_MAPPING_SCHEMA,
        )
        generated_items = _mapping_from_ids(
            collapsed.get_column("item_id"),
            id_column="item_id",
            index_column="item_index",
            schema=ITEM_MAPPING_SCHEMA,
        )
        if self._user_mapping is not None and not self._user_mapping.equals(
            generated_users
        ):
            raise ContractValidationError(
                "supplied user mapping differs from fit-history mapping"
            )
        if self._item_mapping is not None and not self._item_mapping.equals(
            generated_items
        ):
            raise ContractValidationError(
                "supplied item mapping differs from fit-history mapping"
            )
        self._user_mapping = generated_users
        self._item_mapping = generated_items
        self._fit_matrix = _build_confidence_csr(
            collapsed, generated_users, generated_items
        )

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[ImplicitALSFitBatch]:
        if batch_size is not None:
            raise ValueError("implicit ALS fit requires one complete sparse matrix")
        assert self._fit_matrix is not None
        assert self._user_mapping is not None
        assert self._item_mapping is not None
        yield ImplicitALSFitBatch(
            user_items=self._fit_matrix,
            user_mapping=self._user_mapping.clone(),
            item_mapping=self._item_mapping.clone(),
        )

    def _load_predict_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history", "target_users"}:
            raise TypeError(
                "load_predict_data requires history= and target_users="
            )
        self._predict_history, self._predict_history_source = _daily_source(
            kwargs["history"], name="implicit ALS prediction history"
        )
        self._predict_targets, self._target_source = _target_source(
            kwargs["target_users"]
        )
        self._prediction_seen = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_predict_data does not accept arguments")
        if self._user_mapping is None or self._item_mapping is None:
            raise RuntimeError(
                "ALS mappings must come from fit data or a restored artifact"
            )
        assert self._predict_history is not None
        assert self._predict_targets is not None
        targets = self._predict_targets.collect(engine="streaming").sort("user_id")
        validate_id_columns(targets, require_item=False)
        validate_no_nulls(targets, name="implicit ALS target users")
        validate_unique_keys(
            targets, keys=("user_id",), name="implicit ALS target users"
        )
        known = targets.join(self._user_mapping, on="user_id", how="inner").sort(
            "user_id"
        )
        self._prediction_user_ids = known.get_column("user_id").to_numpy().astype(
            np.uint64, copy=False
        )
        self._prediction_user_indices = known.get_column(
            "user_index"
        ).to_numpy().astype(np.int32, copy=False)
        if known.is_empty():
            self._prediction_seen = csr_matrix(
                (0, self._item_mapping.height), dtype=np.float32
            )
            return
        seen = (
            self._predict_history.join(targets.lazy(), on="user_id", how="semi")
            .select("user_id", "item_id")
            .unique()
            .join(self._user_mapping.lazy(), on="user_id", how="inner")
            .join(self._item_mapping.lazy(), on="item_id", how="inner")
            .select("user_index", "item_index")
            .sort(("user_index", "item_index"))
            .collect(engine="streaming")
        )
        model_rows = seen.get_column("user_index").to_numpy().astype(
            np.int64, copy=False
        )
        local_rows = np.searchsorted(
            self._prediction_user_indices.astype(np.int64, copy=False), model_rows
        ).astype(np.int64, copy=False)
        if (
            np.any(local_rows >= known.height)
            or not np.array_equal(
                self._prediction_user_indices[local_rows].astype(
                    np.int64, copy=False
                ),
                model_rows,
            )
        ):
            raise ContractValidationError("prediction history has unmapped users")
        columns = seen.get_column("item_index").to_numpy().astype(
            np.int32, copy=False
        )
        counts = np.bincount(local_rows, minlength=known.height).astype(np.int64)
        indptr = np.empty(known.height + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        self._prediction_seen = csr_matrix(
            (
                np.ones(len(columns), dtype=np.float32),
                columns,
                indptr,
            ),
            shape=(known.height, self._item_mapping.height),
            dtype=np.float32,
        )
        if not self._prediction_seen.has_sorted_indices:
            raise ContractValidationError("prediction seen CSR must be sorted")

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[ImplicitALSPredictBatch]:
        assert self._prediction_seen is not None
        size = batch_size or max(len(self._prediction_user_ids), 1)
        for offset in range(0, len(self._prediction_user_ids), size):
            stop = min(offset + size, len(self._prediction_user_ids))
            yield ImplicitALSPredictBatch(
                user_ids=self._prediction_user_ids[offset:stop].copy(),
                user_indices=self._prediction_user_indices[offset:stop].copy(),
                user_items=self._prediction_seen[offset:stop],
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "implicit_als",
            "seed": self.seed,
            "fit_history": self._fit_source,
            "predict_history": self._predict_history_source,
            "target_users": self._target_source,
            "reference_time": self._reference_time.isoformat(),
            "implicit_als_config": self._config.to_dict(),
            "user_count": (
                self._user_mapping.height if self._user_mapping is not None else None
            ),
            "item_count": (
                self._item_mapping.height if self._item_mapping is not None else None
            ),
            "matrix_nnz": (
                int(self._fit_matrix.nnz) if self._fit_matrix is not None else None
            ),
            "aggregation": {
                "pair": ["user_id", "item_id"],
                "views": "sum",
                "watch_time": "max",
                "is_like": "max",
                "is_favorite": "max",
                "last_dt": "max(dt)",
            },
        }


class ImplicitALSModel(CandidateModel):
    """CPU float32 conjugate-gradient ALS candidate model."""

    def __init__(self, config: ImplicitALSConfig | Mapping[str, Any]) -> None:
        self._config = (
            config
            if isinstance(config, ImplicitALSConfig)
            else ImplicitALSConfig.from_dict(config)
        )
        self._backend: AlternatingLeastSquares | None = None
        self._user_mapping: pl.DataFrame | None = None
        self._item_mapping: pl.DataFrame | None = None
        self._item_ids: np.ndarray[Any, np.dtype[np.int32]] | None = None

    @property
    def source_name(self) -> str:
        return "implicit_als"

    @property
    def config(self) -> ImplicitALSConfig:
        return self._config

    @property
    def backend(self) -> AlternatingLeastSquares:
        if self._backend is None:
            raise RuntimeError("implicit ALS model has not been fitted")
        return self._backend

    @property
    def user_mapping(self) -> pl.DataFrame:
        if self._user_mapping is None:
            raise RuntimeError("implicit ALS model has not been fitted")
        return self._user_mapping.clone()

    @property
    def item_mapping(self) -> pl.DataFrame:
        if self._item_mapping is None:
            raise RuntimeError("implicit ALS model has not been fitted")
        return self._item_mapping.clone()

    def _fit(
        self, loader: CandidateDataLoader[Any, Any], **kwargs: Any
    ) -> None:
        show_progress = kwargs.pop("show_progress", False)
        callback = kwargs.pop("callback", None)
        if kwargs:
            raise TypeError(f"unexpected fit arguments: {sorted(kwargs)}")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        if not isinstance(loader, ImplicitALSDataLoader):
            raise TypeError("ImplicitALSModel requires ImplicitALSDataLoader")
        if (
            loader.config.seed != self._config.seed
            or loader.config.confidence_key() != self._config.confidence_key()
        ):
            raise ValueError(
                "loader and model must share seed and confidence configuration"
            )
        batches = list(loader.iter_fit_batches(batch_size=None))
        if len(batches) != 1:
            raise RuntimeError("implicit ALS loader must yield exactly one fit batch")
        batch = batches[0]
        backend = AlternatingLeastSquares(
            factors=self._config.factors,
            regularization=self._config.regularization,
            alpha=1.0,
            dtype=np.float32,
            use_native=True,
            use_cg=True,
            iterations=self._config.iterations,
            calculate_training_loss=False,
            num_threads=self._config.num_threads,
            random_state=self._config.seed,
        )
        backend.fit(
            batch.user_items,
            show_progress=bool(show_progress),
            callback=callback,
        )
        if backend.user_factors.dtype != np.float32 or backend.item_factors.dtype != np.float32:
            raise ContractValidationError("implicit ALS factors must be float32")
        if not np.isfinite(backend.user_factors).all() or not np.isfinite(
            backend.item_factors
        ).all():
            raise ContractValidationError("implicit ALS factors must be finite")
        self._backend = backend
        self._set_mappings(batch.user_mapping, batch.item_mapping)

    def _set_mappings(
        self, user_mapping: pl.DataFrame, item_mapping: pl.DataFrame
    ) -> None:
        _validate_mapping(
            user_mapping,
            id_column="user_id",
            index_column="user_index",
            schema=USER_MAPPING_SCHEMA,
        )
        _validate_mapping(
            item_mapping,
            id_column="item_id",
            index_column="item_index",
            schema=ITEM_MAPPING_SCHEMA,
        )
        self._user_mapping = user_mapping.clone()
        self._item_mapping = item_mapping.clone()
        self._item_ids = item_mapping.get_column("item_id").to_numpy().astype(
            np.int32, copy=False
        )

    def _recommend_one_with_ties(
        self,
        user_index: int,
        user_items: csr_matrix,
        *,
        k: int,
        initial_ids: np.ndarray[Any, Any] | None = None,
        initial_scores: np.ndarray[Any, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        assert self._backend is not None
        assert self._item_ids is not None
        item_count = len(self._item_ids)
        request = min(item_count, k + 32)
        ids = initial_ids
        scores = initial_scores
        while True:
            if ids is None or scores is None:
                ids, scores = self._backend.recommend(
                    np.int32(user_index),
                    user_items,
                    N=request,
                    filter_already_liked_items=True,
                    recalculate_user=False,
                )
            ids = np.asarray(ids)
            scores = np.asarray(scores)
            valid = (ids >= 0) & np.isfinite(scores)
            ids = ids[valid].astype(np.int64, copy=False)
            scores = scores[valid].astype(np.float32, copy=False)
            if not len(ids):
                return ids, scores
            item_ids = self._item_ids[ids]
            order = np.lexsort((item_ids, -scores))
            ids = ids[order]
            scores = scores[order]
            if len(ids) <= k or request >= item_count or scores[k - 1] != scores[-1]:
                return ids[:k], scores[:k]
            request = min(item_count, max(request + 32, request * 2))
            ids = None
            scores = None

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
        if not isinstance(loader, ImplicitALSDataLoader):
            raise TypeError("ImplicitALSModel requires ImplicitALSDataLoader")
        if self._backend is None or self._item_ids is None:
            raise RuntimeError("fit or from_artifact must be called before predict")
        if not self.user_mapping.equals(loader.user_mapping) or not self.item_mapping.equals(
            loader.item_mapping
        ):
            raise ValueError("prediction loader mappings differ from model mappings")
        outputs: list[pl.DataFrame] = []
        for batch in loader.iter_predict_batches(batch_size=batch_size):
            if not isinstance(batch, ImplicitALSPredictBatch):
                raise TypeError("ImplicitALSModel requires ImplicitALSPredictBatch")
            if not len(batch.user_ids):
                continue
            request = min(len(self._item_ids), k + 32)
            batch_ids, batch_scores = self._backend.recommend(
                batch.user_indices,
                batch.user_items,
                N=request,
                filter_already_liked_items=True,
                recalculate_user=False,
            )
            user_values: list[int] = []
            item_values: list[int] = []
            score_values: list[float] = []
            rank_values: list[int] = []
            for local_index, (user_id, user_index) in enumerate(
                zip(batch.user_ids, batch.user_indices, strict=True)
            ):
                selected_ids, selected_scores = self._recommend_one_with_ties(
                    int(user_index),
                    batch.user_items[local_index],
                    k=k,
                    initial_ids=batch_ids[local_index],
                    initial_scores=batch_scores[local_index],
                )
                count = len(selected_ids)
                if not count:
                    continue
                user_values.extend([int(user_id)] * count)
                item_values.extend(int(value) for value in self._item_ids[selected_ids])
                score_values.extend(float(value) for value in selected_scores)
                rank_values.extend(range(1, count + 1))
            if user_values:
                outputs.append(
                    pl.DataFrame(
                        {
                            "user_id": pl.Series(user_values, dtype=pl.UInt64),
                            "item_id": pl.Series(item_values, dtype=pl.Int32),
                            "score": pl.Series(score_values, dtype=pl.Float64),
                            "rank": pl.Series(rank_values, dtype=pl.UInt32),
                            "source": pl.Series(
                                [self.source_name] * len(user_values),
                                dtype=pl.String,
                            ),
                        },
                        schema=CANDIDATE_SCHEMA,
                    )
                )
        result = (
            pl.concat(outputs, rechunk=True).cast(CANDIDATE_SCHEMA)
            if outputs
            else pl.DataFrame(schema=CANDIDATE_SCHEMA)
        )
        validate_candidate_output(result, k=k, source_name=self.source_name)
        return result

    def save(
        self,
        artifact_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save backend factors, mappings, config, and optional fit metadata."""

        if self._backend is None or self._user_mapping is None or self._item_mapping is None:
            raise RuntimeError("fit must be called before save")
        directory = Path(artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "model": directory / "als_model.npz",
            "users": directory / "user_mapping.parquet",
            "items": directory / "item_mapping.parquet",
            "config": directory / "model_config.json",
        }
        existing = [path.name for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite ALS artifact files: {existing}")
        portable = {
            "artifact_version": 1,
            "model": "implicit_als",
            "source_name": self.source_name,
            "implicit_als_config": self._config.to_dict(),
            "backend": {
                "library": "implicit",
                "version": implicit.__version__,
                "implementation": "implicit.cpu.als.AlternatingLeastSquares",
                "dtype": "float32",
                "use_cg": True,
                "alpha": 1.0,
            },
            "user_count": self._user_mapping.height,
            "item_count": self._item_mapping.height,
            "metadata": dict(metadata or {}),
        }
        try:
            json.dumps(portable, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"ALS artifact metadata is not JSON serializable: {error}") from error
        self._backend.save(paths["model"])
        self._user_mapping.write_parquet(paths["users"], compression="zstd")
        self._item_mapping.write_parquet(paths["items"], compression="zstd")
        paths["config"].write_text(
            json.dumps(portable, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> Self:
        """Restore a portable CPU ALS artifact without invoking fit()."""

        directory = Path(artifact_dir)
        paths = {
            "model": directory / "als_model.npz",
            "users": directory / "user_mapping.parquet",
            "items": directory / "item_mapping.parquet",
            "config": directory / "model_config.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"implicit ALS artifact is missing: {missing}")
        try:
            portable = json.loads(paths["config"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read implicit ALS model config: {error}") from error
        if (
            not isinstance(portable, dict)
            or portable.get("artifact_version") != 1
            or portable.get("model") != "implicit_als"
            or portable.get("source_name") != "implicit_als"
        ):
            raise ContractValidationError("invalid implicit ALS portable config")
        backend_config = portable.get("backend")
        if not isinstance(backend_config, dict) or (
            backend_config.get("implementation")
            != "implicit.cpu.als.AlternatingLeastSquares"
            or backend_config.get("dtype") != "float32"
            or backend_config.get("use_cg") is not True
            or backend_config.get("alpha") != 1.0
        ):
            raise ContractValidationError("invalid implicit ALS backend contract")
        model = cls(ImplicitALSConfig.from_dict(portable["implicit_als_config"]))
        backend = AlternatingLeastSquares.load(str(paths["model"]))
        users = pl.read_parquet(paths["users"])
        items = pl.read_parquet(paths["items"])
        model._backend = backend
        model._set_mappings(users, items)
        if (
            backend.user_factors.shape
            != (users.height, model.config.factors)
            or backend.item_factors.shape
            != (items.height, model.config.factors)
            or backend.user_factors.dtype != np.float32
            or backend.item_factors.dtype != np.float32
            or not np.isfinite(backend.user_factors).all()
            or not np.isfinite(backend.item_factors).all()
        ):
            raise ContractValidationError("restored implicit ALS factors are invalid")
        if backend.alpha != 1.0 or not backend.use_cg:
            raise ContractValidationError("restored implicit ALS solver contract differs")
        if portable.get("user_count") != users.height or portable.get(
            "item_count"
        ) != items.height:
            raise ContractValidationError("restored ALS mapping counts differ")
        return model

    def get_config(self) -> dict[str, Any]:
        return {
            "model": "implicit_als",
            "source_name": self.source_name,
            "implicit_als_config": self._config.to_dict(),
            "backend": "implicit.cpu.als.AlternatingLeastSquares",
            "dtype": "float32",
            "solver": "conjugate_gradient",
            "alpha": 1.0,
            "tie_break": ["score DESC", "item_id ASC"],
            "tie_lookahead": 32,
        }


__all__ = [
    "COLLAPSED_ALS_HISTORY_SCHEMA",
    "ITEM_MAPPING_SCHEMA",
    "USER_MAPPING_SCHEMA",
    "ImplicitALSConfig",
    "ImplicitALSDataLoader",
    "ImplicitALSFitBatch",
    "ImplicitALSModel",
    "ImplicitALSPredictBatch",
]

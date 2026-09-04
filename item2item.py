"""History-only item-to-item co-visitation candidates.

The loader is the only component that reads the immutable daily history
snapshot.  It collapses daily rows to unique user-item interactions and
prepares bounded, deterministic profiles.  The model builds sparse item-item
relations and emits a partial personalized candidate source; global fallback
is deliberately owned by orchestration.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from numbers import Integral
from typing import Any, Final, Self

import numpy as np
import polars as pl
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


class ItemProfile(str, Enum):
    """Interaction subset used to create item pairs."""

    ALL = "all"
    POSITIVE = "positive"


class PairDirection(str, Enum):
    """Co-visitation relation direction."""

    UNDIRECTED = "undirected"
    DIRECTED = "directed"


class PairWeight(str, Enum):
    """Contribution of one user's item pair."""

    UNIFORM = "uniform"
    TIME_DISTANCE = "time_distance"


class PairNormalization(str, Enum):
    """Normalization applied to aggregated pair contributions."""

    RAW = "raw"
    COSINE = "cosine"
    JACCARD = "jaccard"


class SeedStrength(str, Enum):
    """Strength multiplier applied to an inference seed."""

    UNIFORM = "uniform"
    EVENT_STRENGTH = "event_strength"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class Item2ItemConfig:
    """Portable configuration of one item-to-item candidate source."""

    config_id: str
    profile: ItemProfile | str = ItemProfile.ALL
    direction: PairDirection | str = PairDirection.UNDIRECTED
    pair_weight: PairWeight | str = PairWeight.UNIFORM
    normalization: PairNormalization | str = PairNormalization.RAW
    history_cap: int = 10
    pair_half_life_hours: float = 24.0
    min_pair_users: int = 1
    neighbor_k: int = 100
    seed_k: int = 5
    seed_recency_half_life_hours: float | None = 24.0
    seed_strength: SeedStrength | str = SeedStrength.UNIFORM

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise ValueError("config_id must be a non-empty string")
        enum_fields = {
            "profile": ItemProfile,
            "direction": PairDirection,
            "pair_weight": PairWeight,
            "normalization": PairNormalization,
            "seed_strength": SeedStrength,
        }
        for name, enum_type in enum_fields.items():
            try:
                checked = enum_type(getattr(self, name))
            except ValueError as error:
                raise ValueError(
                    f"{name} must be one of {[member.value for member in enum_type]}"
                ) from error
            object.__setattr__(self, name, checked)
        for name in ("history_cap", "min_pair_users", "neighbor_k", "seed_k"):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "pair_half_life_hours",
            _positive_float(
                self.pair_half_life_hours, name="pair_half_life_hours"
            ),
        )
        if self.seed_recency_half_life_hours is not None:
            object.__setattr__(
                self,
                "seed_recency_half_life_hours",
                _positive_float(
                    self.seed_recency_half_life_hours,
                    name="seed_recency_half_life_hours",
                ),
            )
        if self.seed_k > self.history_cap:
            raise ValueError("seed_k must not exceed history_cap")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Validate and construct a config loaded from JSON."""

        if not isinstance(value, Mapping):
            raise TypeError("item2item config must be a mapping")
        allowed = {
            "config_id",
            "profile",
            "direction",
            "pair_weight",
            "normalization",
            "history_cap",
            "pair_half_life_hours",
            "min_pair_users",
            "neighbor_k",
            "seed_k",
            "seed_recency_half_life_hours",
            "seed_strength",
        }
        unexpected = sorted(set(value).difference(allowed))
        if unexpected:
            raise ValueError(f"unexpected item2item config fields: {unexpected}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-serializable configuration."""

        return {
            "config_id": self.config_id,
            "profile": self.profile.value,
            "direction": self.direction.value,
            "pair_weight": self.pair_weight.value,
            "normalization": self.normalization.value,
            "history_cap": self.history_cap,
            "pair_half_life_hours": self.pair_half_life_hours,
            "min_pair_users": self.min_pair_users,
            "neighbor_k": self.neighbor_k,
            "seed_k": self.seed_k,
            "seed_recency_half_life_hours": (
                self.seed_recency_half_life_hours
            ),
            "seed_strength": self.seed_strength.value,
        }

    def relation_key(self) -> tuple[Any, ...]:
        """Return fields that uniquely determine the fitted neighbor table."""

        return (
            self.profile.value,
            self.direction.value,
            self.pair_weight.value,
            self.normalization.value,
            self.history_cap,
            self.pair_half_life_hours,
            self.min_pair_users,
            self.neighbor_k,
        )


COLLAPSED_HISTORY_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "last_dt": pl.Datetime("us"),
        "views": pl.UInt64,
        "watch_time": pl.Int64,
        "is_like": pl.Int32,
        "is_favorite": pl.Int32,
        "is_positive": pl.Int32,
        "history_rank": pl.UInt32,
    }
)
SEEN_PAIR_SCHEMA: Final = pl.Schema(
    {"user_id": pl.UInt64, "item_id": pl.Int32}
)
NEIGHBOR_TABLE_SCHEMA: Final = pl.Schema(
    {
        "item_id": pl.Int32,
        "neighbor_item_id": pl.Int32,
        "score": pl.Float64,
        "co_user_count": pl.UInt32,
        "rank": pl.UInt32,
    }
)


@dataclass(frozen=True)
class Item2ItemPredictBatch:
    """One target-user batch with recent seeds and complete seen pairs."""

    users: pl.DataFrame
    seeds: pl.DataFrame
    seen_pairs: pl.DataFrame


def _collapse_history(history: pl.LazyFrame) -> pl.LazyFrame:
    """Collapse daily history to deterministic unique user-item rows."""

    return (
        history.group_by("user_id", "item_id")
        .agg(
            pl.col("dt").max().alias("last_dt"),
            pl.col("views").sum().cast(pl.UInt64).alias("views"),
            pl.col("watch_time").max().alias("watch_time"),
            pl.col("is_like").max().alias("is_like"),
            pl.col("is_favorite").max().alias("is_favorite"),
            pl.col("is_positive").max().alias("is_positive"),
        )
        .sort(
            ("user_id", "last_dt", "item_id"),
            descending=(False, True, False),
        )
        .with_columns(
            pl.col("item_id")
            .cum_count()
            .over("user_id")
            .cast(pl.UInt32)
            .alias("history_rank")
        )
        .select(COLLAPSED_HISTORY_SCHEMA.names())
        .cast(COLLAPSED_HISTORY_SCHEMA)
    )


def _slice_sorted_users(
    frame: pl.DataFrame,
    user_values: np.ndarray[Any, np.dtype[np.uint64]],
    first_user: int,
    last_user: int,
) -> pl.DataFrame:
    start = int(np.searchsorted(user_values, np.uint64(first_user), side="left"))
    stop = int(np.searchsorted(user_values, np.uint64(last_user), side="right"))
    return frame.slice(start, stop - start)


class Item2ItemDataLoader(
    CandidateDataLoader[pl.DataFrame, Item2ItemPredictBatch]
):
    """Read one shared daily history and prepare item2item-specific batches."""

    def __init__(
        self,
        *,
        reference_time: datetime,
        max_history_items: int = 10,
        max_seed_items: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        if not isinstance(reference_time, datetime):
            raise TypeError("reference_time must be a datetime")
        if reference_time.tzinfo is not None:
            raise ValueError("reference_time must be timezone-naive")
        self._reference_time = reference_time
        self._max_history_items = _positive_int(
            max_history_items, name="max_history_items"
        )
        self._max_seed_items = _positive_int(
            max_seed_items, name="max_seed_items"
        )
        if self._max_seed_items > self._max_history_items:
            raise ValueError("max_seed_items must not exceed max_history_items")
        self._fit_history: pl.LazyFrame | None = None
        self._predict_history: pl.LazyFrame | None = None
        self._predict_targets: pl.LazyFrame | None = None
        self._fit_profiles: pl.DataFrame | None = None
        self._prediction_users: pl.DataFrame | None = None
        self._prediction_seeds: pl.DataFrame | None = None
        self._prediction_seen: pl.DataFrame | None = None
        self._seed_users = np.empty(0, dtype=np.uint64)
        self._seen_users = np.empty(0, dtype=np.uint64)
        self._fit_source = "<not_loaded>"
        self._predict_history_source = "<not_loaded>"
        self._target_source = "<not_loaded>"

    @property
    def reference_time(self) -> datetime:
        return self._reference_time

    @property
    def max_history_items(self) -> int:
        return self._max_history_items

    @property
    def max_seed_items(self) -> int:
        return self._max_seed_items

    @property
    def fit_profiles(self) -> pl.DataFrame:
        if self._fit_profiles is None:
            raise RuntimeError("fit data has not been prepared")
        return self._fit_profiles.clone()

    @property
    def prediction_users(self) -> pl.DataFrame:
        if self._prediction_users is None:
            raise RuntimeError("prediction data has not been prepared")
        return self._prediction_users.clone()

    def _load_fit_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history"}:
            raise TypeError("load_fit_data requires only history=")
        self._fit_history, self._fit_source = _daily_source(
            kwargs["history"], name="item2item fit history"
        )
        self._fit_profiles = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_fit_data does not accept arguments")
        assert self._fit_history is not None
        profiles = (
            _collapse_history(self._fit_history)
            .filter(pl.col("history_rank") <= self._max_history_items)
            .collect(engine="streaming")
        )
        if profiles.is_empty():
            raise ValueError("item2item fit history must not be empty")
        if profiles.schema != COLLAPSED_HISTORY_SCHEMA:
            raise ContractValidationError("invalid collapsed history schema")
        validate_no_nulls(profiles, name="item2item fit profiles")
        validate_unique_keys(
            profiles,
            keys=("user_id", "item_id"),
            name="item2item fit profiles",
        )
        if (profiles.get_column("last_dt") >= self._reference_time).any():
            raise ContractValidationError(
                "item2item fit history contains last_dt at or after reference_time"
            )
        self._fit_profiles = profiles

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[pl.DataFrame]:
        assert self._fit_profiles is not None
        if batch_size is None:
            yield self._fit_profiles.clone()
            return
        for offset in range(0, self._fit_profiles.height, batch_size):
            yield self._fit_profiles.slice(offset, batch_size)

    def _load_predict_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"history", "target_users"}:
            raise TypeError(
                "load_predict_data requires history= and target_users="
            )
        self._predict_history, self._predict_history_source = _daily_source(
            kwargs["history"], name="item2item prediction history"
        )
        self._predict_targets, self._target_source = _target_source(
            kwargs["target_users"]
        )
        self._prediction_users = None
        self._prediction_seeds = None
        self._prediction_seen = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_predict_data does not accept arguments")
        assert self._predict_history is not None
        assert self._predict_targets is not None
        targets = self._predict_targets.collect(engine="streaming").sort("user_id")
        validate_id_columns(targets, require_item=False)
        validate_no_nulls(targets, name="item2item target users")
        validate_unique_keys(
            targets, keys=("user_id",), name="item2item target users"
        )
        target_lazy = targets.lazy()
        target_history = self._predict_history.join(
            target_lazy, on="user_id", how="semi"
        )
        seeds = (
            _collapse_history(target_history)
            .filter(pl.col("history_rank") <= self._max_seed_items)
            .collect(engine="streaming")
        )
        seen = (
            target_history.select("user_id", "item_id")
            .unique()
            .sort(("user_id", "item_id"))
            .select(SEEN_PAIR_SCHEMA.names())
            .cast(SEEN_PAIR_SCHEMA)
            .collect(engine="streaming")
        )
        if seeds.height and (
            seeds.get_column("last_dt") >= self._reference_time
        ).any():
            raise ContractValidationError(
                "item2item prediction history contains future seeds"
            )
        validate_no_nulls(seeds, name="item2item prediction seeds")
        validate_unique_keys(
            seeds,
            keys=("user_id", "item_id"),
            name="item2item prediction seeds",
        )
        validate_no_nulls(seen, name="item2item prediction seen pairs")
        validate_unique_keys(
            seen,
            keys=("user_id", "item_id"),
            name="item2item prediction seen pairs",
        )
        self._prediction_users = targets
        self._prediction_seeds = seeds
        self._prediction_seen = seen
        self._seed_users = seeds.get_column("user_id").to_numpy()
        self._seen_users = seen.get_column("user_id").to_numpy()

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[Item2ItemPredictBatch]:
        assert self._prediction_users is not None
        assert self._prediction_seeds is not None
        assert self._prediction_seen is not None
        size = batch_size or max(self._prediction_users.height, 1)
        for offset in range(0, self._prediction_users.height, size):
            users = self._prediction_users.slice(offset, size)
            first_user = users.get_column("user_id").item(0)
            last_user = users.get_column("user_id").item(-1)
            yield Item2ItemPredictBatch(
                users=users,
                seeds=_slice_sorted_users(
                    self._prediction_seeds,
                    self._seed_users,
                    first_user,
                    last_user,
                ),
                seen_pairs=_slice_sorted_users(
                    self._prediction_seen,
                    self._seen_users,
                    first_user,
                    last_user,
                ),
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "item2item",
            "seed": self.seed,
            "fit_history": self._fit_source,
            "predict_history": self._predict_history_source,
            "target_users": self._target_source,
            "reference_time": self._reference_time.isoformat(),
            "max_history_items": self._max_history_items,
            "max_seed_items": self._max_seed_items,
            "collapse": {
                "last_dt": "max(dt)",
                "views": "sum(views)",
                "binary_and_watch_fields": "max",
                "order": ["last_dt DESC", "item_id ASC"],
            },
        }


@dataclass
class _PairStatistics:
    item_ids: np.ndarray[Any, np.dtype[np.int32]]
    weighted: csr_matrix
    support: csr_matrix
    item_user_counts: np.ndarray[Any, np.dtype[np.float64]]


def _build_pair_statistics(
    profiles: pl.DataFrame, config: Item2ItemConfig
) -> _PairStatistics:
    selected = profiles.filter(pl.col("history_rank") <= config.history_cap)
    if config.profile is ItemProfile.POSITIVE:
        selected = selected.filter(pl.col("is_positive") == 1)
    selected = selected.sort(
        ("user_id", "last_dt", "item_id"),
        descending=(False, True, False),
    )
    if selected.is_empty():
        raise ValueError("selected item2item profile is empty")

    item_ids = selected.get_column("item_id").unique().sort().to_numpy()
    source_items = selected.get_column("item_id").to_numpy()
    item_indices = np.searchsorted(item_ids, source_items).astype(np.int32)
    user_ids = selected.get_column("user_id").to_numpy()
    timestamps = selected.get_column("last_dt").to_numpy().astype(
        "datetime64[us]"
    ).astype(np.int64)
    item_user_counts = np.bincount(
        item_indices, minlength=len(item_ids)
    ).astype(np.float64)
    shape = (len(item_ids), len(item_ids))
    weighted = csr_matrix(shape, dtype=np.float64)
    support = csr_matrix(shape, dtype=np.float64)

    for distance in range(1, config.history_cap):
        same_user = user_ids[:-distance] == user_ids[distance:]
        if config.direction is PairDirection.DIRECTED:
            same_user &= timestamps[:-distance] > timestamps[distance:]
        if not same_user.any():
            continue
        newer = item_indices[:-distance][same_user]
        older = item_indices[distance:][same_user]
        time_delta_hours = (
            timestamps[:-distance][same_user]
            - timestamps[distance:][same_user]
        ).astype(np.float64) / 3_600_000_000.0
        if config.direction is PairDirection.DIRECTED:
            rows = older
            columns = newer
            deltas = time_delta_hours
        else:
            rows = np.concatenate((newer, older))
            columns = np.concatenate((older, newer))
            deltas = np.concatenate((time_delta_hours, time_delta_hours))
        if config.pair_weight is PairWeight.UNIFORM:
            values = np.ones(len(rows), dtype=np.float64)
        else:
            values = np.exp2(-deltas / config.pair_half_life_hours)
        weighted = weighted + csr_matrix(
            (values, (rows, columns)), shape=shape, dtype=np.float64
        )
        support = support + csr_matrix(
            (
                np.ones(len(rows), dtype=np.float64),
                (rows, columns),
            ),
            shape=shape,
            dtype=np.float64,
        )
    weighted.sum_duplicates()
    support.sum_duplicates()
    weighted.sort_indices()
    support.sort_indices()
    if not np.array_equal(weighted.indptr, support.indptr) or not np.array_equal(
        weighted.indices, support.indices
    ):
        raise RuntimeError("weighted and support sparse structures diverged")
    return _PairStatistics(
        item_ids=item_ids,
        weighted=weighted,
        support=support,
        item_user_counts=item_user_counts,
    )


def _neighbor_table_from_pair_statistics(
    statistics: _PairStatistics, config: Item2ItemConfig
) -> pl.DataFrame:
    weighted = statistics.weighted
    support = statistics.support
    degrees = np.diff(weighted.indptr)
    rows = np.repeat(np.arange(weighted.shape[0], dtype=np.int32), degrees)
    columns = weighted.indices
    counts = support.data
    values = weighted.data.copy()
    keep = counts >= config.min_pair_users
    rows = rows[keep]
    columns = columns[keep]
    counts = counts[keep]
    values = values[keep]
    if config.normalization is PairNormalization.COSINE:
        denominator = np.sqrt(
            statistics.item_user_counts[rows]
            * statistics.item_user_counts[columns]
        )
        values /= denominator
    elif config.normalization is PairNormalization.JACCARD:
        denominator = (
            statistics.item_user_counts[rows]
            + statistics.item_user_counts[columns]
            - counts
        )
        values /= denominator

    normalized = csr_matrix(
        (values, (rows, columns)), shape=weighted.shape, dtype=np.float64
    )
    count_matrix = csr_matrix(
        (counts, (rows, columns)), shape=weighted.shape, dtype=np.float64
    )
    normalized.sort_indices()
    count_matrix.sort_indices()
    if not np.array_equal(normalized.indptr, count_matrix.indptr) or not np.array_equal(
        normalized.indices, count_matrix.indices
    ):
        raise RuntimeError("normalized and support sparse structures diverged")

    row_degrees = np.diff(normalized.indptr)
    output_size = int(np.minimum(row_degrees, config.neighbor_k).sum())
    output_items = np.empty(output_size, dtype=np.int32)
    output_neighbors = np.empty(output_size, dtype=np.int32)
    output_scores = np.empty(output_size, dtype=np.float64)
    output_counts = np.empty(output_size, dtype=np.uint32)
    output_ranks = np.empty(output_size, dtype=np.uint32)
    position = 0
    for row_index in np.flatnonzero(row_degrees):
        start = normalized.indptr[row_index]
        stop = normalized.indptr[row_index + 1]
        row_scores = normalized.data[start:stop]
        row_columns = normalized.indices[start:stop]
        row_counts = count_matrix.data[start:stop]
        take = min(len(row_scores), config.neighbor_k)
        if len(row_scores) > take:
            threshold = np.partition(row_scores, len(row_scores) - take)[
                len(row_scores) - take
            ]
            greater = np.flatnonzero(row_scores > threshold)
            equal = np.flatnonzero(row_scores == threshold)
            remaining = take - len(greater)
            equal_order = np.argsort(
                statistics.item_ids[row_columns[equal]], kind="stable"
            )[:remaining]
            selected = np.concatenate((greater, equal[equal_order]))
        else:
            selected = np.arange(len(row_scores))
        order = np.lexsort(
            (
                statistics.item_ids[row_columns[selected]],
                -row_scores[selected],
            )
        )
        selected = selected[order]
        end = position + take
        output_items[position:end] = statistics.item_ids[row_index]
        output_neighbors[position:end] = statistics.item_ids[
            row_columns[selected]
        ]
        output_scores[position:end] = row_scores[selected]
        output_counts[position:end] = row_counts[selected].astype(np.uint32)
        output_ranks[position:end] = np.arange(1, take + 1, dtype=np.uint32)
        position = end

    table = pl.DataFrame(
        {
            "item_id": pl.Series(output_items, dtype=pl.Int32),
            "neighbor_item_id": pl.Series(
                output_neighbors, dtype=pl.Int32
            ),
            "score": pl.Series(output_scores, dtype=pl.Float64),
            "co_user_count": pl.Series(output_counts, dtype=pl.UInt32),
            "rank": pl.Series(output_ranks, dtype=pl.UInt32),
        },
        schema=NEIGHBOR_TABLE_SCHEMA,
    )
    validate_neighbor_table(table, neighbor_k=config.neighbor_k)
    return table


def validate_neighbor_table(table: pl.DataFrame, *, neighbor_k: int) -> None:
    """Validate the persisted fitted-neighbor boundary."""

    _positive_int(neighbor_k, name="neighbor_k")
    if not isinstance(table, pl.DataFrame):
        raise TypeError("neighbor table must be a polars.DataFrame")
    if table.schema != NEIGHBOR_TABLE_SCHEMA:
        raise ContractValidationError(
            f"neighbor table schema must be {NEIGHBOR_TABLE_SCHEMA}"
        )
    validate_no_nulls(table, name="item2item neighbor table")
    validate_unique_keys(
        table,
        keys=("item_id", "neighbor_item_id"),
        name="item2item neighbor table",
    )
    if table.is_empty():
        raise ContractValidationError("item2item neighbor table must not be empty")
    if (table.get_column("item_id") == table.get_column("neighbor_item_id")).any():
        raise ContractValidationError("item2item neighbor table has self-links")
    if not table.get_column("score").is_finite().all() or (
        table.get_column("score") <= 0
    ).any():
        raise ContractValidationError("neighbor scores must be finite and positive")
    if (table.get_column("co_user_count") == 0).any():
        raise ContractValidationError("co_user_count must be positive")
    rank_groups = table.group_by("item_id").agg(
        pl.len().alias("rows"),
        pl.col("rank").min().alias("minimum"),
        pl.col("rank").max().alias("maximum"),
        pl.col("rank").n_unique().alias("unique"),
    )
    invalid = rank_groups.filter(
        (pl.col("minimum") != 1)
        | (pl.col("maximum") != pl.col("rows"))
        | (pl.col("unique") != pl.col("rows"))
        | (pl.col("rows") > neighbor_k)
    )
    if invalid.height:
        raise ContractValidationError(
            "neighbor ranks must be contiguous and bounded per item"
        )
    physical = table.sort(("item_id", "rank"))
    if not table.equals(physical):
        raise ContractValidationError(
            "neighbor rows must be ordered by item_id ASC, rank ASC"
        )
    semantic = table.sort(
        ("item_id", "score", "neighbor_item_id"),
        descending=(False, True, False),
    ).select("item_id", "neighbor_item_id", "score", "co_user_count")
    ranked = table.select(
        "item_id", "neighbor_item_id", "score", "co_user_count"
    )
    if not ranked.equals(semantic):
        raise ContractValidationError(
            "neighbor rank order must be score DESC, neighbor_item_id ASC"
        )


class Item2ItemModel(CandidateModel):
    """Sparse co-visitation model using recent user items as seeds."""

    def __init__(self, config: Item2ItemConfig | Mapping[str, Any]) -> None:
        self._config = (
            config
            if isinstance(config, Item2ItemConfig)
            else Item2ItemConfig.from_dict(config)
        )
        self._neighbors: pl.DataFrame | None = None

    @property
    def source_name(self) -> str:
        return "item2item"

    @property
    def config(self) -> Item2ItemConfig:
        return self._config

    @property
    def neighbor_table(self) -> pl.DataFrame:
        if self._neighbors is None:
            raise RuntimeError("model has not been fitted")
        return self._neighbors.clone()

    @classmethod
    def from_fitted_neighbors(
        cls,
        config: Item2ItemConfig | Mapping[str, Any],
        neighbor_table: pl.DataFrame,
    ) -> Self:
        """Restore fitted state without reading history or repeating fit."""

        model = cls(config)
        validate_neighbor_table(
            neighbor_table, neighbor_k=model.config.neighbor_k
        )
        if (
            neighbor_table.get_column("co_user_count").min()
            < model.config.min_pair_users
        ):
            raise ContractValidationError(
                "neighbor table violates configured min_pair_users"
            )
        model._neighbors = neighbor_table.clone()
        return model

    def with_inference_config(
        self, config: Item2ItemConfig | Mapping[str, Any]
    ) -> Self:
        """Reuse fitted neighbors when only seed-time parameters change."""

        if self._neighbors is None:
            raise RuntimeError("model has not been fitted")
        checked = (
            config
            if isinstance(config, Item2ItemConfig)
            else Item2ItemConfig.from_dict(config)
        )
        if checked.relation_key() != self._config.relation_key():
            raise ValueError(
                "inference config must preserve all fitted relation fields"
            )
        model = type(self)(checked)
        model._neighbors = self._neighbors.clone()
        return model

    def _fit(
        self, loader: CandidateDataLoader[Any, Any], **kwargs: Any
    ) -> None:
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unexpected fit arguments: {sorted(kwargs)}")
        if not isinstance(loader, Item2ItemDataLoader):
            raise TypeError("Item2ItemModel requires Item2ItemDataLoader")
        if self._config.history_cap > loader.max_history_items:
            raise ValueError("loader max_history_items does not cover config")
        if self._config.seed_k > loader.max_seed_items:
            raise ValueError("loader max_seed_items does not cover config")
        batches = list(loader.iter_fit_batches(batch_size=batch_size))
        if not batches:
            raise ValueError("item2item fit history must contain interactions")
        profiles = pl.concat(batches, rechunk=True)
        statistics = _build_pair_statistics(profiles, self._config)
        self._neighbors = _neighbor_table_from_pair_statistics(
            statistics, self._config
        )

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
        if not isinstance(loader, Item2ItemDataLoader):
            raise TypeError("Item2ItemModel requires Item2ItemDataLoader")
        if self._neighbors is None:
            raise RuntimeError("fit must be called before predict")
        if self._config.seed_k > loader.max_seed_items:
            raise ValueError("loader max_seed_items does not cover config")
        outputs: list[pl.DataFrame] = []
        neighbor_lazy = self._neighbors.lazy().select(
            pl.col("item_id").alias("seed_item_id"),
            pl.col("neighbor_item_id").alias("item_id"),
            pl.col("score").alias("neighbor_score"),
        )
        reference = pl.lit(loader.reference_time, dtype=pl.Datetime("us"))
        for batch in loader.iter_predict_batches(batch_size=batch_size):
            if not isinstance(batch, Item2ItemPredictBatch):
                raise TypeError(
                    "Item2ItemModel requires Item2ItemPredictBatch"
                )
            if batch.seeds.is_empty():
                continue
            seeds = batch.seeds.filter(
                pl.col("history_rank") <= self._config.seed_k
            ).lazy()
            age_hours = (
                (reference - pl.col("last_dt"))
                .dt.total_microseconds()
                .cast(pl.Float64)
                / 3_600_000_000.0
            )
            if self._config.seed_recency_half_life_hours is None:
                recency = pl.lit(1.0)
            else:
                recency = (
                    -math.log(2.0)
                    * age_hours
                    / self._config.seed_recency_half_life_hours
                ).exp()
            if self._config.seed_strength is SeedStrength.UNIFORM:
                strength = pl.lit(1.0)
            else:
                strength = (
                    pl.lit(1.0)
                    + pl.col("views").cast(pl.Float64).log1p()
                    + (pl.col("watch_time") > 60).cast(pl.Float64)
                    + pl.col("is_like").cast(pl.Float64)
                    + pl.col("is_favorite").cast(pl.Float64)
                )
            scored = (
                seeds.rename({"item_id": "seed_item_id"})
                .join(neighbor_lazy, on="seed_item_id", how="inner")
                .with_columns(
                    (pl.col("neighbor_score") * recency * strength)
                    .cast(pl.Float64)
                    .alias("score")
                )
                .group_by("user_id", "item_id")
                .agg(pl.col("score").max())
                .join(
                    batch.seen_pairs.lazy(),
                    on=["user_id", "item_id"],
                    how="anti",
                )
                .sort(
                    ("user_id", "score", "item_id"),
                    descending=(False, True, False),
                )
                .group_by("user_id", maintain_order=True)
                .head(k)
                .with_columns(
                    pl.col("item_id")
                    .cum_count()
                    .over("user_id")
                    .cast(pl.UInt32)
                    .alias("rank"),
                    pl.lit(self.source_name).alias("source"),
                )
                .select(CANDIDATE_SCHEMA.names())
                .cast(CANDIDATE_SCHEMA)
                .collect(engine="streaming")
            )
            if scored.height:
                outputs.append(scored)
        if not outputs:
            return pl.DataFrame(schema=CANDIDATE_SCHEMA)
        result = pl.concat(outputs, rechunk=True).cast(CANDIDATE_SCHEMA)
        validate_candidate_output(result, k=k, source_name=self.source_name)
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "model": "item2item",
            "source_name": self.source_name,
            "item2item_config": self._config.to_dict(),
            "pair_backend": "scipy_csr",
            "pair_aggregation": "max_one_contribution_per_unique_user_pair",
            "seed_aggregation": "max",
            "tie_break": ["score DESC", "item_id ASC"],
        }


__all__ = [
    "COLLAPSED_HISTORY_SCHEMA",
    "NEIGHBOR_TABLE_SCHEMA",
    "Item2ItemConfig",
    "Item2ItemDataLoader",
    "Item2ItemModel",
    "Item2ItemPredictBatch",
    "ItemProfile",
    "PairDirection",
    "PairNormalization",
    "PairWeight",
    "SeedStrength",
    "validate_neighbor_table",
]

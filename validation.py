"""Reusable validation for model/loader and Polars boundary contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from numbers import Integral
from typing import Any, Literal

import polars as pl

from interfaces import (
    CANDIDATE_SCHEMA,
    FEATURE_TABLE_REQUIRED_SCHEMA,
    FINAL_RECOMMENDATION_SCHEMA,
    RANKER_OUTPUT_SCHEMA,
    CandidateModel,
    DataLoader,
    DataLoaderState,
    Model,
)


class ContractValidationError(ValueError):
    """Raised when a public pipeline boundary violates its contract."""


def validate_batch_size(batch_size: int | None) -> None:
    """Validate the common batch-size argument without changing its value."""

    if batch_size is None:
        return
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ContractValidationError(
            "batch_size must be a positive integer or None"
        )
    if batch_size <= 0:
        raise ContractValidationError("batch_size must be positive")


def validate_json_config(config: object, *, name: str = "config") -> None:
    """Require a JSON object with finite, serializable values."""

    if not isinstance(config, dict):
        raise ContractValidationError(f"{name} must be a dict")
    try:
        json.dumps(config, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractValidationError(
            f"{name} must be JSON-serializable: {error}"
        ) from error


def validate_model_config(model: Model[Any]) -> None:
    """Validate model config and the stable name of candidate sources."""

    validate_json_config(model.get_config(), name="model config")
    if isinstance(model, CandidateModel):
        source_first = model.source_name
        source_second = model.source_name
        if (
            not isinstance(source_first, str)
            or not source_first
            or source_second != source_first
        ):
            raise ContractValidationError(
                "candidate source_name must be a stable non-empty string"
            )


def validate_loader(loader: DataLoader[Any, Any]) -> None:
    """Validate loader identity, states, seed, and serializable config."""

    if not isinstance(loader, DataLoader):
        raise ContractValidationError("loader must implement DataLoader")
    if isinstance(loader.seed, bool) or not isinstance(loader.seed, int):
        raise ContractValidationError("loader seed must be an explicit integer")
    if not isinstance(loader.fit_state, DataLoaderState):
        raise ContractValidationError("loader fit_state is invalid")
    if not isinstance(loader.predict_state, DataLoaderState):
        raise ContractValidationError("loader predict_state is invalid")
    validate_json_config(loader.get_config(), name="loader config")


BatchEqual = Callable[[Any, Any], bool]


def _batch_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, pl.DataFrame):
        return left.equals(right)
    if isinstance(left, pl.Series):
        return left.equals(right)
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _batch_values_equal(
                getattr(left, field.name), getattr(right, field.name)
            )
            for field in fields(left)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _batch_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _batch_values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    if hasattr(result, "all"):
        try:
            return bool(result.all())
        except (TypeError, ValueError):
            return False
    return False


def validate_deterministic_iteration(
    loader: DataLoader[Any, Any],
    *,
    phase: Literal["fit", "predict"],
    batch_size: int | None = None,
    batch_equal: BatchEqual | None = None,
) -> None:
    """Check that two fresh iterators yield identical batches.

    The relevant loader side must already be prepared.  A custom comparator can
    be supplied for sparse matrices or framework-specific tensor containers.
    """

    validate_loader(loader)
    validate_batch_size(batch_size)
    iterator_factory = (
        loader.iter_fit_batches
        if phase == "fit"
        else loader.iter_predict_batches
    )
    first_iterator = iterator_factory(batch_size=batch_size)
    second_iterator = iterator_factory(batch_size=batch_size)
    if first_iterator is second_iterator:
        raise ContractValidationError(
            f"{phase} iteration must return a fresh iterator on every call"
        )
    first_batches = list(first_iterator)
    second_batches = list(second_iterator)
    comparator = batch_equal or _batch_values_equal
    if len(first_batches) != len(second_batches) or not all(
        comparator(first, second)
        for first, second in zip(first_batches, second_batches, strict=True)
    ):
        raise ContractValidationError(
            f"{phase} batch iteration is not deterministic"
        )


def _require_frame(frame: object, *, name: str) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise ContractValidationError(f"{name} must be a polars.DataFrame")
    return frame


def _validate_exact_schema(
    frame: pl.DataFrame, expected: pl.Schema, *, name: str
) -> None:
    if frame.schema != expected:
        raise ContractValidationError(
            f"{name} schema must be {expected}, got {frame.schema}"
        )


def _validate_required_schema(
    frame: pl.DataFrame, expected: pl.Schema, *, name: str
) -> None:
    problems = {
        column: (dtype, frame.schema.get(column))
        for column, dtype in expected.items()
        if frame.schema.get(column) != dtype
    }
    if problems:
        raise ContractValidationError(
            f"{name} has missing or invalid required columns: {problems}"
        )


def validate_no_nulls(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    name: str = "frame",
) -> None:
    """Reject nulls in all columns or in an explicit column subset."""

    selected = list(columns) if columns is not None else frame.columns
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise ContractValidationError(
            f"{name} is missing columns required for null validation: {missing}"
        )
    null_counts = frame.select(
        pl.col(column).null_count().alias(column) for column in selected
    ).row(0, named=True)
    invalid = {column: count for column, count in null_counts.items() if count}
    if invalid:
        raise ContractValidationError(f"{name} contains null values: {invalid}")


def validate_unique_keys(
    frame: pl.DataFrame, *, keys: Sequence[str], name: str = "frame"
) -> None:
    """Reject duplicate key tuples."""

    missing = sorted(set(keys).difference(frame.columns))
    if missing:
        raise ContractValidationError(
            f"{name} is missing duplicate-check columns: {missing}"
        )
    duplicates = (
        frame.group_by(list(keys))
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicates:
        raise ContractValidationError(
            f"{name} contains {duplicates} duplicate key group(s) for {list(keys)}"
        )


def validate_id_columns(frame: pl.DataFrame, *, require_item: bool = True) -> None:
    """Require exact integer ID dtypes; floating-point IDs are never accepted."""

    expected = {"user_id": pl.UInt64}
    if require_item:
        expected["item_id"] = pl.Int32
    invalid = {
        column: (dtype, frame.schema.get(column))
        for column, dtype in expected.items()
        if frame.schema.get(column) != dtype
    }
    if invalid:
        raise ContractValidationError(f"invalid ID columns: {invalid}")


def validate_candidate_output(
    frame: pl.DataFrame, *, k: int, source_name: str
) -> None:
    """Validate a deterministic, single-source candidate table."""

    frame = _require_frame(frame, name="candidate output")
    validate_batch_size(k)
    if k is None:  # pragma: no cover - k is statically non-optional
        raise ContractValidationError("k must be a positive integer")
    if not isinstance(source_name, str) or not source_name:
        raise ContractValidationError("source_name must be a non-empty string")
    _validate_exact_schema(frame, CANDIDATE_SCHEMA, name="candidate output")
    validate_id_columns(frame)
    validate_no_nulls(frame, name="candidate output")
    validate_unique_keys(
        frame,
        keys=("user_id", "item_id", "source"),
        name="candidate output",
    )

    if frame.height == 0:
        return
    sources = frame.get_column("source").unique().to_list()
    if sources != [source_name]:
        raise ContractValidationError(
            f"candidate source must be constant {source_name!r}, got {sources}"
        )
    if not frame.get_column("score").is_finite().all():
        raise ContractValidationError("candidate scores must be finite")

    over_cap = (
        frame.group_by("user_id")
        .len()
        .filter(pl.col("len") > k)
        .height
    )
    if over_cap:
        raise ContractValidationError(
            f"candidate output exceeds k={k} for {over_cap} user(s)"
        )

    rank_groups = frame.group_by(("user_id", "source")).agg(
        pl.len().alias("row_count"),
        pl.col("rank").min().alias("min_rank"),
        pl.col("rank").max().alias("max_rank"),
        pl.col("rank").n_unique().alias("unique_ranks"),
    )
    invalid_ranks = rank_groups.filter(
        (pl.col("min_rank") != 1)
        | (pl.col("max_rank") != pl.col("row_count"))
        | (pl.col("unique_ranks") != pl.col("row_count"))
    ).height
    if invalid_ranks:
        raise ContractValidationError(
            "candidate ranks must start at 1 and be contiguous per user/source"
        )

    physical = frame.sort(("user_id", "source", "rank"))
    if not frame.equals(physical):
        raise ContractValidationError(
            "candidate rows must be physically ordered by "
            "user_id ASC, source ASC, rank ASC"
        )

    semantic = frame.sort(
        ("user_id", "source", "score", "item_id"),
        descending=(False, False, True, False),
    ).select("user_id", "item_id", "score", "source")
    ranked = frame.select("user_id", "item_id", "score", "source")
    if not ranked.equals(semantic):
        raise ContractValidationError(
            "candidate rank order must be score DESC, item_id ASC"
        )


def validate_feature_table(frame: pl.DataFrame) -> None:
    """Validate prepared ranker features at their external boundary."""

    frame = _require_frame(frame, name="feature table")
    _validate_required_schema(
        frame, FEATURE_TABLE_REQUIRED_SCHEMA, name="feature table"
    )
    validate_id_columns(frame)
    validate_no_nulls(frame, name="feature table")
    validate_unique_keys(
        frame, keys=("user_id", "item_id"), name="feature table"
    )


def validate_ranker_output(frame: pl.DataFrame) -> None:
    """Validate candidate-level scores emitted by a ranker."""

    frame = _require_frame(frame, name="ranker output")
    _validate_exact_schema(frame, RANKER_OUTPUT_SCHEMA, name="ranker output")
    validate_id_columns(frame)
    validate_no_nulls(frame, name="ranker output")
    validate_unique_keys(
        frame, keys=("user_id", "item_id"), name="ranker output"
    )
    if not frame.get_column("ranker_score").is_finite().all():
        raise ContractValidationError("ranker scores must be finite")


def validate_final_recommendations(
    frame: pl.DataFrame, *, expected_k: int | None = None
) -> None:
    """Validate the typed in-memory final recommendation boundary."""

    frame = _require_frame(frame, name="final recommendations")
    validate_batch_size(expected_k)
    _validate_exact_schema(
        frame, FINAL_RECOMMENDATION_SCHEMA, name="final recommendations"
    )
    validate_id_columns(frame, require_item=False)
    validate_no_nulls(frame, name="final recommendations")
    validate_unique_keys(
        frame, keys=("user_id",), name="final recommendations"
    )
    if frame.height == 0:
        return
    invalid_inner_nulls = frame.filter(
        pl.col("item_ids").list.len()
        != pl.col("item_ids").list.drop_nulls().list.len()
    ).height
    if invalid_inner_nulls:
        raise ContractValidationError(
            "final recommendation lists contain null item IDs"
        )
    duplicate_items = frame.filter(
        pl.col("item_ids").list.len() != pl.col("item_ids").list.n_unique()
    ).height
    if duplicate_items:
        raise ContractValidationError(
            "final recommendation lists must contain unique items"
        )
    if expected_k is not None:
        wrong_size = frame.filter(
            pl.col("item_ids").list.len() != expected_k
        ).height
        if wrong_size:
            raise ContractValidationError(
                f"{wrong_size} user(s) do not have exactly {expected_k} items"
            )


def validate_recommendations_against_history(
    frame: pl.DataFrame,
    *,
    target_users: pl.DataFrame,
    history_daily: pl.DataFrame | pl.LazyFrame,
    expected_k: int = 20,
) -> None:
    """Validate the full target universe plus known-item and unseen-pair rules."""

    from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA

    validate_final_recommendations(frame, expected_k=expected_k)
    if target_users.schema != TARGET_USER_SCHEMA:
        raise ContractValidationError(
            f"target users schema must be {TARGET_USER_SCHEMA}, "
            f"got {target_users.schema}"
        )
    validate_no_nulls(target_users, name="target users")
    validate_unique_keys(target_users, keys=("user_id",), name="target users")
    missing_users = target_users.join(frame, on="user_id", how="anti")
    extra_users = frame.select("user_id").join(
        target_users, on="user_id", how="anti"
    )
    if missing_users.height or extra_users.height:
        raise ContractValidationError(
            "recommendation users do not exactly match target users: "
            f"missing={missing_users.height}, extra={extra_users.height}"
        )

    history = (
        history_daily.lazy()
        if isinstance(history_daily, pl.DataFrame)
        else history_daily
    )
    if not isinstance(history, pl.LazyFrame):
        raise TypeError("history_daily must be a Polars DataFrame or LazyFrame")
    if history.collect_schema() != DAILY_INTERACTION_SCHEMA:
        raise ContractValidationError(
            f"history schema must be {DAILY_INTERACTION_SCHEMA}, "
            f"got {history.collect_schema()}"
        )
    recommendation_pairs = (
        frame.explode("item_ids", empty_as_null=True)
        .select(
            "user_id", pl.col("item_ids").cast(pl.Int32).alias("item_id")
        )
        .lazy()
    )
    known_items = history.select("item_id").unique()
    history_pairs = history.select("user_id", "item_id").unique()
    diagnostics = pl.collect_all(
        [
            recommendation_pairs.join(
                known_items, on="item_id", how="anti"
            ).select(pl.len()),
            recommendation_pairs.join(
                history_pairs, on=["user_id", "item_id"], how="semi"
            ).select(pl.len()),
        ],
        engine="streaming",
    )
    unknown_count = diagnostics[0].item()
    seen_count = diagnostics[1].item()
    if unknown_count or seen_count:
        raise ContractValidationError(
            "recommendations violate history semantics: "
            f"unknown_items={unknown_count}, seen_pairs={seen_count}"
        )


__all__ = [
    "ContractValidationError",
    "validate_batch_size",
    "validate_candidate_output",
    "validate_deterministic_iteration",
    "validate_feature_table",
    "validate_final_recommendations",
    "validate_id_columns",
    "validate_json_config",
    "validate_loader",
    "validate_model_config",
    "validate_no_nulls",
    "validate_ranker_output",
    "validate_recommendations_against_history",
    "validate_unique_keys",
]

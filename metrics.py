"""Trusted ranking and candidate metrics for the temporal validation protocol."""

from __future__ import annotations

from typing import Final

import polars as pl

from data_utils import GROUND_TRUTH_SCHEMA, TARGET_USER_SCHEMA
from interfaces import FINAL_RECOMMENDATION_SCHEMA
from validation import (
    ContractValidationError,
    validate_final_recommendations,
    validate_id_columns,
    validate_no_nulls,
    validate_unique_keys,
)

EVALUATION_K: Final = 20


def _validate_ground_truth(frame: pl.DataFrame) -> None:
    if frame.schema != GROUND_TRUTH_SCHEMA:
        raise ContractValidationError(
            f"ground truth schema must be {GROUND_TRUTH_SCHEMA}, got {frame.schema}"
        )
    validate_id_columns(frame)
    validate_no_nulls(frame, name="ground truth")
    validate_unique_keys(frame, keys=("user_id", "item_id"), name="ground truth")


def _validate_target_users(frame: pl.DataFrame) -> None:
    if frame.schema != TARGET_USER_SCHEMA:
        raise ContractValidationError(
            f"target users schema must be {TARGET_USER_SCHEMA}, got {frame.schema}"
        )
    validate_id_columns(frame, require_item=False)
    validate_no_nulls(frame, name="target users")
    validate_unique_keys(frame, keys=("user_id",), name="target users")


def _validate_evaluation_universe(
    target_ground_truth: pl.DataFrame, target_users: pl.DataFrame
) -> None:
    _validate_ground_truth(target_ground_truth)
    _validate_target_users(target_users)
    extra_ground_truth_users = target_ground_truth.select("user_id").unique().join(
        target_users, on="user_id", how="anti"
    )
    if extra_ground_truth_users.height:
        raise ContractValidationError(
            "target ground truth contains users outside the target universe"
        )


def _recommendation_pairs(
    recommendations: pl.DataFrame, target_users: pl.DataFrame
) -> pl.DataFrame:
    if recommendations.schema != FINAL_RECOMMENDATION_SCHEMA:
        raise ContractValidationError(
            "recommendations must use FINAL_RECOMMENDATION_SCHEMA"
        )
    validate_final_recommendations(recommendations)
    extra_users = recommendations.select("user_id").join(
        target_users, on="user_id", how="anti"
    )
    if extra_users.height:
        raise ContractValidationError(
            "recommendations contain users outside the target universe"
        )
    over_cap = recommendations.filter(
        pl.col("item_ids").list.len() > EVALUATION_K
    )
    if over_cap.height:
        raise ContractValidationError(
            f"recommendations must contain at most {EVALUATION_K} items per user"
        )
    return (
        recommendations.explode("item_ids", empty_as_null=True)
        .drop_nulls("item_ids")
        .select(
            "user_id",
            pl.col("item_ids").cast(pl.Int32).alias("item_id"),
        )
    )


def evaluate_precision_at_20(
    recommendations: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, float]:
    """Compute both fixed-denominator macro Precision@20 variants."""

    _validate_evaluation_universe(target_ground_truth, target_users)
    recommendation_pairs = _recommendation_pairs(recommendations, target_users)
    hits = (
        recommendation_pairs.join(
            target_ground_truth, on=["user_id", "item_id"], how="semi"
        )
        .group_by("user_id")
        .len(name="hits")
    )
    all_score = (
        target_users.join(hits, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .select((pl.col("hits") / EVALUATION_K).mean())
        .item()
    )
    labeled_users = target_ground_truth.select("user_id").unique()
    if labeled_users.is_empty():
        labeled_score = 0.0
    else:
        labeled_score = (
            labeled_users.join(hits, on="user_id", how="left")
            .with_columns(pl.col("hits").fill_null(0))
            .select((pl.col("hits") / EVALUATION_K).mean())
            .item()
        )
    return {
        "precision_at_20_all_targets": float(all_score or 0.0),
        "precision_at_20_labeled_users": float(labeled_score or 0.0),
    }


def precision_at_20_all_targets(
    recommendations: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> float:
    """Precision@20 over every target user; empty labels contribute zero."""

    return evaluate_precision_at_20(
        recommendations, target_ground_truth, target_users
    )["precision_at_20_all_targets"]


def precision_at_20_labeled_users(
    recommendations: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> float:
    """Precision@20 over target users with non-empty eligible ground truth."""

    return evaluate_precision_at_20(
        recommendations, target_ground_truth, target_users
    )["precision_at_20_labeled_users"]


def _candidate_pairs(
    candidates: pl.DataFrame, target_users: pl.DataFrame
) -> pl.DataFrame:
    required = {"user_id": pl.UInt64, "item_id": pl.Int32}
    problems = {
        column: (dtype, candidates.schema.get(column))
        for column, dtype in required.items()
        if candidates.schema.get(column) != dtype
    }
    if problems:
        raise ContractValidationError(
            f"candidates have missing or invalid ID columns: {problems}"
        )
    validate_no_nulls(
        candidates, columns=("user_id", "item_id"), name="candidates"
    )
    pairs = candidates.select("user_id", "item_id").unique()
    extra_users = pairs.select("user_id").unique().join(
        target_users, on="user_id", how="anti"
    )
    if extra_users.height:
        raise ContractValidationError(
            "candidates contain users outside the target universe"
        )
    return pairs


def evaluate_candidate_metrics(
    candidates: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, float]:
    """Evaluate a deduplicated candidate union over the full target universe."""

    _validate_evaluation_universe(target_ground_truth, target_users)
    pairs = _candidate_pairs(candidates, target_users)
    candidate_counts = (
        pairs.group_by("user_id")
        .len(name="candidate_count")
        .join(target_users, on="user_id", how="right")
        .with_columns(pl.col("candidate_count").fill_null(0).cast(pl.UInt32))
    )
    count_stats = candidate_counts.select(
        coverage=(pl.col("candidate_count") > 0).mean(),
        mean_candidate_count=pl.col("candidate_count").mean(),
        p50_candidate_count=pl.col("candidate_count").quantile(
            0.50, interpolation="nearest"
        ),
        p90_candidate_count=pl.col("candidate_count").quantile(
            0.90, interpolation="nearest"
        ),
        p95_candidate_count=pl.col("candidate_count").quantile(
            0.95, interpolation="nearest"
        ),
        p99_candidate_count=pl.col("candidate_count").quantile(
            0.99, interpolation="nearest"
        ),
    ).row(0, named=True)

    hit_pairs = pairs.join(
        target_ground_truth, on=["user_id", "item_id"], how="semi"
    )
    relevant_pairs = target_ground_truth.height
    candidate_recall_value = (
        hit_pairs.height / relevant_pairs if relevant_pairs else 0.0
    )
    labeled_users = target_ground_truth.select("user_id").unique()
    hit_counts = hit_pairs.group_by("user_id").len(name="hits")
    candidate_user_hit_rate_value = (
        hit_counts.height / labeled_users.height if labeled_users.height else 0.0
    )

    all_oracle = (
        target_users.join(hit_counts, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .select(
            pl.col("hits")
            .clip(upper_bound=EVALUATION_K)
            .truediv(EVALUATION_K)
            .mean()
        )
        .item()
    )
    if labeled_users.is_empty():
        labeled_oracle = 0.0
    else:
        labeled_oracle = (
            labeled_users.join(hit_counts, on="user_id", how="left")
            .with_columns(pl.col("hits").fill_null(0))
            .select(
                pl.col("hits")
                .clip(upper_bound=EVALUATION_K)
                .truediv(EVALUATION_K)
                .mean()
            )
            .item()
        )

    return {
        "candidate_recall": float(candidate_recall_value),
        "candidate_user_hit_rate": float(candidate_user_hit_rate_value),
        "candidate_oracle_p20_all_targets": float(all_oracle or 0.0),
        "candidate_oracle_p20_labeled_users": float(labeled_oracle or 0.0),
        **{
            key: float(value or 0.0)
            for key, value in count_stats.items()
        },
    }


def evaluate_candidate_metrics_lazy(
    candidates: pl.DataFrame | pl.LazyFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, float]:
    """Evaluate a large candidate union without collecting all candidate rows.

    Only per-user candidate counts and relevant hit counts are materialized.
    This is equivalent to :func:`evaluate_candidate_metrics` for the same
    unique candidate pairs.
    """

    _validate_evaluation_universe(target_ground_truth, target_users)
    lazy = candidates.lazy() if isinstance(candidates, pl.DataFrame) else candidates
    if not isinstance(lazy, pl.LazyFrame):
        raise TypeError("candidates must be a Polars DataFrame or LazyFrame")
    schema = lazy.collect_schema()
    required = {"user_id": pl.UInt64, "item_id": pl.Int32}
    problems = {
        column: (dtype, schema.get(column))
        for column, dtype in required.items()
        if schema.get(column) != dtype
    }
    if problems:
        raise ContractValidationError(
            f"candidates have missing or invalid ID columns: {problems}"
        )
    pairs = lazy.select("user_id", "item_id").unique()
    extra_users = (
        pairs.select("user_id")
        .unique()
        .join(target_users.lazy(), on="user_id", how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if extra_users:
        raise ContractValidationError(
            "candidates contain users outside the target universe"
        )
    candidate_counts = (
        pairs.group_by("user_id")
        .len(name="candidate_count")
        .collect(engine="streaming")
        .join(target_users, on="user_id", how="right")
        .with_columns(pl.col("candidate_count").fill_null(0).cast(pl.UInt32))
    )
    count_stats = candidate_counts.select(
        coverage=(pl.col("candidate_count") > 0).mean(),
        mean_candidate_count=pl.col("candidate_count").mean(),
        p50_candidate_count=pl.col("candidate_count").quantile(
            0.50, interpolation="nearest"
        ),
        p90_candidate_count=pl.col("candidate_count").quantile(
            0.90, interpolation="nearest"
        ),
        p95_candidate_count=pl.col("candidate_count").quantile(
            0.95, interpolation="nearest"
        ),
        p99_candidate_count=pl.col("candidate_count").quantile(
            0.99, interpolation="nearest"
        ),
    ).row(0, named=True)
    hit_counts = (
        pairs.join(
            target_ground_truth.lazy(),
            on=["user_id", "item_id"],
            how="semi",
        )
        .group_by("user_id")
        .len(name="hits")
        .collect(engine="streaming")
    )
    hit_rows = int(hit_counts.get_column("hits").sum() or 0)
    relevant_pairs = target_ground_truth.height
    candidate_recall_value = (
        hit_rows / relevant_pairs if relevant_pairs else 0.0
    )
    labeled_users = target_ground_truth.select("user_id").unique()
    candidate_user_hit_rate_value = (
        hit_counts.height / labeled_users.height if labeled_users.height else 0.0
    )
    all_oracle = (
        target_users.join(hit_counts, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .select(
            pl.col("hits")
            .clip(upper_bound=EVALUATION_K)
            .truediv(EVALUATION_K)
            .mean()
        )
        .item()
    )
    labeled_oracle = (
        labeled_users.join(hit_counts, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .select(
            pl.col("hits")
            .clip(upper_bound=EVALUATION_K)
            .truediv(EVALUATION_K)
            .mean()
        )
        .item()
        if labeled_users.height
        else 0.0
    )
    return {
        "candidate_recall": float(candidate_recall_value),
        "candidate_user_hit_rate": float(candidate_user_hit_rate_value),
        "candidate_oracle_p20_all_targets": float(all_oracle or 0.0),
        "candidate_oracle_p20_labeled_users": float(labeled_oracle or 0.0),
        **{
            key: float(value or 0.0)
            for key, value in count_stats.items()
        },
    }


def candidate_recall(
    candidates: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> float:
    """Micro recall of eligible target ground-truth pairs."""

    return evaluate_candidate_metrics(
        candidates, target_ground_truth, target_users
    )["candidate_recall"]


def candidate_user_hit_rate(
    candidates: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> float:
    """Share of labeled target users with at least one candidate hit."""

    return evaluate_candidate_metrics(
        candidates, target_ground_truth, target_users
    )["candidate_user_hit_rate"]


def candidate_oracle_precision_at_20(
    candidates: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, float]:
    """Return both candidate-set oracle Precision@20 variants."""

    values = evaluate_candidate_metrics(candidates, target_ground_truth, target_users)
    return {
        "candidate_oracle_p20_all_targets": values[
            "candidate_oracle_p20_all_targets"
        ],
        "candidate_oracle_p20_labeled_users": values[
            "candidate_oracle_p20_labeled_users"
        ],
    }


def candidate_count_statistics(
    candidates: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, float]:
    """Coverage and count distribution including zero-candidate targets."""

    values = evaluate_candidate_metrics(candidates, target_ground_truth, target_users)
    keys = (
        "coverage",
        "mean_candidate_count",
        "p50_candidate_count",
        "p90_candidate_count",
        "p95_candidate_count",
        "p99_candidate_count",
    )
    return {key: values[key] for key in keys}


__all__ = [
    "EVALUATION_K",
    "candidate_count_statistics",
    "candidate_oracle_precision_at_20",
    "candidate_recall",
    "candidate_user_hit_rate",
    "evaluate_candidate_metrics",
    "evaluate_candidate_metrics_lazy",
    "evaluate_precision_at_20",
    "precision_at_20_all_targets",
    "precision_at_20_labeled_users",
]

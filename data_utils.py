"""Canonical temporal-fold preparation from immutable raw interactions.

The central invariant is split-before-aggregation: raw timestamped events are
split first, and each fold side is aggregated independently afterwards.  Model
loaders consume the resulting daily history artifact and never raw events.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl

RAW_INTERACTION_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "event_type": pl.Categorical(),
        "watch_time": pl.Int64,
        "date": pl.Datetime("us"),
    }
)
TARGET_USER_SCHEMA: Final = pl.Schema({"user_id": pl.UInt64})
DAILY_INTERACTION_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "date": pl.Date,
        "dt": pl.Datetime("us"),
        "views": pl.UInt32,
        "watch_time": pl.Int64,
        "is_like": pl.Int32,
        "is_favorite": pl.Int32,
        "is_positive": pl.Int32,
    }
)
GROUND_TRUTH_SCHEMA: Final = pl.Schema(
    {"user_id": pl.UInt64, "item_id": pl.Int32}
)

DAILY_KEYS: Final = ("user_id", "item_id", "date")
PAIR_KEYS: Final = ("user_id", "item_id")


class DataPreparationError(ValueError):
    """Raised when source data or a prepared fold violates its contract."""


@dataclass(frozen=True)
class GroundTruthFrames:
    """Lazy ground-truth stages used by the fold materializer."""

    positive_pairs: pl.LazyFrame
    unseen_pairs: pl.LazyFrame
    cold_pairs: pl.LazyFrame
    eligible_pairs: pl.LazyFrame
    target_pairs: pl.LazyFrame


@dataclass(frozen=True)
class FoldPreparationResult:
    """Published fold artifact and its JSON-serializable metadata."""

    output_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any]


def _require_schema(
    actual: pl.Schema, expected: pl.Schema, *, name: str
) -> None:
    if actual != expected:
        raise DataPreparationError(
            f"{name} schema must be {expected}, got {actual}"
        )


def scan_raw_interactions(path: str | Path) -> pl.LazyFrame:
    """Scan raw interactions lazily and validate their exact source schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"raw interactions file does not exist: {source}")
    frame = pl.scan_parquet(source)
    _require_schema(
        frame.collect_schema(), RAW_INTERACTION_SCHEMA, name="raw interactions"
    )
    return frame


def scan_target_users(path: str | Path) -> pl.LazyFrame:
    """Scan target users lazily and validate their exact source schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"target users file does not exist: {source}")
    frame = pl.scan_parquet(source)
    _require_schema(frame.collect_schema(), TARGET_USER_SCHEMA, name="target users")
    return frame


def infer_canonical_cutoff(raw: pl.LazyFrame) -> datetime:
    """Return ``max(raw.date) - 24 hours`` without hardcoded dataset dates."""

    maximum = raw.select(pl.col("date").max()).collect().item()
    if maximum is None:
        raise DataPreparationError("raw interactions must not be empty")
    if not isinstance(maximum, datetime):
        raise DataPreparationError("raw date maximum is not a datetime")
    return maximum - timedelta(days=1)


def split_raw_interactions(
    raw: pl.LazyFrame,
    *,
    cutoff: datetime,
    validation_end_exclusive: datetime | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split raw timestamps into history and a validation interval.

    Canonical preparation leaves ``validation_end_exclusive`` unset, so the
    validation side includes the full raw tail from the cutoff through the
    source maximum.  Rolling folds can pass a half-open interval end.
    """

    if not isinstance(cutoff, datetime):
        raise TypeError("cutoff must be a datetime")
    if (
        validation_end_exclusive is not None
        and validation_end_exclusive <= cutoff
    ):
        raise DataPreparationError(
            "validation_end_exclusive must be later than cutoff"
        )
    history = raw.filter(pl.col("date") < pl.lit(cutoff))
    validation_filter = pl.col("date") >= pl.lit(cutoff)
    if validation_end_exclusive is not None:
        validation_filter &= pl.col("date") < pl.lit(validation_end_exclusive)
    return history, raw.filter(validation_filter)


def aggregate_daily_interactions(raw_side: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate one already-split raw fold side to the shared daily contract."""

    like = (pl.col("event_type") == "like").cast(pl.Int32)
    favorite = (pl.col("event_type") == "favorite").cast(pl.Int32)
    return (
        raw_side.group_by(
            "user_id",
            "item_id",
            pl.col("date").dt.date().alias("date"),
        )
        .agg(
            pl.col("date").min().alias("dt"),
            pl.len().cast(pl.UInt32).alias("views"),
            pl.col("watch_time").max(),
            like.max().alias("is_like"),
            favorite.max().alias("is_favorite"),
        )
        .with_columns(
            (
                (pl.col("watch_time") > 60)
                | (pl.col("is_like") == 1)
                | (pl.col("is_favorite") == 1)
            )
            .cast(pl.Int32)
            .alias("is_positive")
        )
        .select(DAILY_INTERACTION_SCHEMA.names())
        .cast(DAILY_INTERACTION_SCHEMA)
        .sort(DAILY_KEYS)
    )


def build_ground_truth(
    history_daily: pl.LazyFrame,
    validation_daily: pl.LazyFrame,
    target_users: pl.LazyFrame,
) -> GroundTruthFrames:
    """Build deduplicated eligible ground truth and explicit filtering stages."""

    history_pairs = history_daily.select(PAIR_KEYS).unique()
    history_items = history_daily.select("item_id").unique()
    positive_pairs = (
        validation_daily.filter(pl.col("is_positive") == 1)
        .select(PAIR_KEYS)
        .unique()
        .sort(PAIR_KEYS)
    )
    unseen_pairs = positive_pairs.join(
        history_pairs, on=list(PAIR_KEYS), how="anti"
    ).sort(PAIR_KEYS)
    cold_pairs = unseen_pairs.join(
        history_items, on="item_id", how="anti"
    ).sort(PAIR_KEYS)
    eligible_pairs = (
        unseen_pairs.join(history_items, on="item_id", how="semi")
        .select(PAIR_KEYS)
        .cast(GROUND_TRUTH_SCHEMA)
        .sort(PAIR_KEYS)
    )
    target_pairs = (
        eligible_pairs.join(target_users, on="user_id", how="semi")
        .select(PAIR_KEYS)
        .cast(GROUND_TRUTH_SCHEMA)
        .sort(PAIR_KEYS)
    )
    return GroundTruthFrames(
        positive_pairs=positive_pairs,
        unseen_pairs=unseen_pairs,
        cold_pairs=cold_pairs,
        eligible_pairs=eligible_pairs,
        target_pairs=target_pairs,
    )


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sink_parquet(frame: pl.LazyFrame, path: Path) -> None:
    frame.sink_parquet(
        path,
        compression="zstd",
        statistics=True,
        row_group_size=262_144,
        maintain_order=True,
        engine="streaming",
    )


def _scalar_row(frame: pl.LazyFrame) -> dict[str, Any]:
    return frame.collect(engine="streaming").row(0, named=True)


def _raw_diagnostics(raw: pl.LazyFrame) -> dict[str, Any]:
    summary = _scalar_row(
        raw.select(
            rows=pl.len(),
            users=pl.col("user_id").n_unique(),
            items=pl.col("item_id").n_unique(),
            min_date=pl.col("date").min(),
            max_date=pl.col("date").max(),
            null_cells=pl.sum_horizontal(pl.all().null_count()),
        )
    )
    event_counts_frame = (
        raw.group_by("event_type")
        .len(name="rows")
        .sort("event_type")
        .collect(engine="streaming")
    )
    event_counts = {
        str(row["event_type"]): int(row["rows"])
        for row in event_counts_frame.iter_rows(named=True)
    }
    return {
        **{key: _json_value(value) for key, value in summary.items()},
        "event_counts": event_counts,
    }


def _source_target_diagnostics(
    target_users: pl.LazyFrame, raw: pl.LazyFrame
) -> dict[str, int]:
    summary = _scalar_row(
        target_users.select(
            rows=pl.len(),
            unique_users=pl.col("user_id").n_unique(),
            nulls=pl.col("user_id").null_count(),
        )
    )
    if summary["rows"] != summary["unique_users"]:
        raise DataPreparationError("target users must be unique")
    if summary["nulls"]:
        raise DataPreparationError("target users contain null values")
    missing = (
        target_users.join(
            raw.select("user_id").unique(), on="user_id", how="anti"
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if missing:
        raise DataPreparationError(
            f"{missing} target user(s) are absent from raw interactions"
        )
    return {
        "rows": int(summary["rows"]),
        "unique_users": int(summary["unique_users"]),
        "nulls": int(summary["nulls"]),
        "missing_from_raw": int(missing),
    }


def _ordered_key_checks(frame: pl.LazyFrame, keys: tuple[str, ...]) -> dict[str, int]:
    if len(keys) == 1:
        current = pl.col(keys[0])
        previous = current.shift(1)
        duplicate = current == previous
        out_of_order = current < previous
    elif len(keys) == 2:
        first, second = (pl.col(key) for key in keys)
        prev_first, prev_second = first.shift(1), second.shift(1)
        duplicate = (first == prev_first) & (second == prev_second)
        out_of_order = (first < prev_first) | (
            (first == prev_first) & (second < prev_second)
        )
    elif len(keys) == 3:
        first, second, third = (pl.col(key) for key in keys)
        prev_first = first.shift(1)
        prev_second = second.shift(1)
        prev_third = third.shift(1)
        duplicate = (
            (first == prev_first)
            & (second == prev_second)
            & (third == prev_third)
        )
        out_of_order = (
            (first < prev_first)
            | ((first == prev_first) & (second < prev_second))
            | (
                (first == prev_first)
                & (second == prev_second)
                & (third < prev_third)
            )
        )
    else:  # pragma: no cover - internal callers use one to three keys
        raise ValueError("ordered key checks support one to three columns")
    row = _scalar_row(
        frame.select(
            duplicate_rows=duplicate.fill_null(False).sum(),
            out_of_order_rows=out_of_order.fill_null(False).sum(),
        )
    )
    return {key: int(value) for key, value in row.items()}


def _daily_artifact_diagnostics(path: Path) -> dict[str, Any]:
    frame = pl.scan_parquet(path)
    _require_schema(
        frame.collect_schema(), DAILY_INTERACTION_SCHEMA, name=path.name
    )
    summary = _scalar_row(
        frame.select(
            rows=pl.len(),
            users=pl.col("user_id").n_unique(),
            items=pl.col("item_id").n_unique(),
            min_date=pl.col("date").min(),
            max_date=pl.col("date").max(),
            min_dt=pl.col("dt").min(),
            max_dt=pl.col("dt").max(),
            null_cells=pl.sum_horizontal(pl.all().null_count()),
        )
    )
    order = _ordered_key_checks(frame.select(DAILY_KEYS), DAILY_KEYS)
    if summary["null_cells"] or any(order.values()):
        raise DataPreparationError(
            f"invalid daily artifact {path.name}: null/order checks "
            f"{summary['null_cells']=}, {order=}"
        )
    return {
        **{key: _json_value(value) for key, value in summary.items()},
        **order,
    }


def _pair_artifact_diagnostics(path: Path) -> dict[str, Any]:
    frame = pl.scan_parquet(path)
    _require_schema(frame.collect_schema(), GROUND_TRUTH_SCHEMA, name=path.name)
    summary = _scalar_row(
        frame.select(
            rows=pl.len(),
            users=pl.col("user_id").n_unique(),
            items=pl.col("item_id").n_unique(),
            null_cells=pl.sum_horizontal(pl.all().null_count()),
        )
    )
    order = _ordered_key_checks(frame.select(PAIR_KEYS), PAIR_KEYS)
    if summary["null_cells"] or any(order.values()):
        raise DataPreparationError(
            f"invalid pair artifact {path.name}: null/order checks "
            f"{summary['null_cells']=}, {order=}"
        )
    return {
        **{key: int(value) for key, value in summary.items()},
        **order,
    }


def _target_artifact_diagnostics(path: Path) -> dict[str, int]:
    frame = pl.scan_parquet(path)
    _require_schema(frame.collect_schema(), TARGET_USER_SCHEMA, name=path.name)
    summary = _scalar_row(
        frame.select(
            rows=pl.len(),
            users=pl.col("user_id").n_unique(),
            null_cells=pl.sum_horizontal(pl.all().null_count()),
        )
    )
    order = _ordered_key_checks(frame.select("user_id"), ("user_id",))
    if summary["null_cells"] or any(order.values()):
        raise DataPreparationError(
            f"invalid target artifact {path.name}: {summary=}, {order=}"
        )
    return {
        **{key: int(value) for key, value in summary.items()},
        **order,
    }


def _peak_memory_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def prepare_temporal_fold(
    *,
    train_path: str | Path,
    target_users_path: str | Path,
    output_dir: str | Path,
    cutoff: datetime | None = None,
    validation_end_exclusive: datetime | None = None,
    smoke_user_limit: int | None = None,
    run_id: str | None = None,
) -> FoldPreparationResult:
    """Materialize and atomically publish a validated temporal-fold artifact."""

    started = time.perf_counter()
    train_source = Path(train_path)
    target_source = Path(target_users_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if smoke_user_limit is not None:
        if isinstance(smoke_user_limit, bool) or smoke_user_limit <= 0:
            raise ValueError("smoke_user_limit must be a positive integer")
        if output.name == "task01_canonical_data_v1":
            raise DataPreparationError(
                "limited smoke must not use the canonical artifact path"
            )

    raw = scan_raw_interactions(train_source)
    source_targets = scan_target_users(target_source)
    source_raw_diagnostics = _raw_diagnostics(raw)
    source_target_diagnostics = _source_target_diagnostics(source_targets, raw)
    resolved_cutoff = cutoff or infer_canonical_cutoff(raw)
    minimum = datetime.fromisoformat(source_raw_diagnostics["min_date"])
    maximum = datetime.fromisoformat(source_raw_diagnostics["max_date"])
    if not minimum < resolved_cutoff <= maximum:
        raise DataPreparationError(
            f"cutoff must be in ({minimum.isoformat()}, {maximum.isoformat()}]"
        )
    if (
        validation_end_exclusive is not None
        and validation_end_exclusive > maximum + timedelta(microseconds=1)
    ):
        raise DataPreparationError(
            "validation_end_exclusive is later than the source timeline"
        )

    selected_targets = source_targets.sort("user_id")
    mode = "full"
    if smoke_user_limit is not None:
        selected_targets = selected_targets.head(smoke_user_limit)
        raw = raw.join(selected_targets, on="user_id", how="semi")
        mode = "limited_smoke"

    history_raw, validation_raw = split_raw_interactions(
        raw,
        cutoff=resolved_cutoff,
        validation_end_exclusive=validation_end_exclusive,
    )
    split_counts = _scalar_row(
        raw.select(
            history_raw_rows=(pl.col("date") < resolved_cutoff).sum(),
            validation_raw_rows=(
                (pl.col("date") >= resolved_cutoff)
                & (
                    pl.lit(True)
                    if validation_end_exclusive is None
                    else pl.col("date") < validation_end_exclusive
                )
            ).sum(),
        )
    )
    if not split_counts["history_raw_rows"] or not split_counts[
        "validation_raw_rows"
    ]:
        raise DataPreparationError("both history and validation must be non-empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    work = staging / ".work"
    work.mkdir()
    try:
        target_output = staging / "target_users.parquet"
        history_output = staging / "history_daily.parquet"
        validation_output = staging / "validation_daily.parquet"
        ground_truth_output = staging / "ground_truth.parquet"
        target_ground_truth_output = staging / "target_ground_truth.parquet"

        _sink_parquet(
            selected_targets.unique().sort("user_id").cast(TARGET_USER_SCHEMA),
            target_output,
        )
        _sink_parquet(aggregate_daily_interactions(history_raw), history_output)
        _sink_parquet(
            aggregate_daily_interactions(validation_raw), validation_output
        )

        history_daily = pl.scan_parquet(history_output)
        validation_daily = pl.scan_parquet(validation_output)
        artifact_targets = pl.scan_parquet(target_output)
        stages = build_ground_truth(
            history_daily, validation_daily, artifact_targets
        )

        positive_output = work / "positive_pairs.parquet"
        unseen_output = work / "unseen_pairs.parquet"
        cold_output = work / "cold_pairs.parquet"
        _sink_parquet(stages.positive_pairs, positive_output)
        # Reuse materialized positive pairs so the expensive validation-side
        # deduplication is not repeated by the remaining joins.
        positive_pairs = pl.scan_parquet(positive_output)
        history_pairs = history_daily.select(PAIR_KEYS).unique()
        unseen_pairs = positive_pairs.join(
            history_pairs, on=list(PAIR_KEYS), how="anti"
        ).sort(PAIR_KEYS)
        _sink_parquet(unseen_pairs, unseen_output)

        unseen_pairs = pl.scan_parquet(unseen_output)
        history_items = history_daily.select("item_id").unique()
        cold_pairs = unseen_pairs.join(
            history_items, on="item_id", how="anti"
        ).sort(PAIR_KEYS)
        eligible_pairs = (
            unseen_pairs.join(history_items, on="item_id", how="semi")
            .cast(GROUND_TRUTH_SCHEMA)
            .sort(PAIR_KEYS)
        )
        _sink_parquet(cold_pairs, cold_output)
        _sink_parquet(eligible_pairs, ground_truth_output)
        _sink_parquet(
            pl.scan_parquet(ground_truth_output)
            .join(artifact_targets, on="user_id", how="semi")
            .cast(GROUND_TRUTH_SCHEMA)
            .sort(PAIR_KEYS),
            target_ground_truth_output,
        )

        daily_diagnostics = {
            "history": _daily_artifact_diagnostics(history_output),
            "validation": _daily_artifact_diagnostics(validation_output),
        }
        ground_truth_diagnostics = _pair_artifact_diagnostics(
            ground_truth_output
        )
        target_ground_truth_diagnostics = _pair_artifact_diagnostics(
            target_ground_truth_output
        )
        target_artifact_diagnostics = _target_artifact_diagnostics(target_output)

        positive_stats = _pair_artifact_diagnostics(positive_output)
        unseen_stats = _pair_artifact_diagnostics(unseen_output)
        cold_stats = _pair_artifact_diagnostics(cold_output)
        positive_rows = positive_stats["rows"]
        seen_rows = positive_rows - unseen_stats["rows"]
        cold_rows = cold_stats["rows"]
        gt_funnel = {
            "positive_unique_pairs": positive_rows,
            "seen_pairs_removed": seen_rows,
            "seen_pair_share_of_positive": (
                seen_rows / positive_rows if positive_rows else 0.0
            ),
            "unseen_positive_pairs": unseen_stats["rows"],
            "cold_pairs_removed": cold_rows,
            "cold_pair_share_of_positive": (
                cold_rows / positive_rows if positive_rows else 0.0
            ),
            "cold_unique_items": cold_stats["items"],
            "eligible_pairs": ground_truth_diagnostics["rows"],
            "eligible_users": ground_truth_diagnostics["users"],
            "target_eligible_pairs": target_ground_truth_diagnostics["rows"],
            "target_labeled_users": target_ground_truth_diagnostics["users"],
            "target_users": target_artifact_diagnostics["rows"],
        }

        parquet_names = (
            "history_daily.parquet",
            "validation_daily.parquet",
            "ground_truth.parquet",
            "target_ground_truth.parquet",
            "target_users.parquet",
        )
        output_checksums = {
            name: _sha256_file(staging / name) for name in parquet_names
        }
        deterministic_diagnostics = {
            "source_raw": source_raw_diagnostics,
            "source_targets": source_target_diagnostics,
            "split": {
                "cutoff": resolved_cutoff.isoformat(),
                "validation_end_exclusive": (
                    validation_end_exclusive.isoformat()
                    if validation_end_exclusive is not None
                    else None
                ),
                **{key: int(value) for key, value in split_counts.items()},
            },
            "daily": daily_diagnostics,
            "ground_truth_funnel": gt_funnel,
            "ground_truth": ground_truth_diagnostics,
            "target_ground_truth": target_ground_truth_diagnostics,
            "target_users": target_artifact_diagnostics,
            "output_sha256": output_checksums,
        }
        config = {
            "run_id": run_id or output.name,
            "artifact_version": 1,
            "mode": mode,
            "inputs": {
                "train_path": train_source.as_posix(),
                "target_users_path": target_source.as_posix(),
                "train_sha256": _sha256_file(train_source),
                "target_users_sha256": _sha256_file(target_source),
            },
            "split": {
                "cutoff": resolved_cutoff.isoformat(),
                "history_predicate": "date < cutoff",
                "validation_predicate": (
                    "date >= cutoff"
                    if validation_end_exclusive is None
                    else "date >= cutoff AND date < validation_end_exclusive"
                ),
                "validation_end_exclusive": (
                    validation_end_exclusive.isoformat()
                    if validation_end_exclusive is not None
                    else None
                ),
            },
            "aggregation": {
                "split_before_aggregation": True,
                "keys": list(DAILY_KEYS),
                "dt": "min(raw date)",
                "views": "count(raw event rows)",
                "watch_time": "max(watch_time)",
                "is_like": "max(event_type == 'like')",
                "is_favorite": "max(event_type == 'favorite')",
                "is_positive": (
                    "watch_time > 60 OR is_like == 1 OR is_favorite == 1"
                ),
            },
            "ground_truth": {
                "positive_only": True,
                "deduplicate_keys": list(PAIR_KEYS),
                "remove_history_pairs": True,
                "remove_items_absent_from_history": True,
            },
            "smoke_user_limit": smoke_user_limit,
        }
        metrics = {
            "run_id": run_id or output.name,
            "mode": mode,
            "deterministic_diagnostics": deterministic_diagnostics,
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        _write_json(staging / "config.json", config)
        _write_json(staging / "metrics.json", metrics)
        shutil.rmtree(work)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return FoldPreparationResult(output_dir=output, config=config, metrics=metrics)


__all__ = [
    "DAILY_INTERACTION_SCHEMA",
    "GROUND_TRUTH_SCHEMA",
    "RAW_INTERACTION_SCHEMA",
    "TARGET_USER_SCHEMA",
    "DataPreparationError",
    "FoldPreparationResult",
    "GroundTruthFrames",
    "aggregate_daily_interactions",
    "build_ground_truth",
    "infer_canonical_cutoff",
    "prepare_temporal_fold",
    "scan_raw_interactions",
    "scan_target_users",
    "split_raw_interactions",
]

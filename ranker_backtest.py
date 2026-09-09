"""Training-only quantization and complete-query temporal ranker comparisons."""

from __future__ import annotations

import gc
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from catboost.utils import quantize

from data_utils import TARGET_USER_SCHEMA
from experiment_utils import read_json, sha256_file, write_json_atomic
from ranker_data import deterministic_sampling_uniform
from rankers import validate_feature_columns, write_column_description
from validation import ContractValidationError, validate_unique_keys

FOLD_ORDER = ("rolling_1", "rolling_2", "rolling_3", "canonical")
QUANTIZATION_POLICIES = ("frozen_task08", "fit_training")


def load_backtest_config(path: str | Path) -> dict[str, Any]:
    """Validate the deliberately small, predeclared Task 12 experiment."""
    from rankers import CatBoostPointwiseConfig

    value = read_json(path)
    if value.get("kind") != "task12_ranker_backtest" or value.get("mode") not in {
        "full",
        "smoke",
    }:
        raise ValueError("invalid Task 12 kind or mode")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(value.get("run_id", ""))):
        raise ValueError("run_id must be a safe directory name")
    if value.get("seed") != 42:
        raise ValueError("Task 12 fixes seed=42 for a comparable experiment")
    protocol = value["protocol"]
    if protocol != {
        "selection_train_folds": ["rolling_1", "rolling_2"],
        "selection_eval_fold": "rolling_3",
        "canonical_train_folds": ["rolling_1", "rolling_2", "rolling_3"],
        "canonical_eval_fold": "canonical",
    }:
        raise ValueError("Task 12 requires the fixed walk-forward protocol")
    if value["quantization"]["policies"] != list(QUANTIZATION_POLICIES):
        raise ValueError("compare exactly frozen_task08 and fit_training")
    params = CatBoostPointwiseConfig.from_mapping(value["catboost"])
    if params.random_seed != value["seed"] or params.loss_function != "Logloss":
        raise ValueError("Task 12 requires seed 42 and binary Logloss")
    if params.ignored_features or params.scale_pos_weight != 1.0:
        raise ValueError("Task 12 keeps all features and scale_pos_weight=1")
    if value["quantization"]["border_count"] != params.border_count:
        raise ValueError("Pool and model border_count must agree")
    if value["quantization"]["feature_border_type"] != "GreedyLogSum":
        raise ValueError("Task 12 fixes feature_border_type=GreedyLogSum")
    counts = value["selection"]["tree_counts"]
    if (
        not isinstance(counts, list)
        or not counts
        or any(type(n) is not int or n < 1 for n in counts)
    ):
        raise ValueError("tree_counts must be positive integers")
    if counts != sorted(set(counts)) or counts[-1] != params.iterations:
        raise ValueError(
            "tree_counts must be unique, sorted, and include the full budget"
        )
    if value["selection"]["fixed_tree_count"] not in counts:
        raise ValueError("fixed_tree_count must be present in tree_counts")
    for name, number in {
        "target_training_rows": value["training"]["target_rows"],
        "eval_users": value["selection"]["user_count"],
        "batch_size": value["inference"]["batch_size"],
    }.items():
        if type(number) is not int or number <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if value["inference"]["final_k"] != 20:
        raise ValueError("Task 12 uses fixed Precision@20")
    if value["mode"] == "smoke":
        n = value["smoke"]["user_count"]
        if type(n) is not int or not 1 <= n <= 128 or params.iterations > 20:
            raise ValueError("smoke is limited to 128 users and 20 trees")
        if value["training"]["target_rows"] > 30000:
            raise ValueError("smoke training is limited to 30,000 rows")
    elif value["smoke"]["user_count"] is not None or params.task_type != "GPU":
        raise ValueError("full run requires full input scope and the configured GPU")
    for key in (
        "minimum_free_disk_gib",
        "minimum_available_ram_gib",
        "maximum_dsv_gib",
    ):
        number = value["resources"][key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"invalid resource limit: {key}")
    return value


def validate_temporal_training_scope(
    train_folds: Sequence[str], eval_fold: str, cutoffs: Mapping[str, str]
) -> None:
    if (
        not train_folds
        or len(set(train_folds)) != len(train_folds)
        or eval_fold in train_folds
    ):
        raise ValueError("training and evaluation folds must be distinct")
    evaluation = datetime.fromisoformat(cutoffs[eval_fold])
    for fold in train_folds:
        # A fold's labels are from the following complete 24-hour period.
        if datetime.fromisoformat(cutoffs[fold]) + timedelta(days=1) > evaluation:
            raise ValueError(
                f"training labels overlap evaluation history boundary: {fold}"
            )


def select_evaluation_users(
    target_users: pl.DataFrame, *, count: int, seed: int
) -> pl.DataFrame:
    """Sample complete users using IDs only, including users without future GT."""
    if target_users.schema != TARGET_USER_SCHEMA:
        raise ContractValidationError("target user ID dtype must be UInt64")
    validate_unique_keys(target_users, keys=("user_id",), name="target users")
    if type(count) is not int or count <= 0 or count > target_users.height:
        raise ValueError("evaluation user count exceeds the target universe")
    pairs = target_users.with_columns(pl.lit(0, dtype=pl.Int32).alias("item_id"))
    hashed = deterministic_sampling_uniform(pairs, seed=seed, fold="task12:eval_users")
    return (
        target_users.with_columns(pl.Series("__hash", hashed))
        .sort(["__hash", "user_id"])
        .head(count)
        .select("user_id")
        .sort("user_id")
    )


def allocate_training_probabilities(
    counts: Mapping[str, Mapping[str, int]], *, target_rows: int
) -> dict[str, dict[str, float | int]]:
    """Allocate the same expected row budget per fold, retaining every positive."""
    if not counts or type(target_rows) is not int or target_rows <= 0:
        raise ValueError("non-empty folds and a positive row budget are required")
    budget, remainder = divmod(target_rows, len(counts))
    result = {}
    for index, (fold, row) in enumerate(counts.items()):
        positive, total = int(row["positives"]), int(row["rows"])
        allocated = budget + int(index < remainder)
        if positive < 0 or total <= positive or allocated <= positive:
            raise ValueError(
                f"row budget must retain all positives plus negatives: {fold}"
            )
        probability = min(1.0, (allocated - positive) / (total - positive))
        result[fold] = {
            "source_rows": total,
            "positive_rows": positive,
            "target_rows": min(allocated, total),
            "secondary_negative_probability": probability,
        }
    return result


def sample_training_rows(
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    fold: str,
    seed: int,
    negative_probability: float,
) -> pl.DataFrame:
    """Independent second-stage sampling with full inclusion-probability weights."""
    validate_feature_columns(features)
    if not math.isfinite(negative_probability) or not 0 < negative_probability <= 1:
        raise ValueError("negative_probability must be in (0, 1]")
    source_positives = frame.filter(pl.col("label") == 1).height
    selected = frame.filter(pl.col("is_training_sample"))
    if selected.filter(pl.col("label") == 1).height != source_positives:
        raise ContractValidationError("upstream sampling dropped positives")
    if selected.filter(
        pl.col("sampling_probability").is_null()
        | ~pl.col("sampling_probability").is_finite()
        | (pl.col("sampling_probability") <= 0)
        | (pl.col("sampling_probability") > 1)
        | ((pl.col("label") == 1) & (pl.col("sampling_probability") != 1))
    ).height:
        raise ContractValidationError("invalid upstream inclusion probability")
    uniform = deterministic_sampling_uniform(
        selected, seed=seed, fold=f"task12:train:{fold}"
    )
    return (
        selected.with_columns(pl.Series("__uniform", uniform))
        .filter((pl.col("label") == 1) | (pl.col("__uniform") < negative_probability))
        .select(
            "user_id",
            "item_id",
            pl.lit(fold).alias("fold_id"),
            "label",
            (
                1.0
                / (
                    pl.col("sampling_probability").cast(pl.Float64)
                    * pl.when(pl.col("label") == 1)
                    .then(1.0)
                    .otherwise(negative_probability)
                )
            )
            .cast(pl.Float32)
            .alias("sample_weight"),
            *(pl.col(name).cast(pl.Float32) for name in features),
        )
        .sort(["user_id", "item_id"])
    )


def read_numeric_borders(
    path: str | Path, *, feature_count: int
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {i: [] for i in range(feature_count)}
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) not in (2, 3):
            raise ValueError("invalid borders row")
        index, border = int(parts[0]), float(parts[1])
        if index not in result or not math.isfinite(border):
            raise ValueError("invalid feature index or non-finite border")
        result[index].append(border)
    for borders in result.values():
        if borders != sorted(set(borders)):
            raise ValueError("borders must be strictly increasing per feature")
    return result


def quantize_training_parts(
    parts: Sequence[Path],
    *,
    destination: Path,
    features: Sequence[str],
    policy: str,
    frozen_borders: Path,
    border_count: int,
    feature_border_type: str,
    seed: int,
    thread_count: int,
    maximum_dsv_bytes: int,
    progress: Callable[[int, float | None, Any], None] | None = None,
    operation_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Quantize training rows only; persist actual borders and explicit constants.

    Frozen mode ignores borderless features instead of silently fitting their
    thresholds. Fresh mode learns every threshold on these training rows.
    Evaluation uses raw features through the saved model, never fits borders.
    """
    if policy not in QUANTIZATION_POLICIES or not parts:
        raise ValueError("invalid quantization policy or empty input")
    destination.mkdir(parents=True, exist_ok=True)
    dsv, cd = destination / "transport.tsv", destination / "columns.cd"
    write_column_description(cd, feature_columns=features)
    expected = (
        read_numeric_borders(frozen_borders, feature_count=len(features))
        if policy == "frozen_task08"
        else None
    )
    ignored = [index for index, borders in (expected or {}).items() if not borders]
    if expected is not None and any(len(v) > border_count for v in expected.values()):
        raise ValueError("frozen borders exceed the configured border_count")
    if len(ignored) == len(features):
        raise ValueError("frozen borders would ignore every feature")
    try:
        rows, positives, weight_sum = 0, 0, 0.0
        with dsv.open("w", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
            for index, part in enumerate(parts):
                data = pl.read_parquet(
                    part, columns=["label", "sample_weight", *features]
                )
                data.write_csv(
                    stream, separator="\t", include_header=False, float_scientific=True
                )
                stream.flush()
                size = dsv.stat().st_size
                if size > maximum_dsv_bytes:
                    raise ValueError("temporary DSV exceeded the configured size limit")
                rows += data.height
                positives += int(data["label"].sum())
                weight_sum += float(data["sample_weight"].cast(pl.Float64).sum())
                if progress is not None:
                    progress(index, None, {"bytes": size, "rows": rows})
        kwargs = (
            {"input_borders": str(frozen_borders), "ignored_features": ignored}
            if expected is not None
            else {}
        )
        if operation_progress is not None:
            operation_progress("quantize_cpu")
        pool = quantize(
            data_path=str(dsv),
            column_description=str(cd),
            delimiter="\t",
            has_header=False,
            task_type="CPU",
            border_count=border_count,
            feature_border_type=feature_border_type,
            random_seed=seed,
            thread_count=thread_count,
            **kwargs,
        )
        if pool.num_row() != rows or tuple(pool.get_feature_names()) != tuple(features):
            raise ContractValidationError(
                "quantized Pool differs from training contract"
            )
        actual_path = destination / "borders.tsv"
        pool.save_quantization_borders(str(actual_path))
        actual = read_numeric_borders(actual_path, feature_count=len(features))
        if any(len(value) > border_count for value in actual.values()):
            raise ContractValidationError(
                "quantizer exceeded the declared border count"
            )
        if expected is not None and actual != expected:
            raise ContractValidationError(
                "frozen quantization added or changed thresholds"
            )
        if operation_progress is not None:
            operation_progress("save_quantized_pool")
        pool.save(str(destination / "train.quantized"))
        del pool
        gc.collect()
        diagnostics = {
            "policy": policy,
            "rows": rows,
            "positive_rows": positives,
            "weight_sum": weight_sum,
            "feature_count": len(features),
            "border_count": border_count,
            "feature_border_type": feature_border_type,
            "threshold_fit_scope": "training_examples_only"
            if expected is None
            else "frozen_task08_training",
            "input_borders_sha256": sha256_file(frozen_borders)
            if expected is not None
            else None,
            "actual_borders_sha256": sha256_file(actual_path),
            "ignored_borderless_features": [features[i] for i in ignored],
            "features": [
                {"name": name, "border_count": len(actual[i]), "borders": actual[i]}
                for i, name in enumerate(features)
            ],
        }
        write_json_atomic(destination / "quantization.json", diagnostics)
        return diagnostics
    finally:
        dsv.unlink(missing_ok=True)


def select_tree_checkpoint(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select by exact P@20 numerator on one fixed user universe; stable ties."""
    if not curve:
        raise ValueError("checkpoint curve is empty")
    universes = {(row["target_users"], row["labeled_users"]) for row in curve}
    if len(universes) != 1:
        raise ValueError("checkpoint metrics have different evaluation universes")
    return dict(min(curve, key=lambda row: (-row["final_hits"], row["tree_count"])))


def metrics_from_hits(
    *, hits: int, targets: int, labeled: int
) -> dict[str, float | int]:
    if targets <= 0 or not 0 <= labeled <= targets or not 0 <= hits <= 20 * labeled:
        raise ValueError("invalid fixed-denominator Precision@20 counts")
    return {
        "final_hits": hits,
        "target_users": targets,
        "labeled_users": labeled,
        "precision_at_20_all_targets": hits / (20 * targets),
        "precision_at_20_labeled_users": hits / (20 * labeled) if labeled else 0.0,
    }

#!/usr/bin/env python3
"""Train one pointwise CatBoost baseline on immutable task-07 datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import catboost
import polars as pl
from catboost import Pool
from catboost.utils import quantize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from candidate_pipeline import CandidateUnionConfig, generator_columns
from data_utils import GROUND_TRUTH_SCHEMA, TARGET_USER_SCHEMA
from ensemble import RRFConfig, RRFEnsembleModel
from experiment_utils import (
    AtomicBestConfig,
    CheckpointStore,
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from metrics import evaluate_candidate_metrics_lazy, evaluate_precision_at_20
from popularity import candidates_to_recommendations
from rankers import (
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
    ranker_scores_to_candidates,
    validate_feature_columns,
    write_catboost_dsv_part,
    write_column_description,
)
from scripts.run_global_popularity import _final_hit_count, _peak_memory_mb
from validation import ContractValidationError, validate_recommendations_against_history


class CatBoostExperimentError(ValueError):
    """Raised for incompatible task-08 config or immutable input data."""


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatBoostExperimentError(f"{name} must be a positive integer")
    return value


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatBoostExperimentError(f"{name} must be a number")
    result = float(value)
    if not 0 < result <= 1:
        raise CatBoostExperimentError(f"{name} must be in (0, 1]")
    return result


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        frame.write_parquet(temporary, compression="zstd", compression_level=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _directory_size_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _validate_cleanup_target(
    path: Path,
    *,
    marker_name: str,
    run_id: str,
    protected_paths: Sequence[Path],
) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise CatBoostExperimentError(f"cleanup target is not a directory: {path}")
    if len(resolved.parts) < 4 or resolved == Path(resolved.anchor):
        raise CatBoostExperimentError(f"cleanup target is too broad: {path}")
    for protected_path in protected_paths:
        protected = protected_path.resolve()
        if (
            resolved == protected
            or resolved in protected.parents
            or protected in resolved.parents
        ):
            raise CatBoostExperimentError(
                f"cleanup target overlaps protected path: {path} and {protected_path}"
            )
    marker = read_json(resolved / marker_name)
    if marker.get("artifact_version") != 1 or marker.get("run_id") != run_id:
        raise CatBoostExperimentError(
            f"cleanup marker does not belong to run {run_id}: {resolved / marker_name}"
        )
    return resolved


def cleanup_recoverable_state(
    *,
    checkpoint_dir: str | Path,
    best_model_dir: str | Path,
    output_dir: str | Path,
    pool_cache_dir: str | Path,
    dataset_dir: str | Path,
    rrf_artifact_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Remove resume-only state after validating the published final artifact.

    Quantized pools and immutable input datasets are deliberately protected and
    retained. Missing cleanup targets make the operation idempotent.
    """

    output = Path(output_dir)
    config = read_json(output / "config.json")
    metrics = read_json(output / "metrics.json")
    model_path = output / "model" / "model.cbm"
    if (
        config.get("kind") != "task08_catboost_pointwise"
        or config.get("run_id") != run_id
        or metrics.get("run_id") != run_id
        or not model_path.is_file()
        or metrics.get("model_sha256") != sha256_file(model_path)
    ):
        raise CatBoostExperimentError(
            f"refusing cleanup because published artifact is incomplete: {output}"
        )

    protected = [
        output,
        Path(pool_cache_dir),
        Path(dataset_dir),
        Path(rrf_artifact_dir),
    ]
    targets = (
        (Path(checkpoint_dir), "checkpoint.json", "checkpoint"),
        (Path(best_model_dir), "best_model.json", "best_model"),
    )
    removed: dict[str, dict[str, Any]] = {}
    validated_targets: list[tuple[Path, Path, str, int]] = []
    for path, marker_name, name in targets:
        if not path.exists():
            removed[name] = {"removed": False, "bytes": 0}
            continue
        target = _validate_cleanup_target(
            path,
            marker_name=marker_name,
            run_id=run_id,
            protected_paths=protected,
        )
        size_bytes = _directory_size_bytes(target)
        validated_targets.append((path, target, name, size_bytes))

    for path, target, name, size_bytes in validated_targets:
        shutil.rmtree(target)
        removed[name] = {
            "removed": True,
            "bytes": size_bytes,
            "path": path.as_posix(),
        }
    return {
        "removed": removed,
        "freed_bytes": sum(record["bytes"] for record in removed.values()),
        "retained_pool_cache": Path(pool_cache_dir).as_posix(),
    }


def _load_source_config(path: str | Path) -> dict[str, Any]:
    source = read_json(path)
    required = {
        "task07_artifact",
        "task06_rrf_artifact",
        "folds",
        "row_sampling",
        "pool",
        "catboost",
        "inference",
        "seed",
    }
    missing = required.difference(source)
    if missing:
        raise CatBoostExperimentError(f"config lacks fields: {sorted(missing)}")
    folds = source["folds"]
    if not isinstance(folds, Mapping) or set(folds) != {
        "train",
        "early_stopping",
        "canonical",
    }:
        raise CatBoostExperimentError(
            "folds must define train/early_stopping/canonical"
        )
    if len(set(folds.values())) != 3:
        raise CatBoostExperimentError("temporal folds must be distinct")
    sampling = source["row_sampling"]
    if not isinstance(sampling, Mapping):
        raise CatBoostExperimentError("row_sampling must be an object")
    _probability(
        sampling.get("train_negative_keep_probability"),
        name="train_negative_keep_probability",
    )
    _probability(
        sampling.get("eval_negative_keep_probability"),
        name="eval_negative_keep_probability",
    )
    if sampling.get("weighting") != "inverse_sampling_probability":
        raise CatBoostExperimentError("task08 requires inverse sampling weights")
    pool = source["pool"]
    if not isinstance(pool, Mapping):
        raise CatBoostExperimentError("pool must be an object")
    cat_config = CatBoostPointwiseConfig.from_mapping(source["catboost"])
    if (
        _positive_int(pool.get("border_count"), name="pool.border_count")
        != cat_config.border_count
    ):
        raise CatBoostExperimentError("pool and CatBoost border_count must match")
    if pool.get("feature_border_type") not in {
        "Median",
        "Uniform",
        "UniformAndQuantiles",
        "MaxLogSum",
        "MinEntropy",
        "GreedyLogSum",
    }:
        raise CatBoostExperimentError("unsupported pool.feature_border_type")
    if pool.get("quantization_task_type") not in {"CPU", "GPU"}:
        raise CatBoostExperimentError("pool.quantization_task_type must be CPU or GPU")
    _positive_int(pool.get("thread_count"), name="pool.thread_count")
    inference = source["inference"]
    if not isinstance(inference, Mapping):
        raise CatBoostExperimentError("inference must be an object")
    _positive_int(inference.get("batch_size"), name="inference.batch_size")
    if _positive_int(inference.get("final_k"), name="inference.final_k") != 20:
        raise CatBoostExperimentError("task08 requires final_k=20")
    if isinstance(source["seed"], bool) or not isinstance(source["seed"], int):
        raise CatBoostExperimentError("seed must be an integer")
    if source["seed"] != cat_config.random_seed or source["seed"] != sampling.get(
        "seed"
    ):
        raise CatBoostExperimentError("all random seeds must match")
    return source


def _materialized_union_config(rrf_artifact: Path) -> CandidateUnionConfig:
    value = read_json(rrf_artifact / "config.json").get("materialized_union")
    if not isinstance(value, Mapping):
        raise CatBoostExperimentError("task06 artifact lacks materialized union")
    caps = value.get("source_caps")
    order = value.get("source_order")
    if not isinstance(caps, Mapping) or not isinstance(order, list):
        raise CatBoostExperimentError("task06 materialized union is invalid")
    return CandidateUnionConfig.from_mapping(
        {str(source): int(caps[source]) for source in order},
        total_cap=int(value["total_cap"]),
    )


def _eligible_expression(config: CandidateUnionConfig) -> pl.Expr:
    return pl.any_horizontal(
        pl.col(generator_columns(spec.source)["generated"])
        & (pl.col(generator_columns(spec.source)["rank"]) <= spec.cap)
        for spec in config.sources
    )


class FoldContext:
    def __init__(
        self,
        *,
        dataset_root: Path,
        root_manifest: Mapping[str, Any],
        fold: str,
        part_limit: int | None,
        verify_checksums: bool,
    ) -> None:
        fold_value = root_manifest.get("folds", {}).get(fold)
        if not isinstance(fold_value, Mapping):
            raise CatBoostExperimentError(f"task07 manifest lacks fold {fold}")
        manifest_path = dataset_root / str(fold_value["manifest"])
        if sha256_file(manifest_path) != fold_value.get("manifest_sha256"):
            raise ContractValidationError(f"task07 fold manifest differs: {fold}")
        self.fold = fold
        self.root = manifest_path.parent
        self.manifest = read_json(manifest_path)
        expected_parts = self.manifest.get("parts")
        if not isinstance(expected_parts, Mapping):
            raise CatBoostExperimentError(f"task07 fold lacks parts: {fold}")
        all_parts = sorted((self.root / "ranker_data").glob("part-*.parquet"))
        if len(all_parts) != int(self.manifest.get("part_count", -1)):
            raise ContractValidationError(f"task07 part count differs: {fold}")
        self.parts = all_parts[:part_limit] if part_limit is not None else all_parts
        if not self.parts:
            raise CatBoostExperimentError(f"no ranker parts selected: {fold}")
        for part in self.parts:
            expected = expected_parts.get(part.name)
            if not isinstance(expected, str):
                raise ContractValidationError(f"unregistered task07 part: {part}")
            if verify_checksums and sha256_file(part) != expected:
                raise ContractValidationError(f"task07 part checksum differs: {part}")
        target_path = self.root / "target_users.parquet"
        ground_truth_path = self.root / "target_ground_truth.parquet"
        if sha256_file(target_path) != self.manifest.get("target_users_sha256"):
            raise ContractValidationError(f"target user checksum differs: {fold}")
        if sha256_file(ground_truth_path) != self.manifest.get(
            "target_ground_truth_sha256"
        ):
            raise ContractValidationError(f"ground-truth checksum differs: {fold}")
        target_users = pl.read_parquet(target_path)
        ground_truth = pl.read_parquet(ground_truth_path)
        if (
            target_users.schema != TARGET_USER_SCHEMA
            or ground_truth.schema != GROUND_TRUTH_SCHEMA
        ):
            raise ContractValidationError(f"evaluation schema differs: {fold}")
        if part_limit is not None:
            selected_users = (
                pl.scan_parquet([part.as_posix() for part in self.parts])
                .select("user_id")
                .unique()
                .collect(engine="streaming")
                .cast(TARGET_USER_SCHEMA)
                .sort("user_id")
            )
            target_users = target_users.join(selected_users, on="user_id", how="semi")
            ground_truth = ground_truth.join(selected_users, on="user_id", how="semi")
        self.target_users = target_users.sort("user_id")
        self.ground_truth = ground_truth.sort(("user_id", "item_id"))
        history_value = self.manifest.get("history_path")
        if not isinstance(history_value, str):
            raise CatBoostExperimentError(f"task07 fold lacks history path: {fold}")
        self.history_path = Path(history_value)
        if not self.history_path.is_file():
            raise FileNotFoundError(self.history_path)


def _feature_contract(
    dataset_root: Path,
) -> tuple[tuple[str, ...], dict[str, Any], str]:
    schema_path = dataset_root / "feature_schema.json"
    schema = read_json(schema_path)
    features = schema.get("feature_columns")
    if not isinstance(features, list) or schema.get("feature_count") != len(features):
        raise CatBoostExperimentError("task07 feature schema is invalid")
    checked = validate_feature_columns(features)
    return checked, schema, config_sha256(schema)


def _validate_task07(
    dataset_root: Path,
    *,
    folds: Sequence[str],
    part_limit: int | None,
    verify_checksums: bool,
) -> tuple[dict[str, FoldContext], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    root_manifest = read_json(dataset_root / "dataset_manifest.json")
    if root_manifest.get("kind") != "task07_ranker_datasets":
        raise CatBoostExperimentError("input is not a task07 ranker dataset")
    if part_limit is None and root_manifest.get("mode") != "full":
        raise CatBoostExperimentError("full task08 requires full task07 artifact")
    features, feature_schema, schema_file_sha = _feature_contract(dataset_root)
    if schema_file_sha != root_manifest.get("schema_sha256"):
        raise ContractValidationError("task07 root schema checksum differs")
    contexts = {
        fold: FoldContext(
            dataset_root=dataset_root,
            root_manifest=root_manifest,
            fold=fold,
            part_limit=part_limit,
            verify_checksums=verify_checksums,
        )
        for fold in folds
    }
    return contexts, features, feature_schema, root_manifest


def _dsv_projection(feature_columns: Sequence[str]) -> list[str]:
    return [
        "user_id",
        "item_id",
        "label",
        "is_training_sample",
        "sampling_probability",
        *feature_columns,
    ]


def _validate_checkpoint_file(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ContractValidationError(f"completed checkpoint file is corrupt: {path}")


def _materialize_dsv_parts(
    *,
    role: str,
    context: FoldContext,
    feature_columns: Sequence[str],
    probability: float,
    seed: int,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> tuple[list[Path], dict[str, Any]]:
    output_dir = checkpoint_root / "raw_parts" / role
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    totals = {
        "rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "sample_weight_sum": 0.0,
        "size_bytes": 0,
    }
    projection = _dsv_projection(feature_columns)
    for part in context.parts:
        fold_key = f"{context.fold}:{part.stem}"
        output = output_dir / f"{part.stem}.tsv"
        record = checkpoint.get(stage=f"dsv_{role}", config="pointwise", fold=fold_key)
        started = reporter.operation_start(
            stage="pool_materialization",
            config=role,
            fold=context.fold,
            operation=part.stem,
        )
        if record is not None:
            _validate_checkpoint_file(output, record)
            diagnostics = record["diagnostics"]
            reporter.event(
                "operation_resume_skip",
                stage="pool_materialization",
                config=role,
                fold=context.fold,
                operation=part.stem,
            )
        else:
            if output.exists():
                raise FileExistsError(f"unregistered DSV part exists: {output}")
            frame = pl.read_parquet(part, columns=projection)
            diagnostics = write_catboost_dsv_part(
                frame,
                output,
                feature_columns=feature_columns,
                fold=f"{context.fold}:{role}",
                seed=seed,
                negative_keep_probability=probability,
            )
            checkpoint.complete(
                stage=f"dsv_{role}",
                config="pointwise",
                fold=fold_key,
                metadata={
                    "sha256": diagnostics["sha256"],
                    "diagnostics": diagnostics,
                },
            )
        reporter.operation_finish(
            stage="pool_materialization",
            config=role,
            fold=context.fold,
            operation=part.stem,
            started=started,
            rows=diagnostics["rows"],
        )
        reporter.stage_advance()
        outputs.append(output)
        for name in totals:
            totals[name] += diagnostics[name]
    return outputs, totals


def _assemble_dsv(
    *,
    role: str,
    parts: Sequence[Path],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> Path:
    output = checkpoint_root / "raw" / f"{role}.tsv"
    record = checkpoint.get(stage="assemble_dsv", config="pointwise", fold=role)
    started = reporter.operation_start(
        stage="pool_materialization",
        config=role,
        fold=role,
        operation="assemble",
    )
    if record is not None:
        _validate_checkpoint_file(output, record)
        reporter.event(
            "operation_resume_skip",
            stage="pool_materialization",
            config=role,
            fold=role,
            operation="assemble",
        )
    else:
        if output.exists():
            raise FileExistsError(f"unregistered assembled DSV exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
        try:
            with temporary.open("wb") as destination:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        checkpoint.complete(
            stage="assemble_dsv",
            config="pointwise",
            fold=role,
            metadata={
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
            },
        )
    reporter.operation_finish(
        stage="pool_materialization",
        config=role,
        fold=role,
        operation="assemble",
        started=started,
        size_bytes=output.stat().st_size,
    )
    return output


def _validate_pool_cache(
    pool_root: Path,
    *,
    pool_digest: str,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    manifest = read_json(pool_root / "pool_manifest.json")
    if (
        manifest.get("artifact_version") != 1
        or manifest.get("kind") != "task08_quantized_pools"
        or manifest.get("pool_config_sha256") != pool_digest
        or manifest.get("feature_columns") != list(feature_columns)
    ):
        raise ContractValidationError(f"incompatible quantized pool cache: {pool_root}")
    for name, expected in manifest.get("files", {}).items():
        path = pool_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractValidationError(f"quantized pool cache is corrupt: {path}")
    for role in ("train", "eval"):
        pool = Pool(f"quantized://{(pool_root / f'{role}.quantized').resolve()}")
        if pool.num_row() != int(manifest[f"{role}_rows"]):
            raise ContractValidationError(f"quantized {role} row count differs")
        if tuple(pool.get_feature_names()) != tuple(feature_columns):
            raise ContractValidationError(f"quantized {role} feature order differs")
    return manifest


def _prepare_pool_cache(
    *,
    pool_root: Path,
    pool_digest: str,
    feature_columns: Sequence[str],
    train_context: FoldContext,
    eval_context: FoldContext,
    source: Mapping[str, Any],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    if pool_root.exists():
        return _validate_pool_cache(
            pool_root, pool_digest=pool_digest, feature_columns=feature_columns
        )
    sampling = source["row_sampling"]
    total_parts = len(train_context.parts) + len(eval_context.parts)
    stage_started = reporter.stage_start(
        stage="pool_materialization", total=total_parts, unit="part"
    )
    train_parts, train_diagnostics = _materialize_dsv_parts(
        role="train",
        context=train_context,
        feature_columns=feature_columns,
        probability=float(sampling["train_negative_keep_probability"]),
        seed=int(sampling["seed"]),
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    eval_parts, eval_diagnostics = _materialize_dsv_parts(
        role="eval",
        context=eval_context,
        feature_columns=feature_columns,
        probability=float(sampling["eval_negative_keep_probability"]),
        seed=int(sampling["seed"]),
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    reporter.stage_finish(
        stage="pool_materialization",
        started=stage_started,
        train_rows=train_diagnostics["rows"],
        eval_rows=eval_diagnostics["rows"],
    )
    train_dsv = _assemble_dsv(
        role="train",
        parts=train_parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    eval_dsv = _assemble_dsv(
        role="eval",
        parts=eval_parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    pool_work = checkpoint_root / "pool_artifact"
    pool_work.mkdir(parents=True, exist_ok=True)
    column_description = pool_work / "columns.cd"
    write_column_description(column_description, feature_columns=feature_columns)
    pool_config = source["pool"]
    train_path = pool_work / "train.quantized"
    borders_path = pool_work / "borders.tsv"
    record = checkpoint.get(stage="quantize", config="pointwise", fold="train")
    started = reporter.operation_start(
        stage="quantization",
        config="pointwise",
        fold=train_context.fold,
        operation="train",
    )
    if record is not None:
        _validate_checkpoint_file(train_path, record)
        if sha256_file(borders_path) != record.get("borders_sha256"):
            raise ContractValidationError("training quantization borders differ")
    else:
        if train_path.exists() or borders_path.exists():
            raise FileExistsError("unregistered training quantization output exists")
        train_pool = quantize(
            data_path=train_dsv.as_posix(),
            column_description=column_description.as_posix(),
            delimiter="\t",
            has_header=False,
            thread_count=int(pool_config["thread_count"]),
            border_count=int(pool_config["border_count"]),
            feature_border_type=str(pool_config["feature_border_type"]),
            task_type=str(pool_config["quantization_task_type"]),
            random_seed=int(source["seed"]),
        )
        train_pool.save(train_path.as_posix())
        train_pool.save_quantization_borders(borders_path.as_posix())
        checkpoint.complete(
            stage="quantize",
            config="pointwise",
            fold="train",
            metadata={
                "sha256": sha256_file(train_path),
                "borders_sha256": sha256_file(borders_path),
                "rows": train_pool.num_row(),
            },
        )
        del train_pool
        gc.collect()
    reporter.operation_finish(
        stage="quantization",
        config="pointwise",
        fold=train_context.fold,
        operation="train",
        started=started,
    )
    eval_path = pool_work / "eval.quantized"
    record = checkpoint.get(stage="quantize", config="pointwise", fold="eval")
    started = reporter.operation_start(
        stage="quantization",
        config="pointwise",
        fold=eval_context.fold,
        operation="eval",
    )
    if record is not None:
        _validate_checkpoint_file(eval_path, record)
    else:
        if eval_path.exists():
            raise FileExistsError("unregistered evaluation quantization output exists")
        eval_pool = quantize(
            data_path=eval_dsv.as_posix(),
            column_description=column_description.as_posix(),
            delimiter="\t",
            has_header=False,
            thread_count=int(pool_config["thread_count"]),
            input_borders=borders_path.as_posix(),
            task_type=str(pool_config["quantization_task_type"]),
            random_seed=int(source["seed"]),
        )
        eval_pool.save(eval_path.as_posix())
        checkpoint.complete(
            stage="quantize",
            config="pointwise",
            fold="eval",
            metadata={"sha256": sha256_file(eval_path), "rows": eval_pool.num_row()},
        )
        del eval_pool
        gc.collect()
    reporter.operation_finish(
        stage="quantization",
        config="pointwise",
        fold=eval_context.fold,
        operation="eval",
        started=started,
    )
    manifest = {
        "artifact_version": 1,
        "kind": "task08_quantized_pools",
        "pool_config_sha256": pool_digest,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "train_fold": train_context.fold,
        "eval_fold": eval_context.fold,
        "train_rows": train_diagnostics["rows"],
        "eval_rows": eval_diagnostics["rows"],
        "train_diagnostics": train_diagnostics,
        "eval_diagnostics": eval_diagnostics,
        "quantization": dict(pool_config),
        "files": {
            "train.quantized": sha256_file(train_path),
            "eval.quantized": sha256_file(eval_path),
            "borders.tsv": sha256_file(borders_path),
            "columns.cd": sha256_file(column_description),
        },
    }
    write_json_atomic(pool_work / "pool_manifest.json", manifest)
    publish_directory_atomic(pool_work, pool_root)
    return _validate_pool_cache(
        pool_root, pool_digest=pool_digest, feature_columns=feature_columns
    )


class _CatBoostLogBridge:
    def __init__(self, callback: Any) -> None:
        self._callback = callback
        self._buffer = ""
        self._started = time.perf_counter()

    def write(self, value: str) -> int:
        self._buffer += str(value)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            match = re.match(r"\s*(\d+):", line)
            if match:
                self._callback(
                    int(match.group(1)),
                    time.perf_counter() - self._started,
                    line[:500],
                )
        return len(value)

    def flush(self) -> None:
        return None


def _fit_or_restore_model(
    *,
    source: Mapping[str, Any],
    feature_columns: Sequence[str],
    pool_root: Path,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    best_model_root: Path,
    run_id: str,
    digest: str,
    reporter: EventProgressReporter,
) -> CatBoostPointwiseModel:
    model_dir = checkpoint_root / "trained_model"
    record = checkpoint.get(stage="fit", config="pointwise", fold="rolling")
    if record is not None:
        model = CatBoostPointwiseModel.from_artifact(model_dir)
        if sha256_file(model_dir / "model.cbm") != record.get("model_sha256"):
            raise ContractValidationError("completed CatBoost model differs")
        reporter.event(
            "operation_resume_skip",
            stage="fit",
            config="pointwise",
            fold="rolling",
            operation="catboost_fit",
        )
    else:
        loader = (
            CatBoostRankerDataLoader(
                feature_columns=feature_columns, seed=int(source["seed"])
            )
            .load_fit_data(
                train_pool=pool_root / "train.quantized",
                eval_pool=pool_root / "eval.quantized",
            )
            .prepare_fit_data()
        )
        config = CatBoostPointwiseConfig.from_mapping(source["catboost"])
        model = CatBoostPointwiseModel(config, feature_columns=feature_columns)
        iteration_started, callback = reporter.iteration_start(
            stage="fit",
            config=config.config_id,
            fold=str(source["folds"]["early_stopping"]),
            total=config.iterations,
            unit="tree",
        )
        bridge = _CatBoostLogBridge(callback)
        snapshot = checkpoint_root / "catboost_training" / "snapshot.cbsnapshot"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.fit(
                loader,
                train_dir=checkpoint_root / "catboost_training" / "train_dir",
                snapshot_file=snapshot,
                log_cout=bridge,
                log_cerr=bridge,
            )
        except BaseException:
            reporter.iteration_finish(
                stage="fit",
                config=config.config_id,
                fold=str(source["folds"]["early_stopping"]),
                started=iteration_started,
                status="failed",
            )
            raise
        reporter.iteration_finish(
            stage="fit",
            config=config.config_id,
            fold=str(source["folds"]["early_stopping"]),
            started=iteration_started,
            status="completed",
        )
        model.save(model_dir)
        checkpoint.complete(
            stage="fit",
            config="pointwise",
            fold="rolling",
            metadata={
                "model_sha256": sha256_file(model_dir / "model.cbm"),
                "tree_count": model.tree_count,
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
            },
        )
        del loader
        gc.collect()
    best = AtomicBestConfig(best_model_root, run_id=run_id, config_digest=digest)
    best_copy = best_model_root / "model"
    if not best_copy.exists():
        model.save(best_copy)
    restored_best = CatBoostPointwiseModel.from_artifact(best_copy)
    if sha256_file(best_copy / "model.cbm") != sha256_file(model_dir / "model.cbm"):
        raise ContractValidationError("best portable model differs from fitted model")
    eval_scores = restored_best.best_score.get("validation", {})
    eval_value = eval_scores.get(restored_best.config.eval_metric)
    if eval_value is None:
        eval_value = min(eval_scores.values()) if eval_scores else float("inf")
    best.update(
        score=(-float(eval_value),),
        payload={
            "model_artifact": best_copy.as_posix(),
            "model_sha256": sha256_file(best_copy / "model.cbm"),
            "best_iteration": restored_best.best_iteration,
            "eval_metric": restored_best.config.eval_metric,
            "eval_value": float(eval_value),
        },
    )
    return restored_best


def _full_rrf_model(
    winner: RRFEnsembleModel, materialized: CandidateUnionConfig
) -> RRFEnsembleModel:
    return RRFEnsembleModel(
        RRFConfig(
            config_id="task06_weights_on_full_materialized_union",
            candidate_config=materialized,
            weights=winner.config.weights,
            rrf_constant=winner.config.rrf_constant,
            final_k=winner.config.final_k,
        )
    )


def _inference_output_paths(root: Path, fold: str, part: Path) -> dict[str, Path]:
    base = root / "inference_parts" / fold
    return {
        name: base / name / part.name
        for name in ("catboost_full", "catboost_parity", "rrf_full", "rrf_parity")
    }


def _score_inference_part(
    *,
    part: Path,
    fold: str,
    model: CatBoostPointwiseModel,
    winner_rrf: RRFEnsembleModel,
    full_rrf: RRFEnsembleModel,
    materialized: CandidateUnionConfig,
    feature_columns: Sequence[str],
    batch_size: int,
    final_k: int,
) -> dict[str, pl.DataFrame]:
    frame = pl.read_parquet(part, columns=["user_id", "item_id", *feature_columns])
    loader = (
        CatBoostRankerDataLoader(
            feature_columns=feature_columns,
            seed=model.config.random_seed,
        )
        .load_predict_data(frame=frame)
        .prepare_predict_data()
    )
    cat_scores = model.predict(loader, batch_size=batch_size)
    cat_full = ranker_scores_to_candidates(cat_scores, k=final_k)
    rrf_projection = ["user_id", "item_id"]
    for spec in materialized.sources:
        columns = generator_columns(spec.source)
        rrf_projection.extend((columns["generated"], columns["rank"]))
    rrf_frame = frame.select(rrf_projection)
    parity_scores = winner_rrf.score(
        rrf_frame, materialized_config=materialized, validate_features=False
    )
    cat_parity = ranker_scores_to_candidates(
        cat_scores.join(
            parity_scores.select("user_id", "item_id"),
            on=("user_id", "item_id"),
            how="semi",
        ),
        k=final_k,
        source_name="catboost_pointwise_rrf_parity",
    )
    rrf_parity = winner_rrf.rank_candidates(
        rrf_frame, materialized_config=materialized, validate_features=False
    ).filter(pl.col("rank") <= final_k)
    rrf_full = full_rrf.rank_candidates(
        rrf_frame, materialized_config=materialized, validate_features=False
    ).filter(pl.col("rank") <= final_k)
    return {
        "catboost_full": cat_full,
        "catboost_parity": cat_parity,
        "rrf_full": rrf_full,
        "rrf_parity": rrf_parity,
    }


def _evaluate_fold(
    *,
    context: FoldContext,
    model: CatBoostPointwiseModel,
    winner_rrf: RRFEnsembleModel,
    full_rrf: RRFEnsembleModel,
    materialized: CandidateUnionConfig,
    feature_columns: Sequence[str],
    source: Mapping[str, Any],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    artifact_work: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    started_fold = time.perf_counter()
    batch_size = int(source["inference"]["batch_size"])
    final_k = int(source["inference"]["final_k"])
    outputs: dict[str, list[pl.DataFrame]] = {
        "catboost_full": [],
        "catboost_parity": [],
        "rrf_full": [],
        "rrf_parity": [],
    }
    stage_started = reporter.stage_start(
        stage=f"inference_{context.fold}", total=len(context.parts), unit="part"
    )
    for part in context.parts:
        paths = _inference_output_paths(checkpoint_root, context.fold, part)
        record = checkpoint.get(
            stage="inference", config="pointwise", fold=f"{context.fold}:{part.stem}"
        )
        operation_started = reporter.operation_start(
            stage=f"inference_{context.fold}",
            config="pointwise",
            fold=context.fold,
            operation=part.stem,
        )
        if record is not None:
            for name, path in paths.items():
                if not path.is_file() or sha256_file(path) != record.get(
                    "files", {}
                ).get(name):
                    raise ContractValidationError(
                        f"inference checkpoint differs: {path}"
                    )
            reporter.event(
                "operation_resume_skip",
                stage=f"inference_{context.fold}",
                config="pointwise",
                fold=context.fold,
                operation=part.stem,
            )
        else:
            if any(path.exists() for path in paths.values()):
                raise FileExistsError(
                    f"unregistered inference output exists: {part.stem}"
                )
            frames = _score_inference_part(
                part=part,
                fold=context.fold,
                model=model,
                winner_rrf=winner_rrf,
                full_rrf=full_rrf,
                materialized=materialized,
                feature_columns=feature_columns,
                batch_size=batch_size,
                final_k=final_k,
            )
            for name, frame in frames.items():
                _write_parquet_atomic(frame, paths[name])
            checkpoint.complete(
                stage="inference",
                config="pointwise",
                fold=f"{context.fold}:{part.stem}",
                metadata={
                    "files": {name: sha256_file(path) for name, path in paths.items()},
                    "rows": {name: frame.height for name, frame in frames.items()},
                },
            )
        for name, path in paths.items():
            outputs[name].append(pl.read_parquet(path))
        reporter.operation_finish(
            stage=f"inference_{context.fold}",
            config="pointwise",
            fold=context.fold,
            operation=part.stem,
            started=operation_started,
        )
        reporter.stage_advance()
        gc.collect()
    reporter.stage_finish(stage=f"inference_{context.fold}", started=stage_started)
    recommendations: dict[str, pl.DataFrame] = {}
    top_frames: dict[str, pl.DataFrame] = {}
    for name, parts in outputs.items():
        top = pl.concat(parts, rechunk=True).sort(("user_id", "source", "rank"))
        top_frames[name] = top
        recommendations[name] = candidates_to_recommendations(
            top, context.target_users, k=final_k
        )
    history = pl.scan_parquet(context.history_path)
    for name in ("catboost_full", "catboost_parity"):
        validate_recommendations_against_history(
            recommendations[name],
            target_users=context.target_users,
            history_daily=history,
            expected_k=final_k,
        )
    fold_output = artifact_work / "evaluation" / context.fold
    recommendation_hashes: dict[str, str] = {}
    for name, frame in recommendations.items():
        path = fold_output / f"recommendations_{name}.parquet"
        _write_parquet_atomic(frame, path)
        recommendation_hashes[name] = sha256_file(path)
    lazy_features = pl.scan_parquet([part.as_posix() for part in context.parts])
    full_candidate_metrics = evaluate_candidate_metrics_lazy(
        lazy_features.select("user_id", "item_id"),
        context.ground_truth,
        context.target_users,
    )
    parity_candidate_metrics = evaluate_candidate_metrics_lazy(
        lazy_features.filter(
            _eligible_expression(winner_rrf.config.candidate_config)
        ).select("user_id", "item_id"),
        context.ground_truth,
        context.target_users,
    )
    metrics_by_ranker: dict[str, Any] = {}
    for name, frame in recommendations.items():
        precision = evaluate_precision_at_20(
            frame, context.ground_truth, context.target_users
        )
        metrics_by_ranker[name] = {
            **precision,
            "final_hits": _final_hit_count(frame, context.ground_truth),
            "recommendations_sha256": recommendation_hashes[name],
        }
    metrics = {
        "fold": context.fold,
        "target_users": context.target_users.height,
        "target_labeled_users": context.ground_truth.get_column("user_id").n_unique(),
        "target_ground_truth_pairs": context.ground_truth.height,
        "candidate_sets": {
            "full_materialized_union": full_candidate_metrics,
            "task06_rrf_parity": parity_candidate_metrics,
        },
        "rankers": metrics_by_ranker,
        "runtime_seconds": time.perf_counter() - started_fold,
    }
    write_json_atomic(fold_output / "metrics.json", metrics)
    return metrics


def _gpu_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        return {
            "available": True,
            "name": values[0],
            "driver_version": values[1],
            "memory_total_mib": int(values[2]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"available": False, "error": str(error)}


def run_experiment(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    checkpoint_dir: str | Path,
    pool_cache_dir: str | Path,
    best_model_dir: str | Path,
    log_file: str | Path | None,
    smoke_part_limit: int | None,
    verify_input_checksums: bool,
    show_progress: bool,
    cleanup_recoverable_on_success: bool = False,
) -> dict[str, Any]:
    if not run_id:
        raise CatBoostExperimentError("run_id must be non-empty")
    if smoke_part_limit is not None:
        _positive_int(smoke_part_limit, name="smoke_part_limit")
    source = _load_source_config(config_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    dataset_root = Path(source["task07_artifact"])
    rrf_artifact = Path(source["task06_rrf_artifact"])
    folds = source["folds"]
    mode = "smoke" if smoke_part_limit is not None else "full"
    resolved_digest_input = {
        "source": source,
        "source_config_sha256": sha256_file(config_path),
        "run_id": run_id,
        "mode": mode,
        "smoke_part_limit": smoke_part_limit,
    }
    digest = config_sha256(resolved_digest_input)
    checkpoint_root = Path(checkpoint_dir)
    checkpoint = CheckpointStore(checkpoint_root, run_id=run_id, config_digest=digest)
    artifact_work = checkpoint_root / "artifact"
    artifact_work.mkdir(parents=True, exist_ok=True)
    reporter = EventProgressReporter(
        task_name="task08_catboost_pointwise",
        total_phases=6,
        log_file=log_file,
        show_progress=show_progress,
    )
    started = time.perf_counter()
    reporter.event(
        "run_start",
        stage="run",
        config="pointwise",
        fold="all",
        operation="run",
        run_id=run_id,
        mode=mode,
    )
    try:
        phase = reporter.phase_start("validate_inputs")
        contexts, feature_columns, feature_schema, root_manifest = _validate_task07(
            dataset_root,
            folds=[folds["train"], folds["early_stopping"], folds["canonical"]],
            part_limit=smoke_part_limit,
            verify_checksums=verify_input_checksums,
        )
        winner_rrf = RRFEnsembleModel.from_artifact(rrf_artifact / "model")
        materialized = _materialized_union_config(rrf_artifact)
        full_rrf = _full_rrf_model(winner_rrf, materialized)
        reporter.phase_finish(
            "validate_inputs",
            phase,
            feature_count=len(feature_columns),
            task07_schema_sha256=root_manifest["schema_sha256"],
        )

        phase = reporter.phase_start("prepare_quantized_pools")
        pool_digest = config_sha256(
            {
                "task07_schema_sha256": root_manifest["schema_sha256"],
                "task07_fold_manifests": {
                    role: root_manifest["folds"][folds[role]]["manifest_sha256"]
                    for role in ("train", "early_stopping")
                },
                "pool": source["pool"],
                "row_sampling": source["row_sampling"],
                "feature_columns": list(feature_columns),
                "smoke_part_limit": smoke_part_limit,
            }
        )
        pool_manifest = _prepare_pool_cache(
            pool_root=Path(pool_cache_dir),
            pool_digest=pool_digest,
            feature_columns=feature_columns,
            train_context=contexts[folds["train"]],
            eval_context=contexts[folds["early_stopping"]],
            source=source,
            checkpoint=checkpoint,
            checkpoint_root=checkpoint_root,
            reporter=reporter,
        )
        reporter.phase_finish(
            "prepare_quantized_pools",
            phase,
            train_rows=pool_manifest["train_rows"],
            eval_rows=pool_manifest["eval_rows"],
        )

        phase = reporter.phase_start("fit_pointwise_catboost")
        model = _fit_or_restore_model(
            source=source,
            feature_columns=feature_columns,
            pool_root=Path(pool_cache_dir),
            checkpoint=checkpoint,
            checkpoint_root=checkpoint_root,
            best_model_root=Path(best_model_dir),
            run_id=run_id,
            digest=digest,
            reporter=reporter,
        )
        reporter.phase_finish(
            "fit_pointwise_catboost",
            phase,
            tree_count=model.tree_count,
            best_iteration=model.best_iteration,
        )

        phase = reporter.phase_start("evaluate_early_stopping_fold")
        eval_metrics = _evaluate_fold(
            context=contexts[folds["early_stopping"]],
            model=model,
            winner_rrf=winner_rrf,
            full_rrf=full_rrf,
            materialized=materialized,
            feature_columns=feature_columns,
            source=source,
            checkpoint=checkpoint,
            checkpoint_root=checkpoint_root,
            artifact_work=artifact_work,
            reporter=reporter,
        )
        reporter.phase_finish(
            "evaluate_early_stopping_fold",
            phase,
            p20_all=eval_metrics["rankers"]["catboost_full"][
                "precision_at_20_all_targets"
            ],
        )

        phase = reporter.phase_start("evaluate_canonical_once")
        canonical_metrics = _evaluate_fold(
            context=contexts[folds["canonical"]],
            model=model,
            winner_rrf=winner_rrf,
            full_rrf=full_rrf,
            materialized=materialized,
            feature_columns=feature_columns,
            source=source,
            checkpoint=checkpoint,
            checkpoint_root=checkpoint_root,
            artifact_work=artifact_work,
            reporter=reporter,
        )
        reporter.phase_finish(
            "evaluate_canonical_once",
            phase,
            p20_all=canonical_metrics["rankers"]["catboost_full"][
                "precision_at_20_all_targets"
            ],
        )

        phase = reporter.phase_start("publish_artifact")
        final_model = artifact_work / "model"
        if not final_model.exists():
            shutil.copytree(checkpoint_root / "trained_model", final_model)
        restored = CatBoostPointwiseModel.from_artifact(final_model)
        if sha256_file(final_model / "model.cbm") != sha256_file(
            checkpoint_root / "trained_model" / "model.cbm"
        ):
            raise ContractValidationError("published model copy differs")
        importance_path = artifact_work / "feature_importance.parquet"
        if not importance_path.exists():
            _write_parquet_atomic(restored.get_feature_importance(), importance_path)
        write_json_atomic(artifact_work / "feature_schema.json", feature_schema)
        resolved = {
            "artifact_version": 1,
            "kind": "task08_catboost_pointwise",
            "run_id": run_id,
            "mode": mode,
            "source_config_path": Path(config_path).as_posix(),
            "source_config_sha256": sha256_file(config_path),
            "task07_artifact": dataset_root.as_posix(),
            "task07_schema_sha256": root_manifest["schema_sha256"],
            "task06_rrf_artifact": rrf_artifact.as_posix(),
            "folds": dict(folds),
            "row_sampling": dict(source["row_sampling"]),
            "pool_cache": Path(pool_cache_dir).as_posix(),
            "pool_config_sha256": pool_digest,
            "catboost": restored.config.to_dict(),
            "inference": dict(source["inference"]),
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "library_versions": {
                "catboost": catboost.__version__,
                "polars": pl.__version__,
                "python": platform.python_version(),
            },
            "hardware": {"gpu": _gpu_metadata()},
            "canonical_isolation": True,
            "canonical_evaluated_config_count": 1,
            "smoke_part_limit": smoke_part_limit,
            "cleanup_policy": {
                "recoverable_state_on_success": cleanup_recoverable_on_success,
                "retained": ["published_artifact", "quantized_pool_cache"],
                "removed": ["checkpoint", "best_model"],
            },
        }
        write_json_atomic(artifact_work / "config.json", resolved)
        primary = canonical_metrics["rankers"]["catboost_full"]
        candidate = canonical_metrics["candidate_sets"]["full_materialized_union"]
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "precision_at_20_all_targets": primary["precision_at_20_all_targets"],
            "precision_at_20_labeled_users": primary["precision_at_20_labeled_users"],
            **candidate,
            "final_hits": primary["final_hits"],
            "tree_count": restored.tree_count,
            "best_iteration": restored.best_iteration,
            "best_score": restored.best_score,
            "early_stopping": eval_metrics,
            "canonical": canonical_metrics,
            "comparisons": {
                "full_union_catboost_minus_rrf_p20_all": primary[
                    "precision_at_20_all_targets"
                ]
                - canonical_metrics["rankers"]["rrf_full"][
                    "precision_at_20_all_targets"
                ],
                "parity_catboost_minus_rrf_p20_all": canonical_metrics["rankers"][
                    "catboost_parity"
                ]["precision_at_20_all_targets"]
                - canonical_metrics["rankers"]["rrf_parity"][
                    "precision_at_20_all_targets"
                ],
            },
            "model_sha256": sha256_file(final_model / "model.cbm"),
            "feature_importance_sha256": sha256_file(importance_path),
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        write_json_atomic(artifact_work / "metrics.json", metrics)
        publish_directory_atomic(artifact_work, output)
        reporter.phase_finish(
            "publish_artifact",
            phase,
            artifact=output.as_posix(),
            p20_all=metrics["precision_at_20_all_targets"],
        )
        if cleanup_recoverable_on_success:
            cleanup_started = reporter.operation_start(
                stage="cleanup",
                config="pointwise",
                fold="all",
                operation="recoverable_state",
            )
            cleanup_summary = cleanup_recoverable_state(
                checkpoint_dir=checkpoint_root,
                best_model_dir=best_model_dir,
                output_dir=output,
                pool_cache_dir=pool_cache_dir,
                dataset_dir=dataset_root,
                rrf_artifact_dir=rrf_artifact,
                run_id=run_id,
            )
            reporter.operation_finish(
                stage="cleanup",
                config="pointwise",
                fold="all",
                operation="recoverable_state",
                started=cleanup_started,
                freed_bytes=cleanup_summary["freed_bytes"],
                retained_pool_cache=cleanup_summary["retained_pool_cache"],
            )
        reporter.event(
            "run_finish",
            stage="run",
            config="pointwise",
            fold="all",
            operation="run",
            status="completed",
            runtime_seconds=metrics["runtime_seconds"],
            current_metric=metrics["precision_at_20_all_targets"],
            best_metric=metrics["precision_at_20_all_targets"],
            best_config=restored.config.config_id,
        )
        return metrics
    except BaseException as error:
        reporter.event(
            "run_finish",
            stage="run",
            config="pointwise",
            fold="all",
            operation="run",
            status="failed",
            error=repr(error),
            runtime_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        reporter.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one binary-Logloss CatBoost ranker from immutable task-07 "
            "Parquet without invoking candidate models."
        )
    )
    parser.add_argument("--config", default="configs/task08_catboost_pointwise_v1.json")
    parser.add_argument(
        "--output-dir", default="artifacts/task08_catboost_pointwise_v1"
    )
    parser.add_argument("--run-id", default="task08_catboost_pointwise_v1")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/.task08_catboost_pointwise_v1.checkpoint"
    )
    parser.add_argument(
        "--pool-cache-dir", default="artifacts/task08_catboost_pools_v1"
    )
    parser.add_argument(
        "--best-model-dir", default="artifacts/.task08_catboost_pointwise_v1.best-model"
    )
    parser.add_argument("--log-file", default="logs/task08_catboost_pointwise_v1.log")
    parser.add_argument(
        "--smoke-part-limit",
        type=int,
        default=None,
        help="Use only the first N user shards per fold; never a full model run.",
    )
    parser.add_argument(
        "--skip-input-checksums",
        action="store_true",
        help="Skip expensive Parquet checksums; manifest/schema checks remain enabled.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--cleanup-recoverable-on-success",
        action="store_true",
        help=(
            "After atomic artifact publication, delete resume-only checkpoint "
            "and best-model directories while retaining quantized pools."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    metrics = run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
        pool_cache_dir=args.pool_cache_dir,
        best_model_dir=args.best_model_dir,
        log_file=args.log_file,
        smoke_part_limit=args.smoke_part_limit,
        verify_input_checksums=not args.skip_input_checksums,
        show_progress=not args.no_progress,
        cleanup_recoverable_on_success=args.cleanup_recoverable_on_success,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

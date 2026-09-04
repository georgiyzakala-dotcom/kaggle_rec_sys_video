#!/usr/bin/env python3
"""Run bounded walk-forward CatBoost pointwise selection for Task 09."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import shutil
import signal
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
from catboost_selection import (
    SELECTION_ARTIFACT_KIND,
    SELECTION_ARTIFACT_VERSION,
    CatBoostSelectionError,
    FoldPair,
    aggregate_fold_results,
    select_best_result,
    stage_challengers,
    validate_selection_config,
)
from ensemble import RRFEnsembleModel
from experiment_utils import (
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
    write_column_description,
)
from scripts.run_catboost_ranker import (
    FoldContext,
    _assemble_dsv,
    _CatBoostLogBridge,
    _copy_file_atomic,
    _feature_contract,
    _full_rrf_model,
    _gpu_metadata,
    _materialize_dsv_parts,
    _materialized_union_config,
    _validate_pool_cache,
    _validate_task07,
    _write_parquet_atomic,
)
from scripts.run_global_popularity import _final_hit_count, _peak_memory_mb
from validation import ContractValidationError, validate_recommendations_against_history


class IntentionalSelectionStop(RuntimeError):
    """Raised by smoke runs to exercise checkpoint/resume."""


def _raise_graceful_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}; resume state was preserved")


def _copy_directory_atomic(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _directory_size_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _model_directory(checkpoint_root: Path, config_id: str, pair_id: str) -> Path:
    return checkpoint_root / "trained_models" / config_id / pair_id


def _validate_feature_importance(path: Path, *, feature_columns: Sequence[str]) -> None:
    if not path.is_file():
        raise ContractValidationError(f"feature importance is missing: {path}")
    frame = pl.read_parquet(path)
    if (
        frame.columns != ["feature", "importance"]
        or frame.height != len(feature_columns)
        or set(frame.get_column("feature").to_list()) != set(feature_columns)
        or not frame.get_column("importance").is_finite().all()
    ):
        raise ContractValidationError(f"invalid feature importance: {path}")


def _discard_unregistered_file(
    path: Path,
    *,
    checkpoint_record: Mapping[str, Any] | None,
    reporter: EventProgressReporter,
    stage: str,
    config: str,
    fold: str,
    operation: str,
) -> None:
    if checkpoint_record is not None or not path.exists():
        return
    if not path.is_file():
        raise ContractValidationError(f"unregistered output is not a file: {path}")
    path.unlink()
    reporter.event(
        "unregistered_output_discarded",
        stage=stage,
        config=config,
        fold=fold,
        operation=operation,
        path=path.as_posix(),
    )


def _reconcile_dsv_outputs(
    *,
    role: str,
    context: FoldContext,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> None:
    for part in context.parts:
        fold_key = f"{context.fold}:{part.stem}"
        record = checkpoint.get(stage=f"dsv_{role}", config="pointwise", fold=fold_key)
        _discard_unregistered_file(
            checkpoint_root / "raw_parts" / role / f"{part.stem}.tsv",
            checkpoint_record=record,
            reporter=reporter,
            stage="pool_materialization",
            config=role,
            fold=context.fold,
            operation=part.stem,
        )
    assembled_record = checkpoint.get(
        stage="assemble_dsv", config="pointwise", fold=role
    )
    _discard_unregistered_file(
        checkpoint_root / "raw" / f"{role}.tsv",
        checkpoint_record=assembled_record,
        reporter=reporter,
        stage="pool_materialization",
        config=role,
        fold=role,
        operation="assemble",
    )


def _pool_source(
    source: Mapping[str, Any], *, train_probability: float
) -> dict[str, Any]:
    return {
        "row_sampling": {
            "train_negative_keep_probability": train_probability,
            "eval_negative_keep_probability": float(
                source["row_sampling"]["eval_negative_keep_probability"]
            ),
            "seed": int(source["seed"]),
            "weighting": "inverse_sampling_probability",
        },
        "pool": dict(source["pool"]),
        "seed": int(source["seed"]),
    }


def _pool_digest(
    *,
    pair: FoldPair,
    root_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    feature_columns: Sequence[str],
    part_limit: int | None,
    train_probability: float,
) -> str:
    pool_source = _pool_source(source, train_probability=train_probability)
    return config_sha256(
        {
            "task07_schema_sha256": root_manifest["schema_sha256"],
            "task07_fold_manifests": {
                "train": root_manifest["folds"][pair.train_fold]["manifest_sha256"],
                "early_stopping": root_manifest["folds"][pair.eval_fold][
                    "manifest_sha256"
                ],
            },
            "pool": pool_source["pool"],
            "row_sampling": pool_source["row_sampling"],
            "feature_columns": list(feature_columns),
            "smoke_part_limit": part_limit,
        }
    )


def _validate_task09_pair_pool(
    pool_root: Path,
    *,
    pool_digest: str,
    pair: FoldPair,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    manifest = read_json(pool_root / "pool_manifest.json")
    if (
        manifest.get("artifact_version") != 1
        or manifest.get("kind") != "task09_quantized_pair_pools"
        or manifest.get("pool_config_sha256") != pool_digest
        or manifest.get("train_fold") != pair.train_fold
        or manifest.get("eval_fold") != pair.eval_fold
        or manifest.get("feature_columns") != list(feature_columns)
    ):
        raise ContractValidationError(f"incompatible Task 09 pool: {pool_root}")
    for name, expected in manifest.get("files", {}).items():
        path = pool_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractValidationError(f"corrupt Task 09 pool file: {path}")
    for role in ("train", "eval"):
        pool = Pool(f"quantized://{(pool_root / f'{role}.quantized').resolve()}")
        if pool.num_row() != int(manifest[f"{role}_rows"]):
            raise ContractValidationError(f"{role} pool row count differs")
        if tuple(pool.get_feature_names()) != tuple(feature_columns):
            raise ContractValidationError(f"{role} pool feature order differs")
        del pool
    gc.collect()
    return manifest


def _prepare_pair_pool(
    *,
    pair: FoldPair,
    contexts: Mapping[str, FoldContext],
    root_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    feature_columns: Sequence[str],
    part_limit: int | None,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
    allow_task08_pool: bool,
) -> dict[str, Any]:
    pool_root = pair.pool_cache
    digest = _pool_digest(
        pair=pair,
        root_manifest=root_manifest,
        source=source,
        feature_columns=feature_columns,
        part_limit=part_limit,
        train_probability=1.0,
    )
    if pool_root.exists():
        manifest = read_json(pool_root / "pool_manifest.json")
        if manifest.get("kind") == "task08_quantized_pools" and allow_task08_pool:
            checked = _validate_pool_cache(
                pool_root,
                pool_digest=digest,
                feature_columns=feature_columns,
            )
            if (
                checked.get("train_fold") != pair.train_fold
                or checked.get("eval_fold") != pair.eval_fold
            ):
                raise ContractValidationError("Task 08 pool folds differ")
            return checked
        return _validate_task09_pair_pool(
            pool_root,
            pool_digest=digest,
            pair=pair,
            feature_columns=feature_columns,
        )

    pool_source = _pool_source(source, train_probability=1.0)
    stage_started = reporter.stage_start(
        stage=f"pool_{pair.pair_id}",
        total=len(contexts[pair.train_fold].parts)
        + len(contexts[pair.eval_fold].parts),
        unit="part",
    )
    train_role = f"{pair.pair_id}_train_base"
    eval_role = f"{pair.pair_id}_eval_base"
    _reconcile_dsv_outputs(
        role=train_role,
        context=contexts[pair.train_fold],
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    _reconcile_dsv_outputs(
        role=eval_role,
        context=contexts[pair.eval_fold],
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    train_parts, train_diagnostics = _materialize_dsv_parts(
        role=train_role,
        context=contexts[pair.train_fold],
        feature_columns=feature_columns,
        probability=1.0,
        seed=int(source["seed"]),
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    eval_parts, eval_diagnostics = _materialize_dsv_parts(
        role=eval_role,
        context=contexts[pair.eval_fold],
        feature_columns=feature_columns,
        probability=float(source["row_sampling"]["eval_negative_keep_probability"]),
        seed=int(source["seed"]),
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    reporter.stage_finish(
        stage=f"pool_{pair.pair_id}",
        started=stage_started,
        train_rows=train_diagnostics["rows"],
        eval_rows=eval_diagnostics["rows"],
    )
    train_dsv = _assemble_dsv(
        role=train_role,
        parts=train_parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    eval_dsv = _assemble_dsv(
        role=eval_role,
        parts=eval_parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    work = checkpoint_root / "pool_artifacts" / pair.pair_id
    work.mkdir(parents=True, exist_ok=True)
    cd_path = work / "columns.cd"
    write_column_description(cd_path, feature_columns=feature_columns)
    train_path = work / "train.quantized"
    eval_path = work / "eval.quantized"
    borders_path = work / "borders.tsv"
    pool_config = source["pool"]
    train_record = checkpoint.get(
        stage="quantize_base", config=pair.pair_id, fold=pair.train_fold
    )
    operation_started = reporter.operation_start(
        stage="quantization",
        config=pair.pair_id,
        fold=pair.train_fold,
        operation="train",
    )
    if train_record is None:
        for path, operation in (
            (train_path, "train_quantized"),
            (borders_path, "borders"),
        ):
            _discard_unregistered_file(
                path,
                checkpoint_record=train_record,
                reporter=reporter,
                stage="quantization",
                config=pair.pair_id,
                fold=pair.train_fold,
                operation=operation,
            )
        pool = quantize(
            data_path=train_dsv.as_posix(),
            column_description=cd_path.as_posix(),
            delimiter="\t",
            has_header=False,
            thread_count=int(pool_config["thread_count"]),
            border_count=int(pool_config["border_count"]),
            feature_border_type=str(pool_config["feature_border_type"]),
            task_type=str(pool_config["quantization_task_type"]),
            random_seed=int(source["seed"]),
        )
        pool.save(train_path.as_posix())
        pool.save_quantization_borders(borders_path.as_posix())
        checkpoint.complete(
            stage="quantize_base",
            config=pair.pair_id,
            fold=pair.train_fold,
            metadata={
                "sha256": sha256_file(train_path),
                "borders_sha256": sha256_file(borders_path),
                "rows": pool.num_row(),
            },
        )
        del pool
        gc.collect()
    else:
        if (
            not train_path.is_file()
            or sha256_file(train_path) != train_record.get("sha256")
            or sha256_file(borders_path) != train_record.get("borders_sha256")
        ):
            raise ContractValidationError("base training pool checkpoint differs")
    reporter.operation_finish(
        stage="quantization",
        config=pair.pair_id,
        fold=pair.train_fold,
        operation="train",
        started=operation_started,
    )
    eval_record = checkpoint.get(
        stage="quantize_base", config=pair.pair_id, fold=pair.eval_fold
    )
    operation_started = reporter.operation_start(
        stage="quantization",
        config=pair.pair_id,
        fold=pair.eval_fold,
        operation="eval",
    )
    if eval_record is None:
        _discard_unregistered_file(
            eval_path,
            checkpoint_record=eval_record,
            reporter=reporter,
            stage="quantization",
            config=pair.pair_id,
            fold=pair.eval_fold,
            operation="eval_quantized",
        )
        pool = quantize(
            data_path=eval_dsv.as_posix(),
            column_description=cd_path.as_posix(),
            delimiter="\t",
            has_header=False,
            thread_count=int(pool_config["thread_count"]),
            input_borders=borders_path.as_posix(),
            task_type=str(pool_config["quantization_task_type"]),
            random_seed=int(source["seed"]),
        )
        pool.save(eval_path.as_posix())
        checkpoint.complete(
            stage="quantize_base",
            config=pair.pair_id,
            fold=pair.eval_fold,
            metadata={"sha256": sha256_file(eval_path), "rows": pool.num_row()},
        )
        del pool
        gc.collect()
    else:
        if not eval_path.is_file() or sha256_file(eval_path) != eval_record.get(
            "sha256"
        ):
            raise ContractValidationError("base eval pool checkpoint differs")
    reporter.operation_finish(
        stage="quantization",
        config=pair.pair_id,
        fold=pair.eval_fold,
        operation="eval",
        started=operation_started,
    )
    manifest = {
        "artifact_version": 1,
        "kind": "task09_quantized_pair_pools",
        "pool_config_sha256": digest,
        "pair_id": pair.pair_id,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "train_fold": pair.train_fold,
        "eval_fold": pair.eval_fold,
        "train_rows": train_diagnostics["rows"],
        "eval_rows": eval_diagnostics["rows"],
        "train_diagnostics": train_diagnostics,
        "eval_diagnostics": eval_diagnostics,
        "row_sampling": pool_source["row_sampling"],
        "quantization": dict(pool_config),
        "files": {
            "train.quantized": sha256_file(train_path),
            "eval.quantized": sha256_file(eval_path),
            "borders.tsv": sha256_file(borders_path),
            "columns.cd": sha256_file(cd_path),
        },
    }
    write_json_atomic(work / "pool_manifest.json", manifest)
    publish_directory_atomic(work, pool_root)
    return _validate_task09_pair_pool(
        pool_root,
        pool_digest=digest,
        pair=pair,
        feature_columns=feature_columns,
    )


def _validate_negative_pool(
    root: Path,
    *,
    digest: str,
    pair: FoldPair,
    probability: float,
    feature_columns: Sequence[str],
    borders_sha256: str,
) -> dict[str, Any]:
    manifest = read_json(root / "pool_manifest.json")
    if (
        manifest.get("artifact_version") != 1
        or manifest.get("kind") != "task09_quantized_train_pool"
        or manifest.get("pool_config_sha256") != digest
        or manifest.get("pair_id") != pair.pair_id
        or float(manifest.get("train_negative_keep_probability", -1)) != probability
        or manifest.get("feature_columns") != list(feature_columns)
        or manifest.get("borders_sha256") != borders_sha256
    ):
        raise ContractValidationError(f"incompatible negative pool: {root}")
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractValidationError(f"corrupt negative pool file: {path}")
    path = root / "train.quantized"
    pool = Pool(f"quantized://{path.resolve()}")
    if pool.num_row() != int(manifest["train_rows"]):
        raise ContractValidationError("negative pool row count differs")
    if tuple(pool.get_feature_names()) != tuple(feature_columns):
        raise ContractValidationError("negative pool feature order differs")
    del pool
    gc.collect()
    return manifest


def _prepare_negative_pool(
    *,
    pair: FoldPair,
    context: FoldContext,
    probability: float,
    cache_root: Path,
    base_pool_root: Path,
    source: Mapping[str, Any],
    feature_columns: Sequence[str],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    borders = base_pool_root / "borders.tsv"
    borders_sha = sha256_file(borders)
    digest = config_sha256(
        {
            "pair_id": pair.pair_id,
            "train_fold": pair.train_fold,
            "train_manifest": context.manifest,
            "parts": [part.name for part in context.parts],
            "train_negative_keep_probability": probability,
            "feature_columns": list(feature_columns),
            "borders_sha256": borders_sha,
            "seed": source["seed"],
        }
    )
    if cache_root.exists():
        return _validate_negative_pool(
            cache_root,
            digest=digest,
            pair=pair,
            probability=probability,
            feature_columns=feature_columns,
            borders_sha256=borders_sha,
        )
    role = f"{pair.pair_id}_train_neg_{str(probability).replace('.', '')}"
    stage_started = reporter.stage_start(
        stage=f"negative_pool_{pair.pair_id}",
        total=len(context.parts),
        unit="part",
    )
    _reconcile_dsv_outputs(
        role=role,
        context=context,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    parts, diagnostics = _materialize_dsv_parts(
        role=role,
        context=context,
        feature_columns=feature_columns,
        probability=probability,
        seed=int(source["seed"]),
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    reporter.stage_finish(
        stage=f"negative_pool_{pair.pair_id}",
        started=stage_started,
        rows=diagnostics["rows"],
    )
    dsv = _assemble_dsv(
        role=role,
        parts=parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    work = checkpoint_root / "negative_pool_artifacts" / pair.pair_id
    work.mkdir(parents=True, exist_ok=True)
    cd_path = work / "columns.cd"
    write_column_description(cd_path, feature_columns=feature_columns)
    train_path = work / "train.quantized"
    record = checkpoint.get(
        stage="quantize_negative", config=pair.pair_id, fold=pair.train_fold
    )
    operation_started = reporter.operation_start(
        stage="quantization_negative",
        config=pair.pair_id,
        fold=pair.train_fold,
        operation="train",
    )
    if record is None:
        _discard_unregistered_file(
            train_path,
            checkpoint_record=record,
            reporter=reporter,
            stage="quantization_negative",
            config=pair.pair_id,
            fold=pair.train_fold,
            operation="train_quantized",
        )
        pool = quantize(
            data_path=dsv.as_posix(),
            column_description=cd_path.as_posix(),
            delimiter="\t",
            has_header=False,
            thread_count=int(source["pool"]["thread_count"]),
            input_borders=borders.as_posix(),
            task_type=str(source["pool"]["quantization_task_type"]),
            random_seed=int(source["seed"]),
        )
        pool.save(train_path.as_posix())
        checkpoint.complete(
            stage="quantize_negative",
            config=pair.pair_id,
            fold=pair.train_fold,
            metadata={"sha256": sha256_file(train_path), "rows": pool.num_row()},
        )
        del pool
        gc.collect()
    else:
        if not train_path.is_file() or sha256_file(train_path) != record.get("sha256"):
            raise ContractValidationError("negative pool checkpoint differs")
    reporter.operation_finish(
        stage="quantization_negative",
        config=pair.pair_id,
        fold=pair.train_fold,
        operation="train",
        started=operation_started,
    )
    manifest = {
        "artifact_version": 1,
        "kind": "task09_quantized_train_pool",
        "pool_config_sha256": digest,
        "pair_id": pair.pair_id,
        "train_fold": pair.train_fold,
        "train_rows": diagnostics["rows"],
        "train_diagnostics": diagnostics,
        "train_negative_keep_probability": probability,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "borders_sha256": borders_sha,
        "files": {
            "train.quantized": sha256_file(train_path),
            "columns.cd": sha256_file(cd_path),
        },
    }
    write_json_atomic(work / "pool_manifest.json", manifest)
    publish_directory_atomic(work, cache_root)
    return _validate_negative_pool(
        cache_root,
        digest=digest,
        pair=pair,
        probability=probability,
        feature_columns=feature_columns,
        borders_sha256=borders_sha,
    )


def _model_training_signature(config: CatBoostPointwiseConfig) -> dict[str, Any]:
    value = config.to_dict()
    value.pop("config_id")
    value.pop("snapshot_interval_seconds")
    return value


def _bind_model_artifact_config(
    model_dir: Path,
    *,
    config: CatBoostPointwiseConfig,
    feature_columns: Sequence[str],
) -> CatBoostPointwiseModel:
    model = CatBoostPointwiseModel.from_artifact(model_dir)
    if _model_training_signature(model.config) != _model_training_signature(
        config
    ) or model.feature_columns != tuple(feature_columns):
        raise ContractValidationError("portable model training signature differs")
    if model.config.to_dict() != config.to_dict():
        metadata_path = model_dir / "model_config.json"
        metadata = read_json(metadata_path)
        metadata["catboost"] = config.to_dict()
        write_json_atomic(metadata_path, metadata)
        model = CatBoostPointwiseModel.from_artifact(model_dir)
    return model


def _can_reuse_task08(
    *,
    profile: Mapping[str, Any],
    pair: FoldPair,
    task08_root: Path,
    feature_columns: Sequence[str],
    part_limit: int | None,
) -> bool:
    if (
        part_limit is not None
        or pair.train_fold != "rolling_2"
        or pair.eval_fold != "rolling_3"
    ):
        return False
    if float(profile["train_negative_keep_probability"]) != 1.0:
        return False
    if profile["ignored_features"]:
        return False
    task08_config = read_json(task08_root / "config.json")
    if (
        task08_config.get("mode") != "full"
        or task08_config.get("folds", {}).get("train") != pair.train_fold
        or task08_config.get("folds", {}).get("early_stopping") != pair.eval_fold
        or task08_config.get("feature_columns") != list(feature_columns)
    ):
        return False
    prior = CatBoostPointwiseModel.from_artifact(task08_root / "model")
    current = CatBoostPointwiseConfig.from_mapping(profile["catboost"])
    return _model_training_signature(prior.config) == _model_training_signature(current)


def _fit_model(
    *,
    profile: Mapping[str, Any],
    pair: FoldPair,
    train_pool: Path,
    eval_pool: Path,
    feature_columns: Sequence[str],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
    task08_root: Path,
    part_limit: int | None,
) -> tuple[CatBoostPointwiseModel, dict[str, Any], Path]:
    config_id = str(profile["config_id"])
    model_dir = _model_directory(checkpoint_root, config_id, pair.pair_id)
    importance_path = (
        checkpoint_root / "feature_importance" / config_id / f"{pair.pair_id}.parquet"
    )
    config = CatBoostPointwiseConfig.from_mapping(profile["catboost"])
    record = checkpoint.get(stage="fit", config=config_id, fold=pair.pair_id)
    if record is not None:
        model = _bind_model_artifact_config(
            model_dir, config=config, feature_columns=feature_columns
        )
        if sha256_file(model_dir / "model.cbm") != record.get("model_sha256"):
            raise ContractValidationError("completed selection model differs")
        _validate_feature_importance(importance_path, feature_columns=feature_columns)
        reporter.event(
            "operation_resume_skip",
            stage="fit",
            config=config_id,
            fold=pair.pair_id,
            operation="catboost_fit",
        )
        return model, dict(record), model_dir
    if model_dir.exists():
        model = _bind_model_artifact_config(
            model_dir, config=config, feature_columns=feature_columns
        )
        if importance_path.exists():
            _validate_feature_importance(
                importance_path, feature_columns=feature_columns
            )
        else:
            _write_parquet_atomic(model.get_feature_importance(), importance_path)
        recovered_from = (
            task08_root.as_posix()
            if _can_reuse_task08(
                profile=profile,
                pair=pair,
                task08_root=task08_root,
                feature_columns=feature_columns,
                part_limit=part_limit,
            )
            else None
        )
        metadata = {
            "model_sha256": sha256_file(model_dir / "model.cbm"),
            "tree_count": model.tree_count,
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "fit_runtime_seconds": 0.0,
            "physical_runtime_seconds": 0.0,
            "reused_from": recovered_from,
            "recovered_unregistered_model": True,
        }
        checkpoint.complete(
            stage="fit", config=config_id, fold=pair.pair_id, metadata=metadata
        )
        reporter.event(
            "unregistered_output_recovered",
            stage="fit",
            config=config_id,
            fold=pair.pair_id,
            operation="catboost_model",
            model_sha256=metadata["model_sha256"],
        )
        return model, metadata, model_dir
    _discard_unregistered_file(
        importance_path,
        checkpoint_record=record,
        reporter=reporter,
        stage="fit",
        config=config_id,
        fold=pair.pair_id,
        operation="feature_importance",
    )
    if _can_reuse_task08(
        profile=profile,
        pair=pair,
        task08_root=task08_root,
        feature_columns=feature_columns,
        part_limit=part_limit,
    ):
        _copy_directory_atomic(task08_root / "model", model_dir)
        _copy_file_atomic(task08_root / "feature_importance.parquet", importance_path)
        model = _bind_model_artifact_config(
            model_dir, config=config, feature_columns=feature_columns
        )
        _validate_feature_importance(importance_path, feature_columns=feature_columns)
        metadata = {
            "model_sha256": sha256_file(model_dir / "model.cbm"),
            "tree_count": model.tree_count,
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "fit_runtime_seconds": 0.0,
            "physical_runtime_seconds": 0.0,
            "reused_from": task08_root.as_posix(),
        }
        checkpoint.complete(
            stage="fit", config=config_id, fold=pair.pair_id, metadata=metadata
        )
        reporter.event(
            "operation_reuse",
            stage="fit",
            config=config_id,
            fold=pair.pair_id,
            operation="task08_model",
            model_sha256=metadata["model_sha256"],
        )
        return model, metadata, model_dir
    loader = (
        CatBoostRankerDataLoader(
            feature_columns=feature_columns,
            seed=int(profile["catboost"]["random_seed"]),
        )
        .load_fit_data(train_pool=train_pool, eval_pool=eval_pool)
        .prepare_fit_data()
    )
    model = CatBoostPointwiseModel(config, feature_columns=feature_columns)
    started, callback = reporter.iteration_start(
        stage="fit",
        config=config_id,
        fold=pair.pair_id,
        total=config.iterations,
        unit="tree",
    )
    bridge = _CatBoostLogBridge(callback)
    snapshot = (
        checkpoint_root
        / "catboost_training"
        / config_id
        / pair.pair_id
        / "snapshot.cbsnapshot"
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.fit(
            loader,
            train_dir=snapshot.parent / "train_dir",
            snapshot_file=snapshot,
            log_cout=bridge,
            log_cerr=bridge,
        )
    except BaseException:
        reporter.iteration_finish(
            stage="fit",
            config=config_id,
            fold=pair.pair_id,
            started=started,
            status="failed",
        )
        raise
    fit_runtime = time.perf_counter() - started
    reporter.iteration_finish(
        stage="fit",
        config=config_id,
        fold=pair.pair_id,
        started=started,
        status="completed",
    )
    model.save(model_dir)
    _write_parquet_atomic(model.get_feature_importance(), importance_path)
    _validate_feature_importance(importance_path, feature_columns=feature_columns)
    metadata = {
        "model_sha256": sha256_file(model_dir / "model.cbm"),
        "tree_count": model.tree_count,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "fit_runtime_seconds": fit_runtime,
        "physical_runtime_seconds": fit_runtime,
        "reused_from": None,
    }
    checkpoint.complete(
        stage="fit", config=config_id, fold=pair.pair_id, metadata=metadata
    )
    del loader
    gc.collect()
    return model, metadata, model_dir


def _top_candidates_from_parts(
    parts: Sequence[pl.DataFrame], *, final_k: int
) -> pl.DataFrame:
    if not parts:
        raise ContractValidationError("inference produced no candidate parts")
    return (
        pl.concat(parts, rechunk=True)
        .sort(("user_id", "score", "item_id"), descending=(False, True, False))
        .with_columns(
            pl.col("item_id").cum_count().over("user_id").cast(pl.UInt32).alias("rank")
        )
        .filter(pl.col("rank") <= final_k)
        .sort(("user_id", "rank"))
    )


def _evaluate_model(
    *,
    profile: Mapping[str, Any],
    pair_id: str,
    train_fold: str,
    context: FoldContext,
    model: CatBoostPointwiseModel,
    fit_metadata: Mapping[str, Any],
    feature_columns: Sequence[str],
    source: Mapping[str, Any],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
    stage_name: str = "selection_inference",
) -> dict[str, Any]:
    config_id = str(profile["config_id"])
    result_root = checkpoint_root / "results" / config_id / pair_id
    metrics_path = result_root / "metrics.json"
    recommendations_path = result_root / "recommendations.parquet"
    record = checkpoint.get(stage="evaluation", config=config_id, fold=pair_id)
    if record is not None:
        if (
            not metrics_path.is_file()
            or sha256_file(metrics_path) != record.get("metrics_sha256")
            or not recommendations_path.is_file()
            or sha256_file(recommendations_path) != record.get("recommendations_sha256")
        ):
            raise ContractValidationError("completed fold evaluation differs")
        reporter.event(
            "operation_resume_skip",
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation="fold_evaluation",
        )
        return read_json(metrics_path)
    for path, operation in (
        (metrics_path, "fold_metrics"),
        (recommendations_path, "fold_recommendations"),
    ):
        _discard_unregistered_file(
            path,
            checkpoint_record=record,
            reporter=reporter,
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation=operation,
        )
    started = time.perf_counter()
    output_parts: list[pl.DataFrame] = []
    stage_started = reporter.stage_start(
        stage=stage_name, total=len(context.parts), unit="part"
    )
    for part in context.parts:
        output = checkpoint_root / "inference_parts" / config_id / pair_id / part.name
        part_record = checkpoint.get(
            stage="inference", config=config_id, fold=f"{pair_id}:{part.stem}"
        )
        operation_started = reporter.operation_start(
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation=part.stem,
        )
        if part_record is None:
            _discard_unregistered_file(
                output,
                checkpoint_record=part_record,
                reporter=reporter,
                stage=stage_name,
                config=config_id,
                fold=pair_id,
                operation=part.stem,
            )
            frame = pl.read_parquet(
                part, columns=["user_id", "item_id", *feature_columns]
            )
            loader = (
                CatBoostRankerDataLoader(
                    feature_columns=feature_columns,
                    seed=int(source["seed"]),
                )
                .load_predict_data(frame=frame)
                .prepare_predict_data()
            )
            scores = model.predict(
                loader, batch_size=int(source["inference"]["batch_size"])
            )
            top = ranker_scores_to_candidates(
                scores,
                k=int(source["inference"]["final_k"]),
                source_name="catboost_pointwise",
            )
            _write_parquet_atomic(top, output)
            checkpoint.complete(
                stage="inference",
                config=config_id,
                fold=f"{pair_id}:{part.stem}",
                metadata={"sha256": sha256_file(output), "rows": top.height},
            )
            del frame, loader, scores, top
            gc.collect()
        elif not output.is_file() or sha256_file(output) != part_record.get("sha256"):
            raise ContractValidationError(f"inference checkpoint differs: {output}")
        output_parts.append(pl.read_parquet(output))
        reporter.operation_finish(
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation=part.stem,
            started=operation_started,
        )
        reporter.stage_advance()
    reporter.stage_finish(stage=stage_name, started=stage_started)
    top = _top_candidates_from_parts(
        output_parts, final_k=int(source["inference"]["final_k"])
    )
    recommendations = candidates_to_recommendations(
        top,
        context.target_users,
        k=int(source["inference"]["final_k"]),
    )
    validate_recommendations_against_history(
        recommendations,
        target_users=context.target_users,
        history_daily=pl.scan_parquet(context.history_path),
        expected_k=int(source["inference"]["final_k"]),
    )
    precision = evaluate_precision_at_20(
        recommendations, context.ground_truth, context.target_users
    )
    result_root.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(recommendations, recommendations_path)
    inference_runtime = time.perf_counter() - started
    metrics = {
        "config_id": config_id,
        "profile_sha256": profile["profile_sha256"],
        "pair_id": pair_id,
        "train_fold": train_fold,
        "eval_fold": context.fold,
        **precision,
        "final_hits": _final_hit_count(recommendations, context.ground_truth),
        "tree_count": int(fit_metadata["tree_count"]),
        "best_iteration": int(fit_metadata["best_iteration"]),
        "best_score": fit_metadata["best_score"],
        "fit_runtime_seconds": float(fit_metadata["fit_runtime_seconds"]),
        "inference_runtime_seconds": inference_runtime,
        "runtime_seconds": float(fit_metadata["fit_runtime_seconds"])
        + inference_runtime,
        "physical_runtime_seconds": float(fit_metadata["physical_runtime_seconds"])
        + inference_runtime,
        "recommendations_sha256": sha256_file(recommendations_path),
        "model_sha256": fit_metadata["model_sha256"],
        "target_users": context.target_users.height,
        "target_labeled_users": context.ground_truth.get_column("user_id").n_unique(),
        "target_ground_truth_pairs": context.ground_truth.height,
        "reused_from": fit_metadata.get("reused_from"),
    }
    write_json_atomic(metrics_path, metrics)
    checkpoint.complete(
        stage="evaluation",
        config=config_id,
        fold=pair_id,
        metadata={
            "metrics_sha256": sha256_file(metrics_path),
            "recommendations_sha256": sha256_file(recommendations_path),
        },
    )
    return metrics


def _reuse_task08_evaluation(
    *,
    profile: Mapping[str, Any],
    pair: FoldPair,
    task08_root: Path,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    fit_metadata: Mapping[str, Any],
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    config_id = str(profile["config_id"])
    result_root = checkpoint_root / "results" / config_id / pair.pair_id
    metrics_path = result_root / "metrics.json"
    recommendations_path = result_root / "recommendations.parquet"
    record = checkpoint.get(stage="evaluation", config=config_id, fold=pair.pair_id)
    if record is not None:
        if (
            not metrics_path.is_file()
            or sha256_file(metrics_path) != record.get("metrics_sha256")
            or not recommendations_path.is_file()
            or sha256_file(recommendations_path) != record.get("recommendations_sha256")
        ):
            raise ContractValidationError("reused Task 08 evaluation differs")
        return read_json(metrics_path)
    for path, operation in (
        (metrics_path, "fold_metrics"),
        (recommendations_path, "fold_recommendations"),
    ):
        _discard_unregistered_file(
            path,
            checkpoint_record=record,
            reporter=reporter,
            stage="selection_inference",
            config=config_id,
            fold=pair.pair_id,
            operation=operation,
        )
    task08_metrics = read_json(
        task08_root / "evaluation" / pair.eval_fold / "metrics.json"
    )
    ranker = task08_metrics["rankers"]["catboost_full"]
    source_recommendations = (
        task08_root
        / "evaluation"
        / pair.eval_fold
        / "recommendations_catboost_full.parquet"
    )
    if sha256_file(source_recommendations) != ranker["recommendations_sha256"]:
        raise ContractValidationError("Task 08 rolling recommendations differ")
    _copy_file_atomic(source_recommendations, recommendations_path)
    metrics = {
        "config_id": config_id,
        "profile_sha256": profile["profile_sha256"],
        "pair_id": pair.pair_id,
        "train_fold": pair.train_fold,
        "eval_fold": pair.eval_fold,
        "precision_at_20_all_targets": ranker["precision_at_20_all_targets"],
        "precision_at_20_labeled_users": ranker["precision_at_20_labeled_users"],
        "final_hits": ranker["final_hits"],
        "tree_count": int(fit_metadata["tree_count"]),
        "best_iteration": int(fit_metadata["best_iteration"]),
        "best_score": fit_metadata["best_score"],
        "fit_runtime_seconds": 0.0,
        "inference_runtime_seconds": 0.0,
        "runtime_seconds": 0.0,
        "physical_runtime_seconds": 0.0,
        "source_evaluation_runtime_seconds": task08_metrics["runtime_seconds"],
        "recommendations_sha256": sha256_file(recommendations_path),
        "model_sha256": fit_metadata["model_sha256"],
        "target_users": task08_metrics["target_users"],
        "target_labeled_users": task08_metrics["target_labeled_users"],
        "target_ground_truth_pairs": task08_metrics["target_ground_truth_pairs"],
        "reused_from": task08_root.as_posix(),
    }
    write_json_atomic(metrics_path, metrics)
    checkpoint.complete(
        stage="evaluation",
        config=config_id,
        fold=pair.pair_id,
        metadata={
            "metrics_sha256": sha256_file(metrics_path),
            "recommendations_sha256": sha256_file(recommendations_path),
        },
    )
    return metrics


def _candidate_metrics(context: FoldContext) -> dict[str, Any]:
    return evaluate_candidate_metrics_lazy(
        pl.scan_parquet([part.as_posix() for part in context.parts]).select(
            "user_id", "item_id"
        ),
        context.ground_truth,
        context.target_users,
    )


def _evaluate_rrf(
    *,
    context: FoldContext,
    pair_id: str,
    full_rrf: RRFEnsembleModel,
    materialized: CandidateUnionConfig,
    final_k: int,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    root = checkpoint_root / "static_baselines" / "rrf" / pair_id
    metrics_path = root / "metrics.json"
    recommendations_path = root / "recommendations.parquet"
    record = checkpoint.get(stage="static_rrf", config="rrf_full", fold=pair_id)
    if record is not None:
        if (
            not metrics_path.is_file()
            or sha256_file(metrics_path) != record.get("metrics_sha256")
            or not recommendations_path.is_file()
            or sha256_file(recommendations_path) != record.get("recommendations_sha256")
        ):
            raise ContractValidationError("static RRF checkpoint differs")
        return read_json(metrics_path)
    for path, operation in (
        (metrics_path, "fold_metrics"),
        (recommendations_path, "fold_recommendations"),
    ):
        _discard_unregistered_file(
            path,
            checkpoint_record=record,
            reporter=reporter,
            stage="rrf_baseline",
            config="rrf_full",
            fold=pair_id,
            operation=operation,
        )
    started = time.perf_counter()
    parts: list[pl.DataFrame] = []
    projection = ["user_id", "item_id"]
    for spec in materialized.sources:
        columns = generator_columns(spec.source)
        projection.extend((columns["generated"], columns["rank"]))
    stage_started = reporter.stage_start(
        stage="rrf_baseline", total=len(context.parts), unit="part"
    )
    for part in context.parts:
        frame = pl.read_parquet(part, columns=projection)
        parts.append(
            full_rrf.rank_candidates(
                frame, materialized_config=materialized, validate_features=False
            ).filter(pl.col("rank") <= final_k)
        )
        reporter.stage_status(
            stage="rrf_baseline",
            config="rrf_full",
            fold=context.fold,
            operation=part.stem,
        )
        reporter.stage_advance()
    reporter.stage_finish(stage="rrf_baseline", started=stage_started)
    top = pl.concat(parts, rechunk=True).sort(("user_id", "source", "rank"))
    recommendations = candidates_to_recommendations(
        top, context.target_users, k=final_k
    )
    validate_recommendations_against_history(
        recommendations,
        target_users=context.target_users,
        history_daily=pl.scan_parquet(context.history_path),
        expected_k=final_k,
    )
    precision = evaluate_precision_at_20(
        recommendations, context.ground_truth, context.target_users
    )
    root.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(recommendations, recommendations_path)
    metrics = {
        "pair_id": pair_id,
        "fold": context.fold,
        **precision,
        "final_hits": _final_hit_count(recommendations, context.ground_truth),
        "runtime_seconds": time.perf_counter() - started,
        "recommendations_sha256": sha256_file(recommendations_path),
        "candidate_metrics": _candidate_metrics(context),
    }
    write_json_atomic(metrics_path, metrics)
    checkpoint.complete(
        stage="static_rrf",
        config="rrf_full",
        fold=pair_id,
        metadata={
            "metrics_sha256": sha256_file(metrics_path),
            "recommendations_sha256": sha256_file(recommendations_path),
        },
    )
    return metrics


def _update_best_model(
    *,
    best_root: Path,
    run_id: str,
    digest: str,
    profile: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    model_dir: Path,
) -> None:
    pointer = best_root / "best_model.json"
    if pointer.exists():
        current = read_json(pointer)
        if (
            current.get("artifact_version") != 1
            or current.get("run_id") != run_id
            or current.get("config_sha256") != digest
        ):
            raise CatBoostSelectionError(f"incompatible best-model state: {pointer}")
        if current.get("profile_sha256") == profile["profile_sha256"]:
            return
    version = best_root / "versions" / str(profile["profile_sha256"])
    if not version.exists():
        _copy_directory_atomic(model_dir, version)
    value = {
        "artifact_version": 1,
        "run_id": run_id,
        "config_sha256": digest,
        "config_id": profile["config_id"],
        "profile_sha256": profile["profile_sha256"],
        "model_artifact": version.as_posix(),
        "model_sha256": sha256_file(version / "model.cbm"),
        "selection_metrics": dict(aggregate),
    }
    write_json_atomic(pointer, value)


def _validate_cleanup_target(
    path: Path,
    *,
    marker: str,
    run_id: str,
    protected: Sequence[Path],
) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir() or len(resolved.parts) < 4:
        raise CatBoostSelectionError(f"unsafe cleanup target: {path}")
    for value in protected:
        checked = value.resolve()
        if (
            resolved == checked
            or resolved in checked.parents
            or checked in resolved.parents
        ):
            raise CatBoostSelectionError(
                f"cleanup target overlaps protected path: {path}"
            )
    metadata = read_json(resolved / marker)
    if metadata.get("artifact_version") != 1 or metadata.get("run_id") != run_id:
        raise CatBoostSelectionError(f"cleanup marker belongs to another run: {path}")
    return resolved


def cleanup_selection_state(
    *,
    checkpoint_dir: str | Path,
    best_model_dir: str | Path,
    output_dir: str | Path,
    pool_dirs: Sequence[str | Path],
    input_dirs: Sequence[str | Path],
    run_id: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    config = read_json(output / "config.json")
    metrics = read_json(output / "metrics.json")
    model = output / "model" / "model.cbm"
    if (
        config.get("kind") != SELECTION_ARTIFACT_KIND
        or config.get("run_id") != run_id
        or metrics.get("run_id") != run_id
        or not model.is_file()
        or metrics.get("model_sha256") != sha256_file(model)
    ):
        raise CatBoostSelectionError("published Task 09 artifact is incomplete")
    protected = [
        output,
        *(Path(path) for path in pool_dirs),
        *(Path(path) for path in input_dirs),
    ]
    targets = (
        (Path(checkpoint_dir), "checkpoint.json", "checkpoint"),
        (Path(best_model_dir), "best_model.json", "best_model"),
    )
    validated: list[tuple[Path, str, int]] = []
    removed: dict[str, Any] = {}
    for path, marker, name in targets:
        if not path.exists():
            removed[name] = {"removed": False, "bytes": 0}
            continue
        target = _validate_cleanup_target(
            path, marker=marker, run_id=run_id, protected=protected
        )
        validated.append((target, name, _directory_size_bytes(target)))
    for target, name, size in validated:
        shutil.rmtree(target)
        removed[name] = {"removed": True, "bytes": size, "path": target.as_posix()}
    return {
        "removed": removed,
        "freed_bytes": sum(value["bytes"] for value in removed.values()),
        "retained_pools": [Path(path).as_posix() for path in pool_dirs],
    }


def _load_and_validate_source(
    config_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    raw = read_json(config_path)
    dataset = raw.get("task07_artifact")
    if not isinstance(dataset, str) or not dataset:
        raise CatBoostSelectionError("task07_artifact must be configured")
    feature_columns, feature_schema, schema_sha = _feature_contract(Path(dataset))
    source = validate_selection_config(raw, feature_columns=feature_columns)
    if schema_sha != config_sha256(feature_schema):
        raise ContractValidationError("Task 07 feature schema digest is unstable")
    return source, feature_columns, feature_schema, raw


def _profile_base(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_id": profile["config_id"],
        "catboost": copy.deepcopy(profile["catboost"]),
        "train_negative_keep_probability": profile["train_negative_keep_probability"],
        "feature_set": profile["feature_set"],
    }


def run_selection(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    checkpoint_dir: str | Path,
    best_model_dir: str | Path,
    log_file: str | Path | None,
    smoke_part_limit: int | None,
    verify_input_checksums: bool,
    show_progress: bool,
    cleanup_recoverable_on_success: bool,
    stop_after_completed_configs: int | None = None,
) -> dict[str, Any]:
    if not run_id:
        raise CatBoostSelectionError("run_id must be non-empty")
    if smoke_part_limit is not None and smoke_part_limit <= 0:
        raise CatBoostSelectionError("smoke_part_limit must be positive")
    if stop_after_completed_configs is not None and stop_after_completed_configs <= 0:
        raise CatBoostSelectionError("stop_after_completed_configs must be positive")
    config_source = Path(config_path)
    source, feature_columns, feature_schema, raw_source = _load_and_validate_source(
        config_source
    )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    mode = "smoke" if smoke_part_limit is not None else "full"
    if mode == "smoke":
        smoke_paths = [
            output,
            *(pair.pool_cache for pair in source["fold_pairs"]),
            *(
                Path(source["negative_pool_caches"][pair.pair_id])
                for pair in source["fold_pairs"]
            ),
        ]
        if "smoke" not in run_id.lower() or any(
            "smoke" not in path.as_posix().lower() for path in smoke_paths
        ):
            raise CatBoostSelectionError(
                "limited smoke requires a smoke run_id, output, and pool paths"
            )
    digest_input = {
        "source": raw_source,
        "source_config_sha256": sha256_file(config_source),
        "run_id": run_id,
        "mode": mode,
        "smoke_part_limit": smoke_part_limit,
    }
    digest = config_sha256(digest_input)
    checkpoint_root = Path(checkpoint_dir)
    checkpoint = CheckpointStore(checkpoint_root, run_id=run_id, config_digest=digest)
    best_root = Path(best_model_dir)
    reporter = EventProgressReporter(
        task_name="task09_catboost_selection",
        total_phases=7,
        log_file=log_file,
        show_progress=show_progress,
        log_max_bytes=5 * 1024 * 1024,
        log_backup_count=4,
    )
    started = time.perf_counter()
    reporter.event(
        "run_start",
        stage="run",
        config="selection",
        fold="all",
        operation="run",
        run_id=run_id,
        mode=mode,
    )
    try:
        phase = reporter.phase_start("validate_selection_inputs")
        dataset_root = Path(source["task07_artifact"])
        pair_folds = []
        for pair in source["fold_pairs"]:
            pair_folds.extend((pair.train_fold, pair.eval_fold))
        ordered_folds = list(dict.fromkeys(pair_folds))
        contexts, checked_features, _, root_manifest = _validate_task07(
            dataset_root,
            folds=ordered_folds,
            part_limit=smoke_part_limit,
            verify_checksums=verify_input_checksums,
        )
        if checked_features != feature_columns:
            raise ContractValidationError("Task 07 feature order changed")
        task08_root = Path(source["task08_artifact"])
        task08_config = read_json(task08_root / "config.json")
        task08_model_config = read_json(task08_root / "model" / "model_config.json")
        if (
            task08_config.get("kind") != "task08_catboost_pointwise"
            or task08_config.get("feature_columns") != list(feature_columns)
            or task08_model_config.get("model_sha256")
            != sha256_file(task08_root / "model" / "model.cbm")
        ):
            raise ContractValidationError("Task 08 comparison artifact is incompatible")
        rrf_artifact = Path(source["task06_rrf_artifact"])
        winner_rrf = RRFEnsembleModel.from_artifact(rrf_artifact / "model")
        materialized = _materialized_union_config(rrf_artifact)
        full_rrf = _full_rrf_model(winner_rrf, materialized)
        reporter.phase_finish(
            "validate_selection_inputs",
            phase,
            feature_count=len(feature_columns),
            folds=ordered_folds,
            canonical_opened=False,
        )

        phase = reporter.phase_start("prepare_reusable_pools")
        pair_manifests: dict[str, dict[str, Any]] = {}
        for pair in source["fold_pairs"]:
            pair_manifests[pair.pair_id] = _prepare_pair_pool(
                pair=pair,
                contexts=contexts,
                root_manifest=root_manifest,
                source=source,
                feature_columns=feature_columns,
                part_limit=smoke_part_limit,
                checkpoint=checkpoint,
                checkpoint_root=checkpoint_root,
                reporter=reporter,
                allow_task08_pool=smoke_part_limit is None,
            )
        negative_probabilities = {
            float(
                challenger.get("overrides", {}).get("train_negative_keep_probability")
            )
            for stage in source["search"]["stages"]
            for challenger in stage["challengers"]
            if challenger.get("overrides", {}).get("train_negative_keep_probability")
            is not None
            and float(challenger["overrides"]["train_negative_keep_probability"]) < 1.0
        }
        baseline_negative_probability = float(
            source["baseline_resolved"]["train_negative_keep_probability"]
        )
        if baseline_negative_probability < 1.0:
            negative_probabilities.add(baseline_negative_probability)
        if len(negative_probabilities) > 1:
            raise CatBoostSelectionError(
                "one shared negative-pool probability is supported per run"
            )
        negative_probability = next(iter(negative_probabilities), None)
        negative_manifests: dict[str, dict[str, Any]] = {}
        if negative_probability is not None:
            for pair in source["fold_pairs"]:
                negative_manifests[pair.pair_id] = _prepare_negative_pool(
                    pair=pair,
                    context=contexts[pair.train_fold],
                    probability=negative_probability,
                    cache_root=Path(source["negative_pool_caches"][pair.pair_id]),
                    base_pool_root=pair.pool_cache,
                    source=source,
                    feature_columns=feature_columns,
                    checkpoint=checkpoint,
                    checkpoint_root=checkpoint_root,
                    reporter=reporter,
                )
        reporter.phase_finish(
            "prepare_reusable_pools",
            phase,
            base_pool_count=len(pair_manifests),
            negative_pool_count=len(negative_manifests),
        )

        phase = reporter.phase_start("compute_static_rolling_baselines")
        rrf_metrics: dict[str, dict[str, Any]] = {}
        for pair in source["fold_pairs"]:
            rrf_metrics[pair.pair_id] = _evaluate_rrf(
                context=contexts[pair.eval_fold],
                pair_id=pair.pair_id,
                full_rrf=full_rrf,
                materialized=materialized,
                final_k=int(source["inference"]["final_k"]),
                checkpoint=checkpoint,
                checkpoint_root=checkpoint_root,
                reporter=reporter,
            )
        reporter.phase_finish(
            "compute_static_rolling_baselines", phase, baseline="rrf_full"
        )

        phase = reporter.phase_start("walk_forward_selection")
        results_by_id: dict[str, dict[str, Any]] = {}
        profiles_by_id: dict[str, dict[str, Any]] = {}
        selection_order: list[str] = []
        completed_this_process = 0
        total_profiles = 1 + sum(
            len(stage["challengers"]) for stage in source["search"]["stages"]
        )
        selection_started = time.perf_counter()

        def evaluate_profile(profile: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed_this_process
            config_id = str(profile["config_id"])
            if config_id not in selection_order:
                selection_order.append(config_id)
            profiles_by_id[config_id] = profile
            aggregate_path = checkpoint_root / "aggregates" / f"{config_id}.json"
            aggregate_record = checkpoint.get(
                stage="aggregate", config=config_id, fold="rolling"
            )
            if aggregate_record is not None:
                if not aggregate_path.is_file() or sha256_file(
                    aggregate_path
                ) != aggregate_record.get("sha256"):
                    raise ContractValidationError("aggregate checkpoint differs")
                aggregate = read_json(aggregate_path)
                results_by_id[config_id] = aggregate
                return aggregate
            fold_results: list[dict[str, Any]] = []
            for pair in source["fold_pairs"]:
                base_pool = pair.pool_cache
                probability = float(profile["train_negative_keep_probability"])
                if probability < 1.0:
                    if (
                        negative_probability is None
                        or probability != negative_probability
                    ):
                        raise CatBoostSelectionError(
                            "profile lacks compatible negative pool"
                        )
                    train_pool = (
                        Path(source["negative_pool_caches"][pair.pair_id])
                        / "train.quantized"
                    )
                else:
                    train_pool = base_pool / "train.quantized"
                eval_pool = base_pool / "eval.quantized"
                model, fit_metadata, _model_dir = _fit_model(
                    profile=profile,
                    pair=pair,
                    train_pool=train_pool,
                    eval_pool=eval_pool,
                    feature_columns=feature_columns,
                    checkpoint=checkpoint,
                    checkpoint_root=checkpoint_root,
                    reporter=reporter,
                    task08_root=task08_root,
                    part_limit=smoke_part_limit,
                )
                if _can_reuse_task08(
                    profile=profile,
                    pair=pair,
                    task08_root=task08_root,
                    feature_columns=feature_columns,
                    part_limit=smoke_part_limit,
                ):
                    fold_result = _reuse_task08_evaluation(
                        profile=profile,
                        pair=pair,
                        task08_root=task08_root,
                        checkpoint=checkpoint,
                        checkpoint_root=checkpoint_root,
                        fit_metadata=fit_metadata,
                        reporter=reporter,
                    )
                else:
                    fold_result = _evaluate_model(
                        profile=profile,
                        pair_id=pair.pair_id,
                        train_fold=pair.train_fold,
                        context=contexts[pair.eval_fold],
                        model=model,
                        fit_metadata=fit_metadata,
                        feature_columns=feature_columns,
                        source=source,
                        checkpoint=checkpoint,
                        checkpoint_root=checkpoint_root,
                        reporter=reporter,
                    )
                fold_results.append(fold_result)
                del model
                gc.collect()
            aggregate = aggregate_fold_results(config_id, fold_results)
            aggregate.update(
                profile=profile,
                profile_sha256=profile["profile_sha256"],
                deltas_vs_rrf={
                    pair.pair_id: {
                        "precision_at_20_labeled_users": fold_result[
                            "precision_at_20_labeled_users"
                        ]
                        - rrf_metrics[pair.pair_id]["precision_at_20_labeled_users"],
                        "precision_at_20_all_targets": fold_result[
                            "precision_at_20_all_targets"
                        ]
                        - rrf_metrics[pair.pair_id]["precision_at_20_all_targets"],
                    }
                    for pair, fold_result in zip(
                        source["fold_pairs"], fold_results, strict=True
                    )
                },
            )
            write_json_atomic(aggregate_path, aggregate)
            checkpoint.complete(
                stage="aggregate",
                config=config_id,
                fold="rolling",
                metadata={"sha256": sha256_file(aggregate_path)},
            )
            results_by_id[config_id] = aggregate
            completed_this_process += 1
            elapsed = time.perf_counter() - selection_started
            completed_total = len(results_by_id)
            eta = (
                elapsed / completed_this_process * (total_profiles - completed_total)
                if completed_this_process and total_profiles > completed_total
                else 0.0
            )
            reporter.event(
                "selection_progress",
                stage="walk_forward_selection",
                config=config_id,
                fold="rolling",
                operation="aggregate",
                completed=completed_total,
                total=total_profiles,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                current_metric=aggregate["mean_precision_at_20_labeled_users"],
            )
            if (
                stop_after_completed_configs is not None
                and completed_this_process >= stop_after_completed_configs
            ):
                raise IntentionalSelectionStop(
                    f"stopped after {completed_this_process} completed configs"
                )
            return aggregate

        incumbent = dict(source["baseline_resolved"])
        incumbent_result = evaluate_profile(incumbent)
        latest_pair_id = str(source["canonical"]["latest_pair_id"])
        _update_best_model(
            best_root=best_root,
            run_id=run_id,
            digest=digest,
            profile=incumbent,
            aggregate=incumbent_result,
            model_dir=_model_directory(
                checkpoint_root, incumbent["config_id"], latest_pair_id
            ),
        )
        stage_results: list[dict[str, Any]] = []
        for stage in source["search"]["stages"]:
            challengers = stage_challengers(
                incumbent,
                stage,
                feature_sets=source["feature_sets"],
            )
            candidate_results = [incumbent_result]
            for challenger in challengers:
                candidate_results.append(evaluate_profile(challenger))
            winner = select_best_result(
                candidate_results,
                tie_epsilon=float(source["search"]["tie_epsilon"]),
                order=[result["config_id"] for result in candidate_results],
            )
            incumbent = profiles_by_id[winner["config_id"]]
            incumbent_result = results_by_id[winner["config_id"]]
            stage_record = {
                "stage_id": stage["stage_id"],
                "incumbent_config_id": incumbent["config_id"],
                "evaluated_config_ids": [
                    result["config_id"] for result in candidate_results
                ],
                "winner": winner,
            }
            stage_path = checkpoint_root / "stage_winners" / f"{stage['stage_id']}.json"
            write_json_atomic(stage_path, stage_record)
            checkpoint.complete(
                stage="stage_winner",
                config=str(stage["stage_id"]),
                fold="rolling",
                metadata={"sha256": sha256_file(stage_path)},
            )
            stage_results.append(stage_record)
            _update_best_model(
                best_root=best_root,
                run_id=run_id,
                digest=digest,
                profile=incumbent,
                aggregate=incumbent_result,
                model_dir=_model_directory(
                    checkpoint_root, incumbent["config_id"], latest_pair_id
                ),
            )
            reporter.event(
                "stage_winner",
                stage=str(stage["stage_id"]),
                config=incumbent["config_id"],
                fold="rolling",
                operation="select",
                current_metric=incumbent_result["mean_precision_at_20_labeled_users"],
                best_metric=incumbent_result["mean_precision_at_20_labeled_users"],
            )
        winner_model_dir = _model_directory(
            checkpoint_root, incumbent["config_id"], latest_pair_id
        )
        winner = {
            "artifact_version": 1,
            "run_id": run_id,
            "config_sha256": digest,
            "canonical_evaluated": False,
            "profile": incumbent,
            "rolling_metrics": incumbent_result,
            "latest_pair_id": latest_pair_id,
            "model_artifact": winner_model_dir.as_posix(),
            "model_sha256": sha256_file(winner_model_dir / "model.cbm"),
        }
        winner_path = checkpoint_root / "winner.json"
        write_json_atomic(winner_path, winner)
        checkpoint.complete(
            stage="freeze_winner",
            config=incumbent["config_id"],
            fold="rolling",
            metadata={
                "winner_sha256": sha256_file(winner_path),
                "model_sha256": winner["model_sha256"],
            },
        )
        reporter.phase_finish(
            "walk_forward_selection",
            phase,
            best_config=incumbent["config_id"],
            best_metric=incumbent_result["mean_precision_at_20_labeled_users"],
            canonical_opened=False,
        )

        phase = reporter.phase_start("evaluate_frozen_winner_canonical_once")
        canonical_fold = str(source["canonical"]["fold"])
        canonical_contexts, canonical_features, _, canonical_root_manifest = (
            _validate_task07(
                dataset_root,
                folds=[canonical_fold],
                part_limit=smoke_part_limit,
                verify_checksums=verify_input_checksums,
            )
        )
        if canonical_features != feature_columns:
            raise ContractValidationError("canonical feature order differs")
        canonical_context = canonical_contexts[canonical_fold]
        winner_model = CatBoostPointwiseModel.from_artifact(winner_model_dir)
        latest_pair = next(
            pair for pair in source["fold_pairs"] if pair.pair_id == latest_pair_id
        )
        latest_fit = checkpoint.get(
            stage="fit", config=incumbent["config_id"], fold=latest_pair_id
        )
        assert latest_fit is not None
        canonical_pair_id = "canonical_once"
        task08_canonical = read_json(
            task08_root / "evaluation" / canonical_fold / "metrics.json"
        )
        if _can_reuse_task08(
            profile=incumbent,
            pair=latest_pair,
            task08_root=task08_root,
            feature_columns=feature_columns,
            part_limit=smoke_part_limit,
        ):
            source_recommendations = (
                task08_root
                / "evaluation"
                / "canonical"
                / "recommendations_catboost_full.parquet"
            )
            target_root = (
                checkpoint_root / "results" / incumbent["config_id"] / canonical_pair_id
            )
            recommendations_path = target_root / "recommendations.parquet"
            primary = task08_canonical["rankers"]["catboost_full"]
            if sha256_file(source_recommendations) != primary["recommendations_sha256"]:
                raise ContractValidationError(
                    "Task 08 canonical recommendations differ"
                )
            if not recommendations_path.exists():
                _copy_file_atomic(source_recommendations, recommendations_path)
            elif sha256_file(recommendations_path) != primary["recommendations_sha256"]:
                raise ContractValidationError(
                    "reused canonical recommendations checkpoint differs"
                )
            canonical_metrics = {
                "config_id": incumbent["config_id"],
                "profile_sha256": incumbent["profile_sha256"],
                "pair_id": canonical_pair_id,
                "train_fold": latest_pair.train_fold,
                "eval_fold": canonical_fold,
                "precision_at_20_all_targets": primary["precision_at_20_all_targets"],
                "precision_at_20_labeled_users": primary[
                    "precision_at_20_labeled_users"
                ],
                "final_hits": primary["final_hits"],
                "tree_count": int(latest_fit["tree_count"]),
                "best_iteration": int(latest_fit["best_iteration"]),
                "best_score": latest_fit["best_score"],
                "fit_runtime_seconds": 0.0,
                "inference_runtime_seconds": 0.0,
                "runtime_seconds": 0.0,
                "physical_runtime_seconds": 0.0,
                "recommendations_sha256": sha256_file(recommendations_path),
                "model_sha256": latest_fit["model_sha256"],
                "target_users": task08_canonical["target_users"],
                "target_labeled_users": task08_canonical["target_labeled_users"],
                "target_ground_truth_pairs": task08_canonical[
                    "target_ground_truth_pairs"
                ],
                "reused_from": task08_root.as_posix(),
            }
            write_json_atomic(target_root / "metrics.json", canonical_metrics)
        else:
            canonical_metrics = _evaluate_model(
                profile=incumbent,
                pair_id=canonical_pair_id,
                train_fold=latest_pair.train_fold,
                context=canonical_context,
                model=winner_model,
                fit_metadata={
                    **latest_fit,
                    "fit_runtime_seconds": 0.0,
                    "physical_runtime_seconds": 0.0,
                },
                feature_columns=feature_columns,
                source=source,
                checkpoint=checkpoint,
                checkpoint_root=checkpoint_root,
                reporter=reporter,
                stage_name="canonical_inference",
            )
        canonical_candidate_metrics = _candidate_metrics(canonical_context)
        published_winner = {
            **winner,
            "canonical_evaluated": True,
            "canonical_metrics_sha256": config_sha256(canonical_metrics),
            "model_artifact": (output / "model").as_posix(),
        }
        reporter.phase_finish(
            "evaluate_frozen_winner_canonical_once",
            phase,
            config=incumbent["config_id"],
            p20_labeled=canonical_metrics["precision_at_20_labeled_users"],
            canonical_evaluated_config_count=1,
        )

        phase = reporter.phase_start("publish_selection_artifact")
        artifact_work = checkpoint_root / "artifact"
        if artifact_work.exists():
            shutil.rmtree(artifact_work)
        artifact_work.mkdir(parents=True, exist_ok=True)
        _copy_directory_atomic(
            checkpoint_root / "results", artifact_work / "evaluations"
        )
        _copy_directory_atomic(
            checkpoint_root / "trained_models", artifact_work / "selection_models"
        )
        _copy_directory_atomic(
            checkpoint_root / "feature_importance",
            artifact_work / "feature_importance_by_fold",
        )
        _copy_directory_atomic(
            checkpoint_root / "static_baselines", artifact_work / "static_baselines"
        )
        final_model = artifact_work / "model"
        _copy_directory_atomic(winner_model_dir, final_model)
        winner_importance = (
            checkpoint_root
            / "feature_importance"
            / incumbent["config_id"]
            / f"{latest_pair_id}.parquet"
        )
        _copy_file_atomic(
            winner_importance, artifact_work / "feature_importance.parquet"
        )
        selection_root = artifact_work / "selection"
        selection_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(selection_root / "winner.json", published_winner)
        write_json_atomic(
            selection_root / "stage_results.json", {"stages": stage_results}
        )
        leaderboard_rows = []
        for config_id in selection_order:
            result = results_by_id[config_id]
            leaderboard_rows.append(
                {
                    "config_id": config_id,
                    "mean_precision_at_20_labeled_users": result[
                        "mean_precision_at_20_labeled_users"
                    ],
                    "min_precision_at_20_labeled_users": result[
                        "min_precision_at_20_labeled_users"
                    ],
                    "fold_spread_precision_at_20_labeled_users": result[
                        "fold_spread_precision_at_20_labeled_users"
                    ],
                    "mean_precision_at_20_all_targets": result[
                        "mean_precision_at_20_all_targets"
                    ],
                    "mean_tree_count": result["mean_tree_count"],
                    "total_hits": result["total_hits"],
                    "runtime_seconds": result["runtime_seconds"],
                    "physical_runtime_seconds": result["physical_runtime_seconds"],
                    "profile_sha256": result["profile_sha256"],
                    "feature_set": result["profile"]["feature_set"],
                    "train_negative_keep_probability": result["profile"][
                        "train_negative_keep_probability"
                    ],
                }
            )
        _write_parquet_atomic(
            pl.DataFrame(leaderboard_rows), selection_root / "leaderboard.parquet"
        )
        write_json_atomic(artifact_work / "feature_schema.json", feature_schema)
        canonical_rrf = task08_canonical["rankers"]["rrf_full"]
        canonical_task08 = task08_canonical["rankers"]["catboost_full"]
        if mode == "full":
            comparisons: dict[str, Any] = {
                "comparable": True,
                "task08": canonical_task08,
                "rrf_full": canonical_rrf,
                "winner_minus_task08_p20_labeled": canonical_metrics[
                    "precision_at_20_labeled_users"
                ]
                - canonical_task08["precision_at_20_labeled_users"],
                "winner_minus_task08_p20_all": canonical_metrics[
                    "precision_at_20_all_targets"
                ]
                - canonical_task08["precision_at_20_all_targets"],
                "winner_minus_rrf_p20_labeled": canonical_metrics[
                    "precision_at_20_labeled_users"
                ]
                - canonical_rrf["precision_at_20_labeled_users"],
                "winner_minus_rrf_p20_all": canonical_metrics[
                    "precision_at_20_all_targets"
                ]
                - canonical_rrf["precision_at_20_all_targets"],
            }
        else:
            comparisons = {
                "comparable": False,
                "reason": (
                    "smoke metrics use a target-user shard and cannot be compared "
                    "with full Task 08 or RRF canonical metrics"
                ),
            }
        resolved_config = {
            "artifact_version": SELECTION_ARTIFACT_VERSION,
            "kind": SELECTION_ARTIFACT_KIND,
            "run_id": run_id,
            "mode": mode,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": sha256_file(config_source),
            "config_sha256": digest,
            "task07_artifact": dataset_root.as_posix(),
            "task07_schema_sha256": root_manifest["schema_sha256"],
            "task06_rrf_artifact": rrf_artifact.as_posix(),
            "task08_artifact": task08_root.as_posix(),
            "fold_pairs": [
                {
                    "pair_id": pair.pair_id,
                    "train_fold": pair.train_fold,
                    "eval_fold": pair.eval_fold,
                    "pool_cache": pair.pool_cache.as_posix(),
                }
                for pair in source["fold_pairs"]
            ],
            "pool_manifests": pair_manifests,
            "negative_pool_manifests": negative_manifests,
            "search": raw_source["search"],
            "feature_sets": raw_source["feature_sets"],
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "winner_profile": incumbent,
            "canonical_isolation": {
                "selection_folds": ordered_folds,
                "canonical_loaded_after_winner_freeze": True,
                "canonical_evaluated_config_count": 1,
                "canonical_schema_sha256": canonical_root_manifest["schema_sha256"],
            },
            "library_versions": {
                "catboost": catboost.__version__,
                "polars": pl.__version__,
                "python": platform.python_version(),
            },
            "hardware": {"gpu": _gpu_metadata()},
            "smoke_part_limit": smoke_part_limit,
            "cleanup_policy": {
                "recoverable_state_on_success": cleanup_recoverable_on_success,
                "retained": ["published_artifact", "reusable_pool_caches"],
                "removed_when_enabled": ["checkpoint", "best_model"],
            },
        }
        write_json_atomic(artifact_work / "config.json", resolved_config)
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "winner_config_id": incumbent["config_id"],
            "winner_profile_sha256": incumbent["profile_sha256"],
            "selection_primary_metric": "mean_precision_at_20_labeled_users",
            "selection": incumbent_result,
            "selection_stages": stage_results,
            "precision_at_20_all_targets": canonical_metrics[
                "precision_at_20_all_targets"
            ],
            "precision_at_20_labeled_users": canonical_metrics[
                "precision_at_20_labeled_users"
            ],
            "final_hits": canonical_metrics["final_hits"],
            **canonical_candidate_metrics,
            "canonical": canonical_metrics,
            "canonical_evaluated_config_count": 1,
            "comparisons": comparisons,
            "tree_count": winner_model.tree_count,
            "best_iteration": winner_model.best_iteration,
            "model_sha256": sha256_file(final_model / "model.cbm"),
            "feature_importance_sha256": sha256_file(
                artifact_work / "feature_importance.parquet"
            ),
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        write_json_atomic(artifact_work / "metrics.json", metrics)
        publish_directory_atomic(artifact_work, output)
        reporter.phase_finish(
            "publish_selection_artifact",
            phase,
            artifact=output.as_posix(),
            p20_labeled=metrics["precision_at_20_labeled_users"],
        )

        phase = reporter.phase_start("cleanup_recoverable_state")
        cleanup_summary = None
        if cleanup_recoverable_on_success:
            pool_dirs = [pair.pool_cache for pair in source["fold_pairs"]]
            pool_dirs.extend(
                Path(source["negative_pool_caches"][pair.pair_id])
                for pair in source["fold_pairs"]
            )
            cleanup_summary = cleanup_selection_state(
                checkpoint_dir=checkpoint_root,
                best_model_dir=best_root,
                output_dir=output,
                pool_dirs=pool_dirs,
                input_dirs=[dataset_root, rrf_artifact, task08_root],
                run_id=run_id,
            )
        reporter.phase_finish(
            "cleanup_recoverable_state",
            phase,
            freed_bytes=(cleanup_summary or {}).get("freed_bytes", 0),
        )
        reporter.event(
            "run_finish",
            stage="run",
            config=incumbent["config_id"],
            fold="all",
            operation="run",
            status="completed",
            runtime_seconds=metrics["runtime_seconds"],
            current_metric=metrics["precision_at_20_labeled_users"],
            best_metric=incumbent_result["mean_precision_at_20_labeled_users"],
            best_config=incumbent["config_id"],
        )
        return metrics
    except BaseException as error:
        reporter.event(
            "run_finish",
            stage="run",
            config="selection",
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
            "Run bounded pointwise CatBoost hyperparameter selection on two "
            "rolling Task 07 fold pairs, then evaluate one frozen winner on canonical."
        )
    )
    parser.add_argument("--config", default="configs/task09_catboost_selection_v1.json")
    parser.add_argument(
        "--output-dir", default="artifacts/task09_catboost_selection_v1"
    )
    parser.add_argument("--run-id", default="task09_catboost_selection_v1")
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/.task09_catboost_selection_v1.checkpoint",
    )
    parser.add_argument(
        "--best-model-dir",
        default="artifacts/.task09_catboost_selection_v1.best-model",
    )
    parser.add_argument("--log-file", default="logs/task09_catboost_selection_v1.log")
    parser.add_argument(
        "--smoke-part-limit",
        type=int,
        default=None,
        help="Use the first N user shards per fold; never a full experiment.",
    )
    parser.add_argument(
        "--stop-after-completed-configs",
        type=int,
        default=None,
        help="Testing hook: stop after N newly completed profiles and preserve state.",
    )
    parser.add_argument("--skip-input-checksums", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--cleanup-recoverable-on-success", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_graceful_interrupt)
    try:
        metrics = run_selection(
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            checkpoint_dir=args.checkpoint_dir,
            best_model_dir=args.best_model_dir,
            log_file=args.log_file,
            smoke_part_limit=args.smoke_part_limit,
            verify_input_checksums=not args.skip_input_checksums,
            show_progress=not args.no_progress,
            cleanup_recoverable_on_success=args.cleanup_recoverable_on_success,
            stop_after_completed_configs=args.stop_after_completed_configs,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

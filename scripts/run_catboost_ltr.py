#!/usr/bin/env python3
"""Run bounded walk-forward CatBoost learning-to-rank selection for Task 10."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import polars as pl
from catboost import CatBoostError, Pool
from catboost.utils import get_gpu_device_count, quantize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from catboost_ltr import (
    LTR_POOL_ARTIFACT_VERSION,
    LTR_POOL_KIND,
    LTR_SELECTION_ARTIFACT_KIND,
    LTR_SELECTION_ARTIFACT_VERSION,
    CatBoostLTRError,
    LTRFoldPair,
    build_dense_group_mapping,
    inherited_stage_challengers,
    prepare_grouped_rows,
    select_complete_eval_users,
    validate_ltr_selection_config,
    write_grouped_dsv_part,
    write_ltr_column_description,
)
from catboost_selection import aggregate_fold_results, select_best_result
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
    CatBoostLTRConfig,
    CatBoostLTRDataLoader,
    CatBoostRankerDataLoader,
    CatBoostRankerModel,
    ranker_scores_to_candidates,
    validate_pool_groups,
)
from scripts.run_catboost_ranker import (
    FoldContext,
    _CatBoostLogBridge,
    _copy_file_atomic,
    _feature_contract,
    _gpu_metadata,
    _validate_task07,
    _write_parquet_atomic,
)
from scripts.run_catboost_selection import (
    _copy_directory_atomic,
    _directory_size_bytes,
    _top_candidates_from_parts,
    _validate_feature_importance,
)
from scripts.run_global_popularity import _final_hit_count, _peak_memory_mb
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_recommendations_against_history,
)


class IntentionalLTRStop(RuntimeError):
    """Raised by a smoke run after a safe checkpoint to exercise resume."""


class CatBoostFitWorkerError(RuntimeError):
    """Raised when an isolated native CatBoost fit process fails."""

    def __init__(self, message: str, *, returncode: int, output_tail: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.output_tail = output_tail


class ResourceInfeasibleError(CatBoostFitWorkerError):
    """Raised for a predeclared GPU OOM that may exclude one objective profile."""


def _raise_graceful_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}; resume state was preserved")


def _existing_ancestor(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(path)
        current = current.parent
    return current


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "MemAvailable:":
                return int(fields[1]) * 1024
        raise KeyError("MemAvailable")
    except (OSError, KeyError, ValueError) as error:
        raise CatBoostLTRError("cannot determine available RAM") from error


def _windows_host_free_bytes(drive: str) -> int:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        fallback = Path("/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe")
        if fallback.is_file():
            powershell = fallback.as_posix()
    if powershell is None:
        raise CatBoostLTRError(
            "powershell.exe is required for Windows host disk checks"
        )
    command = f"[Console]::Write((Get-PSDrive -Name '{drive}').Free)"
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        value = int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise CatBoostLTRError(
            f"cannot determine free space on Windows host drive {drive}: {error}"
        ) from error
    if value <= 0:
        raise CatBoostLTRError(f"Windows host drive {drive} reported no free space")
    return value


def _check_windows_host_disk(
    source: Mapping[str, Any], *, launch: bool
) -> dict[str, Any] | None:
    limits = source["resources"]
    drive = limits.get("windows_host_drive")
    if drive is None:
        return None
    threshold_name = (
        "min_windows_host_free_gib" if launch else "stop_windows_host_free_gib"
    )
    threshold_gib = int(limits[threshold_name])
    free_bytes = _windows_host_free_bytes(str(drive))
    if free_bytes < threshold_gib * 1024**3:
        boundary = "launch" if launch else "continuation"
        raise CatBoostLTRError(
            f"Windows host drive {drive}: free space is below the {boundary} "
            f"threshold of {threshold_gib} GiB"
        )
    return {
        "drive": str(drive),
        "free_bytes": free_bytes,
        "threshold_gib": threshold_gib,
        "check": "launch" if launch else "continuation",
    }


def _resource_preflight(source: Mapping[str, Any], *, output: Path) -> dict[str, Any]:
    limits = source["resources"]
    if catboost.__version__ != limits["catboost_version"]:
        raise CatBoostLTRError(
            f"catboost=={limits['catboost_version']} is required, got "
            f"{catboost.__version__}"
        )
    available_ram = _available_memory_bytes()
    required_ram = int(limits["min_available_ram_gib"]) * 1024**3
    if available_ram < required_ram:
        raise CatBoostLTRError(
            f"available RAM is below {limits['min_available_ram_gib']} GiB"
        )
    disk_root = _existing_ancestor(output.parent)
    free_disk = shutil.disk_usage(disk_root).free
    required_disk = int(limits["min_free_disk_gib"]) * 1024**3
    if free_disk < required_disk:
        raise CatBoostLTRError(f"free disk is below {limits['min_free_disk_gib']} GiB")
    gpu_count = int(get_gpu_device_count())
    if gpu_count < int(limits["min_gpu_count"]):
        raise CatBoostLTRError(
            f"CatBoost sees {gpu_count} GPU(s), expected at least "
            f"{limits['min_gpu_count']}"
        )
    gpu = _gpu_metadata()
    if not gpu.get("available"):
        raise CatBoostLTRError(f"NVIDIA GPU metadata is unavailable: {gpu}")
    if int(gpu["memory_total_mib"]) < int(limits["min_gpu_memory_mib"]):
        raise CatBoostLTRError(
            f"GPU memory is below {limits['min_gpu_memory_mib']} MiB"
        )
    windows_host = _check_windows_host_disk(source, launch=True)
    return {
        "limits": dict(limits),
        "observed": {
            "available_ram_bytes": available_ram,
            "free_disk_bytes": free_disk,
            "disk_root": disk_root.as_posix(),
            "catboost_gpu_count": gpu_count,
            "gpu": gpu,
            "windows_host_storage": windows_host,
        },
    }


def _load_source(
    path: Path,
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    raw = read_json(path)
    dataset = raw.get("task07_artifact")
    if not isinstance(dataset, str) or not dataset:
        raise CatBoostLTRError("task07_artifact must be configured")
    features, feature_schema, schema_sha = _feature_contract(Path(dataset))
    if schema_sha != config_sha256(feature_schema):
        raise ContractValidationError("Task 07 feature schema digest is unstable")
    source = validate_ltr_selection_config(raw, feature_columns=features)
    return source, features, feature_schema, raw


def _validate_checkpoint_file(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ContractValidationError(f"completed checkpoint file is corrupt: {path}")


def _assemble_dsv(
    *,
    pair_id: str,
    role: str,
    parts: Sequence[Path],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> Path:
    output = checkpoint_root / "raw" / pair_id / f"{role}.tsv"
    record = checkpoint.get(stage="assemble_ltr_dsv", config=pair_id, fold=role)
    started = reporter.operation_start(
        stage="pool_materialization",
        config=pair_id,
        fold=role,
        operation="assemble",
    )
    if record is not None:
        _validate_checkpoint_file(output, record)
        reporter.event(
            "operation_resume_skip",
            stage="pool_materialization",
            config=pair_id,
            fold=role,
            operation="assemble",
        )
    else:
        recovered = output.exists()
        if not recovered:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
            try:
                with temporary.open("wb") as destination:
                    for part in parts:
                        with part.open("rb") as source:
                            shutil.copyfileobj(
                                source, destination, length=8 * 1024 * 1024
                            )
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
        checkpoint.complete(
            stage="assemble_ltr_dsv",
            config=pair_id,
            fold=role,
            metadata={
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
                "recovered_after_atomic_save": recovered,
            },
        )
    reporter.operation_finish(
        stage="pool_materialization",
        config=pair_id,
        fold=role,
        operation="assemble",
        started=started,
        size_bytes=output.stat().st_size,
    )
    return output


def _materialize_grouped_parts(
    *,
    pair: LTRFoldPair,
    role: str,
    context: FoldContext,
    feature_columns: Sequence[str],
    group_mapping: pl.DataFrame,
    selected_users: pl.DataFrame | None,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> tuple[list[Path], dict[str, int]]:
    output_dir = checkpoint_root / "raw_parts" / pair.pair_id / role
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "rows": 0,
        "groups": 0,
        "positive_rows": 0,
        "positive_groups": 0,
        "zero_positive_groups": 0,
        "size_bytes": 0,
        "min_group_size": sys.maxsize,
        "max_group_size": 0,
    }
    outputs: list[Path] = []
    projection = ["user_id", "item_id", "label", *feature_columns]
    if role == "train":
        projection.insert(3, "is_training_sample")
    for part in context.parts:
        key = f"{context.fold}:{part.stem}"
        output = output_dir / f"{part.stem}.tsv"
        record = checkpoint.get(stage=f"ltr_dsv_{role}", config=pair.pair_id, fold=key)
        started = reporter.operation_start(
            stage="pool_materialization",
            config=f"{pair.pair_id}_{role}",
            fold=context.fold,
            operation=part.stem,
        )
        if record is not None:
            if record.get("skipped"):
                diagnostics = None
            else:
                _validate_checkpoint_file(output, record)
                diagnostics = record["diagnostics"]
            reporter.event(
                "operation_resume_skip",
                stage="pool_materialization",
                config=f"{pair.pair_id}_{role}",
                fold=context.fold,
                operation=part.stem,
            )
        else:
            frame = pl.read_parquet(part, columns=projection)
            if selected_users is not None:
                selected_frame = frame.join(selected_users, on="user_id", how="semi")
            else:
                selected_frame = frame
            if selected_frame.height == 0:
                diagnostics = None
                checkpoint.complete(
                    stage=f"ltr_dsv_{role}",
                    config=pair.pair_id,
                    fold=key,
                    metadata={"skipped": True, "source_sha256": sha256_file(part)},
                )
            else:
                recovered = output.exists()
                if recovered:
                    _, diagnostics = prepare_grouped_rows(
                        frame,
                        group_mapping=group_mapping,
                        feature_columns=feature_columns,
                        training_only=role == "train",
                        selected_users=selected_users,
                    )
                    diagnostics.update(
                        size_bytes=output.stat().st_size,
                        sha256=sha256_file(output),
                    )
                else:
                    diagnostics = write_grouped_dsv_part(
                        frame,
                        output,
                        group_mapping=group_mapping,
                        feature_columns=feature_columns,
                        training_only=role == "train",
                        selected_users=selected_users,
                    )
                checkpoint.complete(
                    stage=f"ltr_dsv_{role}",
                    config=pair.pair_id,
                    fold=key,
                    metadata={
                        "sha256": diagnostics["sha256"],
                        "source_sha256": sha256_file(part),
                        "diagnostics": diagnostics,
                        "recovered_after_atomic_save": recovered,
                    },
                )
            del frame, selected_frame
            gc.collect()
        if diagnostics is not None:
            outputs.append(output)
            for name in (
                "rows",
                "groups",
                "positive_rows",
                "positive_groups",
                "zero_positive_groups",
                "size_bytes",
            ):
                totals[name] += int(diagnostics[name])
            totals["min_group_size"] = min(
                totals["min_group_size"], int(diagnostics["min_group_size"])
            )
            totals["max_group_size"] = max(
                totals["max_group_size"], int(diagnostics["max_group_size"])
            )
        reporter.operation_finish(
            stage="pool_materialization",
            config=f"{pair.pair_id}_{role}",
            fold=context.fold,
            operation=part.stem,
            started=started,
            rows=0 if diagnostics is None else diagnostics["rows"],
        )
        reporter.stage_advance()
    if not outputs or totals["rows"] == 0:
        raise ContractValidationError(f"{pair.pair_id} {role} produced no rows")
    return outputs, totals


def _pool_digest(
    *,
    pair: LTRFoldPair,
    root_manifest: Mapping[str, Any],
    feature_columns: Sequence[str],
    group_mapping: pl.DataFrame,
    eval_users: pl.DataFrame,
    pool_config: Mapping[str, Any],
    part_limit: int | None,
) -> str:
    return config_sha256(
        {
            "kind": LTR_POOL_KIND,
            "task07_schema_sha256": root_manifest["schema_sha256"],
            "train_manifest_sha256": root_manifest["folds"][pair.train_fold][
                "manifest_sha256"
            ],
            "eval_manifest_sha256": root_manifest["folds"][pair.eval_fold][
                "manifest_sha256"
            ],
            "feature_columns": list(feature_columns),
            "group_mapping": group_mapping.to_dict(as_series=False),
            "eval_users": eval_users.to_dict(as_series=False),
            "pool": dict(pool_config),
            "borders_sha256": sha256_file(pair.borders_source),
            "smoke_part_limit": part_limit,
        }
    )


def _validate_grouped_pool(
    root: Path,
    *,
    digest: str,
    pair: LTRFoldPair,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    manifest = read_json(root / "pool_manifest.json")
    if (
        manifest.get("artifact_version") != LTR_POOL_ARTIFACT_VERSION
        or manifest.get("kind") != LTR_POOL_KIND
        or manifest.get("pool_config_sha256") != digest
        or manifest.get("pair_id") != pair.pair_id
        or manifest.get("train_fold") != pair.train_fold
        or manifest.get("eval_fold") != pair.eval_fold
        or manifest.get("feature_columns") != list(feature_columns)
    ):
        raise ContractValidationError(f"incompatible grouped pool: {root}")
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractValidationError(f"corrupt grouped pool file: {path}")
    column_lines = (root / "columns.cd").read_text(encoding="utf-8").splitlines()
    if column_lines[:2] != ["0\tLabel", "1\tGroupId"] or any(
        "Weight" in line for line in column_lines
    ):
        raise ContractValidationError("grouped pool column description is invalid")
    for role in ("train", "eval"):
        pool = Pool(f"quantized://{(root / f'{role}.quantized').resolve()}")
        group_diagnostics = validate_pool_groups(pool, name=role)
        if (
            pool.num_row() != int(manifest[f"{role}_rows"])
            or tuple(pool.get_feature_names()) != tuple(feature_columns)
            or group_diagnostics["groups"] != int(manifest[f"{role}_groups"])
            or group_diagnostics["groups"]
            != int(manifest[f"{role}_diagnostics"]["groups"])
        ):
            raise ContractValidationError(f"{role} grouped pool metadata differs")
        if role == "eval" and group_diagnostics["min_group_size"] < 20:
            raise ContractValidationError(
                "eval groups must contain at least 20 candidates for PrecisionAt@20"
            )
        del pool
        gc.collect()
    return manifest


def _prepare_grouped_pool(
    *,
    pair: LTRFoldPair,
    contexts: Mapping[str, FoldContext],
    root_manifest: Mapping[str, Any],
    feature_columns: Sequence[str],
    pool_config: Mapping[str, Any],
    part_limit: int | None,
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    if not pair.borders_source.is_file():
        raise FileNotFoundError(pair.borders_source)
    train_context = contexts[pair.train_fold]
    eval_context = contexts[pair.eval_fold]
    if not train_context.target_users.equals(eval_context.target_users):
        raise ContractValidationError("fold pair target user sets differ")
    group_mapping = build_dense_group_mapping(train_context.target_users)
    labeled_users = eval_context.ground_truth.select("user_id").unique().sort("user_id")
    requested = int(pool_config["eval_group_count"])
    if part_limit is not None:
        requested = min(requested, labeled_users.height)
    eval_users = select_complete_eval_users(
        labeled_users,
        count=requested,
        seed=int(pool_config["eval_group_seed"]),
    )
    digest = _pool_digest(
        pair=pair,
        root_manifest=root_manifest,
        feature_columns=feature_columns,
        group_mapping=group_mapping,
        eval_users=eval_users,
        pool_config=pool_config,
        part_limit=part_limit,
    )
    if pair.pool_cache.exists():
        return _validate_grouped_pool(
            pair.pool_cache,
            digest=digest,
            pair=pair,
            feature_columns=feature_columns,
        )
    stage_started = reporter.stage_start(
        stage=f"pool_{pair.pair_id}",
        total=len(train_context.parts) + len(eval_context.parts),
        unit="part",
    )
    train_parts, train_diagnostics = _materialize_grouped_parts(
        pair=pair,
        role="train",
        context=train_context,
        feature_columns=feature_columns,
        group_mapping=group_mapping,
        selected_users=None,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    eval_parts, eval_diagnostics = _materialize_grouped_parts(
        pair=pair,
        role="eval",
        context=eval_context,
        feature_columns=feature_columns,
        group_mapping=group_mapping,
        selected_users=eval_users,
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
        pair_id=pair.pair_id,
        role="train",
        parts=train_parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    eval_dsv = _assemble_dsv(
        pair_id=pair.pair_id,
        role="eval",
        parts=eval_parts,
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        reporter=reporter,
    )
    work = checkpoint_root / "pool_artifacts" / pair.pair_id
    work.mkdir(parents=True, exist_ok=True)
    cd_path = work / "columns.cd"
    mapping_path = work / "group_map.parquet"
    eval_users_path = work / "eval_users.parquet"
    borders_path = work / "borders.tsv"
    write_ltr_column_description(cd_path, feature_columns=feature_columns)
    if not mapping_path.exists():
        _write_parquet_atomic(group_mapping, mapping_path)
    elif not pl.read_parquet(mapping_path).equals(group_mapping):
        raise ContractValidationError("staged group mapping differs")
    if not eval_users_path.exists():
        _write_parquet_atomic(eval_users, eval_users_path)
    elif not pl.read_parquet(eval_users_path).equals(eval_users):
        raise ContractValidationError("staged eval users differ")
    if not borders_path.exists():
        _copy_file_atomic(pair.borders_source, borders_path)
    elif sha256_file(borders_path) != sha256_file(pair.borders_source):
        raise ContractValidationError("staged quantization borders differ")
    train_path = work / "train.quantized"
    eval_path = work / "eval.quantized"
    for role, dsv, output in (
        ("train", train_dsv, train_path),
        ("eval", eval_dsv, eval_path),
    ):
        record = checkpoint.get(stage="quantize_ltr", config=pair.pair_id, fold=role)
        started = reporter.operation_start(
            stage="quantization",
            config=pair.pair_id,
            fold=role,
            operation="quantize",
        )
        if record is None:
            pool: Pool | None = None
            recovered = False
            if output.exists():
                try:
                    candidate = Pool(f"quantized://{output.resolve()}")
                    candidate_groups = validate_pool_groups(candidate, name=role)
                    expected = (
                        train_diagnostics if role == "train" else eval_diagnostics
                    )
                    if (
                        candidate.num_row() != expected["rows"]
                        or candidate_groups["groups"] != expected["groups"]
                        or tuple(candidate.get_feature_names())
                        != tuple(feature_columns)
                    ):
                        raise ContractValidationError(
                            f"unregistered {role} quantized pool differs"
                        )
                    pool = candidate
                    recovered = True
                except (CatBoostError, OSError, ValueError):
                    output.unlink(missing_ok=True)
                    pool = None
            if pool is None:
                pool = quantize(
                    data_path=dsv.as_posix(),
                    column_description=cd_path.as_posix(),
                    delimiter="\t",
                    has_header=False,
                    thread_count=int(pool_config["thread_count"]),
                    input_borders=borders_path.as_posix(),
                    task_type=str(pool_config["quantization_task_type"]),
                    random_seed=int(pool_config["eval_group_seed"]),
                )
                temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
                try:
                    pool.save(temporary.as_posix())
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
            groups = validate_pool_groups(pool, name=role)
            checkpoint.complete(
                stage="quantize_ltr",
                config=pair.pair_id,
                fold=role,
                metadata={
                    "sha256": sha256_file(output),
                    "rows": pool.num_row(),
                    "groups": groups["groups"],
                    "recovered_after_atomic_save": recovered,
                },
            )
            del pool
            gc.collect()
        else:
            _validate_checkpoint_file(output, record)
        reporter.operation_finish(
            stage="quantization",
            config=pair.pair_id,
            fold=role,
            operation="quantize",
            started=started,
        )
    train_pool = Pool(f"quantized://{train_path.resolve()}")
    eval_pool = Pool(f"quantized://{eval_path.resolve()}")
    train_groups = validate_pool_groups(train_pool, name="train")["groups"]
    eval_groups = validate_pool_groups(eval_pool, name="eval")["groups"]
    if train_groups != train_diagnostics["groups"]:
        raise ContractValidationError("train group IDs collided or were split")
    if eval_groups != eval_diagnostics["groups"]:
        raise ContractValidationError("eval group IDs collided or were split")
    if eval_diagnostics["min_group_size"] < 20:
        raise ContractValidationError(
            "eval groups must contain at least 20 candidates for PrecisionAt@20"
        )
    del train_pool, eval_pool
    gc.collect()
    manifest = {
        "artifact_version": LTR_POOL_ARTIFACT_VERSION,
        "kind": LTR_POOL_KIND,
        "pool_config_sha256": digest,
        "pair_id": pair.pair_id,
        "train_fold": pair.train_fold,
        "eval_fold": pair.eval_fold,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "train_rows": train_diagnostics["rows"],
        "eval_rows": eval_diagnostics["rows"],
        "train_groups": train_groups,
        "eval_groups": eval_groups,
        "train_diagnostics": train_diagnostics,
        "eval_diagnostics": eval_diagnostics,
        "eval_group_sampling": {
            "method": "splitmix64_smallest_hash",
            "seed": int(pool_config["eval_group_seed"]),
            "requested_groups": int(pool_config["eval_group_count"]),
            "materialized_groups": eval_groups,
            "complete_groups": True,
        },
        "weights": {
            "object_weight_column": False,
            "group_weight_column": False,
            "runtime_group_weight_overlay_supported": True,
        },
        "quantization": {
            "task_type": pool_config["quantization_task_type"],
            "source_borders": pair.borders_source.as_posix(),
            "source_borders_sha256": sha256_file(pair.borders_source),
        },
        "files": {
            "train.quantized": sha256_file(train_path),
            "eval.quantized": sha256_file(eval_path),
            "borders.tsv": sha256_file(borders_path),
            "columns.cd": sha256_file(cd_path),
            "group_map.parquet": sha256_file(mapping_path),
            "eval_users.parquet": sha256_file(eval_users_path),
        },
    }
    write_json_atomic(work / "pool_manifest.json", manifest)
    publish_directory_atomic(work, pair.pool_cache)
    return _validate_grouped_pool(
        pair.pool_cache,
        digest=digest,
        pair=pair,
        feature_columns=feature_columns,
    )


def _model_dir(root: Path, config_id: str, pair_id: str) -> Path:
    return root / "trained_models" / config_id / pair_id


def _is_gpu_oom_output(value: str) -> bool:
    lowered = value.lower()
    return "out of memory" in lowered and (
        "ncudalib::toutofmemoryerror" in lowered
        or "cuda_lib/memory_pool" in lowered
        or "cuda error" in lowered
    )


def _terminate_fit_worker(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait()


def _run_fit_worker_process(
    *,
    config: CatBoostLTRConfig,
    feature_columns: Sequence[str],
    train_pool_path: Path,
    eval_pool_path: Path,
    model_dir: Path,
    work_dir: Path,
    output_callback: Any | None,
) -> float:
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / "worker_spec.json"
    result_path = work_dir / "worker_result.json"
    output_path = work_dir / "worker_output.log"
    spec = {
        "artifact_version": 1,
        "config": config.to_dict(),
        "feature_columns": list(feature_columns),
        "train_pool": train_pool_path.resolve().as_posix(),
        "eval_pool": eval_pool_path.resolve().as_posix(),
        "model_dir": model_dir.resolve().as_posix(),
        "train_dir": (work_dir / "train_dir").resolve().as_posix(),
        "snapshot_file": (work_dir / "snapshot.cbsnapshot").resolve().as_posix(),
        "result_file": result_path.resolve().as_posix(),
    }
    write_json_atomic(spec_path, spec)
    command = [
        sys.executable,
        Path(__file__).resolve().as_posix(),
        "--fit-worker-spec",
        spec_path.resolve().as_posix(),
    ]
    started = time.perf_counter()
    tail: deque[str] = deque(maxlen=200)
    with output_path.open("a", encoding="utf-8") as output_stream:
        output_stream.write(
            f"\n--- worker_start config={config.config_id} time={time.time():.6f} ---\n"
        )
        output_stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                output_stream.write(line)
                output_stream.flush()
                tail.append(line.rstrip("\n"))
                if output_callback is not None:
                    output_callback.write(line)
            returncode = process.wait()
        except BaseException:
            _terminate_fit_worker(process)
            raise
    elapsed = time.perf_counter() - started
    if returncode != 0:
        output_tail = "\n".join(tail)
        error_type = (
            ResourceInfeasibleError
            if _is_gpu_oom_output(output_tail)
            else CatBoostFitWorkerError
        )
        raise error_type(
            f"isolated CatBoost fit failed with exit code {returncode}; "
            f"see {output_path}",
            returncode=returncode,
            output_tail=output_tail,
        )
    if not result_path.is_file() or not model_dir.is_dir():
        raise CatBoostFitWorkerError(
            f"isolated CatBoost fit did not publish its result; see {output_path}",
            returncode=returncode,
            output_tail="\n".join(tail),
        )
    result = read_json(result_path)
    if result.get("status") != "completed" or result.get("config") != config.to_dict():
        raise ContractValidationError("isolated CatBoost fit result differs")
    return elapsed


def _fit_worker(spec_path: str | Path) -> None:
    spec = read_json(spec_path)
    if spec.get("artifact_version") != 1:
        raise CatBoostLTRError("invalid fit worker spec")
    config = CatBoostLTRConfig.from_mapping(spec["config"])
    feature_columns = tuple(spec["feature_columns"])
    model_dir = Path(spec["model_dir"])
    if model_dir.exists():
        raise FileExistsError(f"fit worker refuses to overwrite {model_dir}")
    loader = (
        CatBoostLTRDataLoader(
            feature_columns=feature_columns,
            seed=config.random_seed,
        )
        .load_fit_data(
            train_pool=Path(spec["train_pool"]),
            eval_pool=Path(spec["eval_pool"]),
        )
        .prepare_fit_data()
    )
    model = CatBoostRankerModel(config, feature_columns=feature_columns)
    started = time.perf_counter()
    model.fit(
        loader,
        train_dir=Path(spec["train_dir"]),
        snapshot_file=Path(spec["snapshot_file"]),
        log_cout=sys.stdout,
        log_cerr=sys.stderr,
    )
    model.save(model_dir)
    write_json_atomic(
        spec["result_file"],
        {
            "artifact_version": 1,
            "status": "completed",
            "config": config.to_dict(),
            "model_sha256": sha256_file(model_dir / "model.cbm"),
            "tree_count": model.tree_count,
            "best_iteration": model.best_iteration,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def _verify_portable_predictions(
    model: CatBoostRankerModel,
    *,
    model_dir: Path,
    eval_pool_path: Path,
    row_limit: int = 2048,
) -> int:
    restored = CatBoostRankerModel.from_artifact(model_dir)
    pool = Pool(f"quantized://{eval_pool_path.resolve()}")
    hashes = np.asarray(pool.get_group_id_hash(), dtype=np.uint64).reshape(-1)
    starts = np.flatnonzero(
        np.concatenate((np.array([True]), hashes[1:] != hashes[:-1]))
    )
    stops = np.concatenate((starts[1:], np.array([hashes.size])))
    complete = stops[stops <= row_limit]
    rows = int(complete[-1] if complete.size else stops[0])
    sample = pool.slice(list(range(rows)))
    first = model.predict_pool(sample)
    second = restored.predict_pool(sample)
    if not np.array_equal(first, second):
        raise ContractValidationError(
            "portable CatBoost LTR restore changed deterministic predictions"
        )
    del restored, pool, sample, first, second
    gc.collect()
    return rows


def _fit_model(
    *,
    profile: Mapping[str, Any],
    pair: LTRFoldPair,
    feature_columns: Sequence[str],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> tuple[CatBoostRankerModel, dict[str, Any], Path]:
    config_id = str(profile["config_id"])
    model_dir = _model_dir(checkpoint_root, config_id, pair.pair_id)
    importance_path = (
        checkpoint_root / "feature_importance" / config_id / f"{pair.pair_id}.parquet"
    )
    config = CatBoostLTRConfig.from_mapping(profile["catboost"])
    record = checkpoint.get(stage="fit_ltr", config=config_id, fold=pair.pair_id)
    if record is not None:
        model = CatBoostRankerModel.from_artifact(model_dir)
        if (
            model.config.to_dict() != config.to_dict()
            or model.feature_columns != tuple(feature_columns)
            or sha256_file(model_dir / "model.cbm") != record.get("model_sha256")
        ):
            raise ContractValidationError("completed LTR model differs")
        _validate_feature_importance(importance_path, feature_columns=feature_columns)
        if not record.get("portable_restore_verified") or not record.get(
            "deterministic_inference_verified"
        ):
            raise ContractValidationError("model checkpoint lacks restore verification")
        _verify_portable_predictions(
            model,
            model_dir=model_dir,
            eval_pool_path=pair.pool_cache / "eval.quantized",
        )
        reporter.event(
            "operation_resume_skip",
            stage="fit",
            config=config_id,
            fold=pair.pair_id,
            operation="catboost_ltr_fit",
        )
        return model, dict(record), model_dir
    if model_dir.exists():
        model = CatBoostRankerModel.from_artifact(model_dir)
        if model.config.to_dict() != config.to_dict() or model.feature_columns != tuple(
            feature_columns
        ):
            raise ContractValidationError("unregistered LTR model differs")
        if importance_path.exists():
            _validate_feature_importance(
                importance_path, feature_columns=feature_columns
            )
        else:
            _write_parquet_atomic(model.get_feature_importance(), importance_path)
        verification_rows = _verify_portable_predictions(
            model,
            model_dir=model_dir,
            eval_pool_path=pair.pool_cache / "eval.quantized",
        )
        metadata = {
            "model_sha256": sha256_file(model_dir / "model.cbm"),
            "tree_count": model.tree_count,
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "fit_runtime_seconds": 0.0,
            "physical_runtime_seconds": 0.0,
            "group_weight_diagnostics": model.group_weight_diagnostics,
            "portable_restore_verified": True,
            "deterministic_inference_verified": True,
            "portable_verification_rows": verification_rows,
            "recovered_after_atomic_model_save": True,
        }
        checkpoint.complete(
            stage="fit_ltr",
            config=config_id,
            fold=pair.pair_id,
            metadata=metadata,
        )
        reporter.event(
            "operation_recovered",
            stage="fit",
            config=config_id,
            fold=pair.pair_id,
            operation="portable_model",
        )
        return model, metadata, model_dir
    if importance_path.exists():
        raise FileExistsError("unregistered LTR model output exists")
    started, callback = reporter.iteration_start(
        stage="fit",
        config=config_id,
        fold=pair.pair_id,
        total=config.iterations,
        unit="tree",
    )
    bridge = _CatBoostLogBridge(callback)
    worker_dir = checkpoint_root / "catboost_training" / config_id / pair.pair_id
    try:
        fit_runtime = _run_fit_worker_process(
            config=config,
            feature_columns=feature_columns,
            train_pool_path=pair.pool_cache / "train.quantized",
            eval_pool_path=pair.pool_cache / "eval.quantized",
            model_dir=model_dir,
            work_dir=worker_dir,
            output_callback=bridge,
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
    reporter.iteration_finish(
        stage="fit",
        config=config_id,
        fold=pair.pair_id,
        started=started,
        status="completed",
    )
    model = CatBoostRankerModel.from_artifact(model_dir)
    _write_parquet_atomic(model.get_feature_importance(), importance_path)
    _validate_feature_importance(importance_path, feature_columns=feature_columns)
    verification_rows = _verify_portable_predictions(
        model,
        model_dir=model_dir,
        eval_pool_path=pair.pool_cache / "eval.quantized",
    )
    metadata = {
        "model_sha256": sha256_file(model_dir / "model.cbm"),
        "tree_count": model.tree_count,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "fit_runtime_seconds": fit_runtime,
        "physical_runtime_seconds": fit_runtime,
        "group_weight_diagnostics": model.group_weight_diagnostics,
        "portable_restore_verified": True,
        "deterministic_inference_verified": True,
        "portable_verification_rows": verification_rows,
        "recovered_after_atomic_model_save": False,
    }
    checkpoint.complete(
        stage="fit_ltr", config=config_id, fold=pair.pair_id, metadata=metadata
    )
    gc.collect()
    return model, metadata, model_dir


def _evaluate_model(
    *,
    profile: Mapping[str, Any],
    pair_id: str,
    train_fold: str,
    context: FoldContext,
    model: CatBoostRankerModel,
    fit_metadata: Mapping[str, Any],
    feature_columns: Sequence[str],
    source: Mapping[str, Any],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
    stage_name: str,
) -> dict[str, Any]:
    config_id = str(profile["config_id"])
    result_root = checkpoint_root / "results" / config_id / pair_id
    metrics_path = result_root / "metrics.json"
    recommendations_path = result_root / "recommendations.parquet"
    record = checkpoint.get(stage="evaluation_ltr", config=config_id, fold=pair_id)
    if record is not None:
        if (
            not metrics_path.is_file()
            or sha256_file(metrics_path) != record.get("metrics_sha256")
            or not recommendations_path.is_file()
            or sha256_file(recommendations_path) != record.get("recommendations_sha256")
        ):
            raise ContractValidationError("completed LTR evaluation differs")
        reporter.event(
            "operation_resume_skip",
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation="fold_evaluation",
        )
        return read_json(metrics_path)
    if metrics_path.exists() and recommendations_path.exists():
        metrics = read_json(metrics_path)
        recommendations = pl.read_parquet(recommendations_path)
        validate_recommendations_against_history(
            recommendations,
            target_users=context.target_users,
            history_daily=pl.scan_parquet(context.history_path),
            expected_k=int(source["inference"]["final_k"]),
        )
        precision = evaluate_precision_at_20(
            recommendations, context.ground_truth, context.target_users
        )
        if (
            metrics.get("config_id") != config_id
            or metrics.get("profile_sha256") != profile["profile_sha256"]
            or metrics.get("pair_id") != pair_id
            or metrics.get("recommendations_sha256")
            != sha256_file(recommendations_path)
            or any(metrics.get(name) != value for name, value in precision.items())
            or metrics.get("final_hits")
            != _final_hit_count(recommendations, context.ground_truth)
        ):
            raise ContractValidationError("unregistered LTR evaluation differs")
        checkpoint.complete(
            stage="evaluation_ltr",
            config=config_id,
            fold=pair_id,
            metadata={
                "metrics_sha256": sha256_file(metrics_path),
                "recommendations_sha256": sha256_file(recommendations_path),
                "recovered_after_atomic_save": True,
            },
        )
        reporter.event(
            "operation_recovered",
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation="fold_evaluation",
        )
        return metrics
    if metrics_path.exists() or recommendations_path.exists():
        raise FileExistsError("incomplete unregistered LTR evaluation output exists")
    started = time.perf_counter()
    output_parts: list[pl.DataFrame] = []
    stage_started = reporter.stage_start(
        stage=stage_name, total=len(context.parts), unit="part"
    )
    for part in context.parts:
        output = checkpoint_root / "inference_parts" / config_id / pair_id / part.name
        part_key = f"{pair_id}:{part.stem}"
        part_record = checkpoint.get(
            stage="inference_ltr", config=config_id, fold=part_key
        )
        operation_started = reporter.operation_start(
            stage=stage_name,
            config=config_id,
            fold=pair_id,
            operation=part.stem,
        )
        if part_record is None:
            recovered = output.exists()
            if recovered:
                top = pl.read_parquet(output)
                validate_candidate_output(
                    top,
                    k=int(source["inference"]["final_k"]),
                    source_name="catboost_ltr",
                )
            else:
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
                    source_name="catboost_ltr",
                )
                _write_parquet_atomic(top, output)
                del frame, loader, scores
            checkpoint.complete(
                stage="inference_ltr",
                config=config_id,
                fold=part_key,
                metadata={
                    "sha256": sha256_file(output),
                    "rows": top.height,
                    "recovered_after_atomic_save": recovered,
                },
            )
            del top
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
        "group_weight_diagnostics": fit_metadata.get("group_weight_diagnostics"),
    }
    write_json_atomic(metrics_path, metrics)
    checkpoint.complete(
        stage="evaluation_ltr",
        config=config_id,
        fold=pair_id,
        metadata={
            "metrics_sha256": sha256_file(metrics_path),
            "recommendations_sha256": sha256_file(recommendations_path),
            "recovered_after_atomic_save": False,
        },
    )
    return metrics


def _import_recovery_state(
    *,
    source: Mapping[str, Any],
    feature_columns: Sequence[str],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any] | None:
    recovery = source.get("recovery")
    if recovery is None:
        return None
    source_config_path = Path(recovery["source_config"])
    source_checkpoint_root = Path(recovery["source_checkpoint_dir"])
    source_checkpoint_path = source_checkpoint_root / "checkpoint.json"
    amendment_path = Path(recovery["protocol_amendment"])
    expected_files = (
        (source_config_path, recovery["source_config_sha256"]),
        (source_checkpoint_path, recovery["source_checkpoint_sha256"]),
        (amendment_path, recovery["protocol_amendment_sha256"]),
    )
    for path, expected in expected_files:
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractValidationError(f"recovery source checksum differs: {path}")
    source_state = read_json(source_checkpoint_path)
    if (
        source_state.get("artifact_version") != 1
        or source_state.get("run_id") != recovery["source_run_id"]
        or not isinstance(source_state.get("completed"), dict)
    ):
        raise ContractValidationError("recovery source checkpoint is invalid")
    amendment = read_json(amendment_path)
    if (
        amendment.get("artifact_version") != 1
        or amendment.get("run_id") != recovery["source_run_id"]
        or amendment.get("source_checkpoint_sha256_before_prune")
        != recovery["source_checkpoint_sha256"]
    ):
        raise ContractValidationError("recovery protocol amendment differs")
    source_config, source_features, _, _ = _load_source(source_config_path)
    if tuple(source_features) != tuple(feature_columns):
        raise ContractValidationError("recovery feature order differs")
    source_profiles = {
        profile["config_id"]: profile for profile in source_config["objective_profiles"]
    }
    target_profiles = {
        profile["config_id"]: profile for profile in source["objective_profiles"]
    }
    imported_records: list[dict[str, Any]] = []
    for config_id in recovery["import_completed_profiles"]:
        if source_profiles.get(config_id) != target_profiles.get(config_id):
            raise ContractValidationError(
                f"recovery profile is not byte-equivalent: {config_id}"
            )
        for pair in source["fold_pairs"]:
            pair_id = pair.pair_id
            fit_key = CheckpointStore.key(
                stage="fit_ltr", config=config_id, fold=pair_id
            )
            evaluation_key = CheckpointStore.key(
                stage="evaluation_ltr", config=config_id, fold=pair_id
            )
            source_fit = source_state["completed"].get(fit_key)
            source_evaluation = source_state["completed"].get(evaluation_key)
            if not isinstance(source_fit, dict) or not isinstance(
                source_evaluation, dict
            ):
                raise ContractValidationError(
                    f"recovery source lacks completed {config_id} {pair_id}"
                )
            source_model = _model_dir(source_checkpoint_root, config_id, pair_id)
            source_importance = (
                source_checkpoint_root
                / "feature_importance"
                / config_id
                / f"{pair_id}.parquet"
            )
            source_result = source_checkpoint_root / "results" / config_id / pair_id
            if (
                sha256_file(source_model / "model.cbm")
                != source_fit.get("model_sha256")
                or sha256_file(source_result / "metrics.json")
                != source_evaluation.get("metrics_sha256")
                or sha256_file(source_result / "recommendations.parquet")
                != source_evaluation.get("recommendations_sha256")
            ):
                raise ContractValidationError(
                    f"recovery artifact checksum differs: {config_id} {pair_id}"
                )
            restored = CatBoostRankerModel.from_artifact(source_model)
            if restored.config.to_dict() != target_profiles[config_id][
                "catboost"
            ] or restored.feature_columns != tuple(feature_columns):
                raise ContractValidationError(
                    f"recovery model contract differs: {config_id} {pair_id}"
                )
            _validate_feature_importance(
                source_importance, feature_columns=feature_columns
            )
            destination_model = _model_dir(checkpoint_root, config_id, pair_id)
            destination_importance = (
                checkpoint_root
                / "feature_importance"
                / config_id
                / f"{pair_id}.parquet"
            )
            destination_result = checkpoint_root / "results" / config_id / pair_id
            if not destination_model.exists():
                _copy_directory_atomic(source_model, destination_model)
            if not destination_importance.exists():
                _copy_file_atomic(source_importance, destination_importance)
            if not destination_result.exists():
                _copy_directory_atomic(source_result, destination_result)
            if (
                sha256_file(destination_model / "model.cbm")
                != source_fit["model_sha256"]
                or sha256_file(destination_importance) != sha256_file(source_importance)
                or sha256_file(destination_result / "metrics.json")
                != source_evaluation["metrics_sha256"]
                or sha256_file(destination_result / "recommendations.parquet")
                != source_evaluation["recommendations_sha256"]
            ):
                raise ContractValidationError(
                    f"imported recovery artifact differs: {config_id} {pair_id}"
                )
            provenance = {
                "imported_from_run_id": recovery["source_run_id"],
                "source_checkpoint_sha256": recovery["source_checkpoint_sha256"],
            }
            checkpoint.complete(
                stage="fit_ltr",
                config=config_id,
                fold=pair_id,
                metadata={**source_fit, **provenance},
            )
            checkpoint.complete(
                stage="evaluation_ltr",
                config=config_id,
                fold=pair_id,
                metadata={**source_evaluation, **provenance},
            )
            imported_records.append(
                {
                    "config_id": config_id,
                    "pair_id": pair_id,
                    "model_sha256": source_fit["model_sha256"],
                    "metrics_sha256": source_evaluation["metrics_sha256"],
                    "recommendations_sha256": source_evaluation[
                        "recommendations_sha256"
                    ],
                }
            )
            reporter.event(
                "operation_resume_import",
                stage="recovery",
                config=config_id,
                fold=pair_id,
                operation="verified_completed_profile",
                source_run_id=recovery["source_run_id"],
            )
    recovery_root = checkpoint_root / "recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    copied_amendment = recovery_root / "protocol_amendment.json"
    if not copied_amendment.exists():
        _copy_file_atomic(amendment_path, copied_amendment)
    if sha256_file(copied_amendment) != recovery["protocol_amendment_sha256"]:
        raise ContractValidationError("copied recovery protocol amendment differs")
    manifest = {
        "artifact_version": 1,
        "source_run_id": recovery["source_run_id"],
        "source_config_sha256": recovery["source_config_sha256"],
        "source_checkpoint_sha256": recovery["source_checkpoint_sha256"],
        "protocol_amendment_sha256": recovery["protocol_amendment_sha256"],
        "imported_records": imported_records,
    }
    manifest_path = recovery_root / "import_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ContractValidationError("recovery import manifest differs")
    if not manifest_path.exists():
        write_json_atomic(manifest_path, manifest)
    checkpoint.complete(
        stage="recovery_import",
        config=",".join(recovery["import_completed_profiles"]),
        fold="both",
        metadata={"sha256": sha256_file(manifest_path)},
    )
    return manifest


def _run_resource_probe(
    *,
    profile: Mapping[str, Any],
    pair: LTRFoldPair,
    feature_columns: Sequence[str],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
    iterations: int,
) -> dict[str, Any]:
    config_id = str(profile["config_id"])
    record = checkpoint.get(stage="resource_probe", config=config_id, fold=pair.pair_id)
    probe_root = checkpoint_root / "resource_probes" / config_id / pair.pair_id
    evidence_path = probe_root / "evidence.json"
    if record is not None:
        _validate_checkpoint_file(evidence_path, record)
        return read_json(evidence_path)
    base = CatBoostLTRConfig.from_mapping(profile["catboost"])
    probe_source = base.to_dict()
    probe_source.update(
        config_id=f"{config_id}__resource_probe",
        iterations=iterations,
        early_stopping_rounds=1,
        metric_period=1,
    )
    probe = CatBoostLTRConfig.from_mapping(probe_source)
    model_dir = probe_root / "model"
    started = reporter.operation_start(
        stage="resource_probe",
        config=config_id,
        fold=pair.pair_id,
        operation="full_pool_one_iteration_fit",
    )
    try:
        if model_dir.exists():
            restored = CatBoostRankerModel.from_artifact(model_dir)
            if (
                restored.config.to_dict() != probe.to_dict()
                or restored.feature_columns != tuple(feature_columns)
            ):
                raise ContractValidationError(
                    "unregistered resource-probe model differs"
                )
            runtime = 0.0
            recovered = True
        else:
            runtime = _run_fit_worker_process(
                config=probe,
                feature_columns=feature_columns,
                train_pool_path=pair.pool_cache / "train.quantized",
                eval_pool_path=pair.pool_cache / "eval.quantized",
                model_dir=model_dir,
                work_dir=probe_root / "worker",
                output_callback=None,
            )
            recovered = False
    except ResourceInfeasibleError as error:
        evidence = {
            "artifact_version": 1,
            "status": "resource_infeasible",
            "config_id": config_id,
            "fold_pair": pair.pair_id,
            "probe_config": probe.to_dict(),
            "returncode": error.returncode,
            "error": str(error),
            "output_tail": error.output_tail,
        }
        write_json_atomic(evidence_path, evidence)
        checkpoint.complete(
            stage="resource_probe",
            config=config_id,
            fold=pair.pair_id,
            metadata={
                "sha256": sha256_file(evidence_path),
                "status": "resource_infeasible",
            },
        )
        reporter.operation_finish(
            stage="resource_probe",
            config=config_id,
            fold=pair.pair_id,
            operation="full_pool_one_iteration_fit",
            started=started,
            status="resource_infeasible",
        )
        return evidence
    evidence = {
        "artifact_version": 1,
        "status": "passed",
        "config_id": config_id,
        "fold_pair": pair.pair_id,
        "probe_config": probe.to_dict(),
        "runtime_seconds": runtime,
        "model_sha256": sha256_file(model_dir / "model.cbm"),
        "recovered_after_atomic_model_save": recovered,
    }
    write_json_atomic(evidence_path, evidence)
    checkpoint.complete(
        stage="resource_probe",
        config=config_id,
        fold=pair.pair_id,
        metadata={"sha256": sha256_file(evidence_path), "status": "passed"},
    )
    reporter.operation_finish(
        stage="resource_probe",
        config=config_id,
        fold=pair.pair_id,
        operation="full_pool_one_iteration_fit",
        started=started,
        status="passed",
        runtime_seconds=runtime,
    )
    return evidence


def _record_resource_exclusion(
    *,
    profile: Mapping[str, Any],
    pair_id: str,
    reason: str,
    evidence: Mapping[str, Any],
    checkpoint: CheckpointStore,
    checkpoint_root: Path,
    reporter: EventProgressReporter,
) -> dict[str, Any]:
    config_id = str(profile["config_id"])
    payload = {
        "artifact_version": 1,
        "status": "resource_infeasible",
        "config_id": config_id,
        "objective": CatBoostLTRConfig.from_mapping(profile["catboost"]).objective,
        "fold_pair": pair_id,
        "reason": reason,
        "canonical_opened": False,
        "profile": dict(profile),
        "evidence": dict(evidence),
    }
    path = checkpoint_root / "resource_exclusions" / f"{config_id}.json"
    if path.exists() and read_json(path) != payload:
        raise ContractValidationError("resource exclusion checkpoint differs")
    if not path.exists():
        write_json_atomic(path, payload)
    checkpoint.complete(
        stage="resource_exclusion",
        config=config_id,
        fold=pair_id,
        metadata={"sha256": sha256_file(path)},
    )
    reporter.event(
        "resource_exclusion",
        stage="objective_family",
        config=config_id,
        fold=pair_id,
        operation="exclude_profile_and_continue",
        status="resource_infeasible",
        reason=reason,
        canonical_opened=False,
    )
    return payload


def _candidate_metrics(context: FoldContext) -> dict[str, Any]:
    return evaluate_candidate_metrics_lazy(
        pl.scan_parquet([part.as_posix() for part in context.parts]).select(
            "user_id", "item_id"
        ),
        context.ground_truth,
        context.target_users,
    )


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
            raise CatBoostLTRError(f"incompatible best-model state: {pointer}")
        if current.get("profile_sha256") == profile["profile_sha256"]:
            current_model = Path(str(current.get("model_artifact"))) / "model.cbm"
            if (
                not current_model.is_file()
                or sha256_file(current_model) != current.get("model_sha256")
                or sha256_file(current_model) != sha256_file(model_dir / "model.cbm")
            ):
                raise ContractValidationError("current best portable model differs")
            return
    version = best_root / "versions" / str(profile["profile_sha256"])
    if not version.exists():
        _copy_directory_atomic(model_dir, version)
    elif sha256_file(version / "model.cbm") != sha256_file(model_dir / "model.cbm"):
        raise ContractValidationError("versioned best portable model differs")
    write_json_atomic(
        pointer,
        {
            "artifact_version": 1,
            "run_id": run_id,
            "config_sha256": digest,
            "config_id": profile["config_id"],
            "profile_sha256": profile["profile_sha256"],
            "model_artifact": version.as_posix(),
            "model_sha256": sha256_file(version / "model.cbm"),
            "selection_metrics": dict(aggregate),
        },
    )


def _verify_pinned_pointwise(
    source: Mapping[str, Any], pairs: Sequence[LTRFoldPair]
) -> dict[str, Any]:
    comparator = source["pointwise_comparator"]
    task09 = Path(source["task09_artifact"])
    if task09 != Path(comparator["artifact"]):
        raise ContractValidationError("pointwise artifact paths differ")
    winner = read_json(task09 / "selection" / "winner.json")
    if (
        winner.get("profile", {}).get("profile_sha256") != comparator["profile_sha256"]
        or winner.get("profile", {}).get("config_id") != comparator["config_id"]
    ):
        raise ContractValidationError("Task 09 pinned winner differs")
    actual = winner.get("rolling_metrics", {}).get("fold_results", [])
    expected = comparator["fold_results"]
    if len(actual) != len(expected):
        raise ContractValidationError("Task 09 comparator fold count differs")
    fields = (
        "pair_id",
        "precision_at_20_all_targets",
        "precision_at_20_labeled_users",
        "final_hits",
        "tree_count",
    )
    for actual_row, expected_row, pair in zip(actual, expected, pairs, strict=True):
        if any(actual_row.get(name) != expected_row.get(name) for name in fields):
            raise ContractValidationError(
                f"Task 09 comparator differs for {pair.pair_id}"
            )
    aggregate = aggregate_fold_results(comparator["config_id"], expected)
    return {**aggregate, "profile_sha256": comparator["profile_sha256"]}


def _write_checksums(root: Path) -> None:
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    write_json_atomic(root / "checksums.json", {"artifact_version": 1, "files": files})


def verify_ltr_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    config = read_json(root / "config.json")
    metrics = read_json(root / "metrics.json")
    checksums = read_json(root / "checksums.json")
    if (
        config.get("artifact_version") != LTR_SELECTION_ARTIFACT_VERSION
        or config.get("kind") != LTR_SELECTION_ARTIFACT_KIND
        or metrics.get("run_id") != config.get("run_id")
        or metrics.get("canonical_evaluated_config_count") != 1
    ):
        raise ContractValidationError("published Task 10 artifact is incomplete")
    files = checksums.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ContractValidationError("published checksums are missing")
    for name, expected in files.items():
        file_path = root / str(name)
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise ContractValidationError(f"published checksum differs: {name}")
    recovery = config.get("recovery")
    recovery_import = config.get("recovery_import")
    if (recovery is None) != (recovery_import is None):
        raise ContractValidationError("published recovery metadata is incomplete")
    if recovery is not None:
        import_manifest = read_json(root / "recovery" / "import_manifest.json")
        if import_manifest != recovery_import:
            raise ContractValidationError("published recovery import does not replay")
        if (
            sha256_file(root / "recovery" / "protocol_amendment.json")
            != recovery["protocol_amendment_sha256"]
            or import_manifest.get("source_checkpoint_sha256")
            != recovery["source_checkpoint_sha256"]
        ):
            raise ContractValidationError("published recovery evidence differs")
        resource_probe = config.get("resource_probe")
        if (
            not isinstance(resource_probe, Mapping)
            or read_json(root / "recovery" / "resource_probe.json") != resource_probe
        ):
            raise ContractValidationError("published resource probe differs")
    elif config.get("resource_probe") is not None:
        raise ContractValidationError("resource probe exists outside a recovery run")
    model = CatBoostRankerModel.from_artifact(root / "model")
    if sha256_file(root / "model" / "model.cbm") != metrics.get("model_sha256"):
        raise ContractValidationError("published winner model differs")
    if sha256_file(root / "model_config.json") != sha256_file(
        root / "model" / "model_config.json"
    ):
        raise ContractValidationError("published root model config differs")
    recommendations = pl.read_parquet(root / "canonical_recommendations.parquet")
    if (
        recommendations.height != int(metrics["canonical"]["target_users"])
        or recommendations.get_column("user_id").n_unique() != recommendations.height
        or recommendations.get_column("item_ids").list.len().min() != 20
        or recommendations.get_column("item_ids").list.len().max() != 20
    ):
        raise ContractValidationError("published canonical recommendations differ")
    dataset_root = Path(config["task07_artifact"])
    smoke_part_limit = config.get("smoke_part_limit")
    canonical_fold = str(config["canonical"]["fold"])
    selection_folds = list(config["canonical_isolation"]["selection_folds"])
    contexts, feature_columns, _, root_manifest = _validate_task07(
        dataset_root,
        folds=[*selection_folds, canonical_fold],
        part_limit=smoke_part_limit,
        verify_checksums=False,
    )
    if (
        feature_columns != tuple(config["feature_columns"])
        or root_manifest["schema_sha256"] != config["task07_schema_sha256"]
    ):
        raise ContractValidationError("published Task 07 contract differs")
    canonical_context = contexts[canonical_fold]
    validate_recommendations_against_history(
        recommendations,
        target_users=canonical_context.target_users,
        history_daily=pl.scan_parquet(canonical_context.history_path),
        expected_k=20,
    )
    precision = evaluate_precision_at_20(
        recommendations,
        canonical_context.ground_truth,
        canonical_context.target_users,
    )
    if (
        any(metrics.get(name) != value for name, value in precision.items())
        or any(
            metrics["canonical"].get(name) != value for name, value in precision.items()
        )
        or metrics.get("final_hits")
        != _final_hit_count(recommendations, canonical_context.ground_truth)
    ):
        raise ContractValidationError("published canonical metrics do not replay")
    canonical_files = list((root / "evaluations").glob("*/canonical_once/metrics.json"))
    if (
        len(canonical_files) != 1
        or canonical_files[0].parents[1].name != metrics["winner_config_id"]
    ):
        raise ContractValidationError("canonical was not evaluated exactly once")

    stages_value = read_json(root / "selection" / "stage_results.json")
    stages = stages_value.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ContractValidationError("selection stages are missing")
    selection_order: list[str] = []
    for stage in stages:
        for config_id in stage["evaluated_config_ids"]:
            if config_id not in selection_order:
                selection_order.append(config_id)
    runtime_exclusions = config.get("runtime_resource_exclusions", [])
    if not isinstance(runtime_exclusions, list):
        raise ContractValidationError("runtime resource exclusions must be a list")
    excluded_ids: list[str] = []
    for exclusion in runtime_exclusions:
        config_id = str(exclusion.get("config_id"))
        exclusion_path = root / "resource_exclusions" / f"{config_id}.json"
        if (
            exclusion.get("status") != "resource_infeasible"
            or exclusion.get("canonical_opened") is not False
            or config_id in selection_order
            or not exclusion_path.is_file()
            or read_json(exclusion_path) != exclusion
        ):
            raise ContractValidationError("runtime resource exclusion differs")
        excluded_ids.append(config_id)
    recorded_excluded_ids = [
        str(config_id)
        for stage in stages
        for config_id in stage.get("resource_infeasible_config_ids", [])
    ]
    if sorted(recorded_excluded_ids) != sorted(excluded_ids):
        raise ContractValidationError("selection-stage resource exclusions differ")
    replayed: dict[str, dict[str, Any]] = {}
    pair_ids = [str(pair["pair_id"]) for pair in config["fold_pairs"]]
    for config_id in selection_order:
        fold_results = [
            read_json(root / "evaluations" / config_id / pair_id / "metrics.json")
            for pair_id in pair_ids
        ]
        replayed[config_id] = aggregate_fold_results(config_id, fold_results)
    for stage in stages:
        candidates = [replayed[value] for value in stage["evaluated_config_ids"]]
        selected = select_best_result(
            candidates,
            tie_epsilon=float(config["search"]["tie_epsilon"]),
            order=selection_order,
        )
        if selected["config_id"] != stage["incumbent_config_id"]:
            raise ContractValidationError("selection stage tie-break does not replay")
        for name in (
            "mean_precision_at_20_labeled_users",
            "min_precision_at_20_labeled_users",
            "fold_spread_precision_at_20_labeled_users",
            "mean_precision_at_20_all_targets",
            "mean_tree_count",
            "runtime_seconds",
            "physical_runtime_seconds",
            "total_hits",
        ):
            if stage["winner"].get(name) != selected[name]:
                raise ContractValidationError(
                    f"selection stage aggregate differs: {stage['stage_id']} {name}"
                )
    if stages[-1]["incumbent_config_id"] != metrics["winner_config_id"]:
        raise ContractValidationError("published final LTR winner differs")

    pool_config = config["pool"]
    pair_manifests = config["pool_manifests"]
    for pair_source in config["fold_pairs"]:
        pair = LTRFoldPair.from_mapping(pair_source)
        train_context = contexts[pair.train_fold]
        eval_context = contexts[pair.eval_fold]
        if not train_context.target_users.equals(eval_context.target_users):
            raise ContractValidationError("verified fold target users differ")
        mapping = build_dense_group_mapping(train_context.target_users)
        artifact_pool = root / "pools" / pair.pair_id
        if not pl.read_parquet(artifact_pool / "group_map.parquet").equals(mapping):
            raise ContractValidationError("published dense group mapping differs")
        labeled_users = (
            eval_context.ground_truth.select("user_id").unique().sort("user_id")
        )
        requested = int(pool_config["eval_group_count"])
        if smoke_part_limit is not None:
            requested = min(requested, labeled_users.height)
        eval_users = select_complete_eval_users(
            labeled_users,
            count=requested,
            seed=int(pool_config["eval_group_seed"]),
        )
        if not pl.read_parquet(artifact_pool / "eval_users.parquet").equals(eval_users):
            raise ContractValidationError("published complete eval groups differ")
        expected_manifest = pair_manifests[pair.pair_id]
        checked_manifest = _validate_grouped_pool(
            pair.pool_cache,
            digest=str(expected_manifest["pool_config_sha256"]),
            pair=pair,
            feature_columns=feature_columns,
        )
        if checked_manifest != expected_manifest:
            raise ContractValidationError("published grouped pool manifest differs")

    sample = pl.read_parquet(
        canonical_context.parts[0],
        columns=["user_id", "item_id", *feature_columns],
    ).head(2048)
    first_loader = (
        CatBoostRankerDataLoader(feature_columns=feature_columns)
        .load_predict_data(frame=sample)
        .prepare_predict_data()
    )
    second_loader = (
        CatBoostRankerDataLoader(feature_columns=feature_columns)
        .load_predict_data(frame=sample)
        .prepare_predict_data()
    )
    first = model.predict(first_loader, batch_size=257)
    second = model.predict(second_loader, batch_size=509)
    if not first.equals(second):
        raise ContractValidationError("portable model inference is not deterministic")
    return {
        "run_id": config["run_id"],
        "winner_config_id": metrics["winner_config_id"],
        "tree_count": model.tree_count,
        "canonical_evaluated_config_count": 1,
        "checked_files": len(files),
        "selection_aggregation_replayed": True,
        "group_mapping_replayed": True,
        "precision_replayed": precision,
        "final_hits": metrics["final_hits"],
        "portable_inference_deterministic": True,
    }


def _cleanup(
    *,
    checkpoint_root: Path,
    best_root: Path,
    output: Path,
    pool_dirs: Sequence[Path],
    input_dirs: Sequence[Path],
    run_id: str,
) -> dict[str, Any]:
    verify_ltr_artifact(output)
    protected = [
        output.resolve(),
        *(path.resolve() for path in pool_dirs),
        *(path.resolve() for path in input_dirs),
    ]
    targets = (
        (checkpoint_root, "checkpoint.json", "checkpoint"),
        (best_root, "best_model.json", "best_model"),
    )
    validated: list[tuple[Path, str, int]] = []
    removed: dict[str, Any] = {}
    for path, marker, name in targets:
        if not path.exists():
            removed[name] = {"removed": False, "bytes": 0}
            continue
        resolved = path.resolve()
        if len(resolved.parts) < 4 or any(
            resolved == value or resolved in value.parents or value in resolved.parents
            for value in protected
        ):
            raise CatBoostLTRError(f"unsafe cleanup target: {path}")
        metadata = read_json(resolved / marker)
        if metadata.get("artifact_version") != 1 or metadata.get("run_id") != run_id:
            raise CatBoostLTRError(f"cleanup marker belongs to another run: {path}")
        validated.append((resolved, name, _directory_size_bytes(resolved)))
    for target, name, size in validated:
        shutil.rmtree(target)
        removed[name] = {"removed": True, "bytes": size, "path": target.as_posix()}
    return {
        "removed": removed,
        "freed_bytes": sum(value["bytes"] for value in removed.values()),
        "retained_pools": [path.as_posix() for path in pool_dirs],
    }


def run_ltr_selection(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    checkpoint_dir: str | Path,
    best_model_dir: str | Path,
    log_file: str | Path | None,
    smoke_part_limit: int | None,
    smoke_profile_limit: int | None,
    verify_input_checksums: bool,
    show_progress: bool,
    cleanup_recoverable_on_success: bool,
    stop_after_completed_configs: int | None,
) -> dict[str, Any]:
    if not run_id:
        raise CatBoostLTRError("run_id must be non-empty")
    if smoke_part_limit is not None and smoke_part_limit <= 0:
        raise CatBoostLTRError("smoke_part_limit must be positive")
    if smoke_profile_limit is not None and smoke_profile_limit <= 0:
        raise CatBoostLTRError("smoke_profile_limit must be positive")
    if stop_after_completed_configs is not None and stop_after_completed_configs <= 0:
        raise CatBoostLTRError("stop_after_completed_configs must be positive")
    config_source = Path(config_path)
    source, feature_columns, feature_schema, raw_source = _load_source(config_source)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    mode = "smoke" if smoke_part_limit is not None else "full"
    if mode == "full" and smoke_profile_limit is not None:
        raise CatBoostLTRError("smoke_profile_limit is forbidden in full mode")
    if mode == "smoke":
        smoke_paths = [output, *(pair.pool_cache for pair in source["fold_pairs"])]
        if "smoke" not in run_id.lower() or any(
            "smoke" not in path.as_posix().lower() for path in smoke_paths
        ):
            raise CatBoostLTRError(
                "limited smoke requires smoke run_id, output, and pool paths"
            )
    resource_preflight = _resource_preflight(source, output=output)
    digest = config_sha256(
        {
            "source": raw_source,
            "source_config_sha256": sha256_file(config_source),
            "run_id": run_id,
            "mode": mode,
            "smoke_part_limit": smoke_part_limit,
            "smoke_profile_limit": smoke_profile_limit,
        }
    )
    checkpoint_root = Path(checkpoint_dir)
    checkpoint = CheckpointStore(checkpoint_root, run_id=run_id, config_digest=digest)
    best_root = Path(best_model_dir)
    reporter = EventProgressReporter(
        task_name="task10_catboost_ltr",
        total_phases=6,
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
        available_ram_bytes=resource_preflight["observed"]["available_ram_bytes"],
        free_disk_bytes=resource_preflight["observed"]["free_disk_bytes"],
        gpu=resource_preflight["observed"]["gpu"],
        windows_host_storage=resource_preflight["observed"]["windows_host_storage"],
    )
    try:
        phase = reporter.phase_start("validate_selection_inputs")
        dataset_root = Path(source["task07_artifact"])
        folds = list(
            dict.fromkeys(
                fold
                for pair in source["fold_pairs"]
                for fold in (pair.train_fold, pair.eval_fold)
            )
        )
        contexts, checked_features, _, root_manifest = _validate_task07(
            dataset_root,
            folds=folds,
            part_limit=smoke_part_limit,
            verify_checksums=verify_input_checksums,
        )
        if checked_features != feature_columns:
            raise ContractValidationError("Task 07 feature order changed")
        for pair in source["fold_pairs"]:
            if not pair.borders_source.is_file():
                raise FileNotFoundError(pair.borders_source)
        reporter.phase_finish(
            "validate_selection_inputs",
            phase,
            feature_count=len(feature_columns),
            folds=folds,
            canonical_opened=False,
        )

        phase = reporter.phase_start("prepare_grouped_pools")
        pool_manifests: dict[str, dict[str, Any]] = {}
        for pair in source["fold_pairs"]:
            pool_manifests[pair.pair_id] = _prepare_grouped_pool(
                pair=pair,
                contexts=contexts,
                root_manifest=root_manifest,
                feature_columns=feature_columns,
                pool_config=source["pool"],
                part_limit=smoke_part_limit,
                checkpoint=checkpoint,
                checkpoint_root=checkpoint_root,
                reporter=reporter,
            )
        reporter.phase_finish(
            "prepare_grouped_pools",
            phase,
            train_rows=sum(int(row["train_rows"]) for row in pool_manifests.values()),
            eval_rows=sum(int(row["eval_rows"]) for row in pool_manifests.values()),
        )
        recovery_manifest = (
            _import_recovery_state(
                source=source,
                feature_columns=feature_columns,
                checkpoint=checkpoint,
                checkpoint_root=checkpoint_root,
                reporter=reporter,
            )
            if mode == "full"
            else None
        )

        phase = reporter.phase_start("walk_forward_selection")
        results_by_id: dict[str, dict[str, Any]] = {}
        profiles_by_id: dict[str, dict[str, Any]] = {}
        selection_order: list[str] = []
        runtime_resource_exclusions: list[dict[str, Any]] = []
        completed_now = 0

        def evaluate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal completed_now
            config_id = str(profile["config_id"])
            if config_id in results_by_id:
                return results_by_id[config_id]
            was_complete = all(
                checkpoint.get(
                    stage="evaluation_ltr", config=config_id, fold=pair.pair_id
                )
                is not None
                for pair in source["fold_pairs"]
            )
            fold_results: list[dict[str, Any]] = []
            for pair in source["fold_pairs"]:
                host_storage = _check_windows_host_disk(source, launch=False)
                if host_storage is not None:
                    reporter.event(
                        "resource_check",
                        stage="walk_forward_selection",
                        config=config_id,
                        fold=pair.pair_id,
                        operation="windows_host_disk",
                        **host_storage,
                    )
                model, fit_metadata, _ = _fit_model(
                    profile=profile,
                    pair=pair,
                    feature_columns=feature_columns,
                    checkpoint=checkpoint,
                    checkpoint_root=checkpoint_root,
                    reporter=reporter,
                )
                fold_results.append(
                    _evaluate_model(
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
                        stage_name="selection_inference",
                    )
                )
                del model
                gc.collect()
            aggregate = aggregate_fold_results(config_id, fold_results)
            aggregate.update(
                profile=dict(profile), profile_sha256=profile["profile_sha256"]
            )
            results_by_id[config_id] = aggregate
            profiles_by_id[config_id] = dict(profile)
            selection_order.append(config_id)
            observed_best = select_best_result(
                list(results_by_id.values()),
                tie_epsilon=float(source["search"]["tie_epsilon"]),
                order=selection_order,
            )
            reporter.event(
                "config_finish",
                stage="walk_forward_selection",
                config=config_id,
                fold="both",
                operation="aggregate",
                current_metric=aggregate["mean_precision_at_20_labeled_users"],
                best_metric=observed_best["mean_precision_at_20_labeled_users"],
                best_config=observed_best["config_id"],
            )
            if not was_complete:
                completed_now += 1
                if (
                    stop_after_completed_configs is not None
                    and completed_now >= stop_after_completed_configs
                ):
                    raise IntentionalLTRStop(
                        f"stopped after {completed_now} newly completed configs"
                    )
            return aggregate

        latest_pair_id = str(source["canonical"]["latest_pair_id"])

        def complete_stage(stage_result: Mapping[str, Any]) -> None:
            stage_id = str(stage_result["stage_id"])
            stage_path = checkpoint_root / "stage_results" / f"{stage_id}.json"
            record = checkpoint.get(
                stage="stage_winner",
                config=str(stage_result["incumbent_config_id"]),
                fold=stage_id,
            )
            if record is not None:
                _validate_checkpoint_file(stage_path, record)
                if read_json(stage_path) != dict(stage_result):
                    raise ContractValidationError("stage winner checkpoint differs")
            else:
                if stage_path.exists() and read_json(stage_path) != dict(stage_result):
                    raise ContractValidationError("unregistered stage winner differs")
                write_json_atomic(stage_path, stage_result)
                checkpoint.complete(
                    stage="stage_winner",
                    config=str(stage_result["incumbent_config_id"]),
                    fold=stage_id,
                    metadata={"sha256": sha256_file(stage_path)},
                )
            stage_profile = profiles_by_id[str(stage_result["incumbent_config_id"])]
            stage_aggregate = results_by_id[str(stage_result["incumbent_config_id"])]
            _update_best_model(
                best_root=best_root,
                run_id=run_id,
                digest=digest,
                profile=stage_profile,
                aggregate=stage_aggregate,
                model_dir=_model_dir(
                    checkpoint_root,
                    str(stage_result["incumbent_config_id"]),
                    latest_pair_id,
                ),
            )
            reporter.event(
                "stage_winner",
                stage=stage_id,
                config=str(stage_result["incumbent_config_id"]),
                fold="both",
                operation="aggregate",
                best_metric=stage_aggregate["mean_precision_at_20_labeled_users"],
                best_config=str(stage_result["incumbent_config_id"]),
            )

        objective_profiles = list(source["objective_profiles"])
        if smoke_profile_limit is not None:
            objective_profiles = objective_profiles[:smoke_profile_limit]
        objective_results: list[dict[str, Any]] = []
        recovery = source.get("recovery") if mode == "full" else None
        probe_source = recovery["resource_probe"] if recovery is not None else None
        resource_probe_evidence: dict[str, Any] | None = None
        for profile in objective_profiles:
            config_id = str(profile["config_id"])
            if probe_source is not None and config_id == probe_source["profile_id"]:
                exclusion_record = checkpoint.get(
                    stage="resource_exclusion",
                    config=config_id,
                    fold=probe_source["fold_pair"],
                )
                if exclusion_record is not None:
                    exclusion_path = (
                        checkpoint_root / "resource_exclusions" / f"{config_id}.json"
                    )
                    _validate_checkpoint_file(exclusion_path, exclusion_record)
                    exclusion = read_json(exclusion_path)
                    runtime_resource_exclusions.append(exclusion)
                    probe_evidence_path = (
                        checkpoint_root
                        / "resource_probes"
                        / config_id
                        / probe_source["fold_pair"]
                        / "evidence.json"
                    )
                    if probe_evidence_path.is_file():
                        resource_probe_evidence = read_json(probe_evidence_path)
                    reporter.event(
                        "operation_resume_skip",
                        stage="objective_family",
                        config=config_id,
                        fold=probe_source["fold_pair"],
                        operation="resource_exclusion",
                    )
                    continue
            if probe_source is not None and config_id == probe_source["profile_id"]:
                probe_pair = next(
                    pair
                    for pair in source["fold_pairs"]
                    if pair.pair_id == probe_source["fold_pair"]
                )
                _check_windows_host_disk(source, launch=False)
                probe_evidence = _run_resource_probe(
                    profile=profile,
                    pair=probe_pair,
                    feature_columns=feature_columns,
                    checkpoint=checkpoint,
                    checkpoint_root=checkpoint_root,
                    reporter=reporter,
                    iterations=int(probe_source["iterations"]),
                )
                resource_probe_evidence = probe_evidence
                if probe_evidence["status"] == "resource_infeasible":
                    runtime_resource_exclusions.append(
                        _record_resource_exclusion(
                            profile=profile,
                            pair_id=probe_pair.pair_id,
                            reason="full_pool_resource_probe_gpu_oom",
                            evidence=probe_evidence,
                            checkpoint=checkpoint,
                            checkpoint_root=checkpoint_root,
                            reporter=reporter,
                        )
                    )
                    continue
            try:
                objective_results.append(evaluate_profile(profile))
            except ResourceInfeasibleError as error:
                if probe_source is None or config_id != probe_source["profile_id"]:
                    raise
                runtime_resource_exclusions.append(
                    _record_resource_exclusion(
                        profile=profile,
                        pair_id=probe_source["fold_pair"],
                        reason="isolated_full_fit_gpu_oom_after_passed_probe",
                        evidence={
                            "returncode": error.returncode,
                            "error": str(error),
                            "output_tail": error.output_tail,
                        },
                        checkpoint=checkpoint,
                        checkpoint_root=checkpoint_root,
                        reporter=reporter,
                    )
                )
        if not objective_results:
            raise CatBoostLTRError(
                "all objective-family profiles are resource infeasible"
            )
        incumbent_result = select_best_result(
            objective_results,
            tie_epsilon=float(source["search"]["tie_epsilon"]),
            order=selection_order,
        )
        incumbent = profiles_by_id[incumbent_result["config_id"]]
        stage_results = [
            {
                "stage_id": source["objective_stage"]["stage_id"],
                "evaluated_config_ids": [row["config_id"] for row in objective_results],
                "resource_infeasible_config_ids": [
                    row["config_id"] for row in runtime_resource_exclusions
                ],
                "incumbent_config_id": incumbent["config_id"],
                "winner": incumbent_result,
            }
        ]
        complete_stage(stage_results[-1])
        if smoke_profile_limit is None:
            for stage in source["inherited_stages"]:
                challengers = inherited_stage_challengers(incumbent, stage)
                candidates = [incumbent_result]
                candidates.extend(evaluate_profile(profile) for profile in challengers)
                incumbent_result = select_best_result(
                    candidates,
                    tie_epsilon=float(source["search"]["tie_epsilon"]),
                    order=selection_order,
                )
                incumbent = profiles_by_id[incumbent_result["config_id"]]
                stage_results.append(
                    {
                        "stage_id": stage["stage_id"],
                        "evaluated_config_ids": [
                            row["config_id"] for row in candidates
                        ],
                        "incumbent_config_id": incumbent["config_id"],
                        "winner": incumbent_result,
                    }
                )
                complete_stage(stage_results[-1])
        winner_model_dir = _model_dir(
            checkpoint_root, incumbent["config_id"], latest_pair_id
        )
        _update_best_model(
            best_root=best_root,
            run_id=run_id,
            digest=digest,
            profile=incumbent,
            aggregate=incumbent_result,
            model_dir=winner_model_dir,
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
        _check_windows_host_disk(source, launch=False)
        pointwise_aggregate = (
            _verify_pinned_pointwise(source, source["fold_pairs"])
            if mode == "full"
            else aggregate_fold_results(
                source["pointwise_comparator"]["config_id"],
                source["pointwise_comparator"]["fold_results"],
            )
        )
        promotion = (
            select_best_result(
                [pointwise_aggregate, incumbent_result],
                tie_epsilon=float(source["search"]["tie_epsilon"]),
                order=[pointwise_aggregate["config_id"], incumbent_result["config_id"]],
            )["config_id"]
            == incumbent_result["config_id"]
        )
        canonical_fold = str(source["canonical"]["fold"])
        canonical_contexts, canonical_features, _, canonical_manifest = (
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
        winner_model = CatBoostRankerModel.from_artifact(winner_model_dir)
        latest_fit = checkpoint.get(
            stage="fit_ltr", config=incumbent["config_id"], fold=latest_pair_id
        )
        if latest_fit is None:
            raise ContractValidationError("latest winner fit checkpoint is missing")
        canonical_metrics = _evaluate_model(
            profile=incumbent,
            pair_id="canonical_once",
            train_fold=next(
                pair.train_fold
                for pair in source["fold_pairs"]
                if pair.pair_id == latest_pair_id
            ),
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
        _check_windows_host_disk(source, launch=False)
        artifact_work = checkpoint_root / "artifact"
        if artifact_work.exists():
            shutil.rmtree(artifact_work)
        artifact_work.mkdir(parents=True)
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
        if recovery_manifest is not None:
            _copy_directory_atomic(
                checkpoint_root / "recovery", artifact_work / "recovery"
            )
            if resource_probe_evidence is None:
                raise ContractValidationError(
                    "recovery run lacks resource-probe evidence"
                )
            write_json_atomic(
                artifact_work / "recovery" / "resource_probe.json",
                resource_probe_evidence,
            )
        if runtime_resource_exclusions:
            _copy_directory_atomic(
                checkpoint_root / "resource_exclusions",
                artifact_work / "resource_exclusions",
            )
        _copy_directory_atomic(winner_model_dir, artifact_work / "model")
        _copy_file_atomic(
            winner_model_dir / "model_config.json",
            artifact_work / "model_config.json",
        )
        winner_importance = (
            checkpoint_root
            / "feature_importance"
            / incumbent["config_id"]
            / f"{latest_pair_id}.parquet"
        )
        _copy_file_atomic(
            winner_importance, artifact_work / "feature_importance.parquet"
        )
        canonical_recommendations = (
            checkpoint_root
            / "results"
            / incumbent["config_id"]
            / "canonical_once"
            / "recommendations.parquet"
        )
        _copy_file_atomic(
            canonical_recommendations,
            artifact_work / "canonical_recommendations.parquet",
        )
        pool_output = artifact_work / "pools"
        for pair in source["fold_pairs"]:
            destination = pool_output / pair.pair_id
            destination.mkdir(parents=True)
            for name in (
                "pool_manifest.json",
                "group_map.parquet",
                "eval_users.parquet",
                "columns.cd",
            ):
                _copy_file_atomic(pair.pool_cache / name, destination / name)
        selection_root = artifact_work / "selection"
        selection_root.mkdir(parents=True)
        write_json_atomic(selection_root / "winner.json", published_winner)
        write_json_atomic(
            selection_root / "stage_results.json", {"stages": stage_results}
        )
        leaderboard = []
        for config_id in selection_order:
            result = results_by_id[config_id]
            leaderboard.append(
                {
                    "config_id": config_id,
                    "objective": CatBoostLTRConfig.from_mapping(
                        result["profile"]["catboost"]
                    ).objective,
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
                }
            )
        _write_parquet_atomic(
            pl.DataFrame(leaderboard), selection_root / "leaderboard.parquet"
        )
        write_json_atomic(artifact_work / "feature_schema.json", feature_schema)
        if mode == "full":
            task09_metrics = read_json(Path(source["task09_artifact"]) / "metrics.json")
            task08_canonical = read_json(
                Path(source["task08_artifact"])
                / "evaluation"
                / "canonical"
                / "metrics.json"
            )
            comparisons: dict[str, Any] = {
                "comparable": True,
                "promoted_over_pointwise_on_rolling": promotion,
                "pointwise_s12_rolling": pointwise_aggregate,
                "winner_minus_pointwise_rolling_p20_all": incumbent_result[
                    "mean_precision_at_20_all_targets"
                ]
                - pointwise_aggregate["mean_precision_at_20_all_targets"],
                "winner_minus_pointwise_rolling_p20_labeled": incumbent_result[
                    "mean_precision_at_20_labeled_users"
                ]
                - pointwise_aggregate["mean_precision_at_20_labeled_users"],
                "task09_s12_canonical": {
                    "precision_at_20_all_targets": task09_metrics[
                        "precision_at_20_all_targets"
                    ],
                    "precision_at_20_labeled_users": task09_metrics[
                        "precision_at_20_labeled_users"
                    ],
                    "final_hits": task09_metrics["final_hits"],
                },
                "task08": task08_canonical["rankers"]["catboost_full"],
                "rrf_full": task08_canonical["rankers"]["rrf_full"],
                "winner_minus_task09_p20_all": canonical_metrics[
                    "precision_at_20_all_targets"
                ]
                - task09_metrics["precision_at_20_all_targets"],
                "winner_minus_task09_p20_labeled": canonical_metrics[
                    "precision_at_20_labeled_users"
                ]
                - task09_metrics["precision_at_20_labeled_users"],
                "winner_minus_task08_p20_all": canonical_metrics[
                    "precision_at_20_all_targets"
                ]
                - task08_canonical["rankers"]["catboost_full"][
                    "precision_at_20_all_targets"
                ],
                "winner_minus_task08_p20_labeled": canonical_metrics[
                    "precision_at_20_labeled_users"
                ]
                - task08_canonical["rankers"]["catboost_full"][
                    "precision_at_20_labeled_users"
                ],
                "winner_minus_rrf_p20_all": canonical_metrics[
                    "precision_at_20_all_targets"
                ]
                - task08_canonical["rankers"]["rrf_full"][
                    "precision_at_20_all_targets"
                ],
                "winner_minus_rrf_p20_labeled": canonical_metrics[
                    "precision_at_20_labeled_users"
                ]
                - task08_canonical["rankers"]["rrf_full"][
                    "precision_at_20_labeled_users"
                ],
            }
        else:
            comparisons = {
                "comparable": False,
                "reason": "smoke uses limited user shards",
                "promoted_over_pointwise_on_rolling": None,
            }
        resolved_config = {
            "artifact_version": LTR_SELECTION_ARTIFACT_VERSION,
            "kind": LTR_SELECTION_ARTIFACT_KIND,
            "run_id": run_id,
            "mode": mode,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": sha256_file(config_source),
            "config_sha256": digest,
            "task07_artifact": dataset_root.as_posix(),
            "task07_schema_sha256": root_manifest["schema_sha256"],
            "fold_pairs": [
                {
                    "pair_id": pair.pair_id,
                    "train_fold": pair.train_fold,
                    "eval_fold": pair.eval_fold,
                    "pool_cache": pair.pool_cache.as_posix(),
                    "borders_source": pair.borders_source.as_posix(),
                }
                for pair in source["fold_pairs"]
            ],
            "pool_manifests": pool_manifests,
            "pool": raw_source["pool"],
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "winner_profile": incumbent,
            "search": raw_source["search"],
            "objective_exclusions": raw_source["objective_exclusions"],
            "recovery": raw_source.get("recovery"),
            "recovery_import": recovery_manifest,
            "resource_probe": resource_probe_evidence,
            "runtime_resource_exclusions": runtime_resource_exclusions,
            "canonical": raw_source["canonical"],
            "pointwise_comparator": raw_source["pointwise_comparator"],
            "canonical_isolation": {
                "selection_folds": folds,
                "canonical_loaded_after_winner_freeze": True,
                "canonical_evaluated_config_count": 1,
                "canonical_schema_sha256": canonical_manifest["schema_sha256"],
            },
            "library_versions": {
                "catboost": catboost.__version__,
                "polars": pl.__version__,
                "python": platform.python_version(),
            },
            "resources": resource_preflight,
            "smoke_part_limit": smoke_part_limit,
            "smoke_profile_limit": smoke_profile_limit,
            "cleanup_policy": {
                "recoverable_state_on_success": cleanup_recoverable_on_success,
                "retained": ["published_artifact", "reusable_grouped_pools"],
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
            "runtime_resource_exclusions": runtime_resource_exclusions,
            "resource_probe": resource_probe_evidence,
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
            "model_sha256": sha256_file(artifact_work / "model" / "model.cbm"),
            "feature_importance_sha256": sha256_file(
                artifact_work / "feature_importance.parquet"
            ),
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        write_json_atomic(artifact_work / "metrics.json", metrics)
        _write_checksums(artifact_work)
        publish_directory_atomic(artifact_work, output)
        verify_ltr_artifact(output)
        reporter.phase_finish(
            "publish_selection_artifact",
            phase,
            artifact=output.as_posix(),
            p20_labeled=metrics["precision_at_20_labeled_users"],
        )

        phase = reporter.phase_start("cleanup_recoverable_state")
        cleanup_summary = None
        if cleanup_recoverable_on_success:
            cleanup_summary = _cleanup(
                checkpoint_root=checkpoint_root,
                best_root=best_root,
                output=output,
                pool_dirs=[pair.pool_cache for pair in source["fold_pairs"]],
                input_dirs=[
                    dataset_root,
                    Path(source["task06_rrf_artifact"]),
                    Path(source["task08_artifact"]),
                    Path(source["task09_artifact"]),
                ],
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
            "Run bounded group-aware CatBoost selection on two rolling Task 07 "
            "fold pairs, then evaluate one frozen winner on canonical."
        )
    )
    parser.add_argument("--config", default="configs/task10_catboost_ltr_v1.json")
    parser.add_argument("--output-dir", default="artifacts/task10_catboost_ltr_v1")
    parser.add_argument("--run-id", default="task10_catboost_ltr_v1")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/.task10_catboost_ltr_v1.checkpoint"
    )
    parser.add_argument(
        "--best-model-dir", default="artifacts/.task10_catboost_ltr_v1.best-model"
    )
    parser.add_argument("--log-file", default="logs/task10_catboost_ltr_v1.log")
    parser.add_argument(
        "--smoke-part-limit",
        type=int,
        default=None,
        help="Use the first N complete user shards per fold; never a full experiment.",
    )
    parser.add_argument(
        "--smoke-profile-limit",
        type=int,
        default=None,
        help="Evaluate only the first N objective profiles in a smoke run.",
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
    parser.add_argument(
        "--verify-only",
        metavar="ARTIFACT",
        help="Validate a published Task 10 artifact without fitting.",
    )
    parser.add_argument("--fit-worker-spec", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.fit_worker_spec:
        _fit_worker(args.fit_worker_spec)
        return
    if args.verify_only:
        print(
            json.dumps(
                verify_ltr_artifact(args.verify_only),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_graceful_interrupt)
    try:
        metrics = run_ltr_selection(
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            checkpoint_dir=args.checkpoint_dir,
            best_model_dir=args.best_model_dir,
            log_file=args.log_file,
            smoke_part_limit=args.smoke_part_limit,
            smoke_profile_limit=args.smoke_profile_limit,
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

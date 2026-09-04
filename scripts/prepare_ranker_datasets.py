#!/usr/bin/env python3
"""Materialize task-07 fold-specific, CatBoost-ready ranker datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import implicit
import numpy as np
import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment_utils import (
    CheckpointStore,
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from features import (
    HistoryFeatureConfig,
    build_als_factor_norm_lookups,
    build_item_features,
    build_user_features,
    validate_history_cutoff,
)
from implicit_model import ImplicitALSModel
from item2item import Item2ItemConfig
from ranker_data import (
    NegativeSamplingConfig,
    build_ranker_shard,
    feature_column_names,
    ranker_dataset_diagnostics,
)
from validation import ContractValidationError


class RankerDatasetPreparationError(ValueError):
    """Raised when task-07 config or upstream artifacts are incompatible."""


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RankerDatasetPreparationError(f"{name} must be a positive integer")
    return value


def _peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _write_parquet_atomic(
    frame: pl.DataFrame,
    path: Path,
    *,
    compression: str,
    compression_level: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        frame.write_parquet(
            temporary,
            compression=compression,
            compression_level=compression_level,
            statistics=True,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_config(
    source: Mapping[str, Any],
) -> tuple[
    Path,
    tuple[str, ...],
    HistoryFeatureConfig,
    NegativeSamplingConfig,
    tuple[str, ...],
    str,
    int | None,
]:
    task06_value = source.get("task06_artifact")
    if not isinstance(task06_value, str) or not task06_value:
        raise RankerDatasetPreparationError("task06_artifact must be a path")
    task06 = Path(task06_value)
    if not task06.is_dir():
        raise FileNotFoundError(task06)
    fold_value = source.get(
        "fold_order", ["rolling_1", "rolling_2", "rolling_3", "canonical"]
    )
    if not isinstance(fold_value, list) or not all(
        isinstance(value, str) and value for value in fold_value
    ):
        raise RankerDatasetPreparationError("fold_order must be a string list")
    folds = tuple(fold_value)
    if folds != ("rolling_1", "rolling_2", "rolling_3", "canonical"):
        raise RankerDatasetPreparationError(
            "fold_order must contain rolling_1..3 followed by canonical"
        )
    history_config = HistoryFeatureConfig.from_mapping(
        source.get("history_features", {})
    )
    sampling_config = NegativeSamplingConfig.from_mapping(
        source.get("negative_sampling", {})
    )
    rank_sources_value = source.get(
        "union_rank_sources", ["item2item", "implicit_als"]
    )
    if not isinstance(rank_sources_value, list) or not rank_sources_value:
        raise RankerDatasetPreparationError("union_rank_sources must be a list")
    rank_sources = tuple(str(value) for value in rank_sources_value)
    allowed_sources = {"global_popularity", "recency_popularity", "item2item", "implicit_als"}
    if not set(rank_sources) <= allowed_sources or len(set(rank_sources)) != len(
        rank_sources
    ):
        raise RankerDatasetPreparationError("invalid union_rank_sources")
    parquet = source.get("parquet", {})
    if not isinstance(parquet, Mapping):
        raise RankerDatasetPreparationError("parquet must be an object")
    compression = parquet.get("compression", "zstd")
    if compression not in {"zstd", "snappy", "lz4", "uncompressed"}:
        raise RankerDatasetPreparationError("unsupported Parquet compression")
    level = parquet.get("compression_level", 3)
    if level is not None and (isinstance(level, bool) or not isinstance(level, int)):
        raise RankerDatasetPreparationError("compression_level must be integer or null")
    return (
        task06,
        folds,
        history_config,
        sampling_config,
        rank_sources,
        compression,
        level,
    )


def _source_model_path(source_dir: Path, metadata: Mapping[str, Any]) -> Path:
    storage = metadata.get("model_storage")
    value = metadata.get("model_path")
    if storage not in {"embedded", "reference"} or not isinstance(value, str):
        raise RankerDatasetPreparationError(
            f"invalid source model metadata: {source_dir}"
        )
    path = source_dir / value if storage == "embedded" else Path(value)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _fold_source_context(task06: Path, fold: str) -> dict[str, Any]:
    fold_root = task06 / "folds" / fold
    manifest_path = fold_root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("fold") != fold:
        raise RankerDatasetPreparationError(f"fold manifest mismatch: {fold}")
    parts = manifest.get("parts")
    if not isinstance(parts, Mapping) or not parts:
        raise RankerDatasetPreparationError(f"fold has no task06 parts: {fold}")
    ordered_parts = tuple(sorted(str(name) for name in parts))
    if int(manifest.get("part_count", -1)) != len(ordered_parts):
        raise RankerDatasetPreparationError(f"part count differs for {fold}")
    history_value = manifest.get("history_path")
    if not isinstance(history_value, str):
        raise RankerDatasetPreparationError(f"history path is missing for {fold}")
    history = Path(history_value)
    if not history.is_file():
        raise FileNotFoundError(history)
    try:
        cutoff = datetime.fromisoformat(str(manifest["cutoff"]))
    except (KeyError, ValueError) as error:
        raise RankerDatasetPreparationError(f"invalid cutoff for {fold}") from error
    target_users = fold_root / "target_users.parquet"
    ground_truth = fold_root / "target_ground_truth.parquet"
    if not target_users.is_file() or not ground_truth.is_file():
        raise FileNotFoundError(f"task06 evaluation files are missing for {fold}")
    if sha256_file(target_users) != manifest.get("target_users_sha256"):
        raise RankerDatasetPreparationError(
            f"task06 target-user checksum differs for {fold}"
        )
    if sha256_file(ground_truth) != manifest.get("target_ground_truth_sha256"):
        raise RankerDatasetPreparationError(
            f"task06 ground-truth checksum differs for {fold}"
        )
    item2item_dir = fold_root / "sources" / "item2item"
    als_dir = fold_root / "sources" / "implicit_als"
    item2item_metadata = read_json(item2item_dir / "metadata.json")
    als_metadata = read_json(als_dir / "metadata.json")
    item2item_model = _source_model_path(item2item_dir, item2item_metadata)
    als_model = _source_model_path(als_dir, als_metadata)
    seeds = item2item_dir / "seeds.parquet"
    neighbors = item2item_model / "neighbor_table.parquet"
    if not seeds.is_file() or not neighbors.is_file():
        raise FileNotFoundError(f"item2item feature inputs are missing for {fold}")
    item2item_config_source = item2item_metadata.get("model_config", {}).get(
        "item2item_config"
    )
    if not isinstance(item2item_config_source, Mapping):
        raise RankerDatasetPreparationError(
            f"item2item config is missing for {fold}"
        )
    return {
        "fold": fold,
        "fold_root": fold_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "parts": ordered_parts,
        "part_hashes": dict(parts),
        "history": history,
        "cutoff": cutoff,
        "target_users": target_users,
        "ground_truth": ground_truth,
        "seeds": seeds,
        "neighbors": neighbors,
        "item2item_config": Item2ItemConfig.from_dict(item2item_config_source),
        "als_model": als_model,
        "item2item_model": item2item_model,
    }


def _limited_context(
    context: Mapping[str, Any],
    *,
    smoke_user_limit: int | None,
    task06_users_per_shard: int,
) -> dict[str, Any]:
    target_users = pl.read_parquet(context["target_users"]).sort("user_id")
    ground_truth = pl.read_parquet(context["ground_truth"])
    part_names = tuple(context["parts"])
    if smoke_user_limit is None:
        return {
            **context,
            "evaluation_users": target_users,
            "evaluation_ground_truth": ground_truth,
            "selected_parts": part_names,
        }
    evaluation_users = target_users.head(smoke_user_limit)
    evaluation_ground_truth = ground_truth.join(
        evaluation_users, on="user_id", how="semi"
    )
    needed_parts = min(
        len(part_names),
        (evaluation_users.height + task06_users_per_shard - 1)
        // task06_users_per_shard,
    )
    return {
        **context,
        "evaluation_users": evaluation_users,
        "evaluation_ground_truth": evaluation_ground_truth,
        "selected_parts": part_names[:needed_parts],
    }


def _part_source(context: Mapping[str, Any], part_name: str) -> Path:
    return context["fold_root"] / "union_features" / part_name


def _read_union_part(
    context: Mapping[str, Any], part_name: str
) -> pl.DataFrame:
    part = _part_source(context, part_name)
    if not part.is_file():
        raise FileNotFoundError(part)
    frame = pl.read_parquet(part)
    users = context["evaluation_users"]
    if users.height != int(context["manifest"]["target_users"]):
        frame = frame.join(users, on="user_id", how="semi")
    return frame.sort(("user_id", "item_id"))


def _smoke_candidate_items(context: Mapping[str, Any]) -> pl.DataFrame:
    items = [
        _read_union_part(context, name).select("item_id")
        for name in context["selected_parts"]
    ]
    return pl.concat(items).unique().sort("item_id")


def _file_set_sha256(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def _verify_files(paths: Mapping[str, Path], expected: Mapping[str, Any]) -> None:
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected.get(name):
            raise RankerDatasetPreparationError(
                f"completed checkpoint file is missing or corrupt: {path}"
            )


def _prepare_fold_lookups(
    context: Mapping[str, Any],
    *,
    destination: Path,
    history_config: HistoryFeatureConfig,
    compression: str,
    compression_level: int | None,
    smoke_user_limit: int | None,
) -> dict[str, Any]:
    cutoff = context["cutoff"]
    history = context["history"]
    history_sha256 = sha256_file(history)
    if history_sha256 != context.get("history_sha256_expected"):
        raise RankerDatasetPreparationError(
            f"history checksum differs for {context['fold']}"
        )
    history_diagnostics = validate_history_cutoff(history, cutoff=cutoff)
    history_scan = pl.scan_parquet(history)
    if smoke_user_limit is None:
        user_source: pl.LazyFrame = history_scan
        item_source: pl.LazyFrame = history_scan
    else:
        users = context["evaluation_users"].lazy()
        candidate_items = _smoke_candidate_items(context).lazy()
        user_source = history_scan.join(users, on="user_id", how="semi")
        item_source = history_scan.join(candidate_items, on="item_id", how="semi")
    user_features = build_user_features(
        user_source, cutoff=cutoff, config=history_config
    )
    item_features = build_item_features(
        item_source, cutoff=cutoff, config=history_config
    )
    model = ImplicitALSModel.from_artifact(context["als_model"])
    als_user_norms, als_item_norms = build_als_factor_norm_lookups(model)
    del model
    if smoke_user_limit is not None:
        als_user_norms = als_user_norms.join(
            context["evaluation_users"], on="user_id", how="semi"
        )
        candidate_items = _smoke_candidate_items(context)
        als_item_norms = als_item_norms.join(
            candidate_items, on="item_id", how="semi"
        )
    files = {
        "user_features.parquet": destination / "user_features.parquet",
        "item_features.parquet": destination / "item_features.parquet",
        "als_user_norms.parquet": destination / "als_user_norms.parquet",
        "als_item_norms.parquet": destination / "als_item_norms.parquet",
    }
    for frame, path in (
        (user_features, files["user_features.parquet"]),
        (item_features, files["item_features.parquet"]),
        (als_user_norms, files["als_user_norms.parquet"]),
        (als_item_norms, files["als_item_norms.parquet"]),
    ):
        _write_parquet_atomic(
            frame,
            path,
            compression=compression,
            compression_level=compression_level,
        )
    input_model_files = {
        "item2item_neighbor_table": context["neighbors"],
        "item2item_seeds": context["seeds"],
        "als_model": context["als_model"] / "als_model.npz",
        "als_user_mapping": context["als_model"] / "user_mapping.parquet",
        "als_item_mapping": context["als_model"] / "item_mapping.parquet",
        "als_config": context["als_model"] / "model_config.json",
    }
    metadata = {
        "history": history_diagnostics,
        "history_sha256": history_sha256,
        "history_feature_config": history_config.to_dict(),
        "rows": {
            "user_features": user_features.height,
            "item_features": item_features.height,
            "als_user_norms": als_user_norms.height,
            "als_item_norms": als_item_norms.height,
        },
        "files": _file_set_sha256(files),
        "input_models": {
            "item2item_model_path": context["item2item_model"].as_posix(),
            "als_model_path": context["als_model"].as_posix(),
            "sha256": _file_set_sha256(input_model_files),
        },
    }
    write_json_atomic(destination / "lookup_manifest.json", metadata)
    metadata["manifest_sha256"] = sha256_file(
        destination / "lookup_manifest.json"
    )
    return metadata


def _load_fold_lookups(root: Path) -> dict[str, pl.DataFrame]:
    return {
        "user_features": pl.read_parquet(root / "user_features.parquet"),
        "item_features": pl.read_parquet(root / "item_features.parquet"),
        "als_user_norms": pl.read_parquet(root / "als_user_norms.parquet"),
        "als_item_norms": pl.read_parquet(root / "als_item_norms.parquet"),
    }


def _schema_payload(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {"name": name, "dtype": str(dtype)}
            for name, dtype in frame.schema.items()
        ],
        "feature_columns": feature_column_names(frame),
        "column_count": len(frame.columns),
        "feature_count": len(feature_column_names(frame)),
    }


def _aggregate_part_diagnostics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "rows",
        "users",
        "positive_rows",
        "negative_rows",
        "hard_negative_rows",
        "training_rows",
        "training_positive_rows",
        "training_negative_rows",
    )
    result = {field: sum(int(record[field]) for record in records) for field in fields}
    rows = result["rows"]
    training = result["training_rows"]
    result["positive_rate"] = result["positive_rows"] / rows if rows else 0.0
    result["training_positive_rate"] = (
        result["training_positive_rows"] / training if training else 0.0
    )
    return result


def _artifact_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_preparation(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    checkpoint_dir: str | Path,
    log_file: str | Path | None,
    show_progress: bool,
    smoke_user_limit: int | None = None,
    smoke_fold_limit: int | None = None,
) -> dict[str, Any]:
    """Build all configured fold datasets without running candidate models."""

    started = time.perf_counter()
    config_source = Path(config_path)
    output = Path(output_dir)
    checkpoint_root = Path(checkpoint_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    if not run_id:
        raise RankerDatasetPreparationError("run_id must be non-empty")
    if smoke_user_limit is not None:
        _positive_int(smoke_user_limit, name="smoke_user_limit")
        smoke_fold_limit = 1 if smoke_fold_limit is None else smoke_fold_limit
    if smoke_fold_limit is not None:
        _positive_int(smoke_fold_limit, name="smoke_fold_limit")
        if smoke_user_limit is None:
            raise RankerDatasetPreparationError(
                "smoke_fold_limit requires smoke_user_limit"
            )
    if smoke_user_limit is not None and output == Path(
        "artifacts/task07_ranker_dataset_v1"
    ):
        raise RankerDatasetPreparationError(
            "limited smoke cannot publish to the default full artifact path"
        )
    source = read_json(config_source)
    (
        task06,
        configured_folds,
        history_config,
        sampling_config,
        rank_sources,
        compression,
        compression_level,
    ) = _validate_config(source)
    task06_config_path = task06 / "config.json"
    task06_config = read_json(task06_config_path)
    if task06_config.get("kind") != "task06_offline_candidate_datasets":
        raise RankerDatasetPreparationError("task06 artifact kind is invalid")
    users_per_shard = _positive_int(
        task06_config.get("users_per_shard"), name="task06 users_per_shard"
    )
    folds = (
        configured_folds[:smoke_fold_limit]
        if smoke_fold_limit is not None
        else configured_folds
    )
    base_contexts = {
        fold: _fold_source_context(task06, fold) for fold in folds
    }
    for fold, context in base_contexts.items():
        try:
            expected_history_sha = task06_config["folds"][fold]["output_sha256"][
                "history_daily.parquet"
            ]
        except (KeyError, TypeError) as error:
            raise RankerDatasetPreparationError(
                f"task06 config lacks history checksum for {fold}"
            ) from error
        context["history_sha256_expected"] = expected_history_sha
    input_digest = config_sha256(
        {
            "task07_config": source,
            "task06_config_sha256": sha256_file(task06_config_path),
            "fold_manifest_sha256": {
                fold: sha256_file(context["manifest_path"])
                for fold, context in base_contexts.items()
            },
            "smoke_user_limit": smoke_user_limit,
            "smoke_fold_limit": smoke_fold_limit,
        }
    )
    store = CheckpointStore(
        checkpoint_root, run_id=run_id, config_digest=input_digest
    )
    work = checkpoint_root / "artifact"
    work.mkdir(parents=True, exist_ok=True)
    contexts = {
        fold: _limited_context(
            context,
            smoke_user_limit=smoke_user_limit,
            task06_users_per_shard=users_per_shard,
        )
        for fold, context in base_contexts.items()
    }
    total_shards = sum(len(context["selected_parts"]) for context in contexts.values())
    expected_total_rows = (
        sum(int(context["manifest"]["rows"]) for context in contexts.values())
        if smoke_user_limit is None
        else None
    )
    reporter = EventProgressReporter(
        task_name="task07-prepare",
        total_phases=3,
        log_file=log_file,
        show_progress=show_progress,
    )
    mode = "limited_smoke" if smoke_user_limit is not None else "full"
    reporter.event(
        "run_start",
        run_id=run_id,
        config=config_source.as_posix(),
        task06_artifact=task06.as_posix(),
        output=output.as_posix(),
        checkpoint=checkpoint_root.as_posix(),
        mode=mode,
        folds=list(folds),
        shards=total_shards,
        pid=os.getpid(),
    )
    runtime_by_fold: dict[str, dict[str, float]] = {}
    try:
        lookup_phase = reporter.phase_start("fold_feature_lookups")
        lookup_stage = reporter.stage_start(
            stage="fold_feature_lookups", total=len(folds), unit="fold"
        )
        for fold, context in contexts.items():
            fold_started = time.perf_counter()
            destination = work / "folds" / fold / "lookups"
            record = store.get(
                stage="fold_feature_lookups", config="history_and_als", fold=fold
            )
            files = {
                "user_features.parquet": destination / "user_features.parquet",
                "item_features.parquet": destination / "item_features.parquet",
                "als_user_norms.parquet": destination / "als_user_norms.parquet",
                "als_item_norms.parquet": destination / "als_item_norms.parquet",
            }
            if record is not None:
                _verify_files(files, record.get("files", {}))
                if not (destination / "lookup_manifest.json").is_file():
                    raise RankerDatasetPreparationError(
                        f"lookup manifest is missing for {fold}"
                    )
                if sha256_file(destination / "lookup_manifest.json") != record.get(
                    "manifest_sha256"
                ):
                    raise RankerDatasetPreparationError(
                        f"lookup manifest is corrupt for {fold}"
                    )
                reporter.event(
                    "operation_resume_skip",
                    stage="fold_feature_lookups",
                    config="history_and_als",
                    fold=fold,
                )
            else:
                operation_started = reporter.operation_start(
                    stage="fold_feature_lookups",
                    config="history_and_als",
                    fold=fold,
                    operation="aggregate_and_factor_norms",
                )
                metadata = _prepare_fold_lookups(
                    context,
                    destination=destination,
                    history_config=history_config,
                    compression=compression,
                    compression_level=compression_level,
                    smoke_user_limit=smoke_user_limit,
                )
                store.complete(
                    stage="fold_feature_lookups",
                    config="history_and_als",
                    fold=fold,
                    metadata={
                        "files": metadata["files"],
                        "rows": metadata["rows"],
                        "manifest_sha256": metadata["manifest_sha256"],
                    },
                )
                reporter.operation_finish(
                    stage="fold_feature_lookups",
                    config="history_and_als",
                    fold=fold,
                    operation="aggregate_and_factor_norms",
                    started=operation_started,
                    user_rows=metadata["rows"]["user_features"],
                    item_rows=metadata["rows"]["item_features"],
                )
            reporter.stage_advance()
            runtime_by_fold.setdefault(fold, {})["lookups_seconds"] = (
                time.perf_counter() - fold_started
            )
            gc.collect()
        reporter.stage_finish(stage="fold_feature_lookups", started=lookup_stage)
        reporter.phase_finish("fold_feature_lookups", lookup_phase)

        shard_phase = reporter.phase_start("ranker_shard_materialization")
        shard_stage_started = reporter.stage_start(
            stage="ranker_shard_materialization", total=total_shards, unit="shard"
        )
        progress_started = time.perf_counter()
        completed_shards = 0
        completed_rows = 0
        completed_bytes = 0
        fold_manifests: dict[str, Any] = {}
        common_schema: dict[str, Any] | None = None
        for fold, context in contexts.items():
            fold_started = time.perf_counter()
            lookups = _load_fold_lookups(work / "folds" / fold / "lookups")
            all_seeds = pl.read_parquet(context["seeds"])
            ground_truth = context["evaluation_ground_truth"]
            target_output = work / "folds" / fold / "target_users.parquet"
            gt_output = work / "folds" / fold / "target_ground_truth.parquet"
            if not target_output.exists():
                _write_parquet_atomic(
                    context["evaluation_users"],
                    target_output,
                    compression=compression,
                    compression_level=compression_level,
                )
            elif not pl.read_parquet(target_output).equals(
                context["evaluation_users"]
            ):
                raise RankerDatasetPreparationError(
                    f"persisted target users differ for {fold}"
                )
            if not gt_output.exists():
                _write_parquet_atomic(
                    ground_truth,
                    gt_output,
                    compression=compression,
                    compression_level=compression_level,
                )
            elif not pl.read_parquet(gt_output).equals(ground_truth):
                raise RankerDatasetPreparationError(
                    f"persisted ground truth differs for {fold}"
                )
            part_records: list[Mapping[str, Any]] = []
            part_hashes: dict[str, str] = {}
            input_hashes: dict[str, str] = {}
            for part_name in context["selected_parts"]:
                shard_label = f"{fold}:{part_name.removesuffix('.parquet')}"
                output_part = work / "folds" / fold / "ranker_data" / part_name
                record = store.get(
                    stage="ranker_shard_materialization",
                    config="full_union_features",
                    fold=shard_label,
                )
                if record is not None:
                    if not output_part.is_file() or sha256_file(
                        output_part
                    ) != record.get("sha256"):
                        raise RankerDatasetPreparationError(
                            f"completed ranker shard is corrupt: {output_part}"
                        )
                    reporter.event(
                        "operation_resume_skip",
                        stage="ranker_shard_materialization",
                        config="full_union_features",
                        fold=shard_label,
                    )
                else:
                    input_part = _part_source(context, part_name)
                    input_sha = sha256_file(input_part)
                    expected_sha = context["part_hashes"].get(part_name)
                    if input_sha != expected_sha:
                        raise RankerDatasetPreparationError(
                            f"task06 input checksum differs: {input_part}"
                        )
                    operation_started = reporter.operation_start(
                        stage="ranker_shard_materialization",
                        config="full_union_features",
                        fold=shard_label,
                        operation="join_features_labels_and_sampling",
                    )
                    union = _read_union_part(context, part_name)
                    shard_users = union.select("user_id").unique()
                    shard_seeds = all_seeds.join(
                        shard_users, on="user_id", how="semi"
                    )
                    dataset = build_ranker_shard(
                        union,
                        ground_truth=ground_truth,
                        user_features=lookups["user_features"],
                        item_features=lookups["item_features"],
                        item2item_seeds=shard_seeds,
                        item2item_neighbors=context["neighbors"],
                        item2item_config=context["item2item_config"],
                        als_user_norms=lookups["als_user_norms"],
                        als_item_norms=lookups["als_item_norms"],
                        cutoff=context["cutoff"],
                        fold=fold,
                        sampling_config=sampling_config,
                        rank_sources=rank_sources,
                    )
                    schema = _schema_payload(dataset)
                    if common_schema is None:
                        common_schema = schema
                    elif schema != common_schema:
                        raise ContractValidationError(
                            f"ranker feature schema differs in {shard_label}"
                        )
                    diagnostics = ranker_dataset_diagnostics(dataset)
                    _write_parquet_atomic(
                        dataset,
                        output_part,
                        compression=compression,
                        compression_level=compression_level,
                    )
                    output_sha = sha256_file(output_part)
                    record = {
                        **diagnostics,
                        "sha256": output_sha,
                        "input_sha256": input_sha,
                        "bytes": output_part.stat().st_size,
                    }
                    store.complete(
                        stage="ranker_shard_materialization",
                        config="full_union_features",
                        fold=shard_label,
                        metadata=record,
                    )
                    reporter.operation_finish(
                        stage="ranker_shard_materialization",
                        config="full_union_features",
                        fold=shard_label,
                        operation="join_features_labels_and_sampling",
                        started=operation_started,
                        rows=diagnostics["rows"],
                        positives=diagnostics["positive_rows"],
                        training_rows=diagnostics["training_rows"],
                        output_bytes=record["bytes"],
                    )
                    del union, shard_seeds, dataset
                    gc.collect()
                part_records.append(record)
                part_hashes[part_name] = str(record["sha256"])
                input_hashes[part_name] = str(record["input_sha256"])
                completed_shards += 1
                completed_rows += int(record["rows"])
                completed_bytes += int(record["bytes"])
                elapsed = time.perf_counter() - progress_started
                rate = completed_shards / elapsed if elapsed > 0 else 0.0
                remaining = total_shards - completed_shards
                eta = remaining / rate if rate > 0 else None
                projected_bytes = (
                    completed_bytes / completed_rows * expected_total_rows
                    if expected_total_rows is not None and completed_rows > 0
                    else None
                )
                reporter.stage_advance()
                reporter.event(
                    "stage_progress",
                    stage="ranker_shard_materialization",
                    config="full_union_features",
                    fold=fold,
                    operation=part_name,
                    current=completed_shards,
                    total=total_shards,
                    progress=completed_shards / total_shards,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                    bytes_per_row=(
                        completed_bytes / completed_rows if completed_rows else None
                    ),
                    projected_output_bytes=projected_bytes,
                )
            first_part = (
                work
                / "folds"
                / fold
                / "ranker_data"
                / context["selected_parts"][0]
            )
            fold_schema = _schema_payload(pl.read_parquet(first_part, n_rows=1))
            if common_schema is None:
                common_schema = fold_schema
            elif fold_schema != common_schema:
                raise ContractValidationError(f"ranker schema differs for {fold}")
            aggregate = _aggregate_part_diagnostics(part_records)
            if smoke_user_limit is None and aggregate["rows"] != int(
                context["manifest"]["rows"]
            ):
                raise ContractValidationError(
                    f"ranker rows differ from task06 union for {fold}"
                )
            lookup_manifest = read_json(
                work / "folds" / fold / "lookups" / "lookup_manifest.json"
            )
            fold_manifest = {
                "fold": fold,
                "cutoff": context["cutoff"].isoformat(),
                "history_path": context["history"].as_posix(),
                "history_diagnostics": lookup_manifest["history"],
                "task06_manifest_path": context["manifest_path"].as_posix(),
                "task06_manifest_sha256": sha256_file(context["manifest_path"]),
                "target_users": context["evaluation_users"].height,
                "target_ground_truth_pairs": ground_truth.height,
                "target_users_sha256": sha256_file(target_output),
                "target_ground_truth_sha256": sha256_file(gt_output),
                "part_count": len(context["selected_parts"]),
                "input_parts": input_hashes,
                "parts": part_hashes,
                "diagnostics": aggregate,
                "lookup_manifest": "lookups/lookup_manifest.json",
                "lookup_manifest_sha256": sha256_file(
                    work / "folds" / fold / "lookups" / "lookup_manifest.json"
                ),
                "schema_sha256": config_sha256(common_schema),
            }
            write_json_atomic(
                work / "folds" / fold / "fold_manifest.json", fold_manifest
            )
            fold_manifests[fold] = fold_manifest
            runtime_by_fold.setdefault(fold, {})["shards_seconds"] = (
                time.perf_counter() - fold_started
            )
            del lookups, all_seeds
            gc.collect()
        reporter.stage_finish(
            stage="ranker_shard_materialization", started=shard_stage_started
        )
        reporter.phase_finish(
            "ranker_shard_materialization", shard_phase, shards=total_shards
        )

        publish_phase = reporter.phase_start("validate_and_publish")
        if common_schema is None:
            raise RuntimeError("ranker dataset schema was not produced")
        schema_digest = config_sha256(common_schema)
        write_json_atomic(work / "feature_schema.json", common_schema)
        root_manifest = {
            "run_id": run_id,
            "artifact_version": 1,
            "kind": "task07_ranker_datasets",
            "mode": mode,
            "fold_order": list(folds),
            "schema_sha256": schema_digest,
            "folds": {
                fold: {
                    "manifest": f"folds/{fold}/fold_manifest.json",
                    "manifest_sha256": sha256_file(
                        work / "folds" / fold / "fold_manifest.json"
                    ),
                    "rows": fold_manifests[fold]["diagnostics"]["rows"],
                    "positive_rows": fold_manifests[fold]["diagnostics"][
                        "positive_rows"
                    ],
                    "training_rows": fold_manifests[fold]["diagnostics"][
                        "training_rows"
                    ],
                }
                for fold in folds
            },
        }
        write_json_atomic(work / "dataset_manifest.json", root_manifest)
        resolved_config = {
            "run_id": run_id,
            "artifact_version": 1,
            "kind": "task07_ranker_datasets",
            "mode": mode,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": config_sha256(source),
            "task06_artifact": task06.as_posix(),
            "task06_config_sha256": sha256_file(task06_config_path),
            "fold_order": list(folds),
            "history_features": history_config.to_dict(),
            "negative_sampling": sampling_config.to_dict(),
            "union_rank_sources": list(rank_sources),
            "parquet": {
                "compression": compression,
                "compression_level": compression_level,
            },
            "smoke_user_limit": smoke_user_limit,
            "smoke_fold_limit": smoke_fold_limit,
            "candidate_models_invoked": False,
            "library_versions": {
                "python": platform.python_version(),
                "polars": pl.__version__,
                "numpy": np.__version__,
                "implicit": implicit.__version__,
            },
        }
        write_json_atomic(work / "config.json", resolved_config)
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "folds": {
                fold: fold_manifests[fold]["diagnostics"] for fold in folds
            },
            "runtime_by_fold_seconds": runtime_by_fold,
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
            "artifact_size_bytes": 0,
        }
        write_json_atomic(work / "metrics.json", metrics)
        metrics["artifact_size_bytes"] = _artifact_size_bytes(work)
        write_json_atomic(work / "metrics.json", metrics)
        publish_directory_atomic(work, output)
        reporter.phase_finish(
            "validate_and_publish",
            publish_phase,
            output=output.as_posix(),
            artifact_size_bytes=metrics["artifact_size_bytes"],
        )
        reporter.event(
            "run_finish",
            run_id=run_id,
            status="completed",
            duration_seconds=time.perf_counter() - started,
            output=output.as_posix(),
        )
        return metrics
    except BaseException as error:
        reporter.event(
            "run_finish",
            run_id=run_id,
            status="failed",
            duration_seconds=time.perf_counter() - started,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        reporter.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize fold-specific labels and history-only ranker features "
            "from immutable task06 candidate unions."
        )
    )
    parser.add_argument("--config", default="configs/task07_ranker_dataset_v1.json")
    parser.add_argument("--output-dir", default="artifacts/task07_ranker_dataset_v1")
    parser.add_argument("--run-id", default="task07_ranker_dataset_v1")
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/.task07_ranker_dataset_v1.checkpoint",
    )
    parser.add_argument("--log-file", default="logs/task07_ranker_dataset_v1.log")
    parser.add_argument("--smoke-user-limit", type=int)
    parser.add_argument("--smoke-fold-limit", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run_preparation(
        config_path=args.config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file or None,
        show_progress=not args.no_progress,
        smoke_user_limit=args.smoke_user_limit,
        smoke_fold_limit=args.smoke_fold_limit,
    )
    print(
        json.dumps(
            {
                "run_id": metrics["run_id"],
                "mode": metrics["mode"],
                "runtime_seconds": metrics["runtime_seconds"],
                "peak_memory_mb": metrics["peak_memory_mb"],
                "artifact_size_bytes": metrics["artifact_size_bytes"],
                "folds": metrics["folds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

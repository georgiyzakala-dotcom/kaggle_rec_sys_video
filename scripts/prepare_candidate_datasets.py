#!/usr/bin/env python3
"""Materialize task-06 source candidates and cross-scored union datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
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
import scipy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from candidate_pipeline import (
    CandidateUnionConfig,
    attach_implicit_als_cross_scores,
    attach_item2item_cross_scores,
    attach_ranking_cross_scores,
    build_candidate_union,
    iter_target_user_shards,
    run_candidate_model,
    validate_union_against_history,
    validate_union_features,
)
from experiment_utils import (
    CheckpointStore,
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from implicit_model import ImplicitALSConfig, ImplicitALSDataLoader, ImplicitALSModel
from item2item import Item2ItemConfig, Item2ItemDataLoader, Item2ItemModel
from metrics import evaluate_candidate_metrics, evaluate_candidate_metrics_lazy
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
)
from scripts.run_global_popularity import (
    _load_evaluation_frames,
    _peak_memory_mb,
    _validate_fold,
)
from scripts.run_item2item import _limited_history
from validation import validate_candidate_output, validate_model_config

SOURCE_ORDER = (
    "global_popularity",
    "recency_popularity",
    "item2item",
    "implicit_als",
)


class CandidateDatasetPreparationError(ValueError):
    """Raised when the task-06 materialization config is invalid."""


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateDatasetPreparationError(f"{name} must be positive")
    return value


def _recency_grids(
    config: RecencyPopularityConfig,
) -> tuple[list[float], list[float]]:
    value = config.to_dict()
    windows: list[float] = []
    half_lives: list[float] = []
    for key in ("window_hours", "short_window_hours", "long_window_hours"):
        item = value.get(key)
        if item is not None:
            windows.append(float(item))
    for entry in value.get("window_weights", []):
        window = entry.get("window_hours")
        if window is not None:
            windows.append(float(window))
    if value.get("half_life_hours") is not None:
        half_lives.append(float(value["half_life_hours"]))
    return sorted(set(windows)), sorted(set(half_lives))


def _winner_configs(source: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = source.get("winner_artifacts")
    if not isinstance(artifacts, Mapping):
        raise CandidateDatasetPreparationError("winner_artifacts must be an object")
    required = {"task02", "task03", "task04", "task05"}
    if set(artifacts) != required:
        raise CandidateDatasetPreparationError(
            "winner_artifacts must contain task02, task03, task04, and task05"
        )
    task02 = read_json(Path(str(artifacts["task02"])) / "config.json")
    task03 = read_json(Path(str(artifacts["task03"])) / "model_config.json")
    task04 = read_json(Path(str(artifacts["task04"])) / "model_config.json")
    task05 = read_json(Path(str(artifacts["task05"])) / "model_config.json")
    task02_model = task02.get("model_config")
    if not isinstance(task02_model, Mapping):
        raise CandidateDatasetPreparationError("task02 lacks model_config")
    return {
        "global_popularity": PopularityScore(task02_model["score_type"]),
        "recency_popularity": RecencyPopularityConfig.from_dict(
            task03["recency_config"]
        ),
        "item2item": Item2ItemConfig.from_dict(task04["item2item_config"]),
        "implicit_als": ImplicitALSConfig.from_dict(task05["implicit_als_config"]),
    }


def _resolve_external_model(
    source: Mapping[str, Any], *, fold_label: str, model_source: str
) -> Path | None:
    pretrained = source.get("pretrained_models", {})
    if not isinstance(pretrained, Mapping):
        raise CandidateDatasetPreparationError("pretrained_models must be an object")
    fold = pretrained.get(fold_label)
    if fold is None:
        return None
    if not isinstance(fold, Mapping):
        raise CandidateDatasetPreparationError(
            f"pretrained_models.{fold_label} must be an object"
        )
    value = fold.get(model_source)
    if value is None:
        return None
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, Mapping) and isinstance(value.get("pointer"), str):
        pointer_path = Path(value["pointer"])
        pointer = read_json(pointer_path)
        model_path = pointer.get("model_path")
        if not isinstance(model_path, str):
            raise CandidateDatasetPreparationError(
                f"pretrained pointer lacks model_path: {pointer_path}"
            )
        path = pointer_path.parent / model_path
    else:
        raise CandidateDatasetPreparationError(
            f"invalid pretrained model entry for {fold_label}/{model_source}"
        )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _restore_global(path: Path, score: PopularityScore) -> GlobalPopularityModel:
    return GlobalPopularityModel.from_fitted_ranking(
        score, pl.read_parquet(path / "item_ranking.parquet")
    )


def _restore_recency(
    path: Path, config: RecencyPopularityConfig
) -> RecencyPopularityModel:
    return RecencyPopularityModel.from_fitted_ranking(
        config, pl.read_parquet(path / "item_ranking.parquet")
    )


def _restore_item2item(path: Path, config: Item2ItemConfig) -> Item2ItemModel:
    return Item2ItemModel.from_fitted_neighbors(
        config, pl.read_parquet(path / "neighbor_table.parquet")
    )


def _save_non_als_model(model: Any, directory: Path) -> None:
    directory.mkdir(parents=True)
    write_json_atomic(directory / "model_config.json", model.get_config())
    if isinstance(model, (GlobalPopularityModel, RecencyPopularityModel)):
        model.item_ranking.write_parquet(
            directory / "item_ranking.parquet", compression="zstd", statistics=True
        )
    elif isinstance(model, Item2ItemModel):
        model.neighbor_table.write_parquet(
            directory / "neighbor_table.parquet", compression="zstd", statistics=True
        )
    else:
        raise TypeError(f"unsupported model type: {type(model).__name__}")


def _source_model_path(source_dir: Path, metadata: Mapping[str, Any]) -> Path:
    kind = metadata.get("model_storage")
    value = metadata.get("model_path")
    if not isinstance(value, str):
        raise CandidateDatasetPreparationError("source metadata lacks model_path")
    path = Path(value) if kind == "reference" else source_dir / value
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _prediction_seeds(loader: Item2ItemDataLoader) -> pl.DataFrame:
    batches = list(loader.iter_predict_batches(batch_size=None))
    if len(batches) != 1:
        raise RuntimeError("item2item loader must yield one unbounded batch")
    return batches[0].seeds


def _prepare_source(
    *,
    source_name: str,
    model_config: Any,
    external_model: Path | None,
    history: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    cutoff: datetime,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
    destination: Path,
    reporter: EventProgressReporter,
    fold_label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    fitted = external_model is None
    model: Any
    loader: Any
    try:
        if source_name == "global_popularity":
            loader = PopularityDataLoader(seed=seed)
            if fitted:
                loader.load_fit_data(history=history).prepare_fit_data()
                model = GlobalPopularityModel(model_config).fit(loader)
            else:
                model = _restore_global(external_model, model_config)
            loader.load_predict_data(
                history=history, target_users=target_users
            ).prepare_predict_data()
        elif source_name == "recency_popularity":
            windows, half_lives = _recency_grids(model_config)
            loader = RecencyPopularityDataLoader(
                reference_time=cutoff,
                windows_hours=windows,
                half_lives_hours=half_lives,
                seed=seed,
            )
            if fitted:
                loader.load_fit_data(history=history).prepare_fit_data()
                model = RecencyPopularityModel(model_config).fit(loader)
            else:
                model = _restore_recency(external_model, model_config)
            loader.load_predict_data(
                history=history, target_users=target_users
            ).prepare_predict_data()
        elif source_name == "item2item":
            loader = Item2ItemDataLoader(
                reference_time=cutoff,
                max_history_items=model_config.history_cap,
                max_seed_items=model_config.seed_k,
                seed=seed,
            )
            if fitted:
                loader.load_fit_data(history=history).prepare_fit_data()
                model = Item2ItemModel(model_config).fit(loader)
            else:
                model = _restore_item2item(external_model, model_config)
            loader.load_predict_data(
                history=history, target_users=target_users
            ).prepare_predict_data()
        elif source_name == "implicit_als":
            if fitted:
                loader = ImplicitALSDataLoader(
                    config=model_config, reference_time=cutoff, seed=seed
                )
                loader.load_fit_data(history=history).prepare_fit_data()
                model = ImplicitALSModel(model_config)
                iteration_started, callback = reporter.iteration_start(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                    total=model_config.iterations,
                    unit="iter",
                )
                try:
                    model.fit(loader, show_progress=False, callback=callback)
                except BaseException:
                    reporter.iteration_finish(
                        stage="source_materialization",
                        config=source_name,
                        fold=fold_label,
                        started=iteration_started,
                        status="failed",
                    )
                    raise
                reporter.iteration_finish(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                    started=iteration_started,
                    status="completed",
                )
            else:
                model = ImplicitALSModel.from_artifact(external_model)
                loader = ImplicitALSDataLoader(
                    config=model.config,
                    reference_time=cutoff,
                    user_mapping=model.user_mapping,
                    item_mapping=model.item_mapping,
                    seed=seed,
                )
            loader.load_predict_data(
                history=history, target_users=target_users
            ).prepare_predict_data()
        else:  # pragma: no cover - validated config makes this unreachable
            raise CandidateDatasetPreparationError(source_name)

        validate_model_config(model)
        candidates = run_candidate_model(
            model,
            loader,
            k=candidate_k,
            batch_size=predict_batch_size,
        )
        candidate_path = temporary / "candidates.parquet"
        candidates.write_parquet(candidate_path, compression="zstd", statistics=True)
        if isinstance(loader, Item2ItemDataLoader):
            _prediction_seeds(loader).write_parquet(
                temporary / "seeds.parquet", compression="zstd", statistics=True
            )
        if fitted:
            if isinstance(model, ImplicitALSModel):
                model.save(temporary / "model")
            else:
                _save_non_als_model(model, temporary / "model")
            model_storage = "embedded"
            model_path = "model"
        else:
            model_storage = "reference"
            model_path = external_model.as_posix()
        candidate_metrics = evaluate_candidate_metrics(
            candidates, target_ground_truth, target_users
        )
        metadata = {
            "source": source_name,
            "fold": fold_label,
            "cutoff": cutoff.isoformat(),
            "candidate_k": candidate_k,
            "candidate_rows": candidates.height,
            "model_storage": model_storage,
            "model_path": model_path,
            "model_config": model.get_config(),
            "candidate_metrics": candidate_metrics,
            "candidates_sha256": sha256_file(candidate_path),
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json_atomic(temporary / "metadata.json", metadata)
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        return metadata
    except BaseException:
        # Leave only complete atomically published source directories.
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _load_source_model(
    source_name: str,
    source_dir: Path,
    model_config: Any,
) -> Any:
    metadata = read_json(source_dir / "metadata.json")
    path = _source_model_path(source_dir, metadata)
    if source_name == "global_popularity":
        return _restore_global(path, model_config)
    if source_name == "recency_popularity":
        return _restore_recency(path, model_config)
    if source_name == "item2item":
        return _restore_item2item(path, model_config)
    if source_name == "implicit_als":
        return ImplicitALSModel.from_artifact(path)
    raise CandidateDatasetPreparationError(source_name)


def _fold_context(
    fold_path: Path,
    *,
    selection: bool,
    smoke_user_limit: int | None,
    smoke_context_user_limit: int | None,
) -> tuple[
    dict[str, Path],
    dict[str, Any],
    datetime,
    Path | pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    files, manifest = _validate_fold(fold_path, selection=selection)
    cutoff = datetime.fromisoformat(manifest["cutoff"])
    target_users, ground_truth = _load_evaluation_frames(
        files, smoke_user_limit=smoke_user_limit
    )
    history: Path | pl.DataFrame = files["history_daily.parquet"]
    if smoke_context_user_limit is not None:
        history = _limited_history(
            files["history_daily.parquet"],
            target_users,
            context_user_limit=smoke_context_user_limit,
        )
        ground_truth = ground_truth.join(
            history.select("item_id").unique(), on="item_id", how="semi"
        )
    return files, manifest, cutoff, history, target_users, ground_truth


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fold_union_metrics(
    *,
    parts_pattern: str,
    ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, Any]:
    union = pl.scan_parquet(parts_pattern)
    candidate_metrics = evaluate_candidate_metrics_lazy(
        union, ground_truth, target_users
    )
    schema = union.collect_schema()
    sources = [
        column.removeprefix("generated_by_")
        for column in schema.names()
        if column.startswith("generated_by_")
    ]
    hits = union.join(ground_truth.lazy(), on=["user_id", "item_id"], how="semi")
    expressions: list[pl.Expr] = []
    for source_name in sources:
        generated = pl.col(f"generated_by_{source_name}")
        expressions.extend(
            (
                generated.sum().alias(f"source_hits__{source_name}"),
                (generated & (pl.col("source_count") == 1))
                .sum()
                .alias(f"exclusive_hits__{source_name}"),
            )
        )
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            expressions.append(
                (pl.col(f"generated_by_{left}") & pl.col(f"generated_by_{right}"))
                .sum()
                .alias(f"overlap__{left}__{right}")
            )
    contribution = (
        hits.select(expressions).collect(engine="streaming").row(0, named=True)
    )
    availability_columns = [
        column
        for column in schema.names()
        if column.startswith("cross_score_available_")
    ]
    availability = (
        union.select(
            pl.col(column).mean().alias(column) for column in availability_columns
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    return {
        **candidate_metrics,
        "source_relevant_hits": {
            source_name: int(contribution[f"source_hits__{source_name}"] or 0)
            for source_name in sources
        },
        "exclusive_relevant_hits": {
            source_name: int(contribution[f"exclusive_hits__{source_name}"] or 0)
            for source_name in sources
        },
        "pairwise_relevant_hit_overlap": {
            key.removeprefix("overlap__"): int(value or 0)
            for key, value in contribution.items()
            if key.startswith("overlap__")
        },
        "cross_score_availability": {
            key.removeprefix("cross_score_available_"): float(value or 0.0)
            for key, value in availability.items()
        },
    }


def _validate_config(
    source: Mapping[str, Any],
) -> tuple[CandidateUnionConfig, int, int, dict[str, int]]:
    materialized = source.get("materialized_union")
    if not isinstance(materialized, Mapping):
        raise CandidateDatasetPreparationError("materialized_union must be an object")
    source_caps = materialized.get("source_caps")
    if not isinstance(source_caps, Mapping) or tuple(source_caps) != SOURCE_ORDER:
        raise CandidateDatasetPreparationError(
            f"source_caps must use source order {SOURCE_ORDER}"
        )
    union_config = CandidateUnionConfig.from_mapping(
        {str(name): int(value) for name, value in source_caps.items()},
        total_cap=int(materialized["total_cap"]),
    )
    candidate_k = _positive_int(source.get("candidate_k"), name="candidate_k")
    if any(spec.cap > candidate_k for spec in union_config.sources):
        raise CandidateDatasetPreparationError(
            "materialized source cap exceeds candidate_k"
        )
    users_per_shard = _positive_int(
        source.get("users_per_shard"), name="users_per_shard"
    )
    batch_sizes = source.get("predict_batch_sizes")
    if not isinstance(batch_sizes, Mapping) or set(batch_sizes) != set(SOURCE_ORDER):
        raise CandidateDatasetPreparationError(
            "predict_batch_sizes must cover every source"
        )
    checked_batches = {
        name: _positive_int(batch_sizes[name], name=f"predict_batch_sizes.{name}")
        for name in SOURCE_ORDER
    }
    return union_config, candidate_k, users_per_shard, checked_batches


def run_preparation(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    checkpoint_dir: str | Path,
    log_file: str | Path | None,
    show_progress: bool,
    smoke_user_limit: int | None = None,
    smoke_context_user_limit: int | None = None,
) -> dict[str, Any]:
    """Materialize all source candidates and cross-scored union shards."""

    started = time.perf_counter()
    config_source = Path(config_path)
    output = Path(output_dir)
    checkpoint_root = Path(checkpoint_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    if not run_id:
        raise CandidateDatasetPreparationError("run_id must be non-empty")
    if smoke_user_limit is not None:
        _positive_int(smoke_user_limit, name="smoke_user_limit")
    if smoke_context_user_limit is not None:
        _positive_int(smoke_context_user_limit, name="smoke_context_user_limit")
        if smoke_user_limit is None:
            raise CandidateDatasetPreparationError(
                "smoke_context_user_limit requires smoke_user_limit"
            )
    elif smoke_user_limit is not None:
        smoke_context_user_limit = max(2_000, smoke_user_limit * 20)
    if smoke_user_limit is not None and output == Path(
        "artifacts/task06_candidate_datasets_v1"
    ):
        raise CandidateDatasetPreparationError(
            "limited smoke cannot publish to the default full artifact path"
        )
    source = read_json(config_source)
    digest = config_sha256(source)
    union_config, candidate_k, users_per_shard, batch_sizes = _validate_config(source)
    winner_configs = _winner_configs(source)
    seed = int(source.get("seed", 42))
    folds = source.get("folds")
    if not isinstance(folds, Mapping):
        raise CandidateDatasetPreparationError("folds must be an object")
    selection = folds.get("selection")
    canonical = folds.get("canonical")
    if not isinstance(selection, list) or len(selection) != 3:
        raise CandidateDatasetPreparationError("exactly three selection folds required")
    if not isinstance(canonical, str):
        raise CandidateDatasetPreparationError("canonical fold path must be a string")
    fold_specs = [
        (f"rolling_{index + 1}", Path(str(path)), True)
        for index, path in enumerate(selection)
    ] + [("canonical", Path(canonical), False)]
    store = CheckpointStore(checkpoint_root, run_id=run_id, config_digest=digest)
    work = checkpoint_root / "artifact"
    work.mkdir(parents=True, exist_ok=True)
    reporter = EventProgressReporter(
        task_name="task06-prepare",
        total_phases=3,
        log_file=log_file,
        show_progress=show_progress,
    )
    reporter.event(
        "run_start",
        run_id=run_id,
        config=config_source.as_posix(),
        output=output.as_posix(),
        checkpoint=checkpoint_root.as_posix(),
        mode="limited_smoke" if smoke_user_limit is not None else "full",
        pid=os.getpid(),
    )
    fold_runtime: dict[str, float] = {}
    try:
        source_phase = reporter.phase_start("source_materialization")
        source_stage = reporter.stage_start(
            stage="source_materialization",
            total=len(fold_specs) * len(SOURCE_ORDER),
            unit="fold-source",
        )
        fold_manifests: dict[str, dict[str, Any]] = {}
        fold_paths: dict[str, str] = {}
        for fold_label, fold_path, is_selection in fold_specs:
            (
                _files,
                manifest,
                cutoff,
                history,
                target_users,
                ground_truth,
            ) = _fold_context(
                fold_path,
                selection=is_selection,
                smoke_user_limit=smoke_user_limit,
                smoke_context_user_limit=smoke_context_user_limit,
            )
            fold_manifests[fold_label] = manifest
            fold_paths[fold_label] = fold_path.as_posix()
            fold_root = work / "folds" / fold_label
            eval_targets_path = fold_root / "target_users.parquet"
            eval_ground_truth_path = fold_root / "target_ground_truth.parquet"
            if not eval_targets_path.exists():
                _write_parquet_atomic(target_users, eval_targets_path)
            elif not pl.read_parquet(eval_targets_path).equals(target_users):
                raise CandidateDatasetPreparationError(
                    f"persisted target users differ for {fold_label}"
                )
            if not eval_ground_truth_path.exists():
                _write_parquet_atomic(ground_truth, eval_ground_truth_path)
            elif not pl.read_parquet(eval_ground_truth_path).equals(ground_truth):
                raise CandidateDatasetPreparationError(
                    f"persisted ground truth differs for {fold_label}"
                )
            fold_started = time.perf_counter()
            for source_name in SOURCE_ORDER:
                reporter.stage_status(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                    operation="resume_check",
                )
                source_dir = work / "folds" / fold_label / "sources" / source_name
                record = store.get(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                )
                if record is not None:
                    candidate_path = source_dir / "candidates.parquet"
                    if not candidate_path.is_file() or sha256_file(
                        candidate_path
                    ) != record.get("candidates_sha256"):
                        raise CandidateDatasetPreparationError(
                            f"completed source checkpoint is corrupt: {source_dir}"
                        )
                    reporter.event(
                        "operation_resume_skip",
                        stage="source_materialization",
                        config=source_name,
                        fold=fold_label,
                    )
                    reporter.stage_advance()
                    continue
                operation_started = reporter.operation_start(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                    operation="fit_or_restore_predict",
                )
                external = (
                    None
                    if smoke_user_limit is not None
                    else _resolve_external_model(
                        source,
                        fold_label=fold_label,
                        model_source=source_name,
                    )
                )
                metadata = _prepare_source(
                    source_name=source_name,
                    model_config=winner_configs[source_name],
                    external_model=external,
                    history=history,
                    target_users=target_users,
                    target_ground_truth=ground_truth,
                    cutoff=cutoff,
                    candidate_k=candidate_k,
                    predict_batch_size=batch_sizes[source_name],
                    seed=seed,
                    destination=source_dir,
                    reporter=reporter,
                    fold_label=fold_label,
                )
                store.complete(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                    metadata={
                        "candidates_sha256": metadata["candidates_sha256"],
                        "candidate_rows": metadata["candidate_rows"],
                        "model_storage": metadata["model_storage"],
                    },
                )
                reporter.operation_finish(
                    stage="source_materialization",
                    config=source_name,
                    fold=fold_label,
                    operation="fit_or_restore_predict",
                    started=operation_started,
                    candidate_rows=metadata["candidate_rows"],
                    model_storage=metadata["model_storage"],
                )
                reporter.stage_advance()
                gc.collect()
            fold_runtime[f"{fold_label}_sources_seconds"] = (
                time.perf_counter() - fold_started
            )
        reporter.stage_finish(stage="source_materialization", started=source_stage)
        reporter.phase_finish(
            "source_materialization", source_phase, folds=len(fold_specs)
        )

        union_phase = reporter.phase_start("cross_scored_union")
        shard_total = 0
        contexts: dict[str, tuple[Any, ...]] = {}
        for fold_label, fold_path, is_selection in fold_specs:
            context = _fold_context(
                fold_path,
                selection=is_selection,
                smoke_user_limit=smoke_user_limit,
                smoke_context_user_limit=smoke_context_user_limit,
            )
            contexts[fold_label] = context
            shard_total += sum(
                1
                for _ in iter_target_user_shards(
                    context[4], users_per_shard=users_per_shard
                )
            )
        union_stage = reporter.stage_start(
            stage="cross_scored_union", total=shard_total, unit="shard"
        )
        fold_dataset_metrics: dict[str, Any] = {}
        for fold_label, _fold_path, _is_selection in fold_specs:
            (
                _files,
                _manifest,
                cutoff,
                history,
                target_users,
                ground_truth,
            ) = contexts[fold_label]
            fold_started = time.perf_counter()
            source_root = work / "folds" / fold_label / "sources"
            models = {
                source_name: _load_source_model(
                    source_name,
                    source_root / source_name,
                    winner_configs[source_name],
                )
                for source_name in SOURCE_ORDER
            }
            global_ranking = models["global_popularity"].item_ranking
            recency_ranking = models["recency_popularity"].item_ranking
            neighbor_table = models["item2item"].neighbor_table
            all_seeds = pl.read_parquet(source_root / "item2item" / "seeds.parquet")
            parts_dir = work / "folds" / fold_label / "union_features"
            for shard_index, shard_users in iter_target_user_shards(
                target_users, users_per_shard=users_per_shard
            ):
                shard_label = f"{fold_label}:shard_{shard_index:05d}"
                part_path = parts_dir / f"part-{shard_index:05d}.parquet"
                record = store.get(
                    stage="cross_scored_union",
                    config="materialized_max_caps",
                    fold=shard_label,
                )
                if record is not None:
                    if not part_path.is_file() or sha256_file(part_path) != record.get(
                        "sha256"
                    ):
                        raise CandidateDatasetPreparationError(
                            f"completed union shard is corrupt: {part_path}"
                        )
                    reporter.event(
                        "operation_resume_skip",
                        stage="cross_scored_union",
                        config="materialized_max_caps",
                        fold=shard_label,
                    )
                    reporter.stage_advance()
                    continue
                operation_started = reporter.operation_start(
                    stage="cross_scored_union",
                    config="materialized_max_caps",
                    fold=shard_label,
                    operation="union_and_cross_score",
                )
                source_frames: dict[str, pl.DataFrame] = {}
                for source_name in SOURCE_ORDER:
                    source_frames[source_name] = (
                        pl.scan_parquet(
                            source_root / source_name / "candidates.parquet"
                        )
                        .join(shard_users.lazy(), on="user_id", how="semi")
                        .collect(engine="streaming")
                    )
                    validate_candidate_output(
                        source_frames[source_name],
                        k=candidate_k,
                        source_name=source_name,
                    )
                features = build_candidate_union(source_frames, union_config)
                features = attach_ranking_cross_scores(
                    features,
                    source="global_popularity",
                    item_ranking=global_ranking,
                )
                features = attach_ranking_cross_scores(
                    features,
                    source="recency_popularity",
                    item_ranking=recency_ranking,
                )
                shard_seeds = all_seeds.join(shard_users, on="user_id", how="semi")
                features = attach_item2item_cross_scores(
                    features,
                    seeds=shard_seeds,
                    neighbor_table=neighbor_table,
                    config=winner_configs["item2item"],
                    reference_time=cutoff,
                )
                features = attach_implicit_als_cross_scores(
                    features,
                    model=models["implicit_als"],
                    batch_size=batch_sizes["implicit_als"],
                )
                validate_union_features(
                    features,
                    config=union_config,
                    require_cross_scores=SOURCE_ORDER,
                )
                _write_parquet_atomic(features, part_path)
                part_sha256 = sha256_file(part_path)
                store.complete(
                    stage="cross_scored_union",
                    config="materialized_max_caps",
                    fold=shard_label,
                    metadata={
                        "sha256": part_sha256,
                        "rows": features.height,
                        "users": shard_users.height,
                    },
                )
                reporter.operation_finish(
                    stage="cross_scored_union",
                    config="materialized_max_caps",
                    fold=shard_label,
                    operation="union_and_cross_score",
                    started=operation_started,
                    rows=features.height,
                    users=shard_users.height,
                )
                reporter.stage_advance()
                del features, source_frames
                gc.collect()
            parts = sorted(parts_dir.glob("part-*.parquet"))
            if not parts:
                raise RuntimeError(f"no union feature parts for {fold_label}")
            validate_union_against_history(
                pl.scan_parquet((parts_dir / "part-*.parquet").as_posix()),
                history_daily=(
                    history.lazy() if isinstance(history, pl.DataFrame) else history
                ),
            )
            fold_dataset_metrics[fold_label] = _fold_union_metrics(
                parts_pattern=(parts_dir / "part-*.parquet").as_posix(),
                ground_truth=ground_truth,
                target_users=target_users,
            )
            write_json_atomic(
                work / "folds" / fold_label / "dataset_manifest.json",
                {
                    "fold": fold_label,
                    "cutoff": cutoff.isoformat(),
                    "target_users": target_users.height,
                    "target_ground_truth_pairs": ground_truth.height,
                    "history_path": (fold_paths[fold_label] + "/history_daily.parquet"),
                    "target_users_sha256": sha256_file(
                        work / "folds" / fold_label / "target_users.parquet"
                    ),
                    "target_ground_truth_sha256": sha256_file(
                        work / "folds" / fold_label / "target_ground_truth.parquet"
                    ),
                    "part_count": len(parts),
                    "rows": int(
                        sum(
                            pl.scan_parquet(path).select(pl.len()).collect().item()
                            for path in parts
                        )
                    ),
                    "parts": {path.name: sha256_file(path) for path in parts},
                    "metrics": fold_dataset_metrics[fold_label],
                },
            )
            fold_runtime[f"{fold_label}_union_seconds"] = (
                time.perf_counter() - fold_started
            )
            del models, all_seeds
            gc.collect()
        reporter.stage_finish(stage="cross_scored_union", started=union_stage)
        reporter.phase_finish("cross_scored_union", union_phase, shards=shard_total)

        publish_phase = reporter.phase_start("validate_and_publish")
        mode = "limited_smoke" if smoke_user_limit is not None else "full"
        resolved = {
            "run_id": run_id,
            "artifact_version": 1,
            "kind": "task06_offline_candidate_datasets",
            "mode": mode,
            "seed": seed,
            "candidate_k": candidate_k,
            "materialized_union": union_config.to_dict(),
            "users_per_shard": users_per_shard,
            "predict_batch_sizes": batch_sizes,
            "winner_model_configs": {
                name: (
                    value.value
                    if isinstance(value, PopularityScore)
                    else value.to_dict()
                )
                for name, value in winner_configs.items()
            },
            "folds": fold_manifests,
            "fold_paths": fold_paths,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": digest,
            "smoke_user_limit": smoke_user_limit,
            "smoke_context_user_limit": smoke_context_user_limit,
            "library_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "polars": pl.__version__,
                "implicit": implicit.__version__,
            },
        }
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "folds": fold_dataset_metrics,
            "runtime_by_fold_seconds": fold_runtime,
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        write_json_atomic(work / "config.json", resolved)
        write_json_atomic(work / "metrics.json", metrics)
        publish_directory_atomic(work, output)
        reporter.phase_finish(
            "validate_and_publish", publish_phase, output=output.as_posix()
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
            "Materialize fold-specific source candidates and cross-scored "
            "offline union datasets for task06."
        )
    )
    parser.add_argument("--config", default="configs/task06_candidate_ensemble_v1.json")
    parser.add_argument(
        "--output-dir", default="artifacts/task06_candidate_datasets_v1"
    )
    parser.add_argument("--run-id", default="task06_candidate_datasets_v1")
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/.task06_candidate_datasets_v1.checkpoint",
    )
    parser.add_argument("--log-file", default="logs/task06_candidate_datasets_v1.log")
    parser.add_argument("--smoke-user-limit", type=int)
    parser.add_argument("--smoke-context-user-limit", type=int)
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
        smoke_context_user_limit=args.smoke_context_user_limit,
    )
    print(
        json.dumps(
            {
                "run_id": metrics["run_id"],
                "mode": metrics["mode"],
                "runtime_seconds": metrics["runtime_seconds"],
                "peak_memory_mb": metrics["peak_memory_mb"],
                "folds": {
                    name: {
                        key: values[key]
                        for key in (
                            "candidate_recall",
                            "candidate_oracle_p20_all_targets",
                            "mean_candidate_count",
                        )
                    }
                    for name, values in metrics["folds"].items()
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

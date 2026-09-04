#!/usr/bin/env python3
"""Select implicit ALS on rolling folds and evaluate one canonical winner."""

from __future__ import annotations

import os

# Avoid nested BLAS pools before NumPy/SciPy/implicit are imported.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import gc
import hashlib
import json
import logging
import platform
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from statistics import fmean
from typing import Any

import implicit
import numpy as np
import polars as pl
import scipy
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from implicit_model import (
    ImplicitALSConfig,
    ImplicitALSDataLoader,
    ImplicitALSModel,
)
from item2item import Item2ItemConfig, Item2ItemDataLoader, Item2ItemModel
from metrics import evaluate_candidate_metrics, evaluate_precision_at_20
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
    fill_with_global_popularity,
)
from scripts.run_global_popularity import (
    _final_hit_count,
    _load_evaluation_frames,
    _peak_memory_mb,
    _read_json,
    _sha256,
    _validate_fold,
    _write_json,
)
from scripts.run_item2item import _fallback_usage, _limited_history
from validation import (
    validate_candidate_output,
    validate_json_config,
    validate_loader,
    validate_model_config,
    validate_recommendations_against_history,
)


class ImplicitALSExperimentError(ValueError):
    """Raised when task-05 experiment configuration is invalid."""


class _ProgressReporter:
    """Render nested progress bars and write compact, rotating event logs."""

    def __init__(
        self,
        *,
        log_file: str | Path | None,
        show_progress: bool,
        phase_count: int = 9,
    ) -> None:
        self.log_path = Path(log_file) if log_file is not None else None
        self._show_progress = show_progress
        self._logger = logging.getLogger(f"task05.{uuid.uuid4().hex}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler: RotatingFileHandler | None = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handler = RotatingFileHandler(
                self.log_path,
                mode="a",
                maxBytes=1_000_000,
                backupCount=2,
                encoding="utf-8",
            )
            self._handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            self._logger.addHandler(self._handler)
        self._overall = tqdm(
            total=phase_count,
            desc="task05 | starting",
            unit="phase",
            position=0,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        self._stage: Any = None
        self._fit: Any = None

    @staticmethod
    def _field(value: Any) -> str:
        if isinstance(value, float):
            return format(value, ".6f")
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    def event(self, event: str, **fields: Any) -> None:
        values = [f"event={event}"]
        values.extend(
            f"{name}={self._field(value)}"
            for name, value in fields.items()
            if value is not None
        )
        self._logger.info(" ".join(values))

    def phase_start(self, phase: str, **fields: Any) -> float:
        self._overall.set_description_str(f"task05 | {phase}")
        self._overall.refresh()
        self.event("phase_start", phase=phase, **fields)
        return time.perf_counter()

    def phase_finish(self, phase: str, started: float, **fields: Any) -> None:
        self.event(
            "phase_finish",
            phase=phase,
            duration_seconds=time.perf_counter() - started,
            **fields,
        )
        self._overall.update(1)

    def stage_start(self, stage: str, *, total: int) -> float:
        self._stage = tqdm(
            total=total,
            desc=f"{stage} | prepare",
            unit="fold-model",
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=not self._show_progress,
        )
        self.event("stage_start", stage=stage, planned_fold_models=total)
        return time.perf_counter()

    def stage_status(self, stage: str, config: str, operation: str) -> None:
        self._overall.set_description_str(
            f"task05 | {stage} | {config} | {operation}"
        )
        self._overall.refresh()
        if self._stage is not None:
            self._stage.set_description_str(
                f"{stage} | {config} | {operation}"
            )
            self._stage.refresh()

    def stage_advance(self, count: int = 1) -> None:
        if self._stage is not None:
            self._stage.update(count)

    def stage_finish(
        self, stage: str, started: float, *, winner: str
    ) -> None:
        if self._stage is not None:
            self._stage.close()
            self._stage = None
        self.event(
            "stage_finish",
            stage=stage,
            winner=winner,
            duration_seconds=time.perf_counter() - started,
        )

    def fit_start(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        iterations: int,
    ) -> tuple[float, Any]:
        self.stage_status(stage, config, f"fold={fold} fit")
        self.event(
            "fit_start",
            stage=stage,
            config=config,
            fold=fold,
            iterations=iterations,
        )
        self._fit = tqdm(
            total=iterations,
            desc=f"ALS | {config} | fold={fold}",
            unit="iter",
            position=2,
            leave=False,
            dynamic_ncols=True,
            disable=not self._show_progress,
        )

        def callback(iteration: int, elapsed: float, loss: float | None) -> None:
            completed = iteration + 1
            if self._fit is not None:
                self._fit.update(max(0, completed - self._fit.n))
                self._fit.set_postfix_str(f"last={elapsed:.1f}s")
            self.event(
                "fit_iteration",
                stage=stage,
                config=config,
                fold=fold,
                iteration=completed,
                iterations=iterations,
                iteration_seconds=elapsed,
                loss=loss,
            )

        return time.perf_counter(), callback

    def fit_finish(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        started: float,
        status: str,
    ) -> None:
        if self._fit is not None:
            self._fit.close()
            self._fit = None
        self.event(
            "fit_finish",
            stage=stage,
            config=config,
            fold=fold,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )

    def operation_start(
        self, *, stage: str, config: str, fold: str, operation: str
    ) -> float:
        self.stage_status(stage, config, f"fold={fold} {operation}")
        self.event(
            "operation_start",
            stage=stage,
            config=config,
            fold=fold,
            operation=operation,
        )
        return time.perf_counter()

    def operation_finish(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        operation: str,
        started: float,
        **fields: Any,
    ) -> None:
        self.event(
            "operation_finish",
            stage=stage,
            config=config,
            fold=fold,
            operation=operation,
            duration_seconds=time.perf_counter() - started,
            **fields,
        )

    def close(self) -> None:
        if self._fit is not None:
            self._fit.close()
            self._fit = None
        if self._stage is not None:
            self._stage.close()
            self._stage = None
        self._overall.close()
        if self._handler is not None:
            self._handler.flush()
            self._handler.close()
            self._logger.removeHandler(self._handler)
            self._handler = None


@dataclass(frozen=True)
class BestModelSnapshot:
    """Validated rolling winner passed to a best-model callback."""

    stage: str
    config: ImplicitALSConfig
    fold_label: str
    cutoff: datetime
    model: ImplicitALSModel
    fold_metrics: Mapping[str, Any]
    selection_summary: Mapping[str, float]


BestModelCallback = Callable[[BestModelSnapshot], None]


class BestModelCheckpoint:
    """Atomically retain the strongest rolling model seen so far."""

    _SCORE_KEYS = (
        "mean_union_oracle_p20_all_targets_gain",
        "mean_candidate_oracle_p20_all_targets",
        "mean_precision_at_20_all_targets",
    )

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        source_config_sha256: str,
        reporter: _ProgressReporter | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.source_config_sha256 = source_config_sha256
        self.reporter = reporter

    @classmethod
    def _score(cls, summary: Mapping[str, float]) -> tuple[float, ...]:
        return (
            *(float(summary[key]) for key in cls._SCORE_KEYS),
            -float(summary["mean_runtime_seconds"]),
        )

    def _read_pointer(self) -> dict[str, Any] | None:
        pointer = self.directory / "best_model.json"
        if not pointer.exists():
            return None
        value = _read_json(pointer)
        if (
            value.get("artifact_version") != 1
            or value.get("kind") != "task05_rolling_best_model"
            or value.get("run_id") != self.run_id
            or value.get("source_config_sha256")
            != self.source_config_sha256
        ):
            raise ImplicitALSExperimentError(
                f"best-model checkpoint is incompatible: {pointer}"
            )
        score = value.get("score")
        model_path = Path(str(value.get("model_path", "")))
        if (
            not isinstance(score, list)
            or len(score) != 4
            or not all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and np.isfinite(item)
                for item in score
            )
            or len(model_path.parts) != 3
            or model_path.parts[0] != "versions"
            or model_path.parts[2] != "model"
            or not (self.directory / model_path).is_dir()
        ):
            raise ImplicitALSExperimentError(
                f"best-model checkpoint pointer is invalid: {pointer}"
            )
        return value

    def validate_existing(self) -> None:
        """Fail fast when an existing checkpoint belongs to another run."""

        current = self._read_pointer()
        keep_version = None
        if current is not None:
            keep_version = Path(str(current["model_path"])).parts[1]
        versions = self.directory / "versions"
        if versions.is_dir():
            for child in versions.iterdir():
                if child.is_dir() and child.name != keep_version:
                    shutil.rmtree(child)

    def __call__(self, snapshot: BestModelSnapshot) -> None:
        if snapshot.model.config != snapshot.config:
            raise ImplicitALSExperimentError(
                "best-model callback received a model/config mismatch"
            )
        score = self._score(snapshot.selection_summary)
        current = self._read_pointer()
        if current is not None:
            current_score = tuple(float(value) for value in current["score"])
            if score <= current_score:
                if self.reporter is not None:
                    self.reporter.event(
                        "best_model_unchanged",
                        stage=snapshot.stage,
                        config=snapshot.config.config_id,
                        current_config=current["config_id"],
                    )
                return

        started = time.perf_counter()
        if self.reporter is not None:
            self.reporter.stage_status(
                snapshot.stage,
                snapshot.config.config_id,
                "save_best_model",
            )
            self.reporter.event(
                "best_model_save_start",
                stage=snapshot.stage,
                config=snapshot.config.config_id,
                fold=snapshot.fold_label,
                directory=self.directory.as_posix(),
            )

        versions = self.directory / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        version = uuid.uuid4().hex
        temporary = versions / f".staging-{version}"
        published = versions / version
        pointer_temporary = self.directory / f".best_model-{version}.json"
        pointer_published = False
        try:
            temporary.mkdir()
            model_dir = temporary / "model"
            snapshot.model.save(
                model_dir,
                metadata={
                    "checkpoint_kind": "rolling_selection_best",
                    "stage": snapshot.stage,
                    "selection_fold": snapshot.fold_label,
                    "fit_reference_time": snapshot.cutoff.isoformat(),
                    "source_config_sha256": self.source_config_sha256,
                },
            )
            model_files = (
                "model_config.json",
                "als_model.npz",
                "user_mapping.parquet",
                "item_mapping.parquet",
            )
            model_sha256 = {
                name: _sha256(model_dir / name) for name in model_files
            }
            _write_json(
                temporary / "snapshot.json",
                {
                    "stage": snapshot.stage,
                    "config_id": snapshot.config.config_id,
                    "config": snapshot.config.to_dict(),
                    "fold_label": snapshot.fold_label,
                    "cutoff": snapshot.cutoff.isoformat(),
                    "fold_metrics": dict(snapshot.fold_metrics),
                    "selection_summary": dict(snapshot.selection_summary),
                    "model_sha256": model_sha256,
                },
            )
            os.replace(temporary, published)
            pointer_value = {
                "artifact_version": 1,
                "kind": "task05_rolling_best_model",
                "run_id": self.run_id,
                "source_config_sha256": self.source_config_sha256,
                "stage": snapshot.stage,
                "config_id": snapshot.config.config_id,
                "fold_label": snapshot.fold_label,
                "cutoff": snapshot.cutoff.isoformat(),
                "score": list(score),
                "model_path": f"versions/{version}/model",
                "snapshot_path": f"versions/{version}/snapshot.json",
            }
            _write_json(pointer_temporary, pointer_value)
            os.replace(pointer_temporary, self.directory / "best_model.json")
            pointer_published = True
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            if not pointer_published:
                shutil.rmtree(published, ignore_errors=True)
            pointer_temporary.unlink(missing_ok=True)
            raise

        if current is not None:
            old_model_path = Path(str(current["model_path"]))
            if (
                len(old_model_path.parts) == 3
                and old_model_path.parts[0] == "versions"
                and old_model_path.parts[2] == "model"
            ):
                shutil.rmtree(
                    self.directory / old_model_path.parts[0] / old_model_path.parts[1],
                    ignore_errors=True,
                )
        if self.reporter is not None:
            self.reporter.event(
                "best_model_save_finish",
                stage=snapshot.stage,
                config=snapshot.config.config_id,
                fold=snapshot.fold_label,
                duration_seconds=time.perf_counter() - started,
                model_path=pointer_value["model_path"],
            )


@dataclass
class _FixedSourceConfigs:
    global_score: PopularityScore
    recency: RecencyPopularityConfig
    item2item: Item2ItemConfig


@dataclass
class _FoldContext:
    files: dict[str, Path]
    manifest: dict[str, Any]
    cutoff: datetime
    history_source: Path | pl.DataFrame
    target_users: pl.DataFrame
    target_ground_truth: pl.DataFrame
    fallback_candidates: pl.DataFrame
    baseline_hit_cache: Path
    baseline_metrics: dict[str, Any]


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ImplicitALSExperimentError(f"{name} must be a positive integer")
    return value


def _grid(source: Mapping[str, Any], name: str) -> list[Any]:
    value = source.get(name)
    if not isinstance(value, list) or not value:
        raise ImplicitALSExperimentError(f"ablation.{name} must be a non-empty list")
    if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
        raise ImplicitALSExperimentError(f"ablation.{name} must not contain duplicates")
    return value


def _history_lazy(source: Path | pl.DataFrame) -> pl.LazyFrame:
    return source.lazy() if isinstance(source, pl.DataFrame) else pl.scan_parquet(source)


def _dataframe_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.schema).encode())
    digest.update(str(frame.height).encode())
    digest.update(frame.hash_rows(seed=42).to_numpy().tobytes())
    return digest.hexdigest()


def _candidate_hit_pairs(
    candidates: pl.DataFrame, ground_truth: pl.DataFrame
) -> pl.DataFrame:
    return (
        candidates.select("user_id", "item_id")
        .unique()
        .join(ground_truth, on=["user_id", "item_id"], how="semi")
        .sort(("user_id", "item_id"))
    )


def _hit_metrics(
    hit_pairs: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    target_users: pl.DataFrame,
) -> dict[str, float]:
    values = evaluate_candidate_metrics(
        hit_pairs, target_ground_truth, target_users
    )
    return {
        key: values[key]
        for key in (
            "candidate_recall",
            "candidate_user_hit_rate",
            "candidate_oracle_p20_all_targets",
            "candidate_oracle_p20_labeled_users",
        )
    }


def _fixed_source_configs(source: Mapping[str, Any]) -> _FixedSourceConfigs:
    paths = {
        "task02": Path(str(source.get("task02_artifact", ""))),
        "task03": Path(str(source.get("task03_artifact", ""))),
        "task04": Path(str(source.get("task04_artifact", ""))),
    }
    task02_path = paths["task02"] / "config.json"
    task03_path = paths["task03"] / "model_config.json"
    task04_path = paths["task04"] / "model_config.json"
    missing = [
        path.as_posix()
        for path in (task02_path, task03_path, task04_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"fixed-source artifacts are incomplete: {missing}")
    task02 = _read_json(task02_path)
    task03 = _read_json(task03_path)
    task04 = _read_json(task04_path)
    model_config = task02.get("model_config")
    if not isinstance(model_config, dict):
        raise ImplicitALSExperimentError("task02 artifact lacks model_config")
    return _FixedSourceConfigs(
        global_score=PopularityScore(model_config["score_type"]),
        recency=RecencyPopularityConfig.from_dict(task03["recency_config"]),
        item2item=Item2ItemConfig.from_dict(task04["item2item_config"]),
    )


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
        windows.append(float(entry["window_hours"]))
    if value.get("half_life_hours") is not None:
        half_lives.append(float(value["half_life_hours"]))
    return sorted(set(windows)), sorted(set(half_lives))


def _prepare_global_candidates(
    *,
    history: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    score: PopularityScore,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
) -> pl.DataFrame:
    loader = (
        PopularityDataLoader(seed=seed)
        .load_fit_data(history=history)
        .prepare_fit_data()
        .load_predict_data(history=history, target_users=target_users)
        .prepare_predict_data()
    )
    model = GlobalPopularityModel(score).fit(loader)
    result = model.predict(loader, k=candidate_k, batch_size=predict_batch_size)
    validate_candidate_output(result, k=candidate_k, source_name=model.source_name)
    return result


def _prepare_recency_candidates(
    *,
    history: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    cutoff: datetime,
    config: RecencyPopularityConfig,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
) -> pl.DataFrame:
    windows, half_lives = _recency_grids(config)
    loader = (
        RecencyPopularityDataLoader(
            reference_time=cutoff,
            windows_hours=windows,
            half_lives_hours=half_lives,
            seed=seed,
        )
        .load_fit_data(history=history)
        .prepare_fit_data()
        .load_predict_data(history=history, target_users=target_users)
        .prepare_predict_data()
    )
    model = RecencyPopularityModel(config).fit(loader)
    result = model.predict(loader, k=candidate_k, batch_size=predict_batch_size)
    validate_candidate_output(result, k=candidate_k, source_name=model.source_name)
    return result


def _prepare_item2item_candidates(
    *,
    history: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    cutoff: datetime,
    config: Item2ItemConfig,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
) -> pl.DataFrame:
    loader = (
        Item2ItemDataLoader(
            reference_time=cutoff,
            max_history_items=config.history_cap,
            max_seed_items=config.seed_k,
            seed=seed,
        )
        .load_fit_data(history=history)
        .prepare_fit_data()
        .load_predict_data(history=history, target_users=target_users)
        .prepare_predict_data()
    )
    model = Item2ItemModel(config).fit(loader)
    result = model.predict(loader, k=candidate_k, batch_size=predict_batch_size)
    validate_candidate_output(result, k=candidate_k, source_name=model.source_name)
    return result


def _build_fixed_source_cache(
    *,
    history: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    cutoff: datetime,
    fixed: _FixedSourceConfigs,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
    cache_path: Path,
    reporter: _ProgressReporter | None = None,
    scope: str = "fixed_sources",
    fold_label: str = "unknown",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    source_started = (
        reporter.operation_start(
            stage=scope,
            config="task02_global",
            fold=fold_label,
            operation="fit_predict",
        )
        if reporter is not None
        else time.perf_counter()
    )
    global_candidates = _prepare_global_candidates(
        history=history,
        target_users=target_users,
        score=fixed.global_score,
        candidate_k=candidate_k,
        predict_batch_size=predict_batch_size,
        seed=seed,
    )
    if reporter is not None:
        reporter.operation_finish(
            stage=scope,
            config="task02_global",
            fold=fold_label,
            operation="fit_predict",
            started=source_started,
            candidate_rows=global_candidates.height,
        )
    source_started = (
        reporter.operation_start(
            stage=scope,
            config="task03_recency",
            fold=fold_label,
            operation="fit_predict",
        )
        if reporter is not None
        else time.perf_counter()
    )
    recency_candidates = _prepare_recency_candidates(
        history=history,
        target_users=target_users,
        cutoff=cutoff,
        config=fixed.recency,
        candidate_k=candidate_k,
        predict_batch_size=predict_batch_size,
        seed=seed,
    )
    if reporter is not None:
        reporter.operation_finish(
            stage=scope,
            config="task03_recency",
            fold=fold_label,
            operation="fit_predict",
            started=source_started,
            candidate_rows=recency_candidates.height,
        )
    source_started = (
        reporter.operation_start(
            stage=scope,
            config="task04_item2item",
            fold=fold_label,
            operation="fit_predict",
        )
        if reporter is not None
        else time.perf_counter()
    )
    item2item_candidates = _prepare_item2item_candidates(
        history=history,
        target_users=target_users,
        cutoff=cutoff,
        config=fixed.item2item,
        candidate_k=candidate_k,
        predict_batch_size=predict_batch_size,
        seed=seed,
    )
    if reporter is not None:
        reporter.operation_finish(
            stage=scope,
            config="task04_item2item",
            fold=fold_label,
            operation="fit_predict",
            started=source_started,
            candidate_rows=item2item_candidates.height,
        )
    source_hits = {
        "task02": _candidate_hit_pairs(global_candidates, target_ground_truth),
        "task03": _candidate_hit_pairs(recency_candidates, target_ground_truth),
        "task04": _candidate_hit_pairs(item2item_candidates, target_ground_truth),
    }
    union_hits = pl.concat(tuple(source_hits.values()), rechunk=True).unique().sort(
        ("user_id", "item_id")
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    union_hits.write_parquet(cache_path, compression="zstd", statistics=True)
    metrics = {
        "source_hit_rows": {name: frame.height for name, frame in source_hits.items()},
        "union_relevant_hit_rows": union_hits.height,
        **{
            f"baseline_union_{key}": value
            for key, value in _hit_metrics(
                union_hits, target_ground_truth, target_users
            ).items()
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    del recency_candidates, item2item_candidates, source_hits, union_hits
    gc.collect()
    return global_candidates, metrics


def _prepare_als_loader(
    *, context: _FoldContext, config: ImplicitALSConfig, seed: int
) -> ImplicitALSDataLoader:
    loader = (
        ImplicitALSDataLoader(
            config=config,
            reference_time=context.cutoff,
            seed=seed,
        )
        .load_fit_data(history=context.history_source)
        .prepare_fit_data()
        .load_predict_data(
            history=context.history_source,
            target_users=context.target_users,
        )
        .prepare_predict_data()
    )
    validate_loader(loader)
    return loader


def _final_hit_contribution(
    recommendations: pl.DataFrame,
    als_candidates: pl.DataFrame,
    baseline_hits: pl.DataFrame,
    ground_truth: pl.DataFrame,
) -> dict[str, int]:
    final_pairs = (
        recommendations.explode("item_ids", empty_as_null=True)
        .select(
            "user_id", pl.col("item_ids").cast(pl.Int32).alias("item_id")
        )
        .join(ground_truth, on=["user_id", "item_id"], how="semi")
    )
    als_pairs = als_candidates.select("user_id", "item_id").unique()
    return {
        "final_hits_present_in_als_candidates": final_pairs.join(
            als_pairs, on=["user_id", "item_id"], how="semi"
        ).height,
        "final_hits_exclusive_to_als_vs_fixed_union": final_pairs.join(
            baseline_hits, on=["user_id", "item_id"], how="anti"
        ).join(als_pairs, on=["user_id", "item_id"], how="semi").height,
    }


def _evaluate_model(
    *,
    context: _FoldContext,
    loader: ImplicitALSDataLoader,
    config: ImplicitALSConfig,
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
    keep_outputs: bool,
    keep_model: bool = False,
    reporter: _ProgressReporter | None = None,
    stage_name: str = "implicit_als",
    fold_label: str = "unknown",
) -> tuple[
    dict[str, Any],
    ImplicitALSModel | None,
    pl.DataFrame | None,
    pl.DataFrame | None,
]:
    started = time.perf_counter()
    model = ImplicitALSModel(config)
    validate_model_config(model)
    if reporter is None:
        fit_started = time.perf_counter()
        fit_callback = None
    else:
        fit_started, fit_callback = reporter.fit_start(
            stage=stage_name,
            config=config.config_id,
            fold=fold_label,
            iterations=config.iterations,
        )
    try:
        model.fit(loader, show_progress=False, callback=fit_callback)
    except BaseException:
        if reporter is not None:
            reporter.fit_finish(
                stage=stage_name,
                config=config.config_id,
                fold=fold_label,
                started=fit_started,
                status="failed",
            )
        raise
    fit_seconds = time.perf_counter() - fit_started
    if reporter is not None:
        reporter.fit_finish(
            stage=stage_name,
            config=config.config_id,
            fold=fold_label,
            started=fit_started,
            status="completed",
        )
        predict_started = reporter.operation_start(
            stage=stage_name,
            config=config.config_id,
            fold=fold_label,
            operation="predict",
        )
    else:
        predict_started = time.perf_counter()
    candidates = model.predict(
        loader, k=candidate_k, batch_size=predict_batch_size
    )
    predict_seconds = time.perf_counter() - predict_started
    if reporter is not None:
        reporter.operation_finish(
            stage=stage_name,
            config=config.config_id,
            fold=fold_label,
            operation="predict",
            started=predict_started,
            candidate_rows=candidates.height,
        )
    validate_candidate_output(
        candidates, k=candidate_k, source_name=model.source_name
    )
    metric_started = (
        reporter.operation_start(
            stage=stage_name,
            config=config.config_id,
            fold=fold_label,
            operation="metrics_and_fallback",
        )
        if reporter is not None
        else time.perf_counter()
    )
    candidate_metrics = evaluate_candidate_metrics(
        candidates, context.target_ground_truth, context.target_users
    )
    recommendations = fill_with_global_popularity(
        candidates,
        context.fallback_candidates,
        context.target_users,
        _history_lazy(context.history_source),
        k=final_k,
    )
    validate_recommendations_against_history(
        recommendations,
        target_users=context.target_users,
        history_daily=_history_lazy(context.history_source),
        expected_k=final_k,
    )
    precision = evaluate_precision_at_20(
        recommendations, context.target_ground_truth, context.target_users
    )
    baseline_hits = pl.read_parquet(context.baseline_hit_cache)
    als_hits = _candidate_hit_pairs(candidates, context.target_ground_truth)
    union_hits = pl.concat((baseline_hits, als_hits), rechunk=True).unique()
    union_metrics = _hit_metrics(
        union_hits, context.target_ground_truth, context.target_users
    )
    exclusive_hits = als_hits.join(
        baseline_hits, on=["user_id", "item_id"], how="anti"
    ).height
    final_hits = _final_hit_count(recommendations, context.target_ground_truth)
    metrics = {
        "config_id": config.config_id,
        "model_config": config.to_dict(),
        "target_users": context.target_users.height,
        "target_labeled_users": context.target_ground_truth.get_column(
            "user_id"
        ).n_unique(),
        "target_ground_truth_pairs": context.target_ground_truth.height,
        "candidate_rows": candidates.height,
        "final_recommendation_rows": recommendations.height,
        **candidate_metrics,
        **precision,
        "final_hits": final_hits,
        "exclusive_hits": exclusive_hits,
        "als_relevant_hit_rows": als_hits.height,
        **{
            f"union_{key}": value for key, value in union_metrics.items()
        },
        "union_oracle_p20_all_targets_gain": (
            union_metrics["candidate_oracle_p20_all_targets"]
            - context.baseline_metrics[
                "baseline_union_candidate_oracle_p20_all_targets"
            ]
        ),
        **_fallback_usage(recommendations, candidates),
        **_final_hit_contribution(
            recommendations,
            candidates,
            baseline_hits,
            context.target_ground_truth,
        ),
        "fit_runtime_seconds": fit_seconds,
        "predict_runtime_seconds": predict_seconds,
        "metric_runtime_seconds": time.perf_counter() - metric_started,
        "runtime_seconds": time.perf_counter() - started,
    }
    if reporter is not None:
        reporter.operation_finish(
            stage=stage_name,
            config=config.config_id,
            fold=fold_label,
            operation="metrics_and_fallback",
            started=metric_started,
            precision_at_20_all_targets=metrics[
                "precision_at_20_all_targets"
            ],
            union_oracle_gain=metrics[
                "union_oracle_p20_all_targets_gain"
            ],
        )
    if keep_outputs:
        return metrics, model, candidates, recommendations
    if keep_model:
        del candidates, recommendations, baseline_hits, als_hits, union_hits
        gc.collect()
        return metrics, model, None, None
    del model, candidates, recommendations, baseline_hits, als_hits, union_hits
    gc.collect()
    return metrics, None, None, None


def _selection_summary(
    results: Mapping[str, Sequence[dict[str, Any]]],
    configs: Sequence[ImplicitALSConfig],
) -> dict[str, dict[str, float]]:
    keys = (
        "union_oracle_p20_all_targets_gain",
        "candidate_oracle_p20_all_targets",
        "precision_at_20_all_targets",
        "candidate_recall",
        "candidate_user_hit_rate",
        "exclusive_hits",
        "runtime_seconds",
    )
    summary: dict[str, dict[str, float]] = {}
    for config in configs:
        values = results[config.config_id]
        if len(values) != 3:
            raise RuntimeError("every ALS config must have exactly three fold results")
        summary[config.config_id] = {
            f"mean_{key}": float(fmean(item[key] for item in values))
            for key in keys
        }
    return summary


def _choose_config(
    summary: Mapping[str, Mapping[str, float]],
    configs: Sequence[ImplicitALSConfig],
) -> ImplicitALSConfig:
    order = {config.config_id: index for index, config in enumerate(configs)}

    def key(config: ImplicitALSConfig) -> tuple[float, float, float, float, int]:
        values = summary[config.config_id]
        return (
            -values["mean_union_oracle_p20_all_targets_gain"],
            -values["mean_candidate_oracle_p20_all_targets"],
            -values["mean_precision_at_20_all_targets"],
            values["mean_runtime_seconds"],
            order[config.config_id],
        )

    return min(configs, key=key)


def _evaluate_stage(
    *,
    name: str,
    changed_block: str,
    contexts: Sequence[_FoldContext],
    configs: Sequence[ImplicitALSConfig],
    retained_config: ImplicitALSConfig | None,
    retained_loaders: Sequence[ImplicitALSDataLoader] | None,
    previous_results: Mapping[str, Sequence[dict[str, Any]]],
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
    seed: int,
    reporter: _ProgressReporter | None = None,
    best_model_callback: BestModelCallback | None = None,
) -> tuple[
    dict[str, Any],
    ImplicitALSConfig,
    list[ImplicitALSDataLoader],
    dict[str, list[dict[str, Any]]],
]:
    stage_started = time.perf_counter()
    progress_started = (
        reporter.stage_start(name, total=len(configs) * len(contexts))
        if reporter is not None
        else stage_started
    )
    results: dict[str, list[dict[str, Any]]] = {}
    loader_sets: dict[str, list[ImplicitALSDataLoader]] = {}
    prepare_seconds: dict[str, float] = {}
    fit_count = 0
    for config in configs:
        if config.config_id in previous_results:
            results[config.config_id] = list(previous_results[config.config_id])
            assert retained_loaders is not None
            loader_sets[config.config_id] = list(retained_loaders)
            prepare_seconds[config.config_id] = 0.0
            if reporter is not None:
                reporter.event(
                    "config_reused",
                    stage=name,
                    config=config.config_id,
                    fold_models=len(contexts),
                )
                reporter.stage_status(name, config.config_id, "reused")
                reporter.stage_advance(len(contexts))
            continue
        if (
            retained_config is not None
            and retained_loaders is not None
            and config.confidence_key() == retained_config.confidence_key()
        ):
            loaders = list(retained_loaders)
            loader_prepare = 0.0
        else:
            prepare_started = time.perf_counter()
            loaders = []
            for fold_index, context in enumerate(contexts):
                fold_label = f"rolling_{fold_index + 1}"
                operation_started = (
                    reporter.operation_start(
                        stage=name,
                        config=config.config_id,
                        fold=fold_label,
                        operation="prepare_loader",
                    )
                    if reporter is not None
                    else time.perf_counter()
                )
                loader = _prepare_als_loader(
                    context=context, config=config, seed=seed
                )
                loaders.append(loader)
                if reporter is not None:
                    reporter.operation_finish(
                        stage=name,
                        config=config.config_id,
                        fold=fold_label,
                        operation="prepare_loader",
                        started=operation_started,
                        users=loader.user_mapping.height,
                        items=loader.item_mapping.height,
                        interactions=loader.fit_matrix.nnz,
                    )
            loader_prepare = time.perf_counter() - prepare_started
        loader_sets[config.config_id] = loaders
        prepare_seconds[config.config_id] = loader_prepare
        config_results: list[dict[str, Any]] = []
        checkpoint_model: ImplicitALSModel | None = None
        for fold_index, (context, loader) in enumerate(
            zip(contexts, loaders, strict=True), start=1
        ):
            keep_checkpoint_model = (
                best_model_callback is not None
                and fold_index == len(contexts)
            )
            values, evaluated_model, _, _ = _evaluate_model(
                context=context,
                loader=loader,
                config=config,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
                keep_outputs=False,
                keep_model=keep_checkpoint_model,
                reporter=reporter,
                stage_name=name,
                fold_label=f"rolling_{fold_index}",
            )
            if keep_checkpoint_model:
                assert evaluated_model is not None
                checkpoint_model = evaluated_model
            config_results.append(values)
            fit_count += 1
            if reporter is not None:
                reporter.stage_advance()
        results[config.config_id] = config_results
        if best_model_callback is not None:
            completed_configs = [
                candidate
                for candidate in configs
                if candidate.config_id in results
            ]
            completed_summary = _selection_summary(
                results, completed_configs
            )
            current_best = _choose_config(
                completed_summary, completed_configs
            )
            if current_best.config_id == config.config_id:
                assert checkpoint_model is not None
                best_model_callback(
                    BestModelSnapshot(
                        stage=name,
                        config=config,
                        fold_label=f"rolling_{len(contexts)}",
                        cutoff=contexts[-1].cutoff,
                        model=checkpoint_model,
                        fold_metrics=config_results[-1],
                        selection_summary=completed_summary[
                            config.config_id
                        ],
                    )
                )
            del checkpoint_model
            gc.collect()
    summary = _selection_summary(results, configs)
    winner = _choose_config(summary, configs)
    if reporter is not None:
        reporter.stage_finish(
            name, progress_started, winner=winner.config_id
        )
    winner_loaders = loader_sets[winner.config_id]
    stage = {
        "stage": name,
        "changed_block": changed_block,
        "config_order": [config.config_id for config in configs],
        "winner_config_id": winner.config_id,
        "winner_config": winner.to_dict(),
        "new_fit_count": fit_count,
        "loader_prepare_seconds": prepare_seconds,
        "summary": summary,
        "folds": [
            {
                "fold": context.manifest,
                "configs": [
                    results[config.config_id][fold_index]
                    for config in configs
                ],
            }
            for fold_index, context in enumerate(contexts)
        ],
        "runtime_seconds": time.perf_counter() - stage_started,
    }
    return stage, winner, winner_loaders, results


def _token(value: Any) -> str:
    if value is None:
        return "none"
    return format(value, ".12g") if isinstance(value, float) else str(value)


def _variants(
    winner: ImplicitALSConfig,
    *,
    stage: str,
    field: str,
    values: Sequence[Any],
) -> list[ImplicitALSConfig]:
    configs: list[ImplicitALSConfig] = []
    for value in values:
        if getattr(winner, field) == value:
            configs.append(winner)
        else:
            configs.append(
                replace(
                    winner,
                    config_id=f"{stage}_{field}_{_token(value)}",
                    **{field: value},
                )
            )
    return configs


def _validate_source_config(
    source: Mapping[str, Any],
) -> tuple[
    ImplicitALSConfig,
    list[dict[str, Any]],
    list[Any],
    list[Any],
    list[Any],
    list[Any],
]:
    base_raw = source.get("base_config")
    ablation = source.get("ablation")
    if not isinstance(base_raw, dict) or not isinstance(ablation, dict):
        raise ImplicitALSExperimentError("base_config and ablation must be objects")
    base = ImplicitALSConfig.from_dict(base_raw)
    profiles = _grid(ablation, "confidence_profiles")
    required_profile = {
        "name",
        "interaction_weight",
        "view_weight",
        "long_watch_weight",
        "like_weight",
        "favorite_weight",
    }
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or set(profile) != required_profile:
            raise ImplicitALSExperimentError(
                f"confidence_profiles[{index}] has invalid fields"
            )
        if not isinstance(profile["name"], str) or not profile["name"]:
            raise ImplicitALSExperimentError("confidence profile name must be non-empty")
    return (
        base,
        profiles,
        _grid(ablation, "half_life_hours"),
        _grid(ablation, "factors"),
        _grid(ablation, "regularization"),
        _grid(ablation, "iterations"),
    )


def _build_context(
    *,
    files: dict[str, Path],
    manifest: dict[str, Any],
    history: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    fixed: _FixedSourceConfigs,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
    cache_path: Path,
    reporter: _ProgressReporter | None = None,
    scope: str = "fixed_sources",
    fold_label: str = "unknown",
) -> _FoldContext:
    cutoff = datetime.fromisoformat(manifest["cutoff"])
    fallback, baseline = _build_fixed_source_cache(
        history=history,
        target_users=target_users,
        target_ground_truth=target_ground_truth,
        cutoff=cutoff,
        fixed=fixed,
        candidate_k=candidate_k,
        predict_batch_size=predict_batch_size,
        seed=seed,
        cache_path=cache_path,
        reporter=reporter,
        scope=scope,
        fold_label=fold_label,
    )
    return _FoldContext(
        files=files,
        manifest=manifest,
        cutoff=cutoff,
        history_source=history,
        target_users=target_users,
        target_ground_truth=target_ground_truth,
        fallback_candidates=fallback,
        baseline_hit_cache=cache_path,
        baseline_metrics=baseline,
    )


def run_experiment(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    smoke_user_limit: int | None = None,
    smoke_context_user_limit: int | None = None,
    log_file: str | Path | None = None,
    show_progress: bool = False,
    best_model_dir: str | Path | None = None,
    best_model_callback: BestModelCallback | None = None,
) -> dict[str, Any]:
    """Run five rolling-only stages, then one canonical fit/evaluation."""

    started = time.perf_counter()
    config_source = Path(config_path)
    output = Path(output_dir)
    if best_model_dir is not None and best_model_callback is not None:
        raise ImplicitALSExperimentError(
            "provide best_model_dir or best_model_callback, not both"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if smoke_user_limit is not None:
        _positive_int(smoke_user_limit, name="smoke_user_limit")
    if smoke_context_user_limit is not None:
        _positive_int(smoke_context_user_limit, name="smoke_context_user_limit")
        if smoke_user_limit is None:
            raise ImplicitALSExperimentError(
                "smoke_context_user_limit requires smoke_user_limit"
            )
    elif smoke_user_limit is not None:
        smoke_context_user_limit = max(2_000, smoke_user_limit * 20)
    source = _read_json(config_source)
    source_config_sha256 = _sha256(config_source)
    validate_json_config(source, name="implicit ALS experiment config")
    (
        base,
        profiles,
        decay_values,
        factor_values,
        regularization_values,
        iteration_values,
    ) = _validate_source_config(source)
    seed = int(source.get("seed", 42))
    if base.seed != seed:
        raise ImplicitALSExperimentError("base config seed must equal runner seed")
    candidate_k = _positive_int(source.get("candidate_k"), name="candidate_k")
    final_k = _positive_int(source.get("final_k"), name="final_k")
    predict_batch_size = _positive_int(
        source.get("predict_batch_size"), name="predict_batch_size"
    )
    if candidate_k < final_k or final_k != 20:
        raise ImplicitALSExperimentError(
            "candidate_k must be at least final_k and final_k must equal 20"
        )
    if source.get("fallback_score_type") != "relevant_interaction_count":
        raise ImplicitALSExperimentError(
            "fallback_score_type must preserve task02 relevant_interaction_count"
        )
    folds = source.get("folds")
    if not isinstance(folds, dict):
        raise ImplicitALSExperimentError("folds must be an object")
    selection_paths = folds.get("selection")
    canonical_value = folds.get("canonical")
    if not isinstance(selection_paths, list) or len(selection_paths) != 3:
        raise ImplicitALSExperimentError("exactly three selection folds are required")
    if not isinstance(canonical_value, str):
        raise ImplicitALSExperimentError("canonical fold path must be a string")
    fixed = _fixed_source_configs(source)
    checkpoint_directory = (
        Path(best_model_dir) if best_model_dir is not None else None
    )
    checkpoint_writer: BestModelCheckpoint | None = None
    if checkpoint_directory is not None:
        checkpoint_writer = BestModelCheckpoint(
            checkpoint_directory,
            run_id=run_id,
            source_config_sha256=source_config_sha256,
        )
        checkpoint_writer.validate_existing()
        best_model_callback = checkpoint_writer

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    reporter = _ProgressReporter(
        log_file=log_file,
        show_progress=show_progress,
    )
    if checkpoint_writer is not None:
        checkpoint_writer.reporter = reporter
    reporter.event(
        "run_start",
        run_id=run_id,
        config=config_source.as_posix(),
        output=output.as_posix(),
        mode="limited_smoke" if smoke_user_limit is not None else "full",
        pid=os.getpid(),
        best_model_dir=(
            checkpoint_directory.as_posix()
            if checkpoint_directory is not None
            else None
        ),
    )
    phase_runtime: dict[str, float] = {}
    try:
        fixed_started = reporter.phase_start("selection_fixed_sources")
        contexts: list[_FoldContext] = []
        previous_cutoff: datetime | None = None
        cache_dir = staging / "selection_union_hit_cache"
        for fold_index, value in enumerate(selection_paths):
            if not isinstance(value, str):
                raise ImplicitALSExperimentError("selection fold paths must be strings")
            files, manifest = _validate_fold(Path(value), selection=True)
            cutoff = datetime.fromisoformat(manifest["cutoff"])
            if previous_cutoff is not None and cutoff <= previous_cutoff:
                raise ImplicitALSExperimentError(
                    "selection folds must be ordered by increasing cutoff"
                )
            previous_cutoff = cutoff
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
            contexts.append(
                _build_context(
                    files=files,
                    manifest=manifest,
                    history=history,
                    target_users=target_users,
                    target_ground_truth=ground_truth,
                    fixed=fixed,
                    candidate_k=candidate_k,
                    predict_batch_size=predict_batch_size,
                    seed=seed,
                    cache_path=cache_dir / f"fold_{fold_index}.parquet",
                    reporter=reporter,
                    scope="selection_fixed_sources",
                    fold_label=f"rolling_{fold_index + 1}",
                )
            )
        phase_runtime["selection_fixed_sources_seconds"] = (
            time.perf_counter() - fixed_started
        )
        reporter.phase_finish(
            "selection_fixed_sources",
            fixed_started,
            folds=len(contexts),
        )

        stages: list[dict[str, Any]] = []
        profile_configs = [
            replace(
                base,
                config_id=f"stage1_confidence_{profile['name']}",
                **{key: value for key, value in profile.items() if key != "name"},
            )
            for profile in profiles
        ]
        stage_phase_started = reporter.phase_start(
            "stage1_confidence_profile"
        )
        stage, winner, retained_loaders, retained_results = _evaluate_stage(
            name="stage1_confidence_profile",
            changed_block="confidence_weights",
            contexts=contexts,
            configs=profile_configs,
            retained_config=None,
            retained_loaders=None,
            previous_results={},
            candidate_k=candidate_k,
            final_k=final_k,
            predict_batch_size=predict_batch_size,
            seed=seed,
            reporter=reporter,
            best_model_callback=best_model_callback,
        )
        stages.append(stage)
        reporter.phase_finish(
            "stage1_confidence_profile",
            stage_phase_started,
            winner=winner.config_id,
        )

        stage_specs = (
            ("stage2_decay", "half_life_hours", decay_values),
            ("stage3_factors", "factors", factor_values),
            (
                "stage4_regularization",
                "regularization",
                regularization_values,
            ),
            ("stage5_iterations", "iterations", iteration_values),
        )
        for stage_name, field, values in stage_specs:
            stage_phase_started = reporter.phase_start(stage_name)
            configs = _variants(
                winner,
                stage=stage_name,
                field=field,
                values=values,
            )
            previous = {
                winner.config_id: retained_results[winner.config_id]
            }
            stage, new_winner, new_loaders, new_results = _evaluate_stage(
                name=stage_name,
                changed_block=field,
                contexts=contexts,
                configs=configs,
                retained_config=winner,
                retained_loaders=retained_loaders,
                previous_results=previous,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
                seed=seed,
                reporter=reporter,
                best_model_callback=best_model_callback,
            )
            stages.append(stage)
            winner = new_winner
            retained_loaders = new_loaders
            retained_results = new_results
            reporter.phase_finish(
                stage_name,
                stage_phase_started,
                winner=winner.config_id,
            )
            gc.collect()
        phase_runtime["rolling_selection_seconds"] = sum(
            stage["runtime_seconds"] for stage in stages
        )
        selection_baselines = [context.baseline_metrics for context in contexts]
        selection_manifests = [context.manifest for context in contexts]
        positive_mean_gain = fmean(
            retained_results[winner.config_id][index][
                "union_oracle_p20_all_targets_gain"
            ]
            for index in range(3)
        )
        selection_exclusive_hits = sum(
            retained_results[winner.config_id][index]["exclusive_hits"]
            for index in range(3)
        )
        shutil.rmtree(cache_dir)
        del contexts, retained_loaders
        gc.collect()

        # Canonical isolation boundary: no canonical fold file is opened above.
        canonical_started = time.perf_counter()
        canonical_fixed_started = reporter.phase_start(
            "canonical_fixed_sources"
        )
        canonical_files, canonical_manifest = _validate_fold(
            Path(canonical_value), selection=False
        )
        canonical_cutoff = datetime.fromisoformat(canonical_manifest["cutoff"])
        if previous_cutoff is not None and canonical_cutoff <= previous_cutoff:
            raise ImplicitALSExperimentError(
                "canonical cutoff must be later than selection cutoffs"
            )
        target_users, ground_truth = _load_evaluation_frames(
            canonical_files, smoke_user_limit=smoke_user_limit
        )
        canonical_history: Path | pl.DataFrame = canonical_files[
            "history_daily.parquet"
        ]
        if smoke_context_user_limit is not None:
            canonical_history = _limited_history(
                canonical_files["history_daily.parquet"],
                target_users,
                context_user_limit=smoke_context_user_limit,
            )
            ground_truth = ground_truth.join(
                canonical_history.select("item_id").unique(),
                on="item_id",
                how="semi",
            )
        canonical_cache = staging / "canonical_union_hits.parquet"
        canonical_context = _build_context(
            files=canonical_files,
            manifest=canonical_manifest,
            history=canonical_history,
            target_users=target_users,
            target_ground_truth=ground_truth,
            fixed=fixed,
            candidate_k=candidate_k,
            predict_batch_size=predict_batch_size,
            seed=seed,
            cache_path=canonical_cache,
            reporter=reporter,
            scope="canonical_fixed_sources",
            fold_label="canonical",
        )
        reporter.phase_finish(
            "canonical_fixed_sources",
            canonical_fixed_started,
            cutoff=canonical_cutoff.isoformat(),
        )
        canonical_model_started = reporter.phase_start("canonical_als")
        canonical_loader_started = time.perf_counter()
        reporter.event(
            "operation_start",
            stage="canonical_als",
            config=winner.config_id,
            fold="canonical",
            operation="prepare_loader",
        )
        canonical_loader = _prepare_als_loader(
            context=canonical_context, config=winner, seed=seed
        )
        phase_runtime["canonical_loader_prepare_seconds"] = (
            time.perf_counter() - canonical_loader_started
        )
        reporter.event(
            "operation_finish",
            stage="canonical_als",
            config=winner.config_id,
            fold="canonical",
            operation="prepare_loader",
            duration_seconds=phase_runtime[
                "canonical_loader_prepare_seconds"
            ],
            users=canonical_loader.user_mapping.height,
            items=canonical_loader.item_mapping.height,
            interactions=canonical_loader.fit_matrix.nnz,
        )
        canonical_metrics, selected_model, candidates, recommendations = (
            _evaluate_model(
                context=canonical_context,
                loader=canonical_loader,
                config=winner,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
                keep_outputs=True,
                reporter=reporter,
                stage_name="canonical_als",
                fold_label="canonical",
            )
        )
        assert selected_model is not None
        assert candidates is not None
        assert recommendations is not None
        phase_runtime["canonical_fit_evaluate_seconds"] = canonical_metrics[
            "runtime_seconds"
        ]
        reporter.phase_finish(
            "canonical_als",
            canonical_model_started,
            winner=winner.config_id,
            precision_at_20_all_targets=canonical_metrics[
                "precision_at_20_all_targets"
            ],
        )

        artifact_started = reporter.phase_start("artifact_restore_publish")
        save_started = reporter.operation_start(
            stage="artifact_restore_publish",
            config=winner.config_id,
            fold="canonical",
            operation="save_model_and_recommendations",
        )
        recommendations_path = staging / "recommendations.parquet"
        recommendations.write_parquet(
            recommendations_path,
            compression="zstd",
            statistics=True,
            row_group_size=262_144,
        )
        history_checksum = (
            canonical_manifest["output_sha256"]["history_daily.parquet"]
            if smoke_context_user_limit is None
            else _dataframe_checksum(canonical_history)
        )
        selected_model.save(
            staging,
            metadata={
                "fit_reference_time": canonical_cutoff.isoformat(),
                "fit_history_sha256": history_checksum,
                "fit_history_scope": (
                    "full_fold_history"
                    if smoke_context_user_limit is None
                    else "limited_smoke_context"
                ),
            },
        )
        reporter.operation_finish(
            stage="artifact_restore_publish",
            config=winner.config_id,
            fold="canonical",
            operation="save_model_and_recommendations",
            started=save_started,
        )

        restore_started = time.perf_counter()
        restore_log_started = reporter.operation_start(
            stage="artifact_restore_publish",
            config=winner.config_id,
            fold="canonical",
            operation="restore_and_verify",
        )
        restored = ImplicitALSModel.from_artifact(staging)
        restored_loader = (
            ImplicitALSDataLoader(
                config=restored.config,
                reference_time=canonical_cutoff,
                user_mapping=restored.user_mapping,
                item_mapping=restored.item_mapping,
                seed=seed,
            )
            .load_predict_data(
                history=canonical_history, target_users=target_users
            )
            .prepare_predict_data()
        )
        restored_candidates = restored.predict(
            restored_loader, k=candidate_k, batch_size=predict_batch_size
        )
        restored_recommendations = fill_with_global_popularity(
            restored_candidates,
            canonical_context.fallback_candidates,
            target_users,
            _history_lazy(canonical_history),
            k=final_k,
        )
        deterministic_match = recommendations.equals(restored_recommendations)
        if not deterministic_match:
            raise RuntimeError("artifact-restored implicit ALS top-20 differs")
        user_factor_max_abs_diff = float(
            np.max(
                np.abs(
                    selected_model.backend.user_factors
                    - restored.backend.user_factors
                )
            )
        )
        item_factor_max_abs_diff = float(
            np.max(
                np.abs(
                    selected_model.backend.item_factors
                    - restored.backend.item_factors
                )
            )
        )
        phase_runtime["artifact_restore_seconds"] = (
            time.perf_counter() - restore_started
        )
        reporter.operation_finish(
            stage="artifact_restore_publish",
            config=winner.config_id,
            fold="canonical",
            operation="restore_and_verify",
            started=restore_log_started,
            deterministic_match=deterministic_match,
        )
        phase_runtime["canonical_total_seconds"] = (
            time.perf_counter() - canonical_started
        )
        canonical_cache.unlink()

        publish_started = reporter.operation_start(
            stage="artifact_restore_publish",
            config=winner.config_id,
            fold="canonical",
            operation="write_metadata_and_publish",
        )
        output_sha256 = {
            name: _sha256(staging / name)
            for name in (
                "model_config.json",
                "als_model.npz",
                "user_mapping.parquet",
                "item_mapping.parquet",
                "recommendations.parquet",
            )
        }
        mode = "limited_smoke" if smoke_user_limit is not None else "full"
        rolling_best_checkpoint = None
        if checkpoint_directory is not None:
            checkpoint_pointer = checkpoint_directory / "best_model.json"
            if not checkpoint_pointer.is_file():
                raise RuntimeError(
                    "rolling best-model callback did not publish a checkpoint"
                )
            rolling_best_checkpoint = {
                "directory": checkpoint_directory.as_posix(),
                **_read_json(checkpoint_pointer),
            }
        resolved = {
            "run_id": run_id,
            "artifact_version": 1,
            "mode": mode,
            "seed": seed,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "predict_batch_size": predict_batch_size,
            "fallback_score_type": source["fallback_score_type"],
            "base_config": base.to_dict(),
            "ablation": source["ablation"],
            "selection": {
                "fold_count": 3,
                "canonical_isolation": True,
                "primary_metric": (
                    "mean union_oracle_p20_all_targets_gain vs fixed "
                    "task02+task03+task04"
                ),
                "tie_break": [
                    "ALS standalone candidate_oracle_p20_all_targets",
                    "ALS standalone precision_at_20_all_targets after fallback",
                    "lower runtime",
                    "configured order",
                ],
                "selected_config_id": winner.config_id,
                "selected_config": winner.to_dict(),
            },
            "folds": {
                "selection": selection_manifests,
                "canonical": canonical_manifest,
            },
            "fixed_source_configs": {
                "task02": fixed.global_score.value,
                "task03": fixed.recency.to_dict(),
                "task04": fixed.item2item.to_dict(),
            },
            "task02_artifact": source["task02_artifact"],
            "task03_artifact": source["task03_artifact"],
            "task04_artifact": source["task04_artifact"],
            "model_config": selected_model.get_config(),
            "data_loader_config": canonical_loader.get_config(),
            "fit_cutoff": canonical_cutoff.isoformat(),
            "fit_history_sha256": history_checksum,
            "library_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "polars": pl.__version__,
                "implicit": implicit.__version__,
            },
            "threading": {
                "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
                "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "implicit_num_threads": winner.num_threads,
            },
            "smoke_user_limit": smoke_user_limit,
            "smoke_context_user_limit": smoke_context_user_limit,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": source_config_sha256,
            "rolling_best_model_checkpoint": rolling_best_checkpoint,
        }
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "selected_config_id": winner.config_id,
            "selected_config": winner.to_dict(),
            "selection_stages": stages,
            "selection_baseline_union": selection_baselines,
            "selection_mean_union_oracle_p20_all_targets_gain": positive_mean_gain,
            "selection_als_exclusive_hits": selection_exclusive_hits,
            "incremental_value_confirmed": (
                positive_mean_gain > 0 and selection_exclusive_hits > 0
            ),
            "canonical_evaluated_config_count": 1,
            "canonical_fixed_union": canonical_context.baseline_metrics,
            "canonical": canonical_metrics,
            **{
                key: canonical_metrics[key]
                for key in (
                    "precision_at_20_all_targets",
                    "precision_at_20_labeled_users",
                    "candidate_recall",
                    "candidate_user_hit_rate",
                    "candidate_oracle_p20_all_targets",
                    "candidate_oracle_p20_labeled_users",
                    "coverage",
                    "mean_candidate_count",
                    "p50_candidate_count",
                    "p90_candidate_count",
                    "p95_candidate_count",
                    "p99_candidate_count",
                    "final_hits",
                    "exclusive_hits",
                    "fallback_positions",
                    "fallback_users",
                    "union_candidate_recall",
                    "union_candidate_user_hit_rate",
                    "union_candidate_oracle_p20_all_targets",
                    "union_candidate_oracle_p20_labeled_users",
                    "union_oracle_p20_all_targets_gain",
                )
            },
            "deterministic_recommendations_match": deterministic_match,
            "rolling_best_model_checkpoint": rolling_best_checkpoint,
            "artifact_restore_tolerance": {
                "top20_ids_and_ranks": "exact",
                "scores_and_factors_max_abs": 1e-7,
                "observed_user_factor_max_abs_diff": user_factor_max_abs_diff,
                "observed_item_factor_max_abs_diff": item_factor_max_abs_diff,
            },
            "output_sha256": output_sha256,
            "runtime_by_phase_seconds": phase_runtime,
            "runtime_seconds": time.perf_counter() - started,
            "canonical_runtime_seconds": phase_runtime["canonical_total_seconds"],
            "peak_memory_mb": _peak_memory_mb(),
        }
        _write_json(staging / "config.json", resolved)
        _write_json(staging / "metrics.json", metrics)
        os.replace(staging, output)
        reporter.operation_finish(
            stage="artifact_restore_publish",
            config=winner.config_id,
            fold="canonical",
            operation="write_metadata_and_publish",
            started=publish_started,
            output=output.as_posix(),
        )
        reporter.phase_finish(
            "artifact_restore_publish",
            artifact_started,
            output=output.as_posix(),
        )
        reporter.event(
            "run_finish",
            run_id=run_id,
            status="completed",
            duration_seconds=time.perf_counter() - started,
            output=output.as_posix(),
        )
    except BaseException as error:
        reporter.event(
            "run_finish",
            run_id=run_id,
            status="failed",
            duration_seconds=time.perf_counter() - started,
            error_type=type(error).__name__,
            error=str(error),
        )
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        reporter.close()
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select CPU implicit ALS on three rolling folds and evaluate "
            "exactly one winner on canonical holdout."
        )
    )
    parser.add_argument("--config", default="configs/task05_implicit_als_v1.json")
    parser.add_argument("--output-dir", default="artifacts/task05_implicit_als_v1")
    parser.add_argument("--run-id", default="task05_implicit_als_v1")
    parser.add_argument("--smoke-user-limit", type=int)
    parser.add_argument("--smoke-context-user-limit", type=int)
    parser.add_argument(
        "--log-file",
        default="logs/task05_implicit_als_v1.log",
        help=(
            "Compact rotating event log (1 MB, two backups). "
            "Use an empty value to disable file logging."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars; file event logging remains enabled.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--best-model-dir",
        help=(
            "Directory for the atomic rolling best-model checkpoint. "
            "Defaults next to output-dir."
        ),
    )
    checkpoint_group.add_argument(
        "--no-best-model-checkpoint",
        action="store_true",
        help="Disable rolling best-model checkpoint saves.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    best_model_dir = None
    if not args.no_best_model_checkpoint:
        best_model_dir = args.best_model_dir or (
            output_dir.parent / f".{output_dir.name}.best-model"
        )
    metrics = run_experiment(
        config_path=args.config,
        output_dir=output_dir,
        run_id=args.run_id,
        smoke_user_limit=args.smoke_user_limit,
        smoke_context_user_limit=args.smoke_context_user_limit,
        log_file=args.log_file or None,
        show_progress=not args.no_progress,
        best_model_dir=best_model_dir,
    )
    summary = {
        "run_id": metrics["run_id"],
        "mode": metrics["mode"],
        "selected_config_id": metrics["selected_config_id"],
        "precision_at_20_all_targets": metrics["precision_at_20_all_targets"],
        "precision_at_20_labeled_users": metrics[
            "precision_at_20_labeled_users"
        ],
        "candidate_recall": metrics["candidate_recall"],
        "selection_mean_union_oracle_p20_all_targets_gain": metrics[
            "selection_mean_union_oracle_p20_all_targets_gain"
        ],
        "exclusive_hits": metrics["exclusive_hits"],
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_memory_mb": metrics["peak_memory_mb"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

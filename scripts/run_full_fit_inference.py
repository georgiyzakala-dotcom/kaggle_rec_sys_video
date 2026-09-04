#!/usr/bin/env python3
"""Run the checkpointed Task 11 full-history fit and inference pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import catboost
import polars as pl
from catboost import Pool

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from data_utils import TARGET_USER_SCHEMA
from experiment_utils import (
    CheckpointStore,
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from features import HistoryFeatureConfig
from pipeline import (
    SOURCE_ORDER,
    TRAINING_FOLDS,
    FullFitPipelineError,
    atomic_write_parquet,
    build_candidate_shard,
    build_feature_lookups,
    build_feature_shard,
    build_sampling_plan,
    candidate_union_config,
    derive_prediction_times,
    fit_candidate_source,
    load_candidate_source,
    load_feature_lookups,
    load_frozen_candidate_configs,
    materialize_full_history,
    quantize_parquet_parts_via_bounded_dsv,
    recommendations_with_fallback,
    sample_training_part,
    verify_checksums_manifest,
    write_checksums_manifest,
)
from rankers import (
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
)
from submission import (
    read_submission,
    submission_schema,
    validate_submission_against_artifacts,
    write_submission,
)
from validation import (
    ContractValidationError,
    validate_final_recommendations,
    validate_recommendations_against_history,
)

EXPECTED_CATBOOST_VERSION = "1.2.10"
TOTAL_PHASES = 12
GIB = 1024**3


class GracefulStop(RuntimeError):
    """Raised at an atomic boundary after SIGTERM or a resource stop."""


class SimulatedInterruption(GracefulStop):
    """Used only to prove checkpoint resume in the limited smoke run."""


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit frozen Task 11 candidate sources on full history, build one "
            "four-fold supervised CatBoost pool, infer target users, and validate "
            "submission.csv."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--validate-submission-only", action="store_true")
    parser.add_argument("--cleanup-recoverable", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--stop-after-checkpoints",
        type=int,
        help="Smoke-test hook: exit with code 75 after N new checkpoints.",
    )
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _load_config(path: Path) -> dict[str, Any]:
    source = read_json(path)
    required = {
        "artifact_version",
        "kind",
        "run_id",
        "mode",
        "seed",
        "paths",
        "candidates",
        "features",
        "training",
        "catboost",
        "inference",
        "resources",
        "logging",
        "smoke",
    }
    missing = required.difference(source)
    if missing:
        raise FullFitPipelineError(f"Task 11 config lacks keys: {sorted(missing)}")
    if source["artifact_version"] != 1 or source["kind"] != "task11_full_fit_inference":
        raise FullFitPipelineError("invalid Task 11 config identity")
    if source["mode"] not in {"full", "limited_smoke"}:
        raise FullFitPipelineError("mode must be full or limited_smoke")
    if source["seed"] != 42 or source["training"]["fold_order"] != list(TRAINING_FOLDS):
        raise FullFitPipelineError("Task 11 requires seed 42 and the frozen fold order")
    if "scale_pos_weight" in source["catboost"]:
        raise FullFitPipelineError("Task 11 config must not set scale_pos_weight")
    CatBoostPointwiseConfig.from_mapping(source["catboost"])
    if source["mode"] == "full":
        required_model = {
            "loss_function": "Logloss",
            "iterations": 1030,
            "depth": 7,
            "learning_rate": 0.08,
            "l2_leaf_reg": 3.0,
            "border_count": 32,
            "random_seed": 42,
            "task_type": "GPU",
            "devices": "0",
            "thread_count": 8,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "random_strength": 1.0,
        }
        differences = {
            name: source["catboost"].get(name)
            for name, expected in required_model.items()
            if source["catboost"].get(name) != expected
        }
        if differences:
            raise FullFitPipelineError(f"production CatBoost config is not frozen: {differences}")
        candidates = source["candidates"]
        if candidates["source_candidate_k"] != 200 or candidates["materialized_total_cap"] != 800:
            raise FullFitPipelineError("production candidate caps must be 200/800")
        if source["training"]["target_rows"] < 45_000_000 or source["training"]["target_rows"] > 47_000_000:
            raise FullFitPipelineError("production training target must be 45-47 million rows")
        if source["smoke"]["user_limit"] is not None:
            raise FullFitPipelineError("production run cannot limit target users")
    return source


def _target_users(config: Mapping[str, Any]) -> pl.DataFrame:
    path = _resolve(config["paths"]["target_users"])
    frame = pl.read_parquet(path).select("user_id").cast(TARGET_USER_SCHEMA)
    if frame.null_count().item() or frame.get_column("user_id").n_unique() != frame.height:
        raise ContractValidationError("target_user_ids must be unique and non-null")
    frame = frame.sort("user_id")
    limit = config["smoke"].get("user_limit")
    if limit is not None:
        frame = frame.head(int(limit))
    if frame.is_empty():
        raise ContractValidationError("target user universe is empty")
    return frame


def _gpu_diagnostics() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 4:
        raise FullFitPipelineError("unexpected nvidia-smi output")
    return {
        "name": values[0],
        "driver_version": values[1],
        "memory_total_mib": int(values[2]),
        "memory_free_mib": int(values[3]),
    }


def _windows_g_free_gib(*, required: bool) -> float | None:
    executable = Path("/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not executable.is_file():
        if required:
            raise FullFitPipelineError("Windows PowerShell is unavailable for G: monitoring")
        return None
    try:
        result = subprocess.run(
            [
                executable.as_posix(),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[double](Get-PSDrive -Name 'G').Free / 1GB",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return float(result.stdout.strip().splitlines()[-1].replace(",", "."))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        if required:
            raise FullFitPipelineError(f"cannot read Windows G: free space: {error}") from error
        return None


def _rss_gib() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, flags=re.MULTILINE)
    if match is None:
        raise FullFitPipelineError("cannot read current RSS")
    return int(match.group(1)) * 1024 / GIB


class _ResourceGuard:
    def __init__(self, config: Mapping[str, Any], work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.peak_rss_gib = 0.0
        self.peak_vram_used_mib = 0
        self.initial_linux_free_gib: float | None = None
        self.minimum_linux_free_gib: float | None = None
        self.initial_windows_g_free_gib: float | None = None
        self.minimum_windows_g_free_gib: float | None = None
        self.last: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._monitor: threading.Thread | None = None

    def start(self) -> None:
        if self._monitor is not None:
            raise RuntimeError("resource monitor is already active")

        def monitor() -> None:
            while not self._stop_event.wait(2.0):
                try:
                    rss = _rss_gib()
                    gpu = _gpu_diagnostics()
                except (OSError, ValueError, subprocess.SubprocessError):
                    continue
                self.peak_rss_gib = max(self.peak_rss_gib, rss)
                self.peak_vram_used_mib = max(
                    self.peak_vram_used_mib,
                    gpu["memory_total_mib"] - gpu["memory_free_mib"],
                )

        self._monitor = threading.Thread(
            target=monitor, name="task11-resource-monitor", daemon=True
        )
        self._monitor.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor is not None:
            self._monitor.join(timeout=10)
            self._monitor = None

    def summary(self) -> dict[str, Any]:
        return {
            "peak_rss_gib": self.peak_rss_gib,
            "peak_vram_used_mib": self.peak_vram_used_mib,
            "initial_linux_free_gib": self.initial_linux_free_gib,
            "minimum_linux_free_gib": self.minimum_linux_free_gib,
            "linux_disk_growth_gib": (
                self.initial_linux_free_gib - self.minimum_linux_free_gib
                if self.initial_linux_free_gib is not None
                and self.minimum_linux_free_gib is not None
                else None
            ),
            "initial_windows_g_free_gib": self.initial_windows_g_free_gib,
            "minimum_windows_g_free_gib": self.minimum_windows_g_free_gib,
            "windows_g_growth_gib": (
                self.initial_windows_g_free_gib - self.minimum_windows_g_free_gib
                if self.initial_windows_g_free_gib is not None
                and self.minimum_windows_g_free_gib is not None
                else None
            ),
        }

    def check(self, operation: str, *, launch: bool = False) -> dict[str, Any]:
        resources = self.config["resources"]
        rss = _rss_gib()
        process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB
        self.peak_rss_gib = max(self.peak_rss_gib, rss, process_peak)
        linux_free = shutil.disk_usage(self.work_dir.parent).free / GIB
        windows_required = self.config["mode"] == "full"
        windows_free = _windows_g_free_gib(required=windows_required)
        gpu = _gpu_diagnostics()
        self.peak_vram_used_mib = max(
            self.peak_vram_used_mib,
            gpu["memory_total_mib"] - gpu["memory_free_mib"],
        )
        if self.initial_linux_free_gib is None:
            self.initial_linux_free_gib = linux_free
        self.minimum_linux_free_gib = min(
            self.minimum_linux_free_gib or linux_free, linux_free
        )
        if windows_free is not None:
            if self.initial_windows_g_free_gib is None:
                self.initial_windows_g_free_gib = windows_free
            self.minimum_windows_g_free_gib = min(
                self.minimum_windows_g_free_gib or windows_free, windows_free
            )
        available_ram = int(Path("/proc/meminfo").read_text(encoding="utf-8").split("MemAvailable:", 1)[1].split()[0]) * 1024 / GIB
        diagnostics = {
            "operation": operation,
            "timestamp": datetime.now().astimezone().isoformat(),
            "rss_gib": rss,
            "peak_rss_gib": self.peak_rss_gib,
            "peak_vram_used_mib": self.peak_vram_used_mib,
            "available_ram_gib": available_ram,
            "linux_free_gib": linux_free,
            "windows_g_free_gib": windows_free,
            "gpu": gpu,
        }
        self.last = diagnostics
        if self.peak_rss_gib > float(resources["maximum_rss_gib"]):
            raise GracefulStop(
                f"peak RSS {self.peak_rss_gib:.2f} GiB exceeds the configured maximum"
            )
        if launch:
            projected_gate = max(
                float(resources["minimum_windows_g_free_gib"]),
                float(resources["stop_windows_g_free_gib"])
                + float(resources["projected_peak_growth_gib"])
                + float(resources["disk_safety_margin_gib"]),
            )
            if windows_free is not None and windows_free < projected_gate:
                raise FullFitPipelineError(
                    f"Windows G: launch gate failed: {windows_free:.2f} < {projected_gate:.2f} GiB"
                )
            if linux_free < float(resources["minimum_linux_free_gib"]):
                raise FullFitPipelineError("Linux filesystem launch gate failed")
            if available_ram < float(resources["minimum_available_ram_gib"]):
                raise FullFitPipelineError("available RAM launch gate failed")
            if gpu["memory_free_mib"] < int(resources["minimum_free_vram_mib"]):
                raise FullFitPipelineError("free VRAM launch gate failed")
        elif windows_free is not None and windows_free < float(resources["stop_windows_g_free_gib"]):
            raise GracefulStop("Windows G: free space crossed the graceful-stop threshold")
        return diagnostics


def _atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ContractValidationError(f"existing copy differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _files_record(work_dir: Path, paths: Sequence[Path]) -> dict[str, str]:
    files: dict[str, str] = {}
    for value in paths:
        if value.is_dir():
            candidates = sorted(path for path in value.rglob("*") if path.is_file())
        else:
            candidates = [value]
        for path in candidates:
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = path.resolve().relative_to(work_dir.resolve()).as_posix()
            files[relative] = sha256_file(path)
    return files


def _checkpoint_valid(
    store: CheckpointStore,
    work_dir: Path,
    *,
    stage: str,
    config: str,
    fold: str,
) -> dict[str, Any] | None:
    record = store.get(stage=stage, config=config, fold=fold)
    if record is None:
        return None
    files = record.get("files")
    if not isinstance(files, dict):
        raise ContractValidationError("checkpoint record lacks file checksums")
    for relative, digest in files.items():
        path = work_dir / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ContractValidationError(f"checkpoint file differs: {relative}")
    return record


class _CheckpointWriter:
    def __init__(
        self,
        store: CheckpointStore,
        work_dir: Path,
        stop_after: int | None,
    ) -> None:
        self.store = store
        self.work_dir = work_dir
        self.stop_after = stop_after
        self.new_count = 0

    def complete(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        paths: Sequence[Path],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = {"files": _files_record(self.work_dir, paths), **dict(metadata or {})}
        self.store.complete(stage=stage, config=config, fold=fold, metadata=record)
        self.new_count += 1
        if self.stop_after is not None and self.new_count >= self.stop_after:
            raise SimulatedInterruption(
                f"simulated interruption after {self.new_count} new checkpoints"
            )


def _validate_task07_inputs(
    config: Mapping[str, Any], *, verify_checksums: bool
) -> tuple[
    Path,
    list[str],
    dict[str, str],
    dict[str, list[Path]],
    dict[str, Any],
]:
    root = _resolve(config["paths"]["task07_dataset"])
    root_manifest = read_json(root / "dataset_manifest.json")
    schema = read_json(root / "feature_schema.json")
    features = schema.get("feature_columns")
    if not isinstance(features, list) or len(features) != 201 or len(set(features)) != 201:
        raise ContractValidationError("Task 07 feature schema must contain 201 unique features")
    all_dtypes = {
        column["name"]: column["dtype"]
        for column in schema.get("columns", [])
        if isinstance(column, dict) and "name" in column and "dtype" in column
    }
    feature_dtypes = {name: all_dtypes.get(name) for name in features}
    if any(dtype is None for dtype in feature_dtypes.values()):
        raise ContractValidationError("Task 07 feature schema lacks feature dtypes")
    model_config = read_json(_resolve(config["paths"]["task08_model_config"]))
    if model_config.get("feature_columns") != features or model_config.get("tree_count") != 1030:
        raise ContractValidationError("Task 08 model does not match the frozen Task 07 schema/tree budget")
    parts: dict[str, list[Path]] = {}
    limit = config["training"].get("source_part_limit_per_fold")
    provenance: dict[str, Any] = {
        "dataset_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
        "feature_schema_sha256": sha256_file(root / "feature_schema.json"),
        "folds": {},
    }
    for fold in TRAINING_FOLDS:
        relative_manifest = root_manifest["folds"][fold]["manifest"]
        manifest_path = root / relative_manifest
        if sha256_file(manifest_path) != root_manifest["folds"][fold]["manifest_sha256"]:
            raise ContractValidationError(f"Task 07 {fold} manifest checksum differs")
        manifest = read_json(manifest_path)
        names = sorted(manifest["parts"])
        if limit is not None:
            names = names[: int(limit)]
        fold_parts = [manifest_path.parent / "ranker_data" / name for name in names]
        for name, path in zip(names, fold_parts, strict=True):
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_checksums and sha256_file(path) != manifest["parts"][name]:
                raise ContractValidationError(f"Task 07 part checksum differs: {fold}/{name}")
        parts[fold] = fold_parts
        provenance["folds"][fold] = {
            "manifest": relative_manifest,
            "manifest_sha256": sha256_file(manifest_path),
            "part_count": len(fold_parts),
            "source_rows": root_manifest["folds"][fold]["rows"],
            "source_training_rows": root_manifest["folds"][fold]["training_rows"],
            "source_positive_rows": root_manifest["folds"][fold]["positive_rows"],
        }
    return root, list(features), feature_dtypes, parts, provenance


def _prediction_times_json(values: Mapping[str, datetime]) -> dict[str, str]:
    return {name: value.isoformat() for name, value in values.items()}


def _history_for_users(history: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    minimum = users.get_column("user_id").min()
    maximum = users.get_column("user_id").max()
    narrowed = history.filter(pl.col("user_id").is_between(minimum, maximum))
    return narrowed.join(users, on="user_id", how="semi")


def _shards(users: pl.DataFrame, size: int) -> list[tuple[int, pl.DataFrame]]:
    if size <= 0:
        raise ValueError("users_per_shard must be positive")
    return [
        (index, users.slice(offset, size))
        for index, offset in enumerate(range(0, users.height, size))
    ]


def _predict_scores(
    model: CatBoostPointwiseModel,
    frame: pl.DataFrame,
    features: Sequence[str],
    batch_size: int,
) -> pl.DataFrame:
    loader = CatBoostRankerDataLoader(feature_columns=features, seed=42)
    loader.load_predict_data(frame=frame).prepare_predict_data()
    return model.predict(loader, batch_size=batch_size)


def _write_owned_marker(work_dir: Path, run_id: str, digest: str) -> None:
    marker = work_dir / ".task11-owner.json"
    expected = {"artifact_version": 1, "run_id": run_id, "config_sha256": digest}
    if marker.exists():
        if read_json(marker) != expected:
            raise FullFitPipelineError(f"work directory has incompatible owner: {work_dir}")
    else:
        write_json_atomic(marker, expected)


def _copy_logs(log_file: Path | None, destination: Path) -> list[Path]:
    if log_file is None or not log_file.exists():
        return []
    outputs: list[Path] = []
    for source in sorted(log_file.parent.glob(f"{log_file.name}*")):
        if source.is_file():
            target = destination / source.name
            _atomic_copy(source, target)
            outputs.append(target)
    return outputs


def run_pipeline(
    *,
    config_path: Path,
    output_dir: Path,
    work_dir: Path,
    log_file: Path | None,
    show_progress: bool,
    stop_after_checkpoints: int | None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    run_id = str(config["run_id"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed artifact: {output_dir}")
    digest = config_sha256(config)
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_owned_marker(work_dir, run_id, digest)
    publish = work_dir / "publish"
    publish.mkdir(exist_ok=True)
    checkpoint = CheckpointStore(work_dir / "checkpoint", run_id=run_id, config_digest=digest)
    checkpoints = _CheckpointWriter(checkpoint, work_dir, stop_after_checkpoints)
    reporter = EventProgressReporter(
        task_name=run_id,
        total_phases=TOTAL_PHASES,
        log_file=log_file,
        show_progress=show_progress,
        log_max_bytes=int(config["logging"]["max_bytes"]),
        log_backup_count=int(config["logging"]["backup_count"]),
    )
    guard = _ResourceGuard(config, work_dir)
    guard.start()

    def check_resources(operation: str, *, launch: bool = False) -> dict[str, Any]:
        diagnostics = guard.check(operation, launch=launch)
        reporter.event(
            "resource_diagnostics",
            stage="resources",
            config=str(config["mode"]),
            fold="all",
            operation=operation,
            **{name: value for name, value in diagnostics.items() if name != "operation"},
        )
        return diagnostics

    started_run = time.perf_counter()
    reporter_closed = False
    signal_received: list[int] = []

    def stop_handler(signum: int, _frame: Any) -> None:
        signal_received.append(signum)
        raise GracefulStop(f"received signal {signum}")

    old_term = signal.signal(signal.SIGTERM, stop_handler)
    old_int = signal.signal(signal.SIGINT, stop_handler)
    reporter.event("run_start", run_id=run_id, mode=config["mode"], pid=os.getpid())
    try:
        phase = reporter.phase_start("preflight")
        if sys.version_info[:2] != (3, 12):
            raise FullFitPipelineError("Task 11 requires Python 3.12")
        if catboost.__version__ != EXPECTED_CATBOOST_VERSION:
            raise FullFitPipelineError(
                f"Task 11 requires catboost {EXPECTED_CATBOOST_VERSION}, got {catboost.__version__}"
            )
        resources_start = check_resources("launch", launch=True)
        target_users = _target_users(config)
        train_path = _resolve(config["paths"]["train"])
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        missing_targets = target_users.join(
            pl.scan_parquet(train_path).select("user_id").unique().collect(engine="streaming"),
            on="user_id",
            how="anti",
        ).height
        if missing_targets:
            raise ContractValidationError(f"{missing_targets} target users are absent from raw train")
        (
            task07_root,
            feature_columns,
            feature_dtypes,
            fold_parts,
            task07_provenance,
        ) = _validate_task07_inputs(
            config, verify_checksums=True
        )
        del task07_root
        borders_source = _resolve(config["paths"]["task08_borders"])
        expected_borders = config["training"]["quantization"]["input_borders_sha256"]
        if sha256_file(borders_source) != expected_borders:
            raise ContractValidationError("Task 08 quantization borders checksum differs")
        prediction_times = derive_prediction_times(train_path)
        write_json_atomic(publish / "config.json", config)
        target_snapshot = publish / "target_users.parquet"
        if not target_snapshot.exists():
            atomic_write_parquet(target_users, target_snapshot)
        reporter.phase_finish(
            "preflight",
            phase,
            target_users=target_users.height,
            feature_count=len(feature_columns),
            resources=resources_start,
        )

        phase = reporter.phase_start("full_history")
        history_path = publish / "full_history" / "history_daily.parquet"
        history_manifest = publish / "full_history" / "manifest.json"
        if _checkpoint_valid(checkpoint, work_dir, stage="full_history", config="daily", fold="all") is None:
            selected = target_users if config["mode"] == "limited_smoke" else None
            diagnostics = materialize_full_history(train_path, history_path, selected_users=selected)
            manifest = {
                "artifact_version": 1,
                "kind": "full_history_daily_interactions",
                "scope": "all_raw_train" if selected is None else "limited_smoke_target_users",
                "prediction_times": _prediction_times_json(prediction_times),
                "diagnostics": diagnostics,
            }
            write_json_atomic(history_manifest, manifest)
            checkpoints.complete(
                stage="full_history",
                config="daily",
                fold="all",
                paths=[history_path, history_manifest],
                metadata={"rows": diagnostics["rows"]},
            )
        history_diagnostics = read_json(history_manifest)["diagnostics"]
        reporter.phase_finish("full_history", phase, rows=history_diagnostics["rows"])
        check_resources("after_full_history")

        phase = reporter.phase_start("fit_candidate_sources")
        winner_paths = {
            name: _resolve(path)
            for name, path in config["paths"]["winner_artifacts"].items()
        }
        frozen_configs, candidate_provenance = load_frozen_candidate_configs(winner_paths)
        source_stage = reporter.stage_start(stage="candidate_sources", total=len(SOURCE_ORDER), unit="source")
        for source_name in SOURCE_ORDER:
            reporter.stage_status(stage="candidate_sources", config=source_name, fold="full", operation="fit_or_restore")
            operation_started = reporter.operation_start(
                stage="candidate_sources",
                config=source_name,
                fold="full",
                operation="fit_or_restore",
            )
            source_dir = publish / "candidate_models" / source_name
            restored = _checkpoint_valid(
                checkpoint,
                work_dir,
                stage="candidate_source",
                config=source_name,
                fold="full",
            ) is not None
            if not restored:
                callback = None
                iteration = None
                if source_name == "implicit_als":
                    iteration, callback = reporter.iteration_start(
                        stage="candidate_sources",
                        config=source_name,
                        fold="full",
                        total=int(frozen_configs[source_name].iterations),
                        unit="iteration",
                    )
                try:
                    fit_candidate_source(
                        source_name,
                        config=frozen_configs[source_name],
                        history_path=history_path,
                        reference_time=prediction_times["prediction_start"],
                        destination=source_dir,
                        seed=int(config["seed"]),
                        als_callback=callback,
                    )
                finally:
                    if iteration is not None:
                        reporter.iteration_finish(
                            stage="candidate_sources",
                            config=source_name,
                            fold="full",
                            started=iteration,
                            status="finished" if source_dir.exists() else "failed",
                        )
                checkpoints.complete(
                    stage="candidate_source",
                    config=source_name,
                    fold="full",
                    paths=[source_dir],
                )
            reporter.operation_finish(
                stage="candidate_sources",
                config=source_name,
                fold="full",
                operation="fit_or_restore",
                started=operation_started,
                status="restored" if restored else "completed",
            )
            reporter.stage_advance()
            check_resources(f"after_candidate_source:{source_name}")
        reporter.stage_finish(stage="candidate_sources", started=source_stage)
        write_json_atomic(
            publish / "candidate_models" / "manifest.json",
            {
                "artifact_version": 1,
                "frozen_winner_provenance": candidate_provenance,
                "source_order": list(SOURCE_ORDER),
                "prediction_start": prediction_times["prediction_start"].isoformat(),
            },
        )
        reporter.phase_finish("fit_candidate_sources", phase)

        models = {
            name: load_candidate_source(
                name, publish / "candidate_models" / name, frozen_configs[name]
            )
            for name in SOURCE_ORDER
        }
        known_items = (
            pl.scan_parquet(history_path)
            .select("item_id")
            .unique()
            .sort("item_id")
            .collect(engine="streaming")
        )
        history = (
            pl.scan_parquet(history_path)
            .join(target_users.lazy(), on="user_id", how="semi")
            .collect(engine="streaming")
            .sort(("user_id", "item_id", "date"))
        )
        neighbor_table = models["item2item"].neighbor_table
        user_shards = _shards(target_users, int(config["candidates"]["users_per_shard"]))
        union_config = candidate_union_config(
            source_candidate_k=int(config["candidates"]["source_candidate_k"]),
            total_cap=int(config["candidates"]["materialized_total_cap"]),
        )
        phase = reporter.phase_start("candidate_generation")
        candidate_stage = reporter.stage_start(stage="candidate_shards", total=len(user_shards), unit="shard")
        candidate_manifests: list[dict[str, Any]] = []
        for index, users in user_shards:
            part = f"part-{index:05d}"
            reporter.stage_status(stage="candidate_shards", config=part, fold="full", operation="generate_or_restore")
            operation_started = reporter.operation_start(
                stage="candidate_shards",
                config=part,
                fold="full",
                operation="generate_or_restore",
            )
            union_path = publish / "candidate_shards" / f"{part}.parquet"
            seeds_path = publish / "candidate_shards" / f"{part}.seeds.parquet"
            manifest_path = publish / "candidate_shards" / f"{part}.manifest.json"
            restored = _checkpoint_valid(
                checkpoint,
                work_dir,
                stage="candidate_shard",
                config=part,
                fold="full",
            ) is not None
            if not restored:
                history_shard = _history_for_users(history, users)
                union, seeds, diagnostics = build_candidate_shard(
                    models=models,
                    history_shard=history_shard,
                    target_users=users,
                    reference_time=prediction_times["prediction_start"],
                    union_config=union_config,
                    predict_batch_sizes=config["candidates"]["predict_batch_sizes"],
                    known_items=known_items,
                    neighbor_table=neighbor_table,
                    seed=int(config["seed"]),
                    minimum_fallback_k=int(config["inference"]["final_k"]),
                )
                atomic_write_parquet(union, union_path)
                atomic_write_parquet(seeds, seeds_path)
                manifest = {
                    "artifact_version": 1,
                    "part": part,
                    "first_user_id": int(users.get_column("user_id").min()),
                    "last_user_id": int(users.get_column("user_id").max()),
                    "diagnostics": diagnostics,
                    "files": {
                        union_path.name: sha256_file(union_path),
                        seeds_path.name: sha256_file(seeds_path),
                    },
                }
                write_json_atomic(manifest_path, manifest)
                checkpoints.complete(
                    stage="candidate_shard",
                    config=part,
                    fold="full",
                    paths=[union_path, seeds_path, manifest_path],
                    metadata={"rows": union.height, "users": users.height},
                )
                del history_shard, union, seeds
                gc.collect()
            candidate_manifests.append(read_json(manifest_path))
            reporter.operation_finish(
                stage="candidate_shards",
                config=part,
                fold="full",
                operation="generate_or_restore",
                started=operation_started,
                status="restored" if restored else "completed",
                rows=candidate_manifests[-1]["diagnostics"]["rows"],
            )
            reporter.stage_advance()
            check_resources(f"after_candidate_shard:{part}")
        reporter.stage_finish(stage="candidate_shards", started=candidate_stage)
        write_json_atomic(
            publish / "candidate_shards" / "manifest.json",
            {
                "artifact_version": 1,
                "part_count": len(candidate_manifests),
                "target_users": target_users.height,
                "source_candidate_k": union_config.sources[0].cap,
                "materialized_total_cap": union_config.total_cap,
                "rows": sum(value["diagnostics"]["rows"] for value in candidate_manifests),
                "parts": [value["part"] for value in candidate_manifests],
                "source_contribution_rows": {
                    name: sum(value["diagnostics"]["source_contribution_rows"][name] for value in candidate_manifests)
                    for name in SOURCE_ORDER
                },
                "source_users": {
                    name: sum(value["diagnostics"]["source_users"][name] for value in candidate_manifests)
                    for name in SOURCE_ORDER
                },
                "source_user_coverage": {
                    name: sum(value["diagnostics"]["source_users"][name] for value in candidate_manifests) / target_users.height
                    for name in SOURCE_ORDER
                },
                "fallback_min_candidates": min(
                    value["diagnostics"]["fallback_min_candidates"]
                    for value in candidate_manifests
                ),
            },
        )
        reporter.phase_finish("candidate_generation", phase, shards=len(user_shards))
        del history
        gc.collect()

        phase = reporter.phase_start("feature_generation")
        feature_config = HistoryFeatureConfig(
            windows_hours=tuple(config["features"]["user_windows_hours"]),
            trend_windows_hours=tuple(config["features"]["item_trend_windows_hours"]),
        )
        if config["features"]["item_windows_hours"] != config["features"]["user_windows_hours"]:
            raise FullFitPipelineError("frozen user/item history windows must match")
        lookups_dir = publish / "feature_lookups"
        if _checkpoint_valid(checkpoint, work_dir, stage="feature_lookups", config="history", fold="full") is None:
            build_feature_lookups(
                history_path=history_path,
                als_model=models["implicit_als"],
                reference_time=prediction_times["prediction_start"],
                config=feature_config,
                destination=lookups_dir,
            )
            checkpoints.complete(
                stage="feature_lookups",
                config="history",
                fold="full",
                paths=[lookups_dir],
            )
        lookups = load_feature_lookups(lookups_dir)
        feature_stage = reporter.stage_start(stage="feature_shards", total=len(user_shards), unit="shard")
        feature_manifests: list[dict[str, Any]] = []
        for index, _users in user_shards:
            part = f"part-{index:05d}"
            reporter.stage_status(stage="feature_shards", config=part, fold="full", operation="build_or_restore")
            operation_started = reporter.operation_start(
                stage="feature_shards",
                config=part,
                fold="full",
                operation="build_or_restore",
            )
            feature_path = publish / "feature_shards" / f"{part}.parquet"
            manifest_path = publish / "feature_shards" / f"{part}.manifest.json"
            restored = _checkpoint_valid(
                checkpoint,
                work_dir,
                stage="feature_shard",
                config=part,
                fold="full",
            ) is not None
            if not restored:
                union = pl.read_parquet(publish / "candidate_shards" / f"{part}.parquet")
                seeds = pl.read_parquet(publish / "candidate_shards" / f"{part}.seeds.parquet")
                features = build_feature_shard(
                    union,
                    seeds=seeds,
                    lookups=lookups,
                    neighbor_table=neighbor_table,
                    item2item_config=frozen_configs["item2item"],
                    reference_time=prediction_times["prediction_start"],
                    expected_features=feature_columns,
                    expected_feature_dtypes=feature_dtypes,
                )
                atomic_write_parquet(features, feature_path)
                manifest = {
                    "artifact_version": 1,
                    "part": part,
                    "rows": features.height,
                    "feature_count": len(feature_columns),
                    "sha256": sha256_file(feature_path),
                }
                write_json_atomic(manifest_path, manifest)
                checkpoints.complete(
                    stage="feature_shard",
                    config=part,
                    fold="full",
                    paths=[feature_path, manifest_path],
                    metadata={"rows": features.height},
                )
                del union, seeds, features
                gc.collect()
            feature_manifests.append(read_json(manifest_path))
            reporter.operation_finish(
                stage="feature_shards",
                config=part,
                fold="full",
                operation="build_or_restore",
                started=operation_started,
                status="restored" if restored else "completed",
                rows=feature_manifests[-1]["rows"],
            )
            reporter.stage_advance()
            check_resources(f"after_feature_shard:{part}")
        reporter.stage_finish(stage="feature_shards", started=feature_stage)
        source_feature_schema = _resolve(config["paths"]["task07_dataset"]) / "feature_schema.json"
        _atomic_copy(source_feature_schema, publish / "feature_schema.json")
        write_json_atomic(
            publish / "feature_shards" / "manifest.json",
            {
                "artifact_version": 1,
                "part_count": len(feature_manifests),
                "feature_count": len(feature_columns),
                "rows": sum(value["rows"] for value in feature_manifests),
                "parts": [value["part"] for value in feature_manifests],
            },
        )
        reporter.phase_finish("feature_generation", phase, shards=len(user_shards))
        del known_items, lookups, models, neighbor_table
        gc.collect()

        phase = reporter.phase_start("training_sampling")
        sampling_plan_path = work_dir / "sampling_plan.json"
        if _checkpoint_valid(checkpoint, work_dir, stage="sampling_plan", config="four_folds", fold="all") is None:
            sampling_plan = build_sampling_plan(
                fold_parts, target_rows=int(config["training"]["target_rows"])
            )
            sampling_plan["task07_provenance"] = task07_provenance
            write_json_atomic(sampling_plan_path, sampling_plan)
            checkpoints.complete(
                stage="sampling_plan",
                config="four_folds",
                fold="all",
                paths=[sampling_plan_path],
            )
        sampling_plan = read_json(sampling_plan_path)
        sampled_parts: list[Path] = []
        sampled_manifests: list[dict[str, Any]] = []
        total_source_parts = sum(len(values) for values in fold_parts.values())
        sample_stage = reporter.stage_start(stage="training_pool_parts", total=total_source_parts, unit="part")
        for fold_id, fold in enumerate(TRAINING_FOLDS):
            probability = sampling_plan["folds"][fold]["secondary_negative_probability"]
            for source in fold_parts[fold]:
                part_name = source.stem
                key = f"{fold}-{part_name}"
                reporter.stage_status(stage="training_pool_parts", config=part_name, fold=fold, operation="sample_or_restore")
                operation_started = reporter.operation_start(
                    stage="training_pool_parts",
                    config=part_name,
                    fold=fold,
                    operation="sample_or_restore",
                )
                output = work_dir / "sampled_training" / fold / source.name
                manifest_path = output.with_suffix(".manifest.json")
                restored = _checkpoint_valid(
                    checkpoint,
                    work_dir,
                    stage="training_part",
                    config=part_name,
                    fold=fold,
                ) is not None
                if not restored:
                    frame = pl.read_parquet(source)
                    sampled, diagnostics = sample_training_part(
                        frame,
                        feature_columns=feature_columns,
                        fold=fold,
                        fold_id=fold_id,
                        seed=int(config["training"]["sampling_seed"]),
                        secondary_negative_probability=float(probability),
                    )
                    atomic_write_parquet(sampled, output)
                    manifest = {
                        "artifact_version": 1,
                        "key": key,
                        "fold": fold,
                        "source": source.relative_to(REPO_ROOT).as_posix(),
                        "source_sha256": sha256_file(source),
                        "secondary_negative_probability": probability,
                        "diagnostics": diagnostics,
                        "output_sha256": sha256_file(output),
                    }
                    write_json_atomic(manifest_path, manifest)
                    checkpoints.complete(
                        stage="training_part",
                        config=part_name,
                        fold=fold,
                        paths=[output, manifest_path],
                        metadata={"rows": sampled.height},
                    )
                    del frame, sampled
                    gc.collect()
                sampled_parts.append(output)
                sampled_manifests.append(read_json(manifest_path))
                reporter.operation_finish(
                    stage="training_pool_parts",
                    config=part_name,
                    fold=fold,
                    operation="sample_or_restore",
                    started=operation_started,
                    status="restored" if restored else "completed",
                    rows=sampled_manifests[-1]["diagnostics"]["rows"],
                )
                reporter.stage_advance()
                check_resources(f"after_training_part:{key}")
        reporter.stage_finish(stage="training_pool_parts", started=sample_stage)
        actual_by_fold = {
            fold: {
                "rows": sum(
                    value["diagnostics"]["rows"]
                    for value in sampled_manifests
                    if value["fold"] == fold
                ),
                "positive_rows": sum(
                    value["diagnostics"]["positive_rows"]
                    for value in sampled_manifests
                    if value["fold"] == fold
                ),
                "negative_rows": sum(
                    value["diagnostics"]["negative_rows"]
                    for value in sampled_manifests
                    if value["fold"] == fold
                ),
            }
            for fold in TRAINING_FOLDS
        }
        sampling_manifest = {
            **sampling_plan,
            "actual_rows": sum(value["diagnostics"]["rows"] for value in sampled_manifests),
            "actual_positive_rows": sum(value["diagnostics"]["positive_rows"] for value in sampled_manifests),
            "actual_negative_rows": sum(value["diagnostics"]["negative_rows"] for value in sampled_manifests),
            "part_count": len(sampled_manifests),
            "parts": sampled_manifests,
            "actual_by_fold": actual_by_fold,
            "fold_provenance_column": "fold_id",
            "object_weight_column": "sample_weight",
        }
        write_json_atomic(publish / "sampling_manifest.json", sampling_manifest)
        reporter.phase_finish(
            "training_sampling",
            phase,
            rows=sampling_manifest["actual_rows"],
            positive_rows=sampling_manifest["actual_positive_rows"],
        )

        phase = reporter.phase_start("quantized_pool")
        borders_path = publish / "quantization" / "borders.tsv"
        columns_path = publish / "quantization" / "columns.cd"
        pool_path = publish / "quantization" / "train.quantized"
        pool_manifest_path = publish / "final_training_pool_manifest.json"
        _atomic_copy(borders_source, borders_path)
        if _checkpoint_valid(checkpoint, work_dir, stage="quantized_pool", config="final", fold="all") is None:
            temporary_dsv = work_dir / "catboost" / "training.tsv"
            dsv_stage = reporter.stage_start(
                stage="quantized_pool_parts",
                total=len(sampled_parts),
                unit="part",
            )

            def dsv_progress(index: int, size_bytes: int) -> None:
                reporter.stage_status(
                    stage="quantized_pool_parts",
                    config=f"part-{index:05d}",
                    fold="all",
                    operation="write_bounded_dsv",
                )
                reporter.stage_advance()
                check_resources(f"after_dsv_part:{index:05d}")
                reporter.event(
                    "dsv_part_complete",
                    stage="quantized_pool_parts",
                    config=f"part-{index:05d}",
                    fold="all",
                    operation="write_bounded_dsv",
                    size_bytes=size_bytes,
                )

            try:
                pool_diagnostics = quantize_parquet_parts_via_bounded_dsv(
                    sampled_parts,
                    feature_columns=feature_columns,
                    borders_path=borders_path,
                    column_description_path=columns_path,
                    dsv_path=temporary_dsv,
                    output_path=pool_path,
                    maximum_dsv_bytes=int(
                        float(config["resources"]["maximum_temporary_dsv_gib"]) * GIB
                    ),
                    thread_count=int(config["resources"]["thread_count"]),
                    random_seed=int(config["seed"]),
                    progress_callback=dsv_progress,
                )
            finally:
                reporter.stage_finish(
                    stage="quantized_pool_parts", started=dsv_stage
                )
            pool_manifest = {
                "artifact_version": 1,
                "kind": "task11_final_quantized_training_pool",
                "fixed_single_fit": True,
                "eval_pool": None,
                "feature_columns": feature_columns,
                "feature_count": len(feature_columns),
                "diagnostics": pool_diagnostics,
                "quantization": config["training"]["quantization"],
                "borders_provenance": {
                    "source": config["paths"]["task08_borders"],
                    "sha256": sha256_file(borders_path),
                    "task07_feature_schema_sha256": task07_provenance["feature_schema_sha256"],
                },
            }
            write_json_atomic(pool_manifest_path, pool_manifest)
            checkpoints.complete(
                stage="quantized_pool",
                config="final",
                fold="all",
                paths=[pool_path, borders_path, columns_path, pool_manifest_path],
                metadata={"rows": pool_diagnostics["rows"]},
            )
        pool_manifest = read_json(pool_manifest_path)
        restored_pool = Pool(f"quantized://{pool_path.resolve().as_posix()}")
        if restored_pool.num_col() != len(feature_columns):
            raise ContractValidationError("final quantized pool has invalid feature count")
        del restored_pool
        gc.collect()
        reporter.phase_finish("quantized_pool", phase, rows=pool_manifest["diagnostics"]["rows"])
        check_resources("after_quantized_pool")

        phase = reporter.phase_start("catboost_fit")
        model_dir = publish / "model"
        feature_importance_path = publish / "feature_importance.parquet"
        training_manifest_path = publish / "training_manifest.json"
        if _checkpoint_valid(checkpoint, work_dir, stage="catboost_fit", config="fixed", fold="all") is None:
            pointwise_config = CatBoostPointwiseConfig.from_mapping(config["catboost"])
            loader = CatBoostRankerDataLoader(feature_columns=feature_columns, seed=int(config["seed"]))
            loader.load_fit_data(train_pool=pool_path).prepare_fit_data()
            model = CatBoostPointwiseModel(pointwise_config, feature_columns=feature_columns)
            iteration, callback = reporter.iteration_start(
                stage="catboost_fit",
                config=pointwise_config.config_id,
                fold="all_temporal_examples",
                total=pointwise_config.iterations,
                unit="tree",
            )
            bridge = _CatBoostLogBridge(callback)
            snapshot = work_dir / "catboost" / "snapshot.cbsnapshot"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            train_dir = work_dir / "catboost" / "train_dir"
            train_dir.mkdir(parents=True, exist_ok=True)
            fit_finished = False
            try:
                model.fit(
                    loader,
                    fixed_tree_budget=True,
                    train_dir=train_dir,
                    snapshot_file=snapshot,
                    log_cout=bridge,
                    log_cerr=bridge,
                )
                fit_finished = True
            finally:
                reporter.iteration_finish(
                    stage="catboost_fit",
                    config=pointwise_config.config_id,
                    fold="all_temporal_examples",
                    started=iteration,
                    status="finished" if fit_finished else "failed",
                )
            model.save(model_dir)
            atomic_write_parquet(model.get_feature_importance(), feature_importance_path)
            training_manifest = {
                "artifact_version": 1,
                "training_mode": "single_fixed_tree_budget_fit",
                "temporal_folds": list(TRAINING_FOLDS),
                "canonical_used_for_final_training_only": True,
                "local_evaluation_after_canonical_inclusion": False,
                "early_stopping": False,
                "eval_pool": None,
                "tree_count": model.tree_count,
                "feature_count": len(feature_columns),
                "training_pool_sha256": sha256_file(pool_path),
                "sampling_manifest_sha256": sha256_file(publish / "sampling_manifest.json"),
            }
            write_json_atomic(training_manifest_path, training_manifest)
            checkpoints.complete(
                stage="catboost_fit",
                config="fixed",
                fold="all",
                paths=[model_dir, feature_importance_path, training_manifest_path],
                metadata={"tree_count": model.tree_count},
            )
            del loader, model
            gc.collect()
        model = CatBoostPointwiseModel.from_artifact(model_dir)
        if model.tree_count != int(config["catboost"]["iterations"]):
            raise ContractValidationError("portable final model has an invalid tree count")
        reporter.phase_finish("catboost_fit", phase, tree_count=model.tree_count)
        check_resources("after_catboost_fit")

        phase = reporter.phase_start("inference")
        inference_stage = reporter.stage_start(stage="inference_shards", total=len(user_shards), unit="shard")
        recommendation_parts: list[Path] = []
        long_parts: list[Path] = []
        inference_manifests: list[dict[str, Any]] = []
        score_parts: list[Path] = []
        for index, users in user_shards:
            part = f"part-{index:05d}"
            reporter.stage_status(stage="inference_shards", config=part, fold="full", operation="predict_or_restore")
            operation_started = reporter.operation_start(
                stage="inference_shards",
                config=part,
                fold="full",
                operation="predict_or_restore",
            )
            feature_path = publish / "feature_shards" / f"{part}.parquet"
            union_path = publish / "candidate_shards" / f"{part}.parquet"
            rec_path = publish / "recommendation_shards" / f"{part}.parquet"
            long_path = publish / "recommendation_shards" / f"{part}.long.parquet"
            score_path = work_dir / "inference_scores" / f"{part}.parquet"
            manifest_path = publish / "recommendation_shards" / f"{part}.manifest.json"
            restored = _checkpoint_valid(
                checkpoint,
                work_dir,
                stage="inference_shard",
                config=part,
                fold="full",
            ) is not None
            if not restored:
                frame = pl.read_parquet(feature_path)
                scores = _predict_scores(
                    model,
                    frame,
                    feature_columns,
                    int(config["inference"]["batch_size"]),
                )
                union = pl.read_parquet(union_path)
                recommendations, long, diagnostics = recommendations_with_fallback(
                    scores,
                    union=union,
                    target_users=users,
                    k=int(config["inference"]["final_k"]),
                )
                atomic_write_parquet(scores, score_path)
                atomic_write_parquet(recommendations, rec_path)
                atomic_write_parquet(long, long_path)
                manifest = {
                    "artifact_version": 1,
                    "part": part,
                    "diagnostics": diagnostics,
                    "files": {
                        rec_path.name: sha256_file(rec_path),
                        long_path.name: sha256_file(long_path),
                        "temporary_scores_sha256": sha256_file(score_path),
                    },
                }
                write_json_atomic(manifest_path, manifest)
                checkpoints.complete(
                    stage="inference_shard",
                    config=part,
                    fold="full",
                    paths=[score_path, rec_path, long_path, manifest_path],
                    metadata={"users": recommendations.height},
                )
                del frame, scores, union, recommendations, long
                gc.collect()
            recommendation_parts.append(rec_path)
            long_parts.append(long_path)
            score_parts.append(score_path)
            inference_manifests.append(read_json(manifest_path))
            reporter.operation_finish(
                stage="inference_shards",
                config=part,
                fold="full",
                operation="predict_or_restore",
                started=operation_started,
                status="restored" if restored else "completed",
                users=inference_manifests[-1]["diagnostics"]["users"],
            )
            reporter.stage_advance()
            check_resources(f"after_inference_shard:{part}")
        reporter.stage_finish(stage="inference_shards", started=inference_stage)

        restored_model = CatBoostPointwiseModel.from_artifact(model_dir)
        repeat_stage = reporter.stage_start(stage="deterministic_repeat", total=len(user_shards), unit="shard")
        for index, _users in user_shards:
            part = f"part-{index:05d}"
            reporter.stage_status(stage="deterministic_repeat", config=part, fold="full", operation="portable_predict_compare")
            operation_started = reporter.operation_start(
                stage="deterministic_repeat",
                config=part,
                fold="full",
                operation="portable_predict_compare",
            )
            restored = _checkpoint_valid(
                checkpoint,
                work_dir,
                stage="inference_repeat",
                config=part,
                fold="full",
            ) is not None
            if not restored:
                frame = pl.read_parquet(publish / "feature_shards" / f"{part}.parquet")
                expected_scores = pl.read_parquet(work_dir / "inference_scores" / f"{part}.parquet")
                actual_scores = _predict_scores(
                    restored_model,
                    frame,
                    feature_columns,
                    int(config["inference"]["batch_size"]),
                )
                if not actual_scores.equals(expected_scores):
                    raise ContractValidationError(f"portable deterministic inference differs for {part}")
                checkpoints.complete(
                    stage="inference_repeat",
                    config=part,
                    fold="full",
                    paths=[
                        publish / "feature_shards" / f"{part}.parquet",
                        work_dir / "inference_scores" / f"{part}.parquet",
                        model_dir / "model.cbm",
                    ],
                    metadata={"deterministic": True},
                )
                del frame, expected_scores, actual_scores
                gc.collect()
            reporter.operation_finish(
                stage="deterministic_repeat",
                config=part,
                fold="full",
                operation="portable_predict_compare",
                started=operation_started,
                status="restored" if restored else "completed",
                deterministic=True,
            )
            reporter.stage_advance()
            check_resources(f"after_inference_repeat:{part}")
        reporter.stage_finish(stage="deterministic_repeat", started=repeat_stage)
        write_json_atomic(
            publish / "recommendation_shards" / "manifest.json",
            {
                "artifact_version": 1,
                "part_count": len(inference_manifests),
                "parts": [value["part"] for value in inference_manifests],
                "users": sum(value["diagnostics"]["users"] for value in inference_manifests),
                "fallback_users": sum(value["diagnostics"]["fallback_users"] for value in inference_manifests),
                "fallback_positions": sum(value["diagnostics"]["fallback_positions"] for value in inference_manifests),
                "portable_repeat_equal": True,
            },
        )
        reporter.phase_finish("inference", phase, shards=len(user_shards))

        phase = reporter.phase_start("recommendation_aggregation")
        internal_path = publish / "internal_recommendations.parquet"
        long_path = publish / "internal_recommendations_long.parquet"
        internal_manifest_path = publish / "internal_recommendations_manifest.json"
        if _checkpoint_valid(checkpoint, work_dir, stage="recommendation_aggregation", config="top20", fold="full") is None:
            recommendations = pl.concat([pl.read_parquet(path) for path in recommendation_parts]).sort("user_id")
            long_recommendations = pl.concat([pl.read_parquet(path) for path in long_parts]).sort(("user_id", "rank"))
            validate_final_recommendations(recommendations, expected_k=int(config["inference"]["final_k"]))
            validate_recommendations_against_history(
                recommendations,
                target_users=target_users,
                history_daily=pl.scan_parquet(history_path),
                expected_k=int(config["inference"]["final_k"]),
            )
            atomic_write_parquet(recommendations, internal_path)
            atomic_write_parquet(long_recommendations, long_path)
            internal_manifest = {
                "artifact_version": 1,
                "users": recommendations.height,
                "rows_long": long_recommendations.height,
                "items_per_user": int(config["inference"]["final_k"]),
                "schema": {name: str(dtype) for name, dtype in recommendations.schema.items()},
                "known_items_only": True,
                "seen_pairs": 0,
                "deterministic_rank_order": True,
                "files": {
                    internal_path.name: sha256_file(internal_path),
                    long_path.name: sha256_file(long_path),
                },
            }
            write_json_atomic(internal_manifest_path, internal_manifest)
            checkpoints.complete(
                stage="recommendation_aggregation",
                config="top20",
                fold="full",
                paths=[internal_path, long_path, internal_manifest_path],
                metadata={"users": recommendations.height},
            )
            del recommendations, long_recommendations
            gc.collect()
        reporter.phase_finish("recommendation_aggregation", phase, users=target_users.height)
        check_resources("after_recommendation_aggregation")

        phase = reporter.phase_start("submission_validation")
        recommendations = pl.read_parquet(internal_path)
        submission_path = publish / "submission.csv"
        if submission_path.exists():
            if not read_submission(submission_path).equals(recommendations.sort("user_id")):
                raise ContractValidationError("existing staging submission differs")
            serialization = {
                "filename": "submission.csv",
                "rows": recommendations.height,
                "sha256": sha256_file(submission_path),
                "round_trip_equal": True,
            }
        else:
            serialization = write_submission(recommendations, submission_path)
        validation = validate_submission_against_artifacts(
            submission_path,
            recommendations=recommendations,
            target_users=target_users,
            history_daily=pl.scan_parquet(history_path),
            expected_k=int(config["inference"]["final_k"]),
        )
        validation["serialization"] = serialization
        write_json_atomic(publish / "submission_schema.json", submission_schema())
        write_json_atomic(publish / "submission_validation.json", validation)
        reporter.phase_finish("submission_validation", phase, sha256=validation["sha256"])
        check_resources("after_submission_validation")

        phase = reporter.phase_start("publication")
        candidate_summary = read_json(publish / "candidate_shards" / "manifest.json")
        inference_summary = read_json(publish / "recommendation_shards" / "manifest.json")
        runtime = {
            "runtime_seconds": time.perf_counter() - started_run,
            "final_resources": check_resources("before_publication"),
            **guard.summary(),
            "signals_received": signal_received,
        }
        metrics = {
            "artifact_version": 1,
            "run_id": run_id,
            "mode": config["mode"],
            "target_users": target_users.height,
            "recommendation_users": recommendations.height,
            "items_per_user": int(config["inference"]["final_k"]),
            "missing_users": 0,
            "extra_users": 0,
            "duplicate_users": 0,
            "duplicate_items": 0,
            "null_values": 0,
            "unknown_items": 0,
            "seen_pairs": 0,
            "candidate_rows": candidate_summary["rows"],
            "candidate_source_contribution_rows": candidate_summary["source_contribution_rows"],
            "candidate_source_users": candidate_summary["source_users"],
            "candidate_source_user_coverage": candidate_summary["source_user_coverage"],
            "fallback_min_candidates": candidate_summary["fallback_min_candidates"],
            "fallback_users": inference_summary["fallback_users"],
            "fallback_positions": inference_summary["fallback_positions"],
            "portable_repeat_equal": True,
            "submission_sha256": validation["sha256"],
            "runtime": runtime,
        }
        write_json_atomic(publish / "metrics.json", metrics)
        write_json_atomic(publish / "runtime_diagnostics.json", runtime)
        write_json_atomic(
            publish / "completion_summary.json",
            {
                "artifact_version": 1,
                "status": "complete",
                "run_id": run_id,
                "completed_at": datetime.now().astimezone().isoformat(),
                "model": "CatBoostPointwiseModel",
                "tree_count": int(config["catboost"]["iterations"]),
                "feature_count": len(feature_columns),
                "target_users": target_users.height,
                "submission": "submission.csv",
                "kaggle_submission_performed": False,
                "recoverable_work_dir": work_dir.as_posix(),
                "recoverable_cleanup_requires_verified_publication": True,
            },
        )
        reporter.event("run_complete", run_id=run_id, output_dir=output_dir.as_posix())
        reporter.phase_finish("publication", phase, output_dir=output_dir.as_posix())
        reporter.close()
        reporter_closed = True
        _copy_logs(log_file, publish / "logs")
        write_checksums_manifest(publish)
        publish_directory_atomic(publish, output_dir)
        checksum_result = verify_checksums_manifest(output_dir)
        verify_result = verify_artifact(
            config_path=output_dir / "config.json",
            artifact_dir=output_dir,
            check_pid=False,
        )
        return {
            "run_id": run_id,
            "output_dir": output_dir.as_posix(),
            "checksums": checksum_result,
            "verification": verify_result,
        }
    except BaseException as error:
        reporter.event(
            "run_failure",
            run_id=run_id,
            error_type=type(error).__name__,
            error=str(error),
            duration_seconds=time.perf_counter() - started_run,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        guard.stop()
        if not reporter_closed:
            reporter.close()


def _validate_shard_manifest(root: Path, name: str) -> dict[str, Any]:
    manifest = read_json(root / name / "manifest.json")
    part_names = manifest.get("parts")
    if not isinstance(part_names, list) or len(part_names) != manifest.get("part_count"):
        raise ContractValidationError(f"{name} manifest has an invalid part list")
    expected = {f"{part}.parquet" for part in part_names}
    actual = {
        path.name
        for path in (root / name).glob("part-*.parquet")
        if ".seeds." not in path.name and ".long." not in path.name
    }
    if name in {"candidate_shards", "feature_shards", "recommendation_shards"} and expected != actual:
        raise ContractValidationError(f"{name} has missing or extra primary shards")
    return manifest


def _active_task11_processes(run_id: str) -> list[int]:
    active: list[int] = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = [
                value.decode()
                for value in (directory / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (OSError, UnicodeDecodeError):
            continue
        is_runner = any(Path(value).name == "run_full_fit_inference.py" for value in arguments)
        if is_runner and any(run_id in value for value in arguments):
            active.append(int(directory.name))
    return sorted(active)


def verify_artifact(
    *, config_path: Path, artifact_dir: Path, check_pid: bool = True
) -> dict[str, Any]:
    config = _load_config(config_path)
    root = artifact_dir.resolve()
    verify_checksums_manifest(root)
    if root.name != config["run_id"]:
        raise ContractValidationError("artifact directory name differs from run_id")
    target_users = _target_users(config)
    stored_targets = pl.read_parquet(root / "target_users.parquet")
    if not stored_targets.equals(target_users):
        raise ContractValidationError("published target universe differs from source")
    history_path = root / "full_history" / "history_daily.parquet"
    recommendations = pl.read_parquet(root / "internal_recommendations.parquet")
    validate_recommendations_against_history(
        recommendations,
        target_users=target_users,
        history_daily=pl.scan_parquet(history_path),
        expected_k=int(config["inference"]["final_k"]),
    )
    submission_validation = validate_submission_against_artifacts(
        root / "submission.csv",
        recommendations=recommendations,
        target_users=target_users,
        history_daily=pl.scan_parquet(history_path),
        expected_k=int(config["inference"]["final_k"]),
    )
    candidates = _validate_shard_manifest(root, "candidate_shards")
    features = _validate_shard_manifest(root, "feature_shards")
    inference = _validate_shard_manifest(root, "recommendation_shards")
    if not (candidates["part_count"] == features["part_count"] == inference["part_count"]):
        raise ContractValidationError("published shard counts differ")
    model_a = CatBoostPointwiseModel.from_artifact(root / "model")
    model_b = CatBoostPointwiseModel.from_artifact(root / "model")
    feature_columns = read_json(root / "feature_schema.json")["feature_columns"]
    first_part = features["parts"][0]
    frame = pl.read_parquet(root / "feature_shards" / f"{first_part}.parquet")
    score_a = _predict_scores(model_a, frame, feature_columns, int(config["inference"]["batch_size"]))
    score_b = _predict_scores(model_b, frame, feature_columns, int(config["inference"]["batch_size"]))
    if not score_a.equals(score_b):
        raise ContractValidationError("verify-only portable inference is not deterministic")
    completion = read_json(root / "completion_summary.json")
    if completion.get("status") != "complete" or completion.get("kaggle_submission_performed") is not False:
        raise ContractValidationError("completion summary is invalid")
    log_files = list((root / "logs").glob("*"))
    if not log_files or not any("event=run_complete" in path.read_text(encoding="utf-8") for path in log_files):
        raise ContractValidationError("published structured log lacks run_complete")
    pid_file = root.parent / f".{config['run_id']}.pid"
    if check_pid and pid_file.exists():
        raise ContractValidationError(f"Task 11 PID file is still active: {pid_file}")
    active_processes = _active_task11_processes(str(config["run_id"]))
    if check_pid and active_processes:
        raise ContractValidationError(
            f"Task 11 processes are still active: {active_processes}"
        )
    return {
        "artifact": root.as_posix(),
        "target_users": target_users.height,
        "items_per_user": int(config["inference"]["final_k"]),
        "tree_count": model_a.tree_count,
        "feature_count": len(feature_columns),
        "shards": features["part_count"],
        "submission_sha256": submission_validation["sha256"],
        "portable_repeat_equal": True,
        "checksums_valid": True,
        "partial_shards": 0,
        "active_pid_file": False,
        "active_processes": active_processes,
    }


def validate_submission_only(*, config_path: Path, artifact_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    target_users = _target_users(config)
    recommendations = pl.read_parquet(artifact_dir / "internal_recommendations.parquet")
    return validate_submission_against_artifacts(
        artifact_dir / "submission.csv",
        recommendations=recommendations,
        target_users=target_users,
        history_daily=pl.scan_parquet(artifact_dir / "full_history" / "history_daily.parquet"),
        expected_k=int(config["inference"]["final_k"]),
    )


def _cleanup_recoverable(work_dir: Path, config: Mapping[str, Any]) -> None:
    resolved = work_dir.resolve()
    expected_parent = _resolve("artifacts").resolve()
    if resolved.parent != expected_parent or resolved.name != f".{config['run_id']}.work":
        raise FullFitPipelineError("refusing cleanup outside the exact Task 11 owned work path")
    marker = read_json(resolved / ".task11-owner.json")
    if marker.get("run_id") != config["run_id"] or marker.get("config_sha256") != config_sha256(config):
        raise FullFitPipelineError("refusing cleanup with an incompatible ownership marker")
    shutil.rmtree(resolved)


def main() -> int:
    args = _parser().parse_args()
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_dir = _resolve(args.output_dir or Path("artifacts") / config["run_id"])
    work_dir = _resolve(args.work_dir or Path("artifacts") / f".{config['run_id']}.work")
    log_file = _resolve(args.log_file) if args.log_file else _resolve(Path("logs") / f"{config['run_id']}.log")
    if args.verify_only and args.validate_submission_only:
        raise FullFitPipelineError("choose only one verification mode")
    if args.cleanup_recoverable and not args.verify_only:
        raise FullFitPipelineError("recoverable cleanup requires --verify-only")
    if args.stop_after_checkpoints is not None and (
        args.stop_after_checkpoints <= 0 or config["mode"] != "limited_smoke"
    ):
        raise FullFitPipelineError("--stop-after-checkpoints is limited to smoke and must be positive")
    try:
        if args.verify_only:
            result = verify_artifact(
                config_path=output_dir / "config.json",
                artifact_dir=output_dir,
                check_pid=True,
            )
            if args.cleanup_recoverable and work_dir.exists():
                _cleanup_recoverable(work_dir, config)
                result["recoverable_work_cleaned"] = True
        elif args.validate_submission_only:
            result = validate_submission_only(
                config_path=output_dir / "config.json", artifact_dir=output_dir
            )
        else:
            result = run_pipeline(
                config_path=config_path,
                output_dir=output_dir,
                work_dir=work_dir,
                log_file=log_file,
                show_progress=not args.no_progress,
                stop_after_checkpoints=args.stop_after_checkpoints,
            )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except SimulatedInterruption as error:
        print(str(error), file=sys.stderr)
        return 75
    except GracefulStop as error:
        print(f"graceful stop: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())

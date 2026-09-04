"""Reusable observability and atomic state helpers for long experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from tqdm import tqdm


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {source}")
    return value


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def publish_directory_atomic(staging: str | Path, output: str | Path) -> None:
    source = Path(staging)
    destination = Path(output)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


class EventProgressReporter:
    """Nested terminal progress plus a compact rotating event log."""

    def __init__(
        self,
        *,
        task_name: str,
        total_phases: int,
        log_file: str | Path | None,
        show_progress: bool,
        log_max_bytes: int = 1_000_000,
        log_backup_count: int = 2,
    ) -> None:
        if not task_name:
            raise ValueError("task_name must be non-empty")
        if total_phases <= 0:
            raise ValueError("total_phases must be positive")
        if log_max_bytes <= 0 or log_backup_count < 0:
            raise ValueError("invalid rotating-log limits")
        self.task_name = task_name
        self.log_path = Path(log_file) if log_file is not None else None
        self.show_progress = bool(show_progress)
        self._logger = logging.getLogger(f"{task_name}.{uuid.uuid4().hex}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler: RotatingFileHandler | None = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handler = RotatingFileHandler(
                self.log_path,
                mode="a",
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
            self._handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
                )
            )
            self._logger.addHandler(self._handler)
        self._overall = tqdm(
            total=total_phases,
            desc=f"{task_name} | starting",
            unit="phase",
            position=0,
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        self._stage: Any = None
        self._iteration: Any = None

    @staticmethod
    def _field(value: Any) -> str:
        if isinstance(value, float):
            return format(value, ".6f")
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    def event(self, event: str, **fields: Any) -> None:
        contextual = dict(fields)
        contextual.setdefault("stage", contextual.get("phase", "run"))
        contextual.setdefault("config", "all")
        contextual.setdefault("fold", "all")
        contextual.setdefault("operation", event)
        values = [f"event={event}"]
        values.extend(
            f"{name}={self._field(value)}"
            for name, value in contextual.items()
            if value is not None
        )
        self._logger.info(" ".join(values))

    def phase_start(self, phase: str, **fields: Any) -> float:
        self._overall.set_description_str(f"{self.task_name} | {phase}")
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

    def stage_start(
        self,
        *,
        stage: str,
        total: int,
        unit: str = "operation",
    ) -> float:
        if self._stage is not None:
            raise RuntimeError("previous progress stage is still active")
        self._stage = tqdm(
            total=total,
            desc=f"{stage} | starting",
            unit=unit,
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        self.event("stage_start", stage=stage, total=total, unit=unit)
        return time.perf_counter()

    def stage_status(
        self, *, stage: str, config: str, fold: str, operation: str
    ) -> None:
        description = f"{stage} | {config} | {fold} | {operation}"
        self._overall.set_description_str(f"{self.task_name} | {description}")
        self._overall.refresh()
        if self._stage is not None:
            self._stage.set_description_str(description)
            self._stage.refresh()

    def stage_advance(self, count: int = 1) -> None:
        if self._stage is not None:
            self._stage.update(count)

    def stage_finish(self, *, stage: str, started: float, **fields: Any) -> None:
        if self._stage is not None:
            self._stage.close()
            self._stage = None
        self.event(
            "stage_finish",
            stage=stage,
            duration_seconds=time.perf_counter() - started,
            **fields,
        )

    def iteration_start(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        total: int,
        unit: str,
    ) -> tuple[float, Any]:
        if self._iteration is not None:
            raise RuntimeError("previous iteration progress is still active")
        self.stage_status(stage=stage, config=config, fold=fold, operation=f"{unit}s")
        self._iteration = tqdm(
            total=total,
            desc=f"{config} | {fold}",
            unit=unit,
            position=2,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        self.event(
            "iteration_start",
            stage=stage,
            config=config,
            fold=fold,
            total=total,
            unit=unit,
        )

        def callback(
            index: int,
            elapsed: float | None = None,
            value: Any = None,
        ) -> None:
            completed = index + 1
            if self._iteration is not None:
                self._iteration.update(max(0, completed - self._iteration.n))
                if elapsed is not None:
                    self._iteration.set_postfix_str(f"last={elapsed:.1f}s")
            self.event(
                "iteration",
                stage=stage,
                config=config,
                fold=fold,
                iteration=completed,
                total=total,
                elapsed_seconds=elapsed,
                value=value,
            )

        return time.perf_counter(), callback

    def iteration_finish(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        started: float,
        status: str,
    ) -> None:
        if self._iteration is not None:
            self._iteration.close()
            self._iteration = None
        self.event(
            "iteration_finish",
            stage=stage,
            config=config,
            fold=fold,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )

    def operation_start(
        self, *, stage: str, config: str, fold: str, operation: str
    ) -> float:
        self.stage_status(stage=stage, config=config, fold=fold, operation=operation)
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
        if self._iteration is not None:
            self._iteration.close()
            self._iteration = None
        if self._stage is not None:
            self._stage.close()
            self._stage = None
        self._overall.close()
        if self._handler is not None:
            self._handler.flush()
            self._handler.close()
            self._logger.removeHandler(self._handler)
            self._handler = None


class CheckpointStore:
    """Atomic completed-operation registry with strict config compatibility."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        config_digest: str,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not config_digest:
            raise ValueError("config_digest must be non-empty")
        self.directory = Path(directory)
        self.run_id = run_id
        self.config_digest = config_digest
        self.path = self.directory / "checkpoint.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            value = read_json(self.path)
            if (
                value.get("artifact_version") != 1
                or value.get("run_id") != run_id
                or value.get("config_sha256") != config_digest
                or not isinstance(value.get("completed"), dict)
            ):
                raise ValueError(f"incompatible checkpoint: {self.path}")
        else:
            write_json_atomic(
                self.path,
                {
                    "artifact_version": 1,
                    "run_id": run_id,
                    "config_sha256": config_digest,
                    "completed": {},
                },
            )

    @staticmethod
    def key(*, stage: str, config: str, fold: str) -> str:
        if not stage or not config or not fold:
            raise ValueError("stage, config, and fold must be non-empty")
        return f"{stage}::{config}::{fold}"

    def _read(self) -> dict[str, Any]:
        return read_json(self.path)

    def get(self, *, stage: str, config: str, fold: str) -> dict[str, Any] | None:
        value = self._read()["completed"].get(
            self.key(stage=stage, config=config, fold=fold)
        )
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("checkpoint record must be an object")
        return value

    def complete(
        self,
        *,
        stage: str,
        config: str,
        fold: str,
        metadata: Mapping[str, Any],
    ) -> None:
        value = self._read()
        key = self.key(stage=stage, config=config, fold=fold)
        completed = dict(value["completed"])
        record = dict(metadata)
        if key in completed:
            if completed[key] != record:
                raise ValueError(f"checkpoint record already differs: {key}")
            return
        completed[key] = record
        value["completed"] = completed
        write_json_atomic(self.path, value)


class AtomicBestConfig:
    """Atomically retain the best portable parameter payload."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        config_digest: str,
    ) -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.config_digest = config_digest
        self.pointer = self.directory / "best_model.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.pointer.exists():
            self._validate(read_json(self.pointer))

    def _validate(self, value: Mapping[str, Any]) -> None:
        if (
            value.get("artifact_version") != 1
            or value.get("run_id") != self.run_id
            or value.get("config_sha256") != self.config_digest
            or not isinstance(value.get("score"), list)
            or not isinstance(value.get("payload"), dict)
        ):
            raise ValueError(f"incompatible best-model pointer: {self.pointer}")

    def read(self) -> dict[str, Any] | None:
        if not self.pointer.exists():
            return None
        value = read_json(self.pointer)
        self._validate(value)
        return value

    def update(
        self,
        *,
        score: tuple[float, ...],
        payload: Mapping[str, Any],
    ) -> bool:
        if not score or not all(isinstance(value, (int, float)) for value in score):
            raise ValueError("score must contain numeric values")
        current = self.read()
        if current is not None and tuple(current["score"]) >= tuple(score):
            return False
        value = {
            "artifact_version": 1,
            "run_id": self.run_id,
            "config_sha256": self.config_digest,
            "score": list(score),
            "payload": dict(payload),
        }
        write_json_atomic(self.pointer, value)
        return True


def remove_empty_directory(path: str | Path) -> None:
    """Remove only a verified empty staging directory."""

    directory = Path(path)
    if directory.exists():
        directory.rmdir()


__all__ = [
    "AtomicBestConfig",
    "CheckpointStore",
    "EventProgressReporter",
    "config_sha256",
    "publish_directory_atomic",
    "read_json",
    "remove_empty_directory",
    "sha256_file",
    "write_json_atomic",
]

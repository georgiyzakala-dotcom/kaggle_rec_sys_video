"""External resource supervisor for bounded Torch experiments in WSL."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from experiment_utils import EventProgressReporter, read_json, write_json_atomic
from scripts.task13_resources import memory_available_gib, process_tree_rss_gib


def snapshot(config: dict, pid: int | None = None) -> dict:
    limits = config["resources"]
    values = {
        "available_ram_gib": memory_available_gib(),
        "rss_gib": process_tree_rss_gib(pid) if pid else 0.0,
        "free_disk_gib": shutil.disk_usage("artifacts").free / 2**30,
    }
    drive = limits["windows_drive"]
    if drive:
        if not Path(drive).is_mount():
            raise RuntimeError(f"Windows disk mount is unavailable: {drive}")
        values["windows_free_gib"] = shutil.disk_usage(drive).free / 2**30
    if config["device"] == "cuda":
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                "--id=" + str(config["gpu_id"]),
            ],
            text=True,
            timeout=10,
        ).strip()
        free, total = (float(value.strip()) for value in result.split(","))
        values.update(vram_free_mib=free, vram_used_mib=total - free)
    work = Path("artifacts") / f".{config['run_id']}.work"
    output = Path("artifacts") / config["run_id"]
    values["run_disk_gib"] = (
        sum(
            p.stat().st_size
            for root in (work, output)
            if root.exists()
            for p in root.rglob("*")
            if p.is_file() and not p.is_symlink()
        )
        / 2**30
    )
    return values


def violation(values: dict, limits: dict, *, initial=False):
    checks = {
        "available_ram_gib": limits["minimum_available_ram_gib"]
        if initial
        else limits["stop_available_ram_gib"],
        "free_disk_gib": limits["minimum_free_disk_gib"]
        if initial
        else limits["stop_free_disk_gib"],
        "windows_free_gib": limits["windows_minimum_start_free_gib"]
        if initial
        else limits["windows_stop_free_gib"],
        "vram_free_mib": limits["minimum_free_vram_mib"]
        if initial
        else limits["stop_free_vram_mib"],
    }
    for key, floor in checks.items():
        if key in values and values[key] < floor:
            return f"{key}={values[key]:.2f} below {floor}"
    if values["rss_gib"] > limits["maximum_rss_gib"]:
        return "process RSS exceeds configured RAM limit"
    if values["run_disk_gib"] > limits["maximum_run_disk_gib"]:
        return "run files exceed configured disk budget"
    return None


def supervise(command: list[str], config: dict) -> int:
    run_id = config["run_id"]
    fold = config.get("fold", "rolling_1")
    logger = EventProgressReporter(
        task_name=run_id + "_resources",
        total_phases=1,
        log_file=f"logs/{run_id}.resources.log",
        show_progress=False,
    )
    path = Path(f"logs/{run_id}.resources.json")
    previous = read_json(path) if path.exists() else {}
    peaks = previous.get("peaks", {})
    stopped = False
    child = None

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {
        sig: signal.signal(sig, interrupt)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        values = snapshot(config)
        error = violation(values, config["resources"], initial=True)
        logger.event(
            "preflight",
            stage="preflight",
            config=run_id,
            fold=fold,
            **values,
            error=error,
        )
        if error:
            raise RuntimeError(error)
        child = subprocess.Popen(command, start_new_session=True)
        last_log = 0.0
        while child.poll() is None:
            if stopped:
                raise KeyboardInterrupt(
                    "stop requested; repeat the same launch command to resume"
                )
            try:
                values = snapshot(config, child.pid)
            except FileNotFoundError:
                if child.poll() is not None:
                    break
                # Atomic rename/unlink can race with a resource sample.
                continue
            for key in ("rss_gib", "vram_used_mib", "run_disk_gib"):
                peaks[key] = max(peaks.get(key, 0.0), values.get(key, 0.0))
            error = violation(values, config["resources"])
            write_json_atomic(
                path,
                {
                    "status": "running",
                    "worker_pid": child.pid,
                    "timestamp_unix": time.time(),
                    "current": values,
                    "peaks": peaks,
                },
            )
            if time.monotonic() - last_log >= 30:
                logger.event(
                    "resource_sample",
                    stage="training",
                    config=run_id,
                    fold=fold,
                    **values,
                )
                last_log = time.monotonic()
            if error:
                raise RuntimeError(error)
            time.sleep(config["resources"]["poll_seconds"])
        code = child.wait()
        write_json_atomic(
            path,
            {
                "status": "completed" if code == 0 else "failed",
                "exit_code": code,
                "peaks": peaks,
            },
        )
        logger.event(
            "worker_finish",
            config=run_id,
            fold=fold,
            exit_code=code,
            peaks=peaks,
        )
        return code
    except BaseException as error:
        logger.event(
            "worker_stop",
            config=run_id,
            fold=fold,
            error=repr(error),
            peaks=peaks,
        )
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write_json_atomic(
            path, {"status": "stopped", "reason": repr(error), "peaks": peaks}
        )
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        logger.close()

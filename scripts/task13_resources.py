"""External supervisor: resource checks continue during native CatBoost calls."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from experiment_utils import EventProgressReporter, read_json, write_json_atomic


def memory_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("cannot read MemAvailable")


def process_tree_rss_gib(pid: int) -> float:
    try:
        rss = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss / 2**30 + sum(process_tree_rss_gib(int(p)) for p in children)
    except (FileNotFoundError, ProcessLookupError):
        return 0.0


def resources(config: dict, pid: int | None = None) -> dict:
    result = {
        "available_ram_gib": memory_available_gib(),
        "rss_gib": process_tree_rss_gib(pid) if pid else 0.0,
        "free_disk_gib": shutil.disk_usage("artifacts").free / 2**30,
    }
    drive = config["resources"].get("windows_drive")
    if drive:
        if not Path(drive).is_mount():
            raise RuntimeError(f"configured Windows mount is unavailable: {drive}")
        result["windows_free_gib"] = shutil.disk_usage(drive).free / 2**30
    if config["catboost"]["task_type"] == "GPU":
        free, total = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=" + str(config["catboost"]["devices"]),
                ],
                text=True,
                timeout=10,
            )
            .strip()
            .split(",")
        )
        result.update(vram_free_mib=int(free), vram_used_mib=int(total) - int(free))
    return result


def violation(values: dict, limits: dict, *, initial: bool = False) -> str | None:
    floors = {
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
    for key, floor in floors.items():
        if key in values and values[key] < floor:
            return f"{key}={values[key]:.3f} below {floor}"
    if values["rss_gib"] > limits["maximum_rss_gib"]:
        return f"rss_gib={values['rss_gib']:.3f} exceeds {limits['maximum_rss_gib']}"
    return None


def supervise(command: list[str], config: dict) -> int:
    run_id = config["run_id"]
    report = EventProgressReporter(
        task_name=f"{run_id}_resources",
        total_phases=1,
        log_file=f"logs/{run_id}.resources.log",
        show_progress=False,
        log_max_bytes=1_000_000,
        log_backup_count=2,
    )
    status_path = Path(f"logs/{run_id}.resources.json")
    previous = read_json(status_path) if status_path.exists() else {}
    peak = float(previous.get("peak_rss_gib", 0.0))
    peak_vram = int(previous.get("peak_device_vram_used_mib", 0))
    child = None
    stopped = False

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True

    old_handlers = {
        s: signal.signal(s, interrupt)
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        values = resources(config)
        error = violation(values, config["resources"], initial=True)
        report.event("preflight", **values, status="failed" if error else "passed")
        if error:
            raise RuntimeError(error)
        child = subprocess.Popen(command, start_new_session=True)
        report.event("worker_start", pid=child.pid)
        last_log = 0.0
        while child.poll() is None:
            if stopped:
                raise KeyboardInterrupt(
                    "stop requested; rerun identical config to resume"
                )
            values = resources(config, child.pid)
            peak = max(peak, values["rss_gib"])
            peak_vram = max(peak_vram, values.get("vram_used_mib", 0))
            error = violation(values, config["resources"])
            write_json_atomic(
                status_path,
                {
                    **values,
                    "peak_rss_gib": peak,
                    "peak_device_vram_used_mib": peak_vram,
                    "worker_pid": child.pid,
                    "timestamp_unix": time.time(),
                    "status": "stopping" if error else "running",
                },
            )
            if time.monotonic() - last_log >= 30:
                report.event("resource_sample", **values, peak_rss_gib=peak)
                last_log = time.monotonic()
            if error:
                raise RuntimeError(error)
            time.sleep(config["resources"]["poll_seconds"])
        report.event("worker_finish", exit_code=child.returncode, peak_rss_gib=peak)
        current = read_json(status_path) if status_path.exists() else {}
        write_json_atomic(
            status_path,
            {
                **current,
                "status": "completed" if child.returncode == 0 else "failed",
                "exit_code": child.returncode,
            },
        )
        return child.returncode
    except BaseException as error:
        report.event("worker_stop", error=repr(error), peak_rss_gib=peak)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write_json_atomic(
            status_path,
            {
                "status": "stopped",
                "reason": repr(error),
                "peak_rss_gib": peak,
                "peak_device_vram_used_mib": peak_vram,
                "timestamp_unix": time.time(),
            },
        )
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        report.close()

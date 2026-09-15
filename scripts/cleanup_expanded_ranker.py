#!/usr/bin/env python3
"""Preview or remove one completed expanded-ranker run's private work directory."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import stat
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from experiment_utils import config_sha256, read_json, sha256_file, write_json_atomic

KIND = "expanded_sasrec_ranker_submission"
AUDIT_KIND = "expanded_ranker_work_cleanup_v1"
WORK_ENTRIES = {
    "best",
    "checkpoint.json",
    "completed",
    "config.json",
    "evaluation",
    "full_sequences",
    "inference",
    "inputs",
    "models",
    "pools",
    "profiles",
    "publication",
    "sasrec",
    "snapshots",
    "timing.json",
    "training_data",
}
LOCKS = (
    "task15_sasrec_benchmark",
    "task13_history_profiles",
    "task14_full_fit",
    "expanded_ranker",
)


def require_plain_path(path, *, directory=False):
    if path.is_symlink():
        raise ValueError(f"symlink is not allowed: {path}")
    if directory and (not path.is_dir() or path.resolve() != path.absolute()):
        raise ValueError(f"expected an ordinary directory: {path}")


@contextmanager
def run_locks(repo, run_id):
    logs = repo / "logs"
    require_plain_path(logs)
    logs.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for name in (f"{run_id}.launcher", *LOCKS):
            path = logs / f"{name}.lock"
            require_plain_path(path)
            handle = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"pipeline is active ({path.name}); cleanup refused"
                ) from error
        yield


def inventory(root):
    """Count allocated file blocks once; an outside hard link prevents reclamation."""
    device = root.stat().st_dev
    files, directories = [], []
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in sorted([*dirs, *names]):
            path = Path(parent) / name
            s = path.lstat()
            if s.st_dev != device or stat.S_ISLNK(s.st_mode):
                raise ValueError(
                    f"symlink or different filesystem inside cleanup target: {path}"
                )
            if stat.S_ISDIR(s.st_mode):
                if os.path.ismount(path):
                    raise ValueError(f"mount point inside cleanup target: {path}")
                directories.append(path.relative_to(root).as_posix())
            elif stat.S_ISREG(s.st_mode):
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "device": s.st_dev,
                        "inode": s.st_ino,
                        "size": s.st_size,
                        "mtime_ns": s.st_mtime_ns,
                        "links": s.st_nlink,
                        "allocated": s.st_blocks * 512,
                    }
                )
            else:
                raise ValueError(f"special file inside cleanup target: {path}")
    files.sort(key=lambda r: r["path"])
    references = Counter((r["device"], r["inode"]) for r in files)
    unique = {(r["device"], r["inode"]): r for r in files}
    return {
        "files": files,
        "directories": directories,
        "file_count": len(files),
        "unique_file_bytes": sum(r["size"] for r in unique.values()),
        "estimated_reclaimable_bytes": sum(
            r["allocated"] for key, r in unique.items() if references[key] == r["links"]
        ),
        "externally_linked_bytes": sum(
            r["size"] for key, r in unique.items() if references[key] < r["links"]
        ),
    }


def verify_completed(path):
    # Importing the verifier does not fit models, require CUDA or append the experiment log.
    from scripts.run_expanded_ranker import verify_artifact

    return verify_artifact(path)


def verify_owner(work, config):
    unknown = {p.name for p in work.iterdir()} - WORK_ENTRIES
    if unknown:
        raise ValueError(f"unexpected entries in work directory: {sorted(unknown)}")
    for name in ("config.json", "checkpoint.json"):
        require_plain_path(work / name)
    if read_json(work / "config.json") != config:
        raise ValueError("work configuration differs from published artifact")
    checkpoint = read_json(work / "checkpoint.json")
    if checkpoint.get("run_id") != config["run_id"] or checkpoint.get(
        "config_sha256"
    ) != config_sha256(config):
        raise ValueError("checkpoint does not belong to the published run")


def remove_inventory(root, plan, *, show_progress):
    with tqdm(
        total=plan["file_count"],
        desc="Удаление рабочего кэша",
        unit="file",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as bar:
        for record in plan["files"]:
            path = root / record["path"]
            # Check directory components again before unlink; never follow a symlink.
            for parent in path.parents:
                if parent == root:
                    break
                require_plain_path(parent, directory=True)
            s = path.lstat()
            if not stat.S_ISREG(s.st_mode) or any(
                getattr(s, key) != record[field]
                for key, field in (
                    ("st_dev", "device"),
                    ("st_ino", "inode"),
                    ("st_size", "size"),
                    ("st_mtime_ns", "mtime_ns"),
                )
            ):
                raise ValueError(f"file changed during cleanup: {path}")
            path.unlink()
            bar.update(1)
    for relative in sorted(
        plan["directories"], key=lambda p: len(Path(p).parts), reverse=True
    ):
        (root / relative).rmdir()
    root.rmdir()


def cleanup(
    repo, run_id, *, apply=False, show_progress=True, verifier=verify_completed
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("invalid run ID")
    repo = Path(repo).resolve()
    artifacts = repo / "artifacts"
    require_plain_path(artifacts, directory=True)
    output, work = artifacts / run_id, artifacts / f".{run_id}.work"
    trash, audit = (
        artifacts / f".{run_id}.cleanup-trash",
        artifacts / f"{run_id}_cleanup_v1",
    )
    for path in (output, work, trash, audit):
        require_plain_path(path)
    require_plain_path(output, directory=True)
    with run_locks(repo, run_id):
        config = read_json(output / "config.json")
        if config.get("run_id") != run_id or config.get("kind") != KIND:
            raise ValueError(
                "output is not the requested completed expanded-ranker run"
            )
        # A directory name or success log is not enough to permit deletion.
        print("Проверка опубликованных моделей, метрик и submission.csv...", flush=True)
        verified = verifier(output)
        manifest_sha = sha256_file(output / "manifest.json")
        if work.exists() and trash.exists():
            raise ValueError(
                "both work and cleanup-trash exist; manual review required"
            )
        source = trash if trash.exists() else work
        if not source.exists():
            print("Рабочий кэш уже отсутствует; результат проверен.")
            return {
                "status": "already_clean",
                "artifact": str(output),
                "verified": verified,
            }
        require_plain_path(source, directory=True)
        source_stat = source.stat()
        identity = {"device": source_stat.st_dev, "inode": source_stat.st_ino}
        receipt = None
        if audit.exists():
            require_plain_path(audit, directory=True)
            require_plain_path(audit / "config.json")
            receipt = read_json(audit / "config.json")
            if (
                receipt.get("kind") != AUDIT_KIND
                or receipt.get("run_id") != run_id
                or receipt.get("manifest_sha256") != manifest_sha
                or receipt.get("source_identity") != identity
            ):
                raise ValueError(
                    "cleanup receipt does not match this directory/artifact"
                )
        if source == work:
            verify_owner(work, config)
        elif receipt is None:
            raise ValueError("cleanup-trash has no ownership receipt")
        plan = inventory(source)
        summary = {k: v for k, v in plan.items() if k not in ("files", "directories")}
        summary.update(artifact=str(output), target=str(source), run_id=run_id)
        print(f"Удаляемый каталог: {source.relative_to(repo)}")
        print(
            f"Файлов: {plan['file_count']}; уникальный размер: {plan['unique_file_bytes'] / 2**30:.3f} GiB"
        )
        print(
            f"Оценка освобождения в Linux: {plan['estimated_reclaimable_bytes'] / 2**30:.3f} GiB"
        )
        print(
            f"Файлы с hard links вне кэша: {plan['externally_linked_bytes'] / 2**30:.3f} GiB (сохраняются)"
        )
        print(
            "Сохраняются final artifact, submission, модели, метрики, data/, остальные runs и logs/."
        )
        if not apply:
            print(
                "Предварительный просмотр: ничего не удалено. Для очистки добавьте --apply."
            )
            return {**summary, "status": "dry_run", "verified": verified}
        # Preserve the ownership receipt before any rename or unlink, allowing recovery
        # even if interruption follows deletion of work/config.json itself.
        audit.mkdir(exist_ok=True)
        if receipt is None:
            write_json_atomic(
                audit / "config.json",
                {
                    "kind": AUDIT_KIND,
                    "run_id": run_id,
                    "manifest_sha256": manifest_sha,
                    "source_identity": identity,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "work": str(work.relative_to(repo)),
                    "trash": str(trash.relative_to(repo)),
                    "implementation_sha256": sha256_file(__file__),
                },
            )
            write_json_atomic(audit / "plan.json", plan)
        started, free_before = time.perf_counter(), shutil.disk_usage(artifacts).free
        write_json_atomic(audit / "metrics.json", {**summary, "status": "deleting"})
        try:
            if source == work:
                work.rename(trash)
            remove_inventory(trash, plan, show_progress=show_progress)
            if sha256_file(output / "manifest.json") != manifest_sha:
                raise ValueError("published manifest changed during cleanup")
            verified = verifier(output)
            result = {
                **summary,
                "status": "completed",
                "verified": verified,
                "elapsed_seconds": time.perf_counter() - started,
                "observed_linux_free_bytes_delta": shutil.disk_usage(artifacts).free
                - free_before,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json_atomic(audit / "metrics.json", result)
        except BaseException as error:
            write_json_atomic(
                audit / "metrics.json",
                {**summary, "status": "interrupted_or_failed", "error": repr(error)},
            )
            raise
        print(f"Очистка завершена. Отчёт: {audit.relative_to(repo)}")
        print(
            "Для возврата свободного места на G: отдельно сожмите VHDX из Windows; см. DISK_CLEANUP.md."
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="task15_sasrec_ranker_v1")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Remove the verified completed run's work cache; default is preview",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    cleanup(ROOT, args.run_id, apply=args.apply, show_progress=not args.no_progress)


if __name__ == "__main__":
    main()

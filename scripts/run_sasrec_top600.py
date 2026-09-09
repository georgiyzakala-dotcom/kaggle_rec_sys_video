#!/usr/bin/env python3
"""Recompute fold metrics and evaluate ALS600/SASRec600; inference only."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
import torch

from experiment_utils import (
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from implicit_model import ImplicitALSDataLoader, ImplicitALSModel
from sasrec_data import SASRecDataLoader, SequenceStore
from sasrec_model import SASRecCandidateModel
from sasrec_selection import SOURCES
from sasrec_top300 import predict_vectorized_seen
from sasrec_top600 import (
    PRIMARY,
    evaluate_deep_shard,
    summarize_deep,
)
from scripts.run_sasrec_benchmark import BenchmarkPaused, configure_torch
from scripts.run_sasrec_folds import (
    FOLD_NAMES,
    FoldsRunner,
    clean_partial,
    metrics_equal,
    read_users,
    seal,
)
from scripts.run_sasrec_folds import (
    IMPLEMENTATION as FOLD_IMPLEMENTATION,
)
from scripts.run_sasrec_optuna import verify_hashes
from scripts.task15_resources import supervise

IMPLEMENTATION = (
    *FOLD_IMPLEMENTATION,
    "sasrec_top300.py",
    "sasrec_top600.py",
    "scripts/run_sasrec_top600.py",
)


def load_config(path):
    c = read_json(path)
    if c["kind"] != "sasrec_top600_evaluation" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid top600 kind/run ID")
    if c["mode"] not in ("full", "smoke") or c["device"] not in ("cpu", "cuda"):
        raise ValueError("invalid mode/device")
    if c["mode"] == "full":
        if c.get("smoke") or c["device"] != "cuda":
            raise ValueError(
                "full extension requires GPU and full evaluation universes"
            )
    else:
        s = c["smoke"]
        if not (
            1 <= s["shards_per_fold"] <= 2
            and 1 <= s["users_per_shard"] <= 2048
            and s["users_per_shard"] * s["shards_per_fold"] <= 2048
        ):
            raise ValueError("smoke inference must be bounded")
    if not 1 <= c["cpu_threads"] <= 8:
        raise ValueError("CPU thread limit must be 1..8")
    r = c["resources"]
    if not (
        0 < r["maximum_rss_gib"] <= 40
        and 0 < r["maximum_run_disk_gib"] <= 30
        and 0 < r["cuda_memory_fraction"] <= 0.8
    ):
        raise ValueError("invalid resource limits")
    return c


def resolve_config(c):
    c = json.loads(json.dumps(c))
    parent = Path(c["parent_artifact"])
    if sha256_file(parent / "manifest.json") != c["parent_manifest_sha256"]:
        raise ValueError("parent fold artifact changed")
    base = read_json(parent / "config.json")
    parent_manifest = read_json(parent / "manifest.json")["sha256"]
    for name in ("config.json", "metrics.json"):
        if sha256_file(parent / name) != parent_manifest[name]:
            raise ValueError("parent config/metrics checksum differs")
    if c["mode"] == "full" and (
        base["mode"] != "full" or tuple(f["name"] for f in base["folds"]) != FOLD_NAMES
    ):
        raise ValueError("full extension requires the complete four-fold parent")
    c.update(
        recipe=base["recipe"],
        seed=base["seed"],
        input=base["input"],
        folds=base["folds"],
        evaluation=base["evaluation"],
        precision=base["precision"],
        inference_precision="float32",
    )
    c["sequence_caches"] = {}
    for fold in c["folds"]:
        path = Path(
            fold.get("sequence_cache")
            or f"artifacts/.{base['run_id']}.work/folds/{fold['name']}/sequences"
        )
        if not path.exists():
            raise FileNotFoundError(f"required history-only sequence cache: {path}")
        c["sequence_caches"][fold["name"]] = str(path)
    c["implementation_sha256"] = {p: sha256_file(ROOT / p) for p in IMPLEMENTATION}
    c["library_versions"] = {
        "torch": str(torch.__version__),
        "polars": pl.__version__,
        **{p: importlib.metadata.version(p) for p in ("implicit", "numpy", "scipy")},
    }
    return c


class ProgressALSLoader(ImplicitALSDataLoader):
    """Keep the existing ALS scoring path, exposing completed batch progress."""

    check_stop = staticmethod(lambda: None)
    batch_done = staticmethod(lambda: None)

    def _iter_predict_batches(self, *, batch_size):
        for batch in super()._iter_predict_batches(batch_size=batch_size):
            self.check_stop()
            yield batch
            self.batch_done()


class Top600Runner(FoldsRunner):
    """Reuse progress, signals and checkpoint identity, with no fit calls."""

    def __init__(self, c, *, show_progress=True, stop_after_shard=None):
        super().__init__(
            c, show_progress=show_progress, stop_after_shard=stop_after_shard
        )

    def run_fold(self, fold):
        c = self.c
        name = fold["name"]
        original = Path(c["parent_artifact"]) / "folds" / name
        info = c["input"][name]
        destination = self.work / "folds" / name
        destination.mkdir(parents=True, exist_ok=True)
        parts_dir = destination / "parts"
        parts_dir.mkdir(exist_ok=True)
        self.event("stage_start", stage="preflight")
        verify_hashes(original / "trained")
        for path, digest in info["file_sha256"].items():
            self.check_stop()
            if sha256_file(path) != digest:
                raise ValueError(f"fixed input changed: {path}")
            self.event("source_verified", stage="preflight", path=path)
        store = SequenceStore.load(Path(c["sequence_caches"][name]))
        if (
            store.metadata["history_sha256"] != info["history_sha256"]
            or store.metadata["context_user_limit"] is not None
            and c["mode"] == "full"
        ):
            raise ValueError("sequence cache belongs to another history/universe")
        configure_torch(c)
        model = SASRecCandidateModel.from_artifact(
            original / "trained/model", device=c["device"]
        )
        loader = SASRecDataLoader(
            store, max_length=model.config.max_length, seed=c["seed"]
        )
        als = ImplicitALSModel.from_artifact(info["als_model"])
        truth = pl.read_parquet(original / "target_ground_truth.parquet")
        parts = sorted((original / "evaluation").glob("part-*"))
        if c["mode"] == "smoke":
            parts = parts[: c["smoke"]["shards_per_fold"]]
        with self.bar(len(parts), f"{name} | SASRec600 + ALS600 + metrics") as bar:
            for index, old in enumerate(parts):
                self.check_stop()
                destination_part = parts_dir / old.name
                if destination_part.exists():
                    verify_hashes(destination_part)
                    if (
                        read_json(destination_part / "metrics.json")["config_sha256"]
                        != self.digest
                    ):
                        raise ValueError("shard configuration differs")
                    self.event("shard_reused", stage="evaluation", shard=index)
                    bar.update(1)
                    continue
                started = time.perf_counter()
                verify_hashes(old)
                previous = pl.read_parquet(old / "per_user.parquet").sort("user_id")
                if c["mode"] == "smoke":
                    previous = previous.head(c["smoke"]["users_per_shard"])
                users = previous.select("user_id")
                subset = lambda frame, users=users: frame.join(
                    users, on="user_id", how="semi"
                )
                sas200 = subset(pl.read_parquet(old / "sasrec.parquet"))
                als400 = subset(pl.read_parquet(old / "als400.parquet"))
                self.event(
                    "shard_start", stage="inference", shard=index, users=users.height
                )
                loader.load_predict_data(
                    user_ids=users["user_id"].to_numpy()
                ).prepare_predict_data()
                infer_start = time.perf_counter()
                with self.bar(
                    math.ceil(users.height / c["evaluation"]["batch_size"]),
                    "SASRec600 | batches",
                    2,
                ) as batches:

                    def done(n):
                        self.check_stop()
                        batches.update(n - batches.n)

                    deeper = predict_vectorized_seen(
                        model,
                        loader,
                        k=600,
                        batch_size=c["evaluation"]["batch_size"],
                        item_chunk_size=c["evaluation"]["item_chunk_size"],
                        callback=done,
                        check_stop=self.check_stop,
                    )
                inference_seconds = time.perf_counter() - infer_start
                als_start = time.perf_counter()
                history = read_users(
                    Path(fold["artifact"]) / "history_daily.parquet", users
                )
                als_loader = ProgressALSLoader(
                    config=als.config,
                    reference_time=datetime.fromisoformat(info["split"]["cutoff"]),
                    user_mapping=als.user_mapping,
                    item_mapping=als.item_mapping,
                    seed=als.config.seed,
                )
                als_loader.load_predict_data(
                    history=history, target_users=users
                ).prepare_predict_data()
                with self.bar(
                    math.ceil(
                        len(als_loader._prediction_user_ids)
                        / c["evaluation"]["als_batch_size"]
                    ),
                    "ALS600 | batches",
                    2,
                ) as batches:
                    als_loader.check_stop = self.check_stop
                    als_loader.batch_done = lambda: batches.update(1)
                    als_deeper = als.predict(
                        als_loader, k=600, batch_size=c["evaluation"]["als_batch_size"]
                    )
                als_seconds = time.perf_counter() - als_start
                del history, als_loader
                sources = {
                    s: read_users(
                        Path(info["source_root"])
                        / "sources"
                        / s
                        / "candidates.parquet",
                        users,
                    )
                    for s in SOURCES
                }
                stats = evaluate_deep_shard(
                    users,
                    subset(truth),
                    sources,
                    sas200,
                    als400,
                    deeper,
                    als_deeper,
                    store,
                    old_stats=previous,
                )
                current = summarize_deep(stats, c["recipe"]["source_policy"])
                partial = parts_dir / (old.name + ".partial")
                clean_partial(partial)
                deeper.write_parquet(partial / "sasrec600.parquet")
                als_deeper.write_parquet(partial / "als600.parquet")
                stats.write_parquet(partial / "per_user.parquet")
                write_json_atomic(
                    partial / "metrics.json",
                    {
                        "config_sha256": self.digest,
                        "fold": name,
                        "source_part": str(old),
                        "users": users.height,
                        "duration_seconds": time.perf_counter() - started,
                        "inference_seconds": inference_seconds,
                        "als_inference_and_loader_seconds": als_seconds,
                        "candidate_recall": current["candidate_recall"],
                        "original_metrics_recomputed_exactly": True,
                        "top200_prefix_identical": True,
                        "als_top400_prefix_identical": True,
                    },
                )
                seal(partial)
                publish_directory_atomic(partial, destination_part)
                self.save_timing()
                self.event(
                    "shard_finish",
                    stage="evaluation",
                    shard=index,
                    duration_seconds=time.perf_counter() - started,
                    inference_seconds=inference_seconds,
                    als_seconds=als_seconds,
                    current_metric=current["candidate_recall"],
                )
                bar.update(1)
                bar.set_postfix(recall=f"{current['candidate_recall']:.4f}")
                del sources, deeper, als_deeper, previous, stats, sas200, als400
                gc.collect()
                if self.stop_after_shard == (name, index + 1):
                    raise BenchmarkPaused("intentional bounded smoke shard pause")
        del model, loader, store, als
        gc.collect()
        if c["device"] == "cuda":
            torch.cuda.empty_cache()
        stats = pl.concat(
            [pl.read_parquet(parts_dir / p.name / "per_user.parquet") for p in parts]
        )
        users = pl.read_parquet(original / "target_users.parquet").sort("user_id")
        measured = summarize_deep(stats, c["recipe"]["source_policy"])
        if c["mode"] == "full":
            if not stats.select("user_id").equals(users):
                raise ValueError("full extension lost target users")
            if not metrics_equal(
                measured["base_metrics_recomputed"],
                read_json(original / "metrics.json"),
            ):
                raise ValueError("independent full-fold metric audit differs")
        timings = [read_json(parts_dir / p.name / "metrics.json") for p in parts]
        measured.update(
            fold=name,
            split=info["split"],
            ground_truth_funnel=info["ground_truth_funnel"],
            runtime_seconds=sum(t["duration_seconds"] for t in timings),
            inference_seconds=sum(t["inference_seconds"] for t in timings),
            als_inference_and_loader_seconds=sum(
                t["als_inference_and_loader_seconds"] for t in timings
            ),
            full_target_universe=c["mode"] == "full",
            original_metrics_verified=True,
        )
        stats.write_parquet(destination / "per_user.parquet")
        write_json_atomic(destination / "metrics.json", measured)
        seal(destination)
        return measured

    def run(self):
        try:
            self.event("run_start", stage="preflight", fit_enabled=False)
            folds = {}
            for fold in self.c["folds"]:
                self.fold = fold["name"]
                start = self.reporter.phase_start(self.fold, fold=self.fold)
                d = self.work / "folds" / self.fold
                if (d / "manifest.json").exists():
                    verify_hashes(d)
                    folds[self.fold] = read_json(d / "metrics.json")
                    self.event("fold_reused", stage="evaluation")
                else:
                    folds[self.fold] = self.run_fold(fold)
                self.reporter.phase_finish(self.fold, start, fold=self.fold)
                self.save_timing()
            self.fold = "all"
            self.check_stop()
            start = self.reporter.phase_start("publish", fold="all")
            result = self.publish(folds)
            self.reporter.phase_finish("publish", start, fold="all")
            self.event(
                "run_finish",
                stage="publish",
                duration_seconds=self.elapsed(),
                output=str(result),
            )
            return result
        except BaseException as error:
            self.event(
                "run_finish",
                status="paused" if isinstance(error, BenchmarkPaused) else "failed",
                error=repr(error),
            )
            write_json_atomic(
                self.work / "failure.json",
                {"fold": self.fold, "error": repr(error), "resume": "same command"},
            )
            raise
        finally:
            self.save_timing()
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
            self.reporter.close()

    def publish(self, folds):
        partial = self.work / "publish.partial"
        clean_partial(partial)
        shutil.copytree(
            self.work / "folds",
            partial / "folds",
            copy_function=os.link,
            ignore=shutil.ignore_patterns("*.partial"),
        )
        write_json_atomic(partial / "config.json", self.c)
        metric = {
            "run_id": self.c["run_id"],
            "mode": self.c["mode"],
            "folds": folds,
            "model_fits": 0,
            "runtime_seconds": self.elapsed(),
            "primary_policy": PRIMARY,
            "parent_artifact": self.c["parent_artifact"],
            "automatic_promotion": False,
        }
        write_json_atomic(partial / "metrics.json", metric)
        rows = [
            {
                "fold": f,
                "policy": p,
                "candidate_recall": m["candidate_recall"],
                "positive_hits": m["positive_hits"],
                "sum_source_caps": m["sum_source_caps"],
                "mean_candidate_count": m["mean_candidate_count"],
                "candidate_oracle_p20_all_targets": m[
                    "candidate_oracle_p20_all_targets"
                ],
                "candidate_oracle_p20_labeled_users": m[
                    "candidate_oracle_p20_labeled_users"
                ],
            }
            for f, v in folds.items()
            for p, m in v["policies"].items()
        ]
        pl.DataFrame(rows).write_csv(partial / "comparison.csv")
        pl.DataFrame(
            [
                {"fold": f, "source": s, "k": int(k), **m}
                for f, v in folds.items()
                for s, curve in v["sources_depth_curve"].items()
                for k, m in curve.items()
            ]
        ).write_csv(partial / "source_depth_metrics.csv")
        pl.DataFrame(
            [
                {
                    "fold": f,
                    "k": int(k),
                    **{name: value for name, value in m.items() if name != "per_user"},
                    "mean_share_sasrec_in_als": m["per_user"]["share_sasrec_in_als"][
                        "mean"
                    ],
                }
                for f, v in folds.items()
                for k, m in v["overlap_at_depth"].items()
            ]
        ).write_csv(partial / "overlap_metrics.csv")
        seal(partial)
        verify_artifact(partial)
        publish_directory_atomic(partial, self.output)
        append_experiments(self.c, metric, self.output)
        return self.output


def verify_artifact(path):
    root = Path(path)
    verify_hashes(root)
    c, metric = read_json(root / "config.json"), read_json(root / "metrics.json")
    for fold in c["folds"]:
        directory = root / "folds" / fold["name"]
        verify_hashes(directory)
        stats = pl.read_parquet(directory / "per_user.parquet")
        measured = summarize_deep(stats, c["recipe"]["source_policy"])
        if not metrics_equal(measured, metric["folds"][fold["name"]]):
            raise ValueError("published top600 metric aggregation differs")
    return {"verified": True, "folds": list(metric["folds"]), "model_fits": 0}


def append_experiments(c, metric, output):
    if c["mode"] != "full":
        return
    with Path("experiments/results.csv").open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {r["run_id"] for r in reader}
        for name, m in metric["folds"].items():
            row = {
                "run_id": c["run_id"] + "_" + name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "split": name + "_24h_top600_diagnostic",
                "seed": c["seed"],
                "candidate_config": json.dumps(
                    {
                        "parent": c["parent_artifact"],
                        "als_cap": 600,
                        "sasrec_cap": 600,
                        "other_sources_cap": 200,
                    }
                ),
                "ranker_config": "not_trained_candidate_union_only",
                "p20_all_targets": None,
                "p20_labeled_users": None,
                "candidate_recall": m["candidate_recall"],
                "coverage": m["coverage"],
                "runtime": m["runtime_seconds"],
                "artifact_path": str(output / "folds" / name),
                "notes": "inference only; both600 budget1800; both300 vs ALS600 control budget1200; old metrics independently recomputed",
            }
            if row["run_id"] not in existing:
                handle.seek(0, 2)
                csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="ignore"
                ).writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task15_sasrec_top600_v1.json")
    )
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop-after-shard",
        nargs=2,
        metavar=("FOLD", "COUNT"),
        help="Smoke-only resume test hook",
    )
    args = parser.parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify_artifact(args.verify_only), indent=2))
        append_experiments(
            read_json(args.verify_only / "config.json"),
            read_json(args.verify_only / "metrics.json"),
            args.verify_only,
        )
        return 0
    c = load_config(args.config)
    if args.stop_after_shard and c["mode"] != "smoke":
        raise ValueError("pause test hooks are restricted to smoke")
    if args.worker:
        try:
            Top600Runner(
                resolve_config(c),
                show_progress=not args.no_progress,
                stop_after_shard=(
                    args.stop_after_shard[0],
                    int(args.stop_after_shard[1]),
                )
                if args.stop_after_shard
                else None,
            ).run()
        except BenchmarkPaused as error:
            print(str(error), file=sys.stderr)
            return 75
        return 0
    Path("logs").mkdir(exist_ok=True)
    with Path("logs/task15_sasrec_benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Task15 GPU job is running") from error
        if (Path("artifacts") / c["run_id"]).exists():
            raise FileExistsError("completed artifact exists; use --verify-only")
        command = [
            sys.executable,
            str(Path(__file__)),
            "--config",
            str(args.config),
            "--worker",
        ]
        if args.no_progress:
            command.append("--no-progress")
        if args.stop_after_shard:
            command.extend(["--stop-after-shard", *args.stop_after_shard])
        return supervise(command, c)


if __name__ == "__main__":
    raise SystemExit(main())

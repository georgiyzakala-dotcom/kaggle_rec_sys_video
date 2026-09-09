#!/usr/bin/env python3
"""Sequential, resumable Optuna selection of SASRec on rolling_1 next-day labels."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import math
import os
import resource
import shutil
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import polars as pl
import torch
from tqdm import tqdm

from experiment_utils import (
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from sasrec_data import SASRecDataLoader, SequenceStore, prepare_sequence_store
from sasrec_model import SASRecCandidateModel, SASRecConfig, save_torch_atomic
from sasrec_selection import SOURCES, SelectionEvaluator, id_sample, standalone_top20
from scripts.run_sasrec_benchmark import (
    BenchmarkPaused,
    capture_rng,
    configure_torch,
    restore_rng,
)
from scripts.task15_resources import supervise

IMPLEMENTATION = (
    "sasrec_data.py",
    "sasrec_model.py",
    "sasrec_selection.py",
    "interfaces.py",
    "data_utils.py",
    "metrics.py",
    "validation.py",
    "experiment_utils.py",
    "scripts/run_sasrec_benchmark.py",
    "scripts/run_sasrec_optuna.py",
    "scripts/task15_resources.py",
    "scripts/task13_resources.py",
)
PARAMETERS = (
    "embedding_dim",
    "max_length",
    "num_blocks",
    "dropout",
    "negative_count",
    "learning_rate",
    "weight_decay",
)


def load_config(path):
    import re

    c = read_json(path)
    if c["kind"] != "sasrec_optuna_selection" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid Optuna kind/run ID")
    if c["fold"] != "rolling_1" or c["mode"] not in ("full", "smoke"):
        raise ValueError("Optuna selection is restricted to rolling_1")
    if c["device"] not in ("cpu", "cuda") or c["precision"] not in (
        "float32",
        "bfloat16",
    ):
        raise ValueError("unsupported device/precision")
    if c["mode"] == "full":
        if (
            c["device"] != "cuda"
            or c["data"]["context_user_limit"] is not None
            or c["training"]["max_users"] is not None
            or not c["data"]["preserve_full_catalog"]
        ):
            raise ValueError("full search requires GPU/all history users/full catalog")
    elif not (
        1 <= c["data"]["context_user_limit"] <= 4096
        and 1 <= c["training"]["max_users"] <= 2048
        and c["search"]["trials"] <= 2
        and c["search"]["max_epochs"] <= 2
        and c["evaluation"]["users"] <= 64
    ):
        raise ValueError("smoke must be explicitly bounded in users/trials/epochs")
    s = c["search"]
    if not (
        1 <= s["trials"] <= 32
        and 1 <= s["max_epochs"] <= 20
        and 0 < s["timeout_hours"] <= 12
    ):
        raise ValueError("search budget is limited to 32 trials/20 epochs/12 hours")
    if not (
        1 <= s["evaluate_every"] <= s["max_epochs"]
        and s["patience_evaluations"] >= 1
        and s["startup_trials"] >= 1
        and s["prune_warmup_epochs"] >= 0
    ):
        raise ValueError("invalid evaluation/pruning schedule")
    if set(c["space"]) != set(PARAMETERS) or set(c["baseline"]) != set(PARAMETERS):
        raise ValueError("unexpected/missing search parameter")
    for name, spec in c["space"].items():
        if "choices" in spec:
            if not spec["choices"] or c["baseline"][name] not in spec["choices"]:
                raise ValueError("baseline must lie within categorical search space")
        elif not 0 < spec["low"] <= c["baseline"][name] <= spec["high"]:
            raise ValueError("invalid log search bounds/baseline")
    for dim in c["space"]["embedding_dim"]["choices"]:
        for length in c["space"]["max_length"]["choices"]:
            if dim > 128 or length > 100:
                raise ValueError("initial search supports d<=128 and length<=100")
            SASRecConfig(
                embedding_dim=dim, max_length=length, num_heads=c["model"]["num_heads"]
            )
    limits = c["resources"]
    if not (
        0 < limits["maximum_rss_gib"] <= 40
        and 0 < limits["maximum_run_disk_gib"] <= 30
        and 0 < limits["cuda_memory_fraction"] <= 0.8
        and 0 < limits["poll_seconds"] <= 5
        and 1 <= c["cpu_threads"] <= 8
    ):
        raise ValueError("invalid RAM/disk/GPU/thread limits")
    if (
        not 1 <= c["evaluation"]["users"] <= 16384
        or c["evaluation"]["candidate_k"] != 200
    ):
        raise ValueError("selection uses top200 and at most 16384 target users")
    for key in ("batch_size", "query_chunk_size"):
        if type(c["training"][key]) is not int or c["training"][key] < 1:
            raise ValueError("invalid training batch size")
    return c


def resolve_config(config):
    c = json.loads(json.dumps(config))
    fold = Path(c["data"]["fold_artifact"])
    split = read_json(fold / "config.json")["split"]
    d = read_json(fold / "metrics.json")["deterministic_diagnostics"]
    if (
        not split["validation_end_exclusive"]
        or (
            datetime.fromisoformat(split["validation_end_exclusive"])
            - datetime.fromisoformat(split["cutoff"])
        ).total_seconds()
        != 86400
    ):
        raise ValueError("Optuna requires an exact next-day target window")
    dataset = Path(c["data"]["source_fold"])
    manifest = read_json(dataset / "dataset_manifest.json")
    if (
        manifest["fold"] != "rolling_1"
        or manifest["cutoff"] != split["cutoff"]
        or Path(manifest["history_path"]).resolve()
        != (fold / "history_daily.parquet").resolve()
    ):
        raise ValueError("fixed sources belong to another fold/history")
    files = {
        str(fold / name): d["output_sha256"][name]
        for name in (
            "history_daily.parquet",
            "target_ground_truth.parquet",
            "target_users.parquet",
        )
    }
    files[str(dataset / "dataset_manifest.json")] = sha256_file(
        dataset / "dataset_manifest.json"
    )
    for source in SOURCES:
        directory = dataset / "sources" / source
        metadata = read_json(directory / "metadata.json")
        if (
            metadata["source"] != source
            or metadata["fold"] != "rolling_1"
            or metadata["cutoff"] != split["cutoff"]
            or metadata["candidate_k"] != 200
        ):
            raise ValueError("fixed source provenance/budget mismatch")
        files[str(directory / "candidates.parquet")] = metadata["candidates_sha256"]
        files[str(directory / "metadata.json")] = sha256_file(
            directory / "metadata.json"
        )
    c["input"] = {
        "cutoff": split["cutoff"],
        "validation_end_exclusive": split["validation_end_exclusive"],
        "history_sha256": d["output_sha256"]["history_daily.parquet"],
        "file_sha256": files,
        "ground_truth_funnel": d["ground_truth_funnel"],
        "history_users": d["daily"]["history"]["users"],
        "history_items": d["daily"]["history"]["items"],
    }
    c["implementation_sha256"] = {p: sha256_file(ROOT / p) for p in IMPLEMENTATION}
    c["library_versions"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "optuna": optuna.__version__,
        "numpy": np.__version__,
        "polars": pl.__version__,
    }
    c["objective"] = "best_equal_budget_union_delta_micro_recall"
    c["canonical_used_for_selection"] = False
    return c


def recipe(c, params):
    model = dict(
        c["model"],
        **{
            k: params[k]
            for k in ("embedding_dim", "max_length", "num_blocks", "dropout")
        },
    )
    training = dict(
        c["training"],
        **{k: params[k] for k in ("negative_count", "learning_rate", "weight_decay")},
    )
    return model, training


def resume_config(current, work, *, compatibility_path=None):
    """Keep original checkpoint provenance for an explicitly reviewed runner fix.

    Only an exact old->new runner hash pair is allowed. Hyperparameters, inputs,
    libraries and every other implementation file must still match exactly.
    """
    path = work / "config.json"
    if not path.exists():
        return current, None
    frozen = read_json(path)
    if config_sha256(frozen) == config_sha256(current):
        return frozen, None
    compatibility_path = (
        compatibility_path
        or ROOT / "configs/task15_sasrec_optuna_resume_compatibility.json"
    )
    rules = (
        read_json(compatibility_path)
        if compatibility_path.exists()
        else {"transitions": []}
    )
    runner = "scripts/run_sasrec_optuna.py"
    for rule in rules["transitions"]:
        if (
            frozen["implementation_sha256"].get(runner) != rule["from_sha256"]
            or current["implementation_sha256"].get(runner) != rule["to_sha256"]
        ):
            continue
        expected = json.loads(json.dumps(current))
        expected["implementation_sha256"][runner] = rule["from_sha256"]
        if config_sha256(expected) == config_sha256(frozen):
            return frozen, {
                **rule,
                "runner": runner,
                "original_config_sha256": config_sha256(frozen),
                "runtime_config_sha256": config_sha256(current),
                "compatibility_manifest_sha256": sha256_file(compatibility_path),
                "training_and_selection_unchanged": True,
            }
    raise ValueError("config/implementation/input changed; use a new run ID")


def tell_result(study, trial, result):
    """COMPLETE supplies an objective; PRUNED uses its last reported value."""
    if result["trial_number"] != trial.number:
        raise ValueError("terminal result belongs to another trial")
    state = optuna.trial.TrialState[result["state"]]
    if state == optuna.trial.TrialState.COMPLETE:
        return study.tell(trial, values=result["value"], state=state)
    if state in (optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL):
        return study.tell(trial, state=state)
    raise ValueError("result must have a terminal trial state")


def suggest(study, trial, c):
    # Independent TPE with a seed per trial AND parameter. Restarting between
    # individual suggestions cannot advance/lose an opaque sampler RNG state.
    params = {}
    for index, name in enumerate(PARAMETERS):
        study.sampler = optuna.samplers.TPESampler(
            seed=c["seed"] + trial.number * 1009 + index,
            n_startup_trials=c["search"]["startup_trials"],
        )
        spec = c["space"][name]
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=True
            )
    return params


def trial_handle(study, frozen):
    # Optuna's public Trial constructor takes its internal storage ID. Keep
    # this small version-pinned adapter isolated and cover running-trial resume.
    return optuna.trial.Trial(study, frozen._trial_id)


def hash_directory(directory):
    return {
        str(p.relative_to(directory)): sha256_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }


def verify_hashes(directory):
    for name, digest in read_json(directory / "manifest.json")["sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"checksum mismatch: {directory / name}")


def prepare_inputs(c, work, event):
    fold = Path(c["data"]["fold_artifact"])
    cache = work / "sequences"
    external = (
        Path(c["data"]["sequence_cache"]) if c["data"]["sequence_cache"] else None
    )
    if cache.exists():
        store = SequenceStore.load(cache)
    elif external is not None and external.exists():
        store = SequenceStore.load(external)
        event("sequence_cache_reused", stage="data", path=str(external))
    else:
        partial = work / "sequences.partial"
        if partial.exists():
            shutil.rmtree(partial)
        prepare_sequence_store(
            fold / "history_daily.parquet",
            partial,
            cutoff=datetime.fromisoformat(c["input"]["cutoff"]),
            expected_sha256=c["input"]["history_sha256"],
            context_user_limit=c["data"]["context_user_limit"],
            preserve_full_catalog=c["data"]["preserve_full_catalog"],
            seed=c["seed"],
            event=lambda name, **kw: event(name, stage="data", **kw),
        )
        publish_directory_atomic(partial, cache)
        store = SequenceStore.load(cache)
    if (
        store.metadata["history_sha256"] != c["input"]["history_sha256"]
        or store.metadata["cutoff"] != c["input"]["cutoff"]
        or store.metadata["context_user_limit"] != c["data"]["context_user_limit"]
        or store.metadata["full_catalog"] != c["data"]["preserve_full_catalog"]
    ):
        raise ValueError("sequence cache belongs to another history/universe")
    if c["mode"] == "full" and (
        len(store.user_ids) != c["input"]["history_users"]
        or len(store.item_ids) != c["input"]["history_items"]
    ):
        raise ValueError("full selection sequence cache is incomplete")
    data = work / "evaluation"
    if not data.exists():
        for path, digest in c["input"]["file_sha256"].items():
            if sha256_file(path) != digest:
                raise ValueError(f"immutable input checksum mismatch: {path}")
            event("input_verified", stage="data", path=path)
        targets = pl.read_parquet(fold / "target_users.parquet")
        if c["mode"] == "smoke":
            targets = targets.join(
                pl.DataFrame({"user_id": store.user_ids}), on="user_id", how="semi"
            )
        users = id_sample(targets, c["evaluation"]["users"], c["seed"])
        truth = (
            pl.scan_parquet(fold / "target_ground_truth.parquet")
            .join(users.lazy(), on="user_id", how="semi")
            .collect()
        )
        original_truth_count = truth.height
        if c["mode"] == "smoke" and not c["data"]["preserve_full_catalog"]:
            truth = truth.join(
                pl.DataFrame({"item_id": store.item_ids}), on="item_id", how="semi"
            )
        partial = work / "evaluation.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        users.write_parquet(partial / "users.parquet")
        truth.write_parquet(partial / "truth.parquet")
        for source in SOURCES:
            frame = (
                pl.scan_parquet(
                    Path(c["data"]["source_fold"])
                    / "sources"
                    / source
                    / "candidates.parquet"
                )
                .join(users.lazy(), on="user_id", how="semi")
                .collect(engine="streaming")
            )
            if c["mode"] == "smoke" and not c["data"]["preserve_full_catalog"]:
                frame = frame.join(
                    pl.DataFrame({"item_id": store.item_ids}), on="item_id", how="semi"
                )
                # Preserve the original source ordering after catalog restriction.
                frame = frame.sort("user_id", "rank").with_columns(
                    pl.col("item_id")
                    .cum_count()
                    .over("user_id")
                    .cast(pl.UInt32)
                    .alias("rank")
                )
            frame.write_parquet(partial / f"{source}.parquet")
            event("source_sample_ready", stage="data", source=source, rows=frame.height)
        write_json_atomic(
            partial / "metadata.json",
            {
                "sampling": "ID_hash_only"
                if c["mode"] == "full"
                else "smoke_context_ID_hash",
                "target_users": users.height,
                "sample_positive_pairs_before_smoke_catalog_filter": original_truth_count,
                "sample_positive_pairs": truth.height,
                "full_fold_ground_truth_funnel": c["input"]["ground_truth_funnel"],
            },
        )
        write_json_atomic(
            partial / "manifest.json", {"sha256": hash_directory(partial)}
        )
        publish_directory_atomic(partial, data)
    verify_hashes(data)
    evaluator = SelectionEvaluator(
        pl.read_parquet(data / "users.parquet"),
        pl.read_parquet(data / "truth.parquet"),
        {s: pl.read_parquet(data / f"{s}.parquet") for s in SOURCES},
        store,
    )
    write_json_atomic(
        work / "baseline_metrics.json",
        {
            "union": evaluator.baseline_metrics,
            "sources": evaluator.source_metrics,
            "universe": read_json(data / "metadata.json"),
        },
    )
    return store, evaluator


class SearchRunner:
    def __init__(
        self, c, *, show_progress=True, stop_after_epoch=None, stop_after_trial=None
    ):
        self.output = Path("artifacts") / c["run_id"]
        self.work = Path("artifacts") / f".{c['run_id']}.work"
        if self.output.exists():
            raise FileExistsError(self.output)
        self.work.mkdir(parents=True, exist_ok=True)
        self.c, runtime_patch = resume_config(c, self.work)
        self.digest = config_sha256(self.c)
        if not (self.work / "config.json").exists():
            write_json_atomic(self.work / "config.json", self.c)
        if runtime_patch:
            write_json_atomic(self.work / "runtime_patch.json", runtime_patch)
        self.show_progress = show_progress
        self.stop_after_epoch, self.stop_after_trial = (
            stop_after_epoch,
            stop_after_trial,
        )
        self.started = time.perf_counter()
        timing = self.work / "timing.json"
        self.previous_seconds = (
            read_json(timing)["active_seconds"] if timing.exists() else 0.0
        )
        self.stopped = False
        self.reporter = EventProgressReporter(
            task_name=c["run_id"],
            total_phases=3,
            log_file=f"logs/{c['run_id']}.log",
            show_progress=show_progress,
        )
        self.handlers = {
            s: signal.signal(s, self.interrupt)
            for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }

    def interrupt(self, signum, frame):
        self.stopped = True

    def check_stop(self):
        if self.stopped:
            raise BenchmarkPaused(
                "stop requested; repeat identical launcher/config to resume"
            )

    def elapsed(self):
        return self.previous_seconds + time.perf_counter() - self.started

    def budget_expired(self):
        return self.elapsed() >= self.c["search"]["timeout_hours"] * 3600

    def save_timing(self):
        write_json_atomic(self.work / "timing.json", {"active_seconds": self.elapsed()})

    def event(self, name, **fields):
        fields.setdefault("config", self.c["run_id"])
        self.reporter.event(name, fold="rolling_1", **fields)

    def bar(self, total, desc, position, *, initial=0):
        return tqdm(
            total=total,
            initial=initial,
            desc=desc,
            position=position,
            leave=position <= 1,
            disable=not self.show_progress,
            dynamic_ncols=True,
        )

    def run(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            configure_torch(self.c)
            before = self.reporter.phase_start("data", fold="rolling_1")
            self.store, self.evaluator = prepare_inputs(self.c, self.work, self.event)
            self.event(
                "baseline_ready", stage="data", metrics=self.evaluator.baseline_metrics
            )
            self.reporter.phase_finish("data", before, fold="rolling_1")
            storage = optuna.storages.RDBStorage(
                url="sqlite:///" + str((self.work / "study.sqlite3").resolve()),
                engine_kwargs={"connect_args": {"timeout": 60}},
            )
            self.study = optuna.create_study(
                study_name=self.c["run_id"],
                storage=storage,
                direction="maximize",
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=self.c["search"]["startup_trials"],
                    n_warmup_steps=self.c["search"]["prune_warmup_epochs"],
                    interval_steps=self.c["search"]["evaluate_every"],
                    n_min_trials=3 if self.c["mode"] == "full" else 1,
                ),
            )
            existing_digest = self.study.user_attrs.get("config_sha256")
            if existing_digest is not None and existing_digest != self.digest:
                raise ValueError("Optuna study provenance differs")
            self.study.set_user_attr("config_sha256", self.digest)
            if not self.study.trials:
                self.study.enqueue_trial(self.c["baseline"])
            before = self.reporter.phase_start("search", fold="rolling_1")
            finished = sum(t.state.is_finished() for t in self.study.trials)
            with self.bar(
                self.c["search"]["trials"], "rolling_1 | trials", 1, initial=finished
            ) as bar:
                while finished < self.c["search"]["trials"]:
                    self.check_stop()
                    running = [
                        t
                        for t in self.study.trials
                        if t.state == optuna.trial.TrialState.RUNNING
                    ]
                    if len(running) > 1:
                        raise ValueError(
                            "sequential search has multiple running trials"
                        )
                    if not running and self.budget_expired() and finished:
                        break
                    trial = (
                        trial_handle(self.study, running[0])
                        if running
                        else self.study.ask()
                    )
                    params = suggest(self.study, trial, self.c)
                    result = self.run_trial(trial, params)
                    tell_result(self.study, trial, result)
                    (
                        self.work
                        / "trials"
                        / f"trial_{trial.number:04d}"
                        / "checkpoint.pt"
                    ).unlink(missing_ok=True)
                    self.save_timing()
                    finished += 1
                    bar.update(1)
                    best = self.best_result()
                    bar.set_postfix(best=f"{best['value']:.6f}" if best else "none")
                    self.event(
                        "trial_finish",
                        stage="search",
                        config=f"trial_{trial.number:04d}",
                        state=result["state"],
                        current_metric=result.get("value"),
                        best_metric=best["value"] if best else None,
                        best_config=f"trial_{best['trial_number']:04d}"
                        if best
                        else None,
                        stop_reason=result["stop_reason"],
                        duration_seconds=result["duration_seconds"],
                    )
                    if self.stop_after_trial == finished:
                        raise BenchmarkPaused(
                            "intentional smoke pause after terminal trial"
                        )
            self.reporter.phase_finish("search", before, fold="rolling_1")
            self.check_stop()
            before = self.reporter.phase_start("publish", fold="rolling_1")
            result = self.publish()
            self.reporter.phase_finish("publish", before, fold="rolling_1")
            self.event(
                "run_finish",
                stage="publish",
                status="completed",
                output=str(result),
                duration_seconds=self.elapsed(),
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
                {
                    "error": repr(error),
                    "resume": "same command; incomplete epoch restarts from last checkpoint",
                },
            )
            raise
        finally:
            self.save_timing()
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
            self.reporter.close()

    def best_result(self):
        candidates = []
        for p in (self.work / "trials").glob("trial_*/result.json"):
            value = read_json(p)
            if value.get("best") is not None:
                candidates.append(value)
        return (
            max(candidates, key=lambda r: (r["value"], -r["trial_number"]))
            if candidates
            else None
        )

    def commit_best_pointer(self, directory, best, number, params):
        if best is None:
            return
        current = self.work / "best_so_far.json"
        prior = read_json(current) if current.exists() else None
        if (
            prior is None
            or best["value"] > prior["value"]
            or prior["trial_number"] == number
        ):
            write_json_atomic(
                current,
                {
                    **best,
                    "trial_number": number,
                    "directory": str(directory / best["path"]),
                    "params": params,
                },
            )
        # Pointer is committed before deleting a previous generation. The
        # checkpoint also references the new directory at this boundary.
        for old in directory.glob("best_epoch_*"):
            if old.name != best["path"] and old.is_dir():
                shutil.rmtree(old)

    def run_trial(self, trial, params):
        directory = self.work / "trials" / f"trial_{trial.number:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "result.json").exists():
            self.event(
                "terminal_trial_recovered", stage="search", config=directory.name
            )
            return read_json(directory / "result.json")
        model_config, training = recipe(self.c, params)
        self.event(
            "trial_start", stage="training", config=directory.name, params=params
        )
        write_json_atomic(
            directory / "config.json",
            {
                "params": params,
                "model": model_config,
                "training": training,
                "config_sha256": self.digest,
            },
        )
        configure_torch(self.c)
        started = time.perf_counter()
        model = SASRecCandidateModel(
            self.store.item_ids,
            SASRecConfig(**model_config),
            device=self.c["device"],
            history_sha256=self.c["input"]["history_sha256"],
        )
        optimizer = torch.optim.AdamW(
            model.encoder.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            fused=self.c["device"] == "cuda",
            foreach=False,
        )
        loader = SASRecDataLoader(
            self.store, max_length=model_config["max_length"], seed=self.c["seed"]
        )
        checkpoint = directory / "checkpoint.pt"
        records, best, epoch, previous_duration = [], None, 0, 0.0

        def save_checkpoint():
            save_torch_atomic(
                checkpoint,
                {
                    "config_sha256": self.digest,
                    "params": params,
                    "epoch": epoch,
                    "records": records,
                    "best": best,
                    "duration_seconds": previous_duration
                    + time.perf_counter()
                    - started,
                    "model": model.encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng(use_cuda=self.c["device"] == "cuda"),
                },
            )
            write_json_atomic(
                directory / "checkpoint.json",
                {"epoch": epoch, "best": best, "config_sha256": self.digest},
            )
            write_json_atomic(directory / "epoch_metrics.json", {"epochs": records})
            self.save_timing()

        try:
            if checkpoint.exists():
                state = torch.load(
                    checkpoint, map_location=self.c["device"], weights_only=True
                )
                if state["config_sha256"] != self.digest or state["params"] != params:
                    raise ValueError("incompatible trial checkpoint")
                model.encoder.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                restore_rng(state["rng"])
                records, best, epoch, previous_duration = (
                    state["records"],
                    state["best"],
                    state["epoch"],
                    state["duration_seconds"],
                )
                del state
                self.event(
                    "checkpoint_restored",
                    stage="training",
                    config=directory.name,
                    epoch=epoch,
                )
                if best:
                    verify_hashes(directory / best["path"])
                    self.commit_best_pointer(directory, best, trial.number, params)
            else:
                save_checkpoint()
            # An interruption after checkpoint but before report/prune is recovered
            # by processing the recorded evaluation before starting another epoch.
            terminal_reason = self.process_evaluation(trial, records, best)
            with self.bar(
                self.c["search"]["max_epochs"],
                f"trial {trial.number} | epochs",
                2,
                initial=epoch,
            ) as epoch_bar:
                while (
                    epoch < self.c["search"]["max_epochs"] and terminal_reason is None
                ):
                    self.check_stop()
                    epoch += 1
                    before = time.perf_counter()
                    self.event(
                        "epoch_start",
                        stage="training",
                        config=directory.name,
                        epoch=epoch,
                    )
                    with self.bar(
                        min(
                            len(self.store.eligible_users),
                            training["max_users"] or len(self.store.eligible_users),
                        ),
                        f"epoch {epoch} | windows",
                        3,
                    ) as prep_bar:

                        def prepared(done, total):
                            self.check_stop()
                            prep_bar.update(done - prep_bar.n)

                        loader.load_fit_data().prepare_fit_data(
                            epoch=epoch,
                            negative_count=training["negative_count"],
                            max_users=training["max_users"],
                            memory_budget_bytes=int(
                                self.c["resources"]["epoch_arrays_gib"] * 2**30
                            ),
                            progress=prepared,
                        )
                    preparation = time.perf_counter() - before
                    batches = math.ceil(
                        loader.epoch_metadata["users"] / training["batch_size"]
                    )
                    with self.bar(batches, f"epoch {epoch} | train", 3) as batch_bar:
                        last_log = time.monotonic()

                        def batch_done(
                            index,
                            loss,
                            seconds,
                            targets,
                            norm,
                            batches=batches,
                            epoch=epoch,
                        ):
                            nonlocal last_log
                            batch_bar.update(1)
                            batch_bar.set_postfix(loss=f"{loss:.4f}", refresh=False)
                            if (
                                index in (1, batches)
                                or time.monotonic() - last_log >= 30
                            ):
                                self.event(
                                    "batch_progress",
                                    stage="training",
                                    config=directory.name,
                                    epoch=epoch,
                                    batch=index,
                                    batches=batches,
                                    loss=loss,
                                    step_seconds=seconds,
                                )
                                last_log = time.monotonic()

                        model.fit(
                            loader,
                            optimizer=optimizer,
                            batch_size=training["batch_size"],
                            precision=self.c["precision"],
                            gradient_clip_norm=training["gradient_clip_norm"],
                            query_chunk_size=training["query_chunk_size"],
                            callback=batch_done,
                            check_stop=self.check_stop,
                        )
                    record = {
                        **loader.epoch_metadata,
                        **model.last_fit_metrics,
                        "preparation_seconds": preparation,
                    }
                    loader.fit_arrays = None
                    if (
                        epoch % self.c["search"]["evaluate_every"] == 0
                        or epoch == self.c["search"]["max_epochs"]
                        or self.budget_expired()
                    ):
                        evaluation, candidates = self.evaluate(
                            model, loader, trial.number, epoch
                        )
                        record["evaluation"] = evaluation
                        if best is None or evaluation["objective"] > best["value"]:
                            name = f"best_epoch_{epoch:03d}"
                            partial = directory / (name + ".partial")
                            if partial.exists():
                                shutil.rmtree(partial)
                            # This name can only be an uncommitted result of the
                            # same incomplete epoch; the checkpoint is authoritative.
                            if (directory / name).exists():
                                shutil.rmtree(directory / name)
                            partial.mkdir()
                            model.save(partial / "model")
                            candidates.write_parquet(partial / "candidates.parquet")
                            standalone_top20(
                                candidates,
                                self.evaluator.sources["global_popularity"],
                                self.evaluator.users,
                            ).write_parquet(partial / "recommendations.parquet")
                            write_json_atomic(partial / "metrics.json", evaluation)
                            active = torch.where(
                                (loader.predict_arrays[0] != 0).any(1)
                            )[0][:8]
                            if not len(active):
                                raise ValueError(
                                    "no active sequence for portable verification"
                                )
                            probe = (
                                loader.predict_arrays[0][active]
                                .long()
                                .to(self.c["device"])
                            )
                            with torch.inference_mode():
                                indices = (
                                    torch.arange(len(probe), device=self.c["device"])
                                    % len(self.store.item_ids)
                                    + 1
                                )
                                scores = (
                                    model.score_pairs(
                                        model.encoder.encode_users(probe), indices
                                    )
                                    .cpu()
                                    .numpy()
                                )
                            np.savez(
                                partial / "portable_probe.npz",
                                inputs=probe.cpu().numpy(),
                                item_indices=indices.cpu().numpy(),
                                scores=scores,
                            )
                            write_json_atomic(
                                partial / "manifest.json",
                                {"sha256": hash_directory(partial)},
                            )
                            publish_directory_atomic(partial, directory / name)
                            best = {
                                "value": evaluation["objective"],
                                "epoch": epoch,
                                "path": name,
                                "policy": evaluation["selected_policy"],
                            }
                    records.append(record)
                    save_checkpoint()
                    self.commit_best_pointer(directory, best, trial.number, params)
                    self.event(
                        "epoch_finish",
                        stage="training",
                        config=directory.name,
                        epoch=epoch,
                        duration_seconds=time.perf_counter() - before,
                        current_metric=record.get("evaluation", {}).get("objective"),
                        best_metric=best["value"] if best else None,
                        loss=record["training_loss"],
                    )
                    epoch_bar.update(1)
                    epoch_bar.set_postfix(
                        best=f"{best['value']:.6f}" if best else "pending"
                    )
                    if self.stop_after_epoch == (trial.number, epoch):
                        raise BenchmarkPaused(
                            "intentional smoke pause after epoch checkpoint"
                        )
                    terminal_reason = self.process_evaluation(trial, records, best)
            if best is None:
                raise ValueError("trial completed without a validation checkpoint")
            result = {
                "trial_number": trial.number,
                "params": params,
                "state": "PRUNED" if terminal_reason == "median_pruner" else "COMPLETE",
                "stop_reason": terminal_reason or "max_epochs",
                "value": best["value"],
                "best": best,
                "epochs_completed": epoch,
                "model_parameters": sum(p.numel() for p in model.encoder.parameters()),
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                if self.c["device"] == "cuda"
                else None,
                "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
                if self.c["device"] == "cuda"
                else None,
                "duration_seconds": previous_duration + time.perf_counter() - started,
            }
            write_json_atomic(directory / "result.json", result)
            return result
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
            result = {
                "trial_number": trial.number,
                "params": params,
                "state": "FAIL",
                "stop_reason": repr(error),
                "best": None,
                "duration_seconds": previous_duration + time.perf_counter() - started,
            }
            write_json_atomic(directory / "result.json", result)
            return result
        finally:
            optimizer = model = loader = None
            gc.collect()
            if self.c["device"] == "cuda":
                torch.cuda.empty_cache()

    def process_evaluation(self, trial, records, best):
        if not records or "evaluation" not in records[-1]:
            return None
        record = records[-1]
        epoch = record["epoch"]
        # Recovery may revisit this evaluation; Optuna reports are immutable.
        if epoch not in trial.study.trials[trial.number].intermediate_values:
            trial.report(best["value"], step=epoch)
        if trial.should_prune():
            return "median_pruner"
        stale = sum("evaluation" in r and r["epoch"] > best["epoch"] for r in records)
        if stale >= self.c["search"]["patience_evaluations"]:
            return "early_stopping"
        if self.budget_expired():
            return "time_budget"
        return None

    def evaluate(self, model, loader, number, epoch):
        self.check_stop()
        loader.load_predict_data(
            user_ids=self.evaluator.users["user_id"].to_numpy()
        ).prepare_predict_data()
        before = time.perf_counter()
        e = self.c["evaluation"]
        with self.bar(
            math.ceil(self.evaluator.users.height / e["batch_size"]),
            f"epoch {epoch} | recall@200",
            3,
        ) as bar:

            def retrieved(index):
                self.check_stop()
                bar.update(1)

            candidates = model.predict(
                loader,
                k=200,
                batch_size=e["batch_size"],
                item_chunk_size=e["item_chunk_size"],
                callback=retrieved,
            )
        retrieval_seconds = time.perf_counter() - before
        result = self.evaluator.evaluate(candidates)
        result["retrieval_seconds"] = retrieval_seconds
        result["evaluation_seconds"] = time.perf_counter() - before
        self.event(
            "evaluation_finish",
            stage="validation",
            config=f"trial_{number:04d}",
            epoch=epoch,
            current_metric=result["objective"],
            policy=result["selected_policy"],
            standalone_recall=result["native_sasrec"]["candidate_recall"],
            duration_seconds=result["evaluation_seconds"],
        )
        return result, candidates

    def publish(self):
        best = self.best_result()
        if best is None:
            raise ValueError(
                "no successful validation checkpoint; inspect failed trials"
            )
        directory = (
            self.work
            / "trials"
            / f"trial_{best['trial_number']:04d}"
            / best["best"]["path"]
        )
        verify_hashes(directory)
        selected = read_json(directory / "metrics.json")
        partial = self.work / "publish"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        shutil.copytree(directory / "model", partial / "model")
        for source, destination in (
            (directory / "candidates.parquet", "selection_candidates.parquet"),
            (directory / "portable_probe.npz", "portable_probe.npz"),
            (directory / "recommendations.parquet", "recommendations.parquet"),
            (self.work / "baseline_metrics.json", "baseline_metrics.json"),
            (self.work / "evaluation/users.parquet", "selection_users.parquet"),
            (self.work / "evaluation/truth.parquet", "selection_truth.parquet"),
        ):
            shutil.copy2(source, partial / destination)
        model_config, training = recipe(self.c, best["params"])
        write_json_atomic(
            partial / "best_recipe.json",
            {
                "trial_number": best["trial_number"],
                "model": model_config,
                "training": dict(training, epochs=best["best"]["epoch"]),
                "source_policy": best["best"]["policy"],
                "selection_fold": "rolling_1",
                "cutoff": self.c["input"]["cutoff"],
                "target_end_exclusive": self.c["input"]["validation_end_exclusive"],
                "seed": self.c["seed"],
                "objective": self.c["objective"],
                "objective_value": best["value"],
                "canonical_used_for_selection": False,
            },
        )
        write_json_atomic(partial / "config.json", self.c)
        if (self.work / "runtime_patch.json").exists():
            shutil.copy2(
                self.work / "runtime_patch.json", partial / "runtime_patch.json"
            )
        write_json_atomic(
            partial / "trials.json",
            {
                "trials": [
                    read_json(p)
                    for p in sorted((self.work / "trials").glob("trial_*/result.json"))
                ]
            },
        )
        with (
            sqlite3.connect(self.work / "study.sqlite3") as source,
            sqlite3.connect(partial / "study.sqlite3") as destination,
        ):
            source.backup(destination)
        metrics = {
            "run_id": self.c["run_id"],
            "mode": self.c["mode"],
            "split": "rolling_1_ID_sample_next_day_selection",
            "best_trial": best,
            "selection": selected,
            "candidate_recall": selected["native_sasrec"]["candidate_recall"],
            "coverage": selected["native_sasrec"]["coverage"],
            **selected["standalone_with_global_fallback"],
            "precision_semantics": "SASRec top20 with independent global fallback on selection sample; not ranker score",
            "baseline": self.evaluator.baseline_metrics,
            "runtime_seconds": self.elapsed(),
            "peak_rss_gib_current_session": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            / 2**20,
            "trials_finished": len(
                [t for t in self.study.trials if t.state.is_finished()]
            ),
            "search_budget_exhausted": self.budget_expired(),
            "canonical_used_for_selection": False,
            "full_fold_metrics_measured": False,
            "promote_over_baseline": False,
            "promotion_note": "Selection estimate only; freeze recipe then compare across rolling folds.",
        }
        write_json_atomic(partial / "metrics.json", metrics)
        write_json_atomic(
            partial / "manifest.json", {"sha256": hash_directory(partial)}
        )
        verification = verify_artifact(partial)
        self.event("portable_verified", stage="publish", **verification)
        self.check_stop()
        publish_directory_atomic(partial, self.output)
        append_experiment(self.c, metrics, self.output)
        print(
            f"Saved {self.output}/best_recipe.json | best delta recall={best['value']:.6f} | trial={best['trial_number']} epoch={best['best']['epoch']}"
        )
        return self.output


def verify_artifact(directory):
    verify_hashes(directory)
    c = read_json(directory / "config.json")
    m = read_json(directory / "metrics.json")
    torch.set_num_threads(c["cpu_threads"])
    model = SASRecCandidateModel.from_artifact(directory / "model")
    if (
        model.history_sha256 != c["input"]["history_sha256"]
        or m["canonical_used_for_selection"]
    ):
        raise ValueError("incorrect portable model provenance")
    probe = np.load(directory / "portable_probe.npz", allow_pickle=False)
    with torch.inference_mode():
        scores = model.score_pairs(
            model.encoder.encode_users(torch.from_numpy(probe["inputs"]).long()),
            torch.from_numpy(probe["item_indices"]).long(),
        ).numpy()
    if not np.allclose(scores, probe["scores"], atol=1e-4, rtol=1e-4):
        raise ValueError("portable CPU inference differs")
    from metrics import evaluate_candidate_metrics, evaluate_precision_at_20

    measured = evaluate_candidate_metrics(
        pl.read_parquet(directory / "selection_candidates.parquet"),
        pl.read_parquet(directory / "selection_truth.parquet"),
        pl.read_parquet(directory / "selection_users.parquet"),
    )
    if measured["candidate_recall"] != m["candidate_recall"]:
        raise ValueError("saved selection recall differs from candidate pairs")
    recommendations = pl.read_parquet(directory / "recommendations.parquet")
    users = pl.read_parquet(directory / "selection_users.parquet")
    if (
        not recommendations.select("user_id")
        .sort("user_id")
        .equals(users.sort("user_id"))
        or recommendations.filter(pl.col("item_ids").list.len() != 20).height
    ):
        raise ValueError(
            "portable recommendations do not cover every selection user with top20"
        )
    precision = evaluate_precision_at_20(
        recommendations, pl.read_parquet(directory / "selection_truth.parquet"), users
    )
    if any(
        not math.isclose(value, m[key], rel_tol=1e-14, abs_tol=1e-15)
        for key, value in precision.items()
    ):
        raise ValueError("saved standalone Precision@20 differs from recommendations")
    return {
        "verified": True,
        "portable_scores_equal": True,
        "candidate_recall": measured["candidate_recall"],
    }


def append_experiment(c, m, output):
    if c["mode"] != "full":
        return
    row = {
        "run_id": c["run_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": m["split"],
        "seed": c["seed"],
        "candidate_config": json.dumps(m["best_trial"]["params"]),
        "ranker_config": "standalone_sasrec_with_global_fallback",
        "p20_all_targets": m["precision_at_20_all_targets"],
        "p20_labeled_users": m["precision_at_20_labeled_users"],
        "candidate_recall": m["candidate_recall"],
        "coverage": m["coverage"],
        "runtime": m["runtime_seconds"],
        "artifact_path": str(output),
        "notes": "Optuna selection sample only; not canonical/ranker improvement; see equal-budget union and ALS overlap in metrics.json",
    }
    path = Path("experiments/results.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or list(row)
        if any(r["run_id"] == c["run_id"] for r in reader):
            return
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task15_sasrec_optuna_v1.json")
    )
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop-after-epoch",
        nargs=2,
        type=int,
        metavar=("TRIAL", "EPOCH"),
        help="Smoke-only checkpoint/pause hook",
    )
    parser.add_argument(
        "--stop-after-trial", type=int, help="Smoke-only terminal-trial pause hook"
    )
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_artifact(args.verify_only)
        append_experiment(
            read_json(args.verify_only / "config.json"),
            read_json(args.verify_only / "metrics.json"),
            args.verify_only,
        )
        print(json.dumps(result, indent=2))
        return 0
    c = load_config(args.config)
    if (args.stop_after_epoch is not None or args.stop_after_trial is not None) and c[
        "mode"
    ] != "smoke":
        raise ValueError("pause testing hooks are restricted to smoke")
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if args.worker:
        try:
            SearchRunner(
                resolve_config(c),
                show_progress=not args.no_progress,
                stop_after_epoch=tuple(args.stop_after_epoch)
                if args.stop_after_epoch
                else None,
                stop_after_trial=args.stop_after_trial,
            ).run()
        except BenchmarkPaused as error:
            print(str(error), file=sys.stderr)
            return 75
        return 0
    # Same lock as the SASRec benchmark prevents concurrent GPU jobs in Task15.
    with Path("logs/task15_sasrec_benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Task15 GPU job is running") from error
        if (Path("artifacts") / c["run_id"]).exists():
            raise FileExistsError("completed search exists; use --verify-only")
        command = [
            sys.executable,
            str(Path(__file__)),
            "--config",
            str(args.config),
            "--worker",
        ]
        if args.no_progress:
            command.append("--no-progress")
        if args.stop_after_epoch:
            command.extend(["--stop-after-epoch", *map(str, args.stop_after_epoch)])
        if args.stop_after_trial:
            command.extend(["--stop-after-trial", str(args.stop_after_trial)])
        return supervise(command, c)


if __name__ == "__main__":
    raise SystemExit(main())

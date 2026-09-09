#!/usr/bin/env python3
"""Frozen SASRec fit and full-user candidate evaluation on four temporal folds."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import itertools
import json
import math
import os
import re
import resource
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import implicit
import numpy as np
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
from implicit_model import ImplicitALSDataLoader, ImplicitALSModel
from sasrec_data import SASRecDataLoader, SequenceStore, prepare_sequence_store
from sasrec_evaluation import evaluate_shard, summarize
from sasrec_model import SASRecCandidateModel, SASRecConfig, save_torch_atomic
from sasrec_selection import POLICIES, SOURCES, id_sample
from scripts.run_sasrec_benchmark import (
    BenchmarkPaused,
    capture_rng,
    configure_torch,
    restore_rng,
)
from scripts.run_sasrec_optuna import hash_directory, verify_hashes
from scripts.run_sasrec_optuna import recipe as selection_recipe
from scripts.run_sasrec_optuna import verify_artifact as verify_selection
from scripts.task15_resources import supervise

FOLD_NAMES = ("rolling_1", "rolling_2", "rolling_3", "canonical")
IMPLEMENTATION = (
    "scripts/run_sasrec_folds.py",
    "sasrec_evaluation.py",
    "sasrec_model.py",
    "sasrec_data.py",
    "sasrec_selection.py",
    "implicit_model.py",
    "metrics.py",
    "validation.py",
    "interfaces.py",
    "data_utils.py",
    "experiment_utils.py",
    "scripts/run_sasrec_benchmark.py",
    "scripts/run_sasrec_optuna.py",
    "scripts/task15_resources.py",
    "scripts/task13_resources.py",
)


def load_config(path):
    c = read_json(path)
    if c["kind"] != "sasrec_frozen_fold_evaluation" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid frozen-fold config/run ID")
    if c["mode"] not in ("full", "smoke") or c["device"] not in ("cuda", "cpu"):
        raise ValueError("invalid mode/device")
    if c["precision"] not in ("float32", "bfloat16") or not 1 <= c["cpu_threads"] <= 8:
        raise ValueError("invalid precision/threads")
    names = [f["name"] for f in c["folds"]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in FOLD_NAMES for n in names)
    ):
        raise ValueError("invalid folds")
    if c["mode"] == "full":
        if tuple(names) != FOLD_NAMES or c["device"] != "cuda" or c.get("smoke"):
            raise ValueError(
                "full run requires all four ordered folds, GPU and full universes"
            )
        if not c["reuse_selection_fold_model"]:
            raise ValueError("reuse the already fully trained rolling_1 winner")
    else:
        s = c["smoke"]
        if not (
            1 <= s["context_users"] <= 4096
            and 1 <= s["training_users"] <= 2048
            and 1 <= s["evaluation_users"] <= 64
            and 1 <= s["epochs"] <= 2
        ):
            raise ValueError("smoke must be explicitly bounded")
        if c["reuse_selection_fold_model"]:
            raise ValueError("smoke must exercise fitting, not reuse selection weights")
    e, r = c["evaluation"], c["resources"]
    if not (
        1 <= e["users_per_shard"] <= 2048
        and 1 <= e["batch_size"] <= 128
        and 1 <= e["als_batch_size"] <= 128
        and 1 <= e["item_chunk_size"] <= 32768
    ):
        raise ValueError("invalid bounded evaluation batches")
    if not (
        0 < r["maximum_rss_gib"] <= 40
        and 0 < r["maximum_run_disk_gib"] <= 30
        and 0 < r["cuda_memory_fraction"] <= 0.8
        and 0 < r["poll_seconds"] <= 5
    ):
        raise ValueError("invalid resource budget")
    return c


def resolve_config(c):
    c = json.loads(json.dumps(c))
    selection = Path(c["selection_artifact"])
    if sha256_file(selection / "best_recipe.json") != c["selection_recipe_sha256"]:
        raise ValueError("selection recipe changed")
    if sha256_file(selection / "manifest.json") != c["selection_manifest_sha256"]:
        raise ValueError("selection artifact changed")
    recipe = read_json(selection / "best_recipe.json")
    selected_config = read_json(selection / "config.json")
    winner = read_json(selection / "metrics.json")["best_trial"]
    architecture, training = selection_recipe(selected_config, winner["params"])
    if (
        recipe["trial_number"] != winner["trial_number"]
        or recipe["model"] != architecture
        or recipe["training"] != {**training, "epochs": winner["best"]["epoch"]}
        or recipe["source_policy"] != winner["best"]["policy"]
    ):
        raise ValueError("recipe differs from the published winning trial")
    if (
        recipe["selection_fold"] != "rolling_1"
        or recipe["canonical_used_for_selection"]
    ):
        raise ValueError("recipe must have been selected only on rolling_1")
    if recipe["seed"] != c["seed"] or recipe["source_policy"] not in POLICIES:
        raise ValueError("invalid frozen seed/policy")
    if c["mode"] == "full" and (
        recipe["training"]["max_users"] is not None
        or selected_config["mode"] != "full"
        or c["precision"] != selected_config["precision"]
    ):
        raise ValueError(
            "full evaluation requires full-fit selection recipe and same precision"
        )
    SASRecConfig(**recipe["model"])
    if not 1 <= recipe["training"]["epochs"] <= 20:
        raise ValueError("invalid frozen epoch count")
    c["recipe"] = recipe
    c["training"] = dict(recipe["training"])
    if c["mode"] == "smoke":
        c["training"].update(
            epochs=c["smoke"]["epochs"], max_users=c["smoke"]["training_users"]
        )
    c["input"] = {}
    for fold in c["folds"]:
        name, root = fold["name"], Path(fold["artifact"])
        split = read_json(root / "config.json")["split"]
        diagnostics = read_json(root / "metrics.json")["deterministic_diagnostics"]
        source_root = Path(c["source_dataset"]) / "folds" / name
        manifest = read_json(source_root / "dataset_manifest.json")
        if (
            manifest["fold"] != name
            or manifest["cutoff"] != split["cutoff"]
            or Path(manifest["history_path"]).resolve()
            != (root / "history_daily.parquet").resolve()
        ):
            raise ValueError("fixed sources belong to a different fold/history")
        if name == "rolling_1" and split["cutoff"] != recipe["cutoff"]:
            raise ValueError("selection and evaluation rolling_1 differ")
        if (
            split.get("validation_end_exclusive")
            and (
                datetime.fromisoformat(split["validation_end_exclusive"])
                - datetime.fromisoformat(split["cutoff"])
            ).total_seconds()
            != 86400
        ):
            raise ValueError("fold target is not the next 24 hours")
        files = {
            str(root / n): diagnostics["output_sha256"][n]
            for n in (
                "history_daily.parquet",
                "target_users.parquet",
                "target_ground_truth.parquet",
            )
        }
        files[str(source_root / "dataset_manifest.json")] = sha256_file(
            source_root / "dataset_manifest.json"
        )
        source_metrics = {}
        for source in SOURCES:
            d = source_root / "sources" / source
            meta = read_json(d / "metadata.json")
            if (
                meta["source"] != source
                or meta["candidate_k"] != 200
                or meta["fold"] != name
                or meta["cutoff"] != split["cutoff"]
            ):
                raise ValueError("incorrect source metadata")
            files[str(d / "metadata.json")] = sha256_file(d / "metadata.json")
            files[str(d / "candidates.parquet")] = meta["candidates_sha256"]
            source_metrics[source] = meta.get("candidate_metrics")
            if source == "implicit_als":
                model_dir = Path(meta["model_path"])
                if meta["model_storage"] == "embedded":
                    model_dir = d / model_dir
                for n in (
                    "model_config.json",
                    "als_model.npz",
                    "user_mapping.parquet",
                    "item_mapping.parquet",
                ):
                    files[str(model_dir / n)] = sha256_file(model_dir / n)
        c["input"][name] = {
            "split": split,
            "file_sha256": files,
            "source_root": str(source_root),
            "history_sha256": diagnostics["output_sha256"]["history_daily.parquet"],
            "ground_truth_funnel": diagnostics["ground_truth_funnel"],
            "history_users": diagnostics["daily"]["history"]["users"],
            "history_items": diagnostics["daily"]["history"]["items"],
            "als_model": str(model_dir),
            "reference_source_metrics": source_metrics,
            "reference_baseline_metrics": manifest["metrics"],
        }
    if c["mode"] == "full":
        cutoffs = [
            datetime.fromisoformat(c["input"][n]["split"]["cutoff"]) for n in FOLD_NAMES
        ]
        if any(
            (b - a).total_seconds() != 86400 for a, b in itertools.pairwise(cutoffs)
        ):
            raise ValueError("expected three rolling folds followed by canonical")
    c["implementation_sha256"] = {p: sha256_file(ROOT / p) for p in IMPLEMENTATION}
    c["library_versions"] = {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "implicit": implicit.__version__,
    }
    return c


def seal(directory):
    write_json_atomic(
        directory / "manifest.json", {"sha256": hash_directory(directory)}
    )


def clean_partial(path):
    # Only runner-owned, unpublished staging directories are replaceable.
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_users(path, users):
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("user_id").is_between(
                int(users["user_id"].min()), int(users["user_id"].max())
            )
        )
        .join(users.lazy(), on="user_id", how="semi")
        .collect(engine="streaming")
    )


class FoldsRunner:
    def __init__(
        self, c, *, show_progress=True, stop_after_epoch=None, stop_after_shard=None
    ):
        self.c, self.show_progress = c, show_progress
        self.stop_after_epoch, self.stop_after_shard = (
            stop_after_epoch,
            stop_after_shard,
        )
        self.work = Path("artifacts") / f".{c['run_id']}.work"
        self.output = Path("artifacts") / c["run_id"]
        if self.output.exists():
            raise FileExistsError("completed artifact exists; use --verify-only")
        self.work.mkdir(parents=True, exist_ok=True)
        self.digest = config_sha256(c)
        path = self.work / "config.json"
        if path.exists() and config_sha256(read_json(path)) != self.digest:
            raise ValueError("config/inputs/implementation changed; use a new run ID")
        if not path.exists():
            write_json_atomic(path, c)
        self.started = time.perf_counter()
        self.previous = (
            read_json(self.work / "timing.json")["active_seconds"]
            if (self.work / "timing.json").exists()
            else 0.0
        )
        self.reporter = EventProgressReporter(
            task_name=c["run_id"],
            total_phases=len(c["folds"]) + 1,
            log_file=f"logs/{c['run_id']}.log",
            show_progress=show_progress,
        )
        self.stopped = False
        self.fold = "all"
        self.handlers = {
            s: signal.signal(s, self.interrupt)
            for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }

    def interrupt(self, signum, frame):
        self.stopped = True

    def check_stop(self):
        if self.stopped:
            raise BenchmarkPaused("stop requested; repeat the same command to resume")

    def event(self, name, **fields):
        self.reporter.event(
            name,
            fold=self.fold,
            config=f"frozen_trial_{self.c['recipe']['trial_number']}",
            **fields,
        )

    def bar(self, total, desc, position=1, initial=0):
        return tqdm(
            total=total,
            initial=initial,
            desc=desc,
            position=position,
            dynamic_ncols=True,
            leave=position == 1,
            disable=not self.show_progress,
        )

    def elapsed(self):
        return self.previous + time.perf_counter() - self.started

    def save_timing(self):
        write_json_atomic(self.work / "timing.json", {"active_seconds": self.elapsed()})

    def prepare(self, fold, directory):
        c, info = self.c, self.c["input"][fold["name"]]
        self.event("stage_start", stage="data")
        for path, digest in info["file_sha256"].items():
            self.check_stop()
            if sha256_file(path) != digest:
                raise ValueError(f"immutable input changed: {path}")
            self.event("input_verified", stage="data", path=path)
        cache = directory / "sequences"
        if c["mode"] == "full" and fold.get("sequence_cache"):
            cache = Path(fold["sequence_cache"])
        if not cache.exists():
            partial = directory / "sequences.partial"
            if partial.exists():
                shutil.rmtree(partial)
            prepare_sequence_store(
                Path(fold["artifact"]) / "history_daily.parquet",
                partial,
                cutoff=datetime.fromisoformat(info["split"]["cutoff"]),
                expected_sha256=info["history_sha256"],
                context_user_limit=c["smoke"]["context_users"]
                if c["mode"] == "smoke"
                else None,
                preserve_full_catalog=True,
                seed=c["seed"],
                event=lambda n, **kw: self.event(n, stage="data", **kw),
            )
            publish_directory_atomic(partial, cache)
        store = SequenceStore.load(cache)
        if (
            store.metadata["history_sha256"] != info["history_sha256"]
            or store.metadata["cutoff"] != info["split"]["cutoff"]
        ):
            raise ValueError("sequence cache belongs to another fold")
        if c["mode"] == "full" and (
            len(store.user_ids) != info["history_users"]
            or len(store.item_ids) != info["history_items"]
            or store.metadata["context_user_limit"] is not None
        ):
            raise ValueError("full evaluation requires complete history users/catalog")
        users = pl.read_parquet(Path(fold["artifact"]) / "target_users.parquet").sort(
            "user_id"
        )
        if c["mode"] == "smoke":
            # Only the smoke samples history users; the full run keeps cold users.
            users = id_sample(
                users.join(
                    pl.DataFrame({"user_id": store.user_ids}), on="user_id", how="semi"
                ),
                c["smoke"]["evaluation_users"],
                c["seed"],
            )
        if users.is_empty():
            raise ValueError("empty evaluation universe")
        truth = read_users(
            Path(fold["artifact"]) / "target_ground_truth.parquet", users
        )
        users.write_parquet(directory / "target_users.parquet")
        truth.write_parquet(directory / "target_ground_truth.parquet")
        self.event(
            "stage_finish",
            stage="data",
            training_users=len(store.eligible_users),
            target_users=users.height,
            positive_pairs=truth.height,
        )
        return store, users, truth

    def train(self, fold, directory, store):
        c, training = self.c, self.c["training"]
        trained = directory / "trained"
        configure_torch(c)
        if trained.exists():
            verify_hashes(trained)
            self.event("trained_model_reused", stage="training")
            return SASRecCandidateModel.from_artifact(
                trained / "model", device=c["device"]
            )
        start = time.perf_counter()
        self.event("stage_start", stage="training", epochs=training["epochs"])
        reuse = c["reuse_selection_fold_model"] and fold["name"] == "rolling_1"
        records, epoch = [], 0
        if reuse:
            model = SASRecCandidateModel.from_artifact(
                Path(c["selection_artifact"]) / "model", device=c["device"]
            )
            if (
                model.history_sha256 != store.metadata["history_sha256"]
                or model.get_config()["architecture"] != c["recipe"]["model"]
            ):
                raise ValueError("selected model belongs to another history/recipe")
            epoch = training["epochs"]
        else:
            model = SASRecCandidateModel(
                store.item_ids,
                SASRecConfig(**c["recipe"]["model"]),
                device=c["device"],
                history_sha256=store.metadata["history_sha256"],
            )
            optimizer = torch.optim.AdamW(
                model.encoder.parameters(),
                lr=training["learning_rate"],
                weight_decay=training["weight_decay"],
                fused=c["device"] == "cuda",
                foreach=False,
            )
            loader = SASRecDataLoader(
                store, max_length=model.config.max_length, seed=c["seed"]
            )
            checkpoint = directory / "checkpoint.pt"

            def checkpoint_save():
                save_torch_atomic(
                    checkpoint,
                    {
                        "config_sha256": self.digest,
                        "fold": self.fold,
                        "epoch": epoch,
                        "records": records,
                        "model": model.encoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": capture_rng(use_cuda=c["device"] == "cuda"),
                    },
                )
                write_json_atomic(
                    directory / "checkpoint.json",
                    {"epoch": epoch, "config_sha256": self.digest, "fold": self.fold},
                )
                self.save_timing()

            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if state["config_sha256"] != self.digest or state["fold"] != self.fold:
                    raise ValueError("incompatible epoch checkpoint")
                model.encoder.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                restore_rng(state["rng"])
                epoch, records = state["epoch"], state["records"]
                del state
                self.event("checkpoint_restored", stage="training", epoch=epoch)
            else:
                checkpoint_save()
            with self.bar(
                training["epochs"], f"{self.fold} | epochs", initial=epoch
            ) as epochs:
                while epoch < training["epochs"]:
                    self.check_stop()
                    before = time.perf_counter()
                    number = epoch + 1
                    self.event("epoch_start", stage="training", epoch=number)
                    total = min(
                        len(store.eligible_users),
                        training["max_users"] or len(store.eligible_users),
                    )
                    with self.bar(
                        total, f"epoch {number} | windows + negatives", 2
                    ) as windows:

                        def prepared(done, total):
                            self.check_stop()
                            windows.update(done - windows.n)

                        loader.load_fit_data().prepare_fit_data(
                            epoch=number,
                            negative_count=training["negative_count"],
                            max_users=training["max_users"],
                            memory_budget_bytes=int(
                                c["resources"]["epoch_arrays_gib"] * 2**30
                            ),
                            progress=prepared,
                        )
                    preparation = time.perf_counter() - before
                    batches = math.ceil(
                        loader.epoch_metadata["users"] / training["batch_size"]
                    )
                    with self.bar(batches, f"epoch {number} | train", 2) as batch_bar:
                        last_log = time.monotonic()

                        def batch_done(
                            index,
                            loss,
                            seconds,
                            targets,
                            norm,
                            batches=batches,
                            number=number,
                        ):
                            nonlocal last_log
                            batch_bar.update(1)
                            batch_bar.set_postfix(loss=f"{loss:.4f}", refresh=False)
                            if (
                                index == 1
                                or index == batches
                                or time.monotonic() - last_log >= 30
                            ):
                                self.event(
                                    "batch_progress",
                                    stage="training",
                                    epoch=number,
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
                            precision=c["precision"],
                            gradient_clip_norm=training["gradient_clip_norm"],
                            query_chunk_size=training["query_chunk_size"],
                            callback=batch_done,
                            check_stop=self.check_stop,
                        )
                    records.append(
                        {
                            "epoch": number,
                            **loader.epoch_metadata,
                            **model.last_fit_metrics,
                            "preparation_seconds": preparation,
                        }
                    )
                    loader.fit_arrays = None
                    epoch = number
                    checkpoint_save()
                    epochs.update(1)
                    self.event(
                        "epoch_finish",
                        stage="training",
                        epoch=epoch,
                        duration_seconds=time.perf_counter() - before,
                        current_metric=records[-1]["training_loss"],
                        best_metric=min(r["training_loss"] for r in records),
                    )
                    if self.stop_after_epoch == (self.fold, epoch):
                        raise BenchmarkPaused("intentional smoke epoch pause")
            del optimizer, loader
            gc.collect()
        partial = directory / "trained.partial"
        clean_partial(partial)
        model.save(partial / "model")
        # Portable inference probe is small and independent of validation labels.
        loader = SASRecDataLoader(
            store, max_length=model.config.max_length, seed=c["seed"]
        )
        ids = store.user_ids[store.eligible_users[:8]]
        loader.load_predict_data(user_ids=ids).prepare_predict_data()
        batch = next(loader.iter_predict_batches(batch_size=8))
        inputs = batch.inputs.to(c["device"], dtype=torch.int64)
        indices = torch.ones(len(inputs), dtype=torch.int64, device=c["device"])
        with torch.inference_mode():
            model.encoder.eval()
            scores = (
                model.score_pairs(model.encoder.encode_users(inputs), indices)
                .cpu()
                .numpy()
            )
        np.savez(
            partial / "portable_probe.npz",
            inputs=inputs.cpu().numpy(),
            item_indices=indices.cpu().numpy(),
            scores=scores,
        )
        write_json_atomic(
            partial / "metrics.json",
            {
                "epochs": epoch,
                "records": records,
                "fit_reused": reuse,
                "selection_artifact": c["selection_artifact"] if reuse else None,
                "labels_used_for_fit_or_stopping": False,
                "runtime_seconds_current_session": time.perf_counter() - start,
                "training_seconds_recorded": sum(
                    r["train_seconds"] + r["preparation_seconds"] for r in records
                ),
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                if c["device"] == "cuda"
                else None,
            },
        )
        seal(partial)
        publish_directory_atomic(partial, trained)
        (directory / "checkpoint.pt").unlink(missing_ok=True)
        self.event(
            "stage_finish",
            stage="training",
            epochs=epoch,
            fit_reused=reuse,
            duration_seconds=time.perf_counter() - start,
        )
        return model

    def evaluate(self, fold, directory, store, users, truth, model):
        c, info = self.c, self.c["input"][fold["name"]]
        e = c["evaluation"]
        parts = directory / "evaluation"
        parts.mkdir(exist_ok=True)
        self.event("stage_start", stage="evaluation", users=users.height)
        als = ImplicitALSModel.from_artifact(info["als_model"])
        if not np.array_equal(als.item_mapping["item_id"].to_numpy(), store.item_ids):
            raise ValueError("ALS catalog differs from fold sequence vocabulary")
        loader = SASRecDataLoader(
            store, max_length=model.config.max_length, seed=c["seed"]
        )
        total = math.ceil(users.height / e["users_per_shard"])
        with self.bar(total, f"{self.fold} | evaluation shards") as bar:
            for index in range(total):
                self.check_stop()
                part = parts / f"part-{index:05d}"
                batch_users = users.slice(
                    index * e["users_per_shard"], e["users_per_shard"]
                )
                if part.exists():
                    verify_hashes(part)
                    meta = read_json(part / "metrics.json")
                    if (
                        meta["config_sha256"] != self.digest
                        or meta["fold"] != self.fold
                    ):
                        raise ValueError("evaluation shard provenance differs")
                    if not pl.read_parquet(
                        part / "per_user.parquet", columns=["user_id"]
                    ).equals(batch_users):
                        raise ValueError("evaluation shard user universe differs")
                    bar.update(1)
                    self.event("shard_reused", stage="evaluation", shard=index)
                    continue
                started = time.perf_counter()
                self.event(
                    "shard_start",
                    stage="evaluation",
                    shard=index,
                    users=batch_users.height,
                )
                sources = {
                    s: read_users(
                        Path(info["source_root"])
                        / "sources"
                        / s
                        / "candidates.parquet",
                        batch_users,
                    )
                    for s in SOURCES
                }
                batch_truth = truth.join(batch_users, on="user_id", how="semi")
                partial = parts / f"part-{index:05d}.partial"
                clean_partial(partial)
                loader.load_predict_data(
                    user_ids=batch_users["user_id"].to_numpy()
                ).prepare_predict_data()
                before = time.perf_counter()
                with self.bar(
                    math.ceil(batch_users.height / e["batch_size"]),
                    "SASRec | retrieval",
                    2,
                ) as retrieval:

                    def retrieved(done):
                        self.check_stop()
                        retrieval.update(done - retrieval.n)

                    native = model.predict(
                        loader,
                        k=200,
                        batch_size=e["batch_size"],
                        item_chunk_size=e["item_chunk_size"],
                        callback=retrieved,
                    )
                native.write_parquet(partial / "sasrec.parquet")
                sasrec_seconds = time.perf_counter() - before
                before = time.perf_counter()
                # Public loader with shard history preserves existing ALS seen semantics.
                history = read_users(
                    Path(fold["artifact"]) / "history_daily.parquet", batch_users
                )
                als_loader = ImplicitALSDataLoader(
                    config=als.config,
                    reference_time=datetime.fromisoformat(info["split"]["cutoff"]),
                    user_mapping=als.user_mapping,
                    item_mapping=als.item_mapping,
                    seed=als.config.seed,
                )
                als_loader.load_predict_data(
                    history=history, target_users=batch_users
                ).prepare_predict_data()
                self.check_stop()
                als400 = als.predict(als_loader, k=400, batch_size=e["als_batch_size"])
                als400.write_parquet(partial / "als400.parquet")
                als_seconds = time.perf_counter() - before
                del als_loader, history
                stats, recommendations = evaluate_shard(
                    batch_users, batch_truth, sources, native, als400, store
                )
                stats.write_parquet(partial / "per_user.parquet")
                for name, recs in recommendations.items():
                    recs.write_parquet(partial / f"top20_{name}.parquet")
                metric = summarize(stats, c["recipe"]["source_policy"])
                write_json_atomic(
                    partial / "metrics.json",
                    {
                        "config_sha256": self.digest,
                        "fold": self.fold,
                        "target_users": batch_users.height,
                        "sasrec_retrieval_seconds": sasrec_seconds,
                        "als400_inference_and_loader_seconds": als_seconds,
                        "duration_seconds": time.perf_counter() - started,
                        "candidate_recall": metric["candidate_recall"],
                        "frozen_policy_delta_recall": metric[
                            "frozen_policy_delta_recall"
                        ],
                    },
                )
                seal(partial)
                publish_directory_atomic(partial, part)
                self.save_timing()
                bar.update(1)
                bar.set_postfix(recall=f"{metric['candidate_recall']:.4f}")
                self.event(
                    "shard_finish",
                    stage="evaluation",
                    shard=index,
                    duration_seconds=time.perf_counter() - started,
                    sasrec_seconds=sasrec_seconds,
                    als400_seconds=als_seconds,
                    current_metric=metric["frozen_policy_delta_recall"],
                )
                del sources, native, als400, stats, recommendations
                gc.collect()
                if self.stop_after_shard == (self.fold, index + 1):
                    raise BenchmarkPaused("intentional smoke shard pause")
        del als, loader
        stats = pl.concat(
            [
                pl.read_parquet(p / "per_user.parquet")
                for p in sorted(parts.glob("part-*"))
                if p.is_dir() and not p.name.endswith(".partial")
            ]
        )
        if not stats.select("user_id").equals(users):
            raise ValueError("fold evaluation does not cover the exact target universe")
        stats.write_parquet(directory / "per_user.parquet")
        metric = summarize(stats, c["recipe"]["source_policy"])
        if c["mode"] == "full":
            refs = {"baseline_800": info["reference_baseline_metrics"]}
            actual = {"baseline_800": metric["policies"]["baseline_800"]}
            for s in SOURCES:
                refs[s], actual[s] = (
                    info["reference_source_metrics"][s],
                    metric["sources"][s]["200"],
                )
            for name, reference in refs.items():
                for key in (
                    "candidate_recall",
                    "coverage",
                    "mean_candidate_count",
                    "candidate_oracle_p20_all_targets",
                    "candidate_oracle_p20_labeled_users",
                ):
                    if not math.isclose(
                        actual[name][key], reference[key], rel_tol=1e-12, abs_tol=1e-14
                    ):
                        raise ValueError(
                            f"full-fold baseline reproduction differs: {name}/{key}"
                        )
        timings = [
            read_json(parts / f"part-{i:05d}" / "metrics.json") for i in range(total)
        ]
        metric.update(
            fold=self.fold,
            split=info["split"],
            ground_truth_funnel=info["ground_truth_funnel"],
            mode=c["mode"],
            all_target_users_evaluated=c["mode"] == "full",
            selection_fold_diagnostic=self.fold == "rolling_1",
            canonical_previously_opened=self.fold == "canonical",
            training=read_json(directory / "trained" / "metrics.json"),
            evaluation_seconds=sum(t["duration_seconds"] for t in timings),
            sasrec_retrieval_seconds=sum(
                t["sasrec_retrieval_seconds"] for t in timings
            ),
            als400_inference_and_loader_seconds=sum(
                t["als400_inference_and_loader_seconds"] for t in timings
            ),
        )
        write_json_atomic(directory / "metrics.json", metric)
        write_json_atomic(
            directory / "complete.json",
            {
                "config_sha256": self.digest,
                "fold": self.fold,
                "sha256": {
                    n: sha256_file(directory / n)
                    for n in (
                        "metrics.json",
                        "per_user.parquet",
                        "target_users.parquet",
                        "target_ground_truth.parquet",
                    )
                },
            },
        )
        self.event(
            "stage_finish",
            stage="evaluation",
            current_metric=metric["frozen_policy_delta_recall"],
            native_recall=metric["candidate_recall"],
        )
        return metric

    def run(self):
        try:
            self.event(
                "run_start",
                stage="preflight",
                resume=(self.work / "timing.json").exists(),
            )
            verify_selection(Path(self.c["selection_artifact"]))
            metrics = {}
            for fold in self.c["folds"]:
                self.fold = fold["name"]
                start = self.reporter.phase_start(self.fold, fold=self.fold)
                directory = self.work / "folds" / self.fold
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "complete.json").exists():
                    complete = read_json(directory / "complete.json")
                    if complete["config_sha256"] != self.digest:
                        raise ValueError("completed fold config differs")
                    for name, digest in complete["sha256"].items():
                        if sha256_file(directory / name) != digest:
                            raise ValueError("completed fold checksum differs")
                    metrics[self.fold] = read_json(directory / "metrics.json")
                    self.event("fold_reused", stage="fold")
                else:
                    store, users, truth = self.prepare(fold, directory)
                    self.check_stop()
                    model = self.train(fold, directory, store)
                    self.check_stop()
                    metrics[self.fold] = self.evaluate(
                        fold, directory, store, users, truth, model
                    )
                    del model, store, users, truth
                    gc.collect()
                    if self.c["device"] == "cuda":
                        torch.cuda.empty_cache()
                self.reporter.phase_finish(self.fold, start, fold=self.fold)
                self.save_timing()
            self.fold = "all"
            self.check_stop()
            start = self.reporter.phase_start("publish", fold="all")
            result = self.publish(metrics)
            self.reporter.phase_finish("publish", start, fold="all")
            self.event(
                "run_finish",
                stage="publish",
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
                    "fold": self.fold,
                    "error": repr(error),
                    "resume": "same command; last complete epoch/shard/fold is retained",
                },
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
        # Hard links avoid a second copy while keeping every published artifact portable.
        for fold in self.c["folds"]:
            source = self.work / "folds" / fold["name"]
            destination = partial / "folds" / fold["name"]
            shutil.copytree(
                source,
                destination,
                copy_function=os.link,
                ignore=shutil.ignore_patterns(
                    "sequences", "*.partial", "checkpoint.pt"
                ),
            )
        write_json_atomic(partial / "config.json", self.c)
        write_json_atomic(partial / "frozen_recipe.json", self.c["recipe"])
        heldout = [folds[n] for n in ("rolling_2", "rolling_3") if n in folds]
        summary = {
            "run_id": self.c["run_id"],
            "mode": self.c["mode"],
            "folds": folds,
            "runtime_seconds": self.elapsed(),
            "peak_rss_gib_current_session": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            / 2**20,
            "primary_policy": self.c["recipe"]["source_policy"],
            "canonical_used_for_selection": False,
            "policy_reselected": False,
            "rolling_2_3_mean_frozen_policy_delta_recall": sum(
                m["frozen_policy_delta_recall"] for m in heldout
            )
            / len(heldout)
            if heldout
            else None,
            "rolling_2_3_mean_add_minus_als400_recall": sum(
                m["add_source_minus_als400_recall"] for m in heldout
            )
            / len(heldout)
            if heldout
            else None,
            "automatic_source_promotion": False,
            "limitations": [
                "rolling_1 selected recipe/epoch; its metrics are diagnostic",
                "canonical was previously used in this project",
                "short observed timeline; single seed",
                "candidate recall and standalone P20 do not establish trained ranker improvement",
            ],
        }
        write_json_atomic(partial / "metrics.json", summary)
        rows = []
        for name, m in folds.items():
            for policy, values in m["policies"].items():
                rows.append({"fold": name, "policy": policy, **values})
        pl.DataFrame(rows).write_csv(partial / "comparison.csv")
        (partial / "report.md").write_text(report(summary), encoding="utf-8")
        seal(partial)
        verify_artifact(partial)
        publish_directory_atomic(partial, self.output)
        append_experiments(self.c, summary, self.output)
        return self.output


def report(summary):
    lines = [
        "# Frozen SASRec candidate evaluation",
        "",
        "Recipe/epochs/policy were frozen on rolling_1. No ranker was trained.",
        "",
        "| Fold | SASRec recall@200 | ALS recall@200 | Baseline800 | Frozen blend800 | Add1000 | ALS400 control1000 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in summary["folds"].items():
        p = m["policies"]
        values = [
            m["candidate_recall"],
            m["sources"]["implicit_als"]["200"]["candidate_recall"],
            *[
                p[n]["candidate_recall"]
                for n in (
                    "baseline_800",
                    m["frozen_policy"],
                    "add_source_1000",
                    "als400_control_1000",
                )
            ],
        ]
        lines.append(
            "| " + name + " | " + " | ".join(f"{v:.8f}" for v in values) + " |"
        )
    lines.extend(
        [
            "",
            "See metrics.json for both standalone Precision@20 denominators, oracle metrics, source depth curves and overlap.",
            "See comparison.csv for equal source budgets and actual unique candidate counts.",
            "",
            *summary["limitations"],
            "",
        ]
    )
    return "\n".join(lines)


def metrics_equal(actual, saved):
    """Counts are exact; parallel floating reductions may differ by a few ULPs."""
    if isinstance(actual, dict):
        return isinstance(saved, dict) and all(
            k in saved and metrics_equal(v, saved[k]) for k, v in actual.items()
        )
    if isinstance(actual, float) and isinstance(saved, (int, float)):
        return math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-14)
    return actual == saved


def verify_artifact(directory):
    directory = Path(directory)
    verify_hashes(directory)
    c, metrics = (
        read_json(directory / "config.json"),
        read_json(directory / "metrics.json"),
    )
    torch.set_num_threads(c["cpu_threads"])
    for fold in c["folds"]:
        d = directory / "folds" / fold["name"]
        verify_hashes(d / "trained")
        for part in sorted((d / "evaluation").glob("part-*")):
            verify_hashes(part)
        users = pl.read_parquet(d / "target_users.parquet")
        stats = pl.read_parquet(d / "per_user.parquet")
        if not stats.select("user_id").equals(users):
            raise ValueError("portable per-user universe differs")
        actual = summarize(stats, c["recipe"]["source_policy"])
        saved = metrics["folds"][fold["name"]]
        if not metrics_equal(actual, saved):
            raise ValueError("portable metric aggregation differs")
        model = SASRecCandidateModel.from_artifact(d / "trained" / "model")
        if (
            model.history_sha256 != c["input"][fold["name"]]["history_sha256"]
            or model.get_config()["architecture"] != c["recipe"]["model"]
            or read_json(d / "trained" / "metrics.json")["epochs"]
            != c["training"]["epochs"]
        ):
            raise ValueError("portable model belongs to another fold")
        probe = np.load(d / "trained" / "portable_probe.npz", allow_pickle=False)
        with torch.inference_mode():
            scores = model.score_pairs(
                model.encoder.encode_users(torch.from_numpy(probe["inputs"]).long()),
                torch.from_numpy(probe["item_indices"]).long(),
            ).numpy()
        if not np.allclose(scores, probe["scores"], atol=1e-4, rtol=1e-4):
            raise ValueError("portable CPU scores differ")
        del model
        gc.collect()
    return {
        "verified": True,
        "folds": list(metrics["folds"]),
        "portable_scores_equal": True,
    }


def append_experiments(c, metrics, output):
    if c["mode"] != "full":
        return
    path = Path("experiments/results.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {r["run_id"] for r in reader}
        for fold, m in metrics["folds"].items():
            run_id = c["run_id"] + "_" + fold
            if run_id in existing:
                continue
            row = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "split": fold + "_24h_frozen_sasrec",
                "seed": c["seed"],
                "candidate_config": json.dumps(c["recipe"]),
                "ranker_config": "standalone_sasrec_with_global_fallback",
                "p20_all_targets": m["precision_at_20_all_targets"],
                "p20_labeled_users": m["precision_at_20_labeled_users"],
                "candidate_recall": m["candidate_recall"],
                "coverage": m["coverage"],
                "runtime": m["evaluation_seconds"]
                + m["training"]["training_seconds_recorded"],
                "artifact_path": str(output / "folds" / fold),
                "notes": "frozen recipe; full target universe; no ranker refit; ALS400 inference only; rolling_1 selection diagnostic",
            }
            if not fields:
                fields = list(row)
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
        "--config", type=Path, default=Path("configs/task15_sasrec_folds_v1.json")
    )
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop-after-epoch",
        nargs=2,
        metavar=("FOLD", "EPOCH"),
        help="Smoke-only epoch pause",
    )
    parser.add_argument(
        "--stop-after-shard",
        nargs=2,
        metavar=("FOLD", "COUNT"),
        help="Smoke-only shard pause",
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
    if (args.stop_after_epoch or args.stop_after_shard) and c["mode"] != "smoke":
        raise ValueError("pause testing hooks are restricted to smoke")
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if args.worker:
        try:
            FoldsRunner(
                resolve_config(c),
                show_progress=not args.no_progress,
                stop_after_epoch=(
                    args.stop_after_epoch[0],
                    int(args.stop_after_epoch[1]),
                )
                if args.stop_after_epoch
                else None,
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
        for name in ("stop_after_epoch", "stop_after_shard"):
            if getattr(args, name):
                command.extend(["--" + name.replace("_", "-"), *getattr(args, name)])
        return supervise(command, c)


if __name__ == "__main__":
    raise SystemExit(main())

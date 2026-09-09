#!/usr/bin/env python3
"""One GPU training benchmark; no Optuna search or full-fold quality selection."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import random
import re
import resource
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from sasrec_data import SASRecDataLoader, SequenceStore, prepare_sequence_store
from sasrec_model import SASRecCandidateModel, SASRecConfig, save_torch_atomic
from scripts.task15_resources import supervise
from validation import validate_candidate_output

IMPLEMENTATION = (
    "sasrec_data.py",
    "sasrec_model.py",
    "scripts/run_sasrec_benchmark.py",
    "scripts/task15_resources.py",
    "experiment_utils.py",
    "interfaces.py",
    "data_utils.py",
    "validation.py",
    "scripts/task13_resources.py",
)


class BenchmarkPaused(Exception):
    pass


def load_config(path: Path) -> dict:
    c = read_json(path)
    if c["kind"] != "sasrec_training_benchmark" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid benchmark kind/run ID")
    if c["fold"] != "rolling_1" or c["mode"] not in ("benchmark", "smoke"):
        raise ValueError("benchmark requires rolling_1 and explicit mode")
    if c["device"] not in ("cpu", "cuda") or c["precision"] not in (
        "float32",
        "bfloat16",
    ):
        raise ValueError("invalid device/precision")
    SASRecConfig(**c["model"])
    training = c["training"]
    for name in ("epochs", "batch_size", "negative_count", "query_chunk_size"):
        if type(training[name]) is not int or training[name] <= 0:
            raise ValueError(f"invalid training.{name}")
    if (
        training["learning_rate"] <= 0
        or training["weight_decay"] < 0
        or training["gradient_clip_norm"] <= 0
    ):
        raise ValueError("invalid optimizer settings")
    if c["mode"] == "benchmark":
        if c["device"] != "cuda" or not 1 <= training["epochs"] <= 5:
            raise ValueError("full benchmark requires CUDA and 1..5 epochs")
        if (
            c["data"]["context_user_limit"] is not None
            or training["max_users"] is not None
        ):
            raise ValueError("full benchmark cannot silently sample training users")
        if not c["data"]["preserve_full_catalog"]:
            raise ValueError("full benchmark requires the full history catalog")
    else:
        if not 1 <= training["epochs"] <= 2 or not 1 <= training["max_users"] <= 2048:
            raise ValueError("smoke requires <=2 epochs and <=2048 training users")
        if not 1 <= c["data"]["context_user_limit"] <= 4096:
            raise ValueError("smoke requires <=4096 context users")
    for key in ("users", "batch_size", "item_chunk_size", "candidate_k"):
        if type(c["retrieval_probe"][key]) is not int or c["retrieval_probe"][key] <= 0:
            raise ValueError(f"invalid retrieval_probe.{key}")
    if c["retrieval_probe"]["users"] > 512:
        raise ValueError("benchmark retrieval probe is limited to 512 users")
    limits = c["resources"]
    if not 0 < limits["cuda_memory_fraction"] <= 0.8:
        raise ValueError("reserve at least 20% of device VRAM")
    if (
        not 0 < limits["maximum_rss_gib"] <= 40
        or not 0 < limits["maximum_run_disk_gib"] <= 20
    ):
        raise ValueError("benchmark is limited to 40GiB RSS and 20GiB run files")
    if not 0 < limits["poll_seconds"] <= 5 or not 1 <= c["cpu_threads"] <= 8:
        raise ValueError("invalid polling/thread limits")
    return c


def resolve_config(c: dict) -> dict:
    c = json.loads(json.dumps(c))
    fold = Path(c["data"]["fold_artifact"])
    split = read_json(fold / "config.json")["split"]
    diagnostics = read_json(fold / "metrics.json")["deterministic_diagnostics"]
    c["input"] = {
        "cutoff": split["cutoff"],
        "validation_end_exclusive": split["validation_end_exclusive"],
        "history_path": str(fold / "history_daily.parquet"),
        "history_sha256": diagnostics["output_sha256"]["history_daily.parquet"],
        "target_users_path": str(fold / "target_users.parquet"),
        "target_users_sha256": diagnostics["output_sha256"]["target_users.parquet"],
        "full_history_users": diagnostics["daily"]["history"]["users"],
        "full_history_items": diagnostics["daily"]["history"]["items"],
    }
    if split["validation_end_exclusive"] is None:
        raise ValueError("rolling benchmark must specify the next-day validation end")
    if (
        datetime.fromisoformat(split["validation_end_exclusive"])
        - datetime.fromisoformat(split["cutoff"])
    ).total_seconds() != 86400:
        raise ValueError("rolling target period must be 24 hours")
    c["implementation_sha256"] = {p: sha256_file(ROOT / p) for p in IMPLEMENTATION}
    c["library_versions"] = {
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "numpy": np.__version__,
        "polars": pl.__version__,
    }
    c["validation_labels_read"] = False
    return c


def configure_torch(c):
    torch.set_num_threads(c["cpu_threads"])
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.use_deterministic_algorithms(c["device"] == "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if c["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required; CPU fallback is disabled for the GPU benchmark"
            )
        torch.cuda.set_device(c["gpu_id"])
        torch.cuda.manual_seed_all(c["seed"])
        torch.cuda.set_per_process_memory_fraction(
            c["resources"]["cuda_memory_fraction"], c["gpu_id"]
        )
        if c["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested BF16 is unavailable")
        torch.cuda.reset_peak_memory_stats()


def capture_rng(*, use_cuda=False):
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if use_cuda else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def verify_artifact(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    for name, digest in manifest["sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"benchmark artifact checksum mismatch: {name}")
    metrics = read_json(directory / "metrics.json")
    c = read_json(directory / "config.json")
    torch.set_num_threads(min(8, c["cpu_threads"]))
    if len(metrics["epochs"]) != c["training"]["epochs"]:
        raise ValueError("incomplete benchmark epoch count")
    if (
        metrics["precision_at_20_all_targets"] is not None
        or metrics["precision_at_20_labeled_users"] is not None
        or metrics["candidate_recall"] is not None
        or metrics["validation_labels_read"]
        or c["validation_labels_read"]
    ):
        raise ValueError("benchmark cannot claim unmeasured quality metrics")
    model = SASRecCandidateModel.from_artifact(directory / "model")
    history = c["input"]["history_sha256"]
    if metrics["sequence_metadata"]["history_sha256"] != history or (
        model.history_sha256 is not None and model.history_sha256 != history
    ):
        raise ValueError("benchmark/model history provenance differs")
    probe = np.load(directory / "portable_probe.npz", allow_pickle=False)
    with torch.inference_mode():
        queries = model.encoder.encode_users(torch.from_numpy(probe["inputs"]).long())
        scores = model.score_pairs(
            queries, torch.from_numpy(probe["item_indices"]).long()
        ).numpy()
    if not np.allclose(scores, probe["scores"], atol=1e-4, rtol=1e-4):
        raise ValueError("portable CPU scores differ from saved inference probe")
    if bool((model.encoder.item_embedding.weight[0] != 0).any()):
        raise ValueError("nonzero padding embedding")
    candidates = pl.read_parquet(directory / "probe_candidates.parquet")
    validate_candidate_output(
        candidates, k=c["retrieval_probe"]["candidate_k"], source_name="sasrec"
    )
    if not set(candidates["item_id"].to_list()).issubset(set(model.item_ids.tolist())):
        raise ValueError("probe contains unknown items")
    return {
        "verified": True,
        "epochs": len(metrics["epochs"]),
        "portable_scores_equal": True,
    }


def run_worker(c: dict, *, show_progress=True, stop_after_epoch=None) -> Path:
    session_started = time.perf_counter()
    run_id = c["run_id"]
    output = Path("artifacts") / run_id
    work = Path("artifacts") / f".{run_id}.work"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    digest = config_sha256(c)
    work.mkdir(parents=True, exist_ok=True)
    if (work / "config.json").exists():
        if config_sha256(read_json(work / "config.json")) != digest:
            raise ValueError(
                "config, implementation or input changed; use a new run ID"
            )
    else:
        write_json_atomic(work / "config.json", c)
    reporter = EventProgressReporter(
        task_name=run_id,
        total_phases=4,
        log_file=f"logs/{run_id}.log",
        show_progress=show_progress,
    )
    stopped = False

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True

    def check_stop():
        if stopped:
            raise BenchmarkPaused("stop requested; latest completed epoch is safe")

    handlers = {
        sig: signal.signal(sig, interrupt)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    context = {"config": run_id, "fold": "rolling_1"}

    def event(name, **fields):
        reporter.event(name, **context, **fields)

    try:
        configure_torch(c)
        started = reporter.phase_start("data", **context)
        cache = work / "sequences"
        if cache.exists():
            store = SequenceStore.load(cache)
            event("sequence_cache_restored", stage="data")
        else:
            staging = work / "sequences.partial"
            if staging.exists():
                shutil.rmtree(staging)
            store = prepare_sequence_store(
                Path(c["input"]["history_path"]),
                staging,
                cutoff=datetime.fromisoformat(c["input"]["cutoff"]),
                expected_sha256=c["input"]["history_sha256"],
                seed=c["seed"],
                context_user_limit=c["data"]["context_user_limit"],
                preserve_full_catalog=c["data"]["preserve_full_catalog"],
                event=lambda name, **fields: event(name, stage="data", **fields),
            )
            write_json_atomic(
                work / "preparation.json", {"seconds": time.perf_counter() - started}
            )
            publish_directory_atomic(staging, cache)
            store = SequenceStore.load(cache, verify=False)
        if not len(store.eligible_users):
            raise ValueError("no eligible training users")
        event("data_ready", stage="data", **store.metadata)
        reporter.phase_finish("data", started, **context)
        check_stop()

        model = SASRecCandidateModel(
            store.item_ids, SASRecConfig(**c["model"]), device=c["device"]
        )
        optimizer = torch.optim.AdamW(
            model.encoder.parameters(),
            lr=c["training"]["learning_rate"],
            weight_decay=c["training"]["weight_decay"],
            fused=c["device"] == "cuda",
            foreach=False,
        )
        loader = SASRecDataLoader(
            store, max_length=c["model"]["max_length"], seed=c["seed"]
        )
        checkpoint = work / "checkpoint.pt"
        records = []
        last_epoch = 0
        checkpoint_seconds = (
            read_json(work / "checkpoint_timing.json")["seconds"]
            if (work / "checkpoint_timing.json").exists()
            else 0.0
        )

        def checkpoint_save(epoch):
            nonlocal checkpoint_seconds
            before = time.perf_counter()
            save_torch_atomic(
                checkpoint,
                {
                    "config_sha256": digest,
                    "epoch": epoch,
                    "records": records,
                    "model": model.encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng(use_cuda=c["device"] == "cuda"),
                },
            )
            checkpoint_seconds += time.perf_counter() - before
            write_json_atomic(
                work / "checkpoint_timing.json", {"seconds": checkpoint_seconds}
            )
            write_json_atomic(
                work / "checkpoint.json",
                {
                    "epoch": epoch,
                    "config_sha256": digest,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                },
            )

        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=c["device"], weights_only=True)
            if state["config_sha256"] != digest:
                raise ValueError("incompatible training checkpoint")
            model.encoder.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
            last_epoch, records = state["epoch"], state["records"]
            del state
            event("checkpoint_restored", stage="training", completed_epochs=last_epoch)
        else:
            checkpoint_save(0)

        training_start = reporter.phase_start("training", **context)
        epoch_bar = tqdm(
            total=c["training"]["epochs"],
            initial=last_epoch,
            desc="rolling_1 | epochs",
            unit="epoch",
            position=1,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        try:
            for epoch in range(last_epoch + 1, c["training"]["epochs"] + 1):
                check_stop()
                before = time.perf_counter()
                event(
                    "epoch_start",
                    stage="training",
                    epoch=epoch,
                    operation="prepare_windows",
                )
                prepare_bar = tqdm(
                    total=min(
                        len(store.eligible_users),
                        c["training"]["max_users"] or len(store.eligible_users),
                    ),
                    desc=f"epoch {epoch} | windows + negatives",
                    unit="user",
                    position=2,
                    leave=False,
                    disable=not show_progress,
                    dynamic_ncols=True,
                )

                def prepared(done, total, bar=prepare_bar):
                    check_stop()
                    bar.update(done - bar.n)

                try:
                    loader.load_fit_data().prepare_fit_data(
                        epoch=epoch,
                        negative_count=c["training"]["negative_count"],
                        max_users=c["training"]["max_users"],
                        memory_budget_bytes=int(
                            c["resources"]["epoch_arrays_gib"] * 2**30
                        ),
                        progress=prepared,
                    )
                finally:
                    prepare_bar.close()
                preparation_seconds = time.perf_counter() - before
                total_batches = math.ceil(
                    loader.epoch_metadata["users"] / c["training"]["batch_size"]
                )
                event(
                    "epoch_prepared",
                    stage="training",
                    **loader.epoch_metadata,
                    seconds=preparation_seconds,
                    total_batches=total_batches,
                )
                batch_bar = tqdm(
                    total=total_batches,
                    desc=f"epoch {epoch} | train",
                    unit="batch",
                    position=2,
                    leave=False,
                    disable=not show_progress,
                    dynamic_ncols=True,
                )
                last_log = time.monotonic()

                def batch_done(
                    index,
                    loss,
                    seconds,
                    targets,
                    norm,
                    bar=batch_bar,
                    batches=total_batches,
                    epoch_number=epoch,
                ):
                    nonlocal last_log
                    bar.update(1)
                    bar.set_postfix(
                        loss=f"{loss:.4f}",
                        step_ms=f"{seconds * 1000:.0f}",
                        refresh=False,
                    )
                    if (
                        index == 1
                        or index == batches
                        or time.monotonic() - last_log >= 30
                    ):
                        event(
                            "batch_progress",
                            stage="training",
                            epoch=epoch_number,
                            batch=index,
                            batches=batches,
                            training_loss=loss,
                            step_seconds=seconds,
                            targets=targets,
                            gradient_norm=norm,
                        )
                        last_log = time.monotonic()

                try:
                    model.fit(
                        loader,
                        optimizer=optimizer,
                        batch_size=c["training"]["batch_size"],
                        precision=c["precision"],
                        gradient_clip_norm=c["training"]["gradient_clip_norm"],
                        query_chunk_size=c["training"]["query_chunk_size"],
                        callback=batch_done,
                        check_stop=check_stop,
                    )
                finally:
                    batch_bar.close()
                record = {
                    **loader.epoch_metadata,
                    **model.last_fit_metrics,
                    "preparation_seconds": preparation_seconds,
                }
                records.append(record)
                loader.fit_arrays = None
                gc.collect()
                checkpoint_save(epoch)
                write_json_atomic(work / "epoch_metrics.json", {"epochs": records})
                best_loss = min(row["training_loss"] for row in records)
                event(
                    "epoch_finish",
                    stage="training",
                    epoch=epoch,
                    operation="checkpoint_complete",
                    duration_seconds=time.perf_counter() - before,
                    current_metric=record["training_loss"],
                    best_metric=best_loss,
                    best_config=run_id,
                )
                epoch_bar.update(1)
                epoch_bar.set_postfix(
                    loss=f"{record['training_loss']:.4f}",
                    epoch_s=f"{record['train_seconds'] + preparation_seconds:.1f}",
                )
                tqdm.write(
                    f"Epoch {epoch}/{c['training']['epochs']}: train={record['train_seconds']:.1f}s; data={preparation_seconds:.1f}s; steady step={record['steady_step_seconds_mean'] * 1000:.1f}ms; loss={record['training_loss']:.5f}"
                )
                if stop_after_epoch == epoch:
                    raise BenchmarkPaused(
                        f"intentional pause after checkpoint epoch {epoch}"
                    )
        finally:
            epoch_bar.close()
        reporter.phase_finish("training", training_start, **context)
        check_stop()

        probe_start = reporter.phase_start("retrieval_probe", **context)
        if (
            sha256_file(c["input"]["target_users_path"])
            != c["input"]["target_users_sha256"]
        ):
            raise ValueError("target user checksum mismatch")
        targets = pl.read_parquet(c["input"]["target_users_path"])
        positive_users = pl.DataFrame(
            {"user_id": store.user_ids[np.diff(store.positive_offsets) > 0]}
        )
        targets = (
            targets.join(positive_users, on="user_id", how="semi")
            .with_columns(pl.col("user_id").hash(seed=c["seed"]).alias("sample"))
            .sort("sample", "user_id")
            .head(c["retrieval_probe"]["users"])
            .drop("sample")
        )
        if targets.is_empty():
            raise ValueError(
                "no target users with positive history for retrieval timing"
            )
        loader.load_predict_data(
            user_ids=targets["user_id"].to_numpy()
        ).prepare_predict_data()
        retrieval_bar = tqdm(
            total=math.ceil(targets.height / c["retrieval_probe"]["batch_size"]),
            desc="full-catalog retrieval probe",
            unit="batch",
            position=1,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        retrieval_start = time.perf_counter()
        try:
            candidates = model.predict(
                loader,
                k=c["retrieval_probe"]["candidate_k"],
                batch_size=c["retrieval_probe"]["batch_size"],
                item_chunk_size=c["retrieval_probe"]["item_chunk_size"],
                callback=lambda n: (check_stop(), retrieval_bar.update(1)),
            )
        finally:
            retrieval_bar.close()
        if c["device"] == "cuda":
            torch.cuda.synchronize()
        retrieval_seconds = time.perf_counter() - retrieval_start
        validate_candidate_output(
            candidates, k=c["retrieval_probe"]["candidate_k"], source_name="sasrec"
        )
        for user_id, frame in candidates.partition_by("user_id", as_dict=True).items():
            index = int(np.searchsorted(store.user_ids, np.uint64(user_id[0])))
            indices = np.searchsorted(store.item_ids, frame["item_id"].to_numpy()) + 1
            if np.intersect1d(indices, store.seen_for(index)).size:
                raise ValueError("retrieval returned a history-seen pair")
        event(
            "retrieval_finish",
            stage="retrieval_probe",
            seconds=retrieval_seconds,
            users=targets.height,
            candidate_rows=candidates.height,
        )
        reporter.phase_finish("retrieval_probe", probe_start, **context)

        publish_start = reporter.phase_start("publish", **context)
        staging = work / "publish"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        model.save(staging / "model")
        candidates.write_parquet(staging / "probe_candidates.parquet")
        # Portable verification scores for eight complete history sequences.
        probe_inputs = loader.predict_arrays[0][:8]
        model.encoder.eval()
        with torch.inference_mode():
            encoded = model.encoder.encode_users(
                probe_inputs.to(c["device"], dtype=torch.int64)
            )
            item_indices = (
                torch.arange(1, len(probe_inputs) + 1, device=c["device"])
                % len(store.item_ids)
                + 1
            )
            probe_scores = model.score_pairs(encoded, item_indices).cpu().numpy()
        np.savez(
            staging / "portable_probe.npz",
            inputs=probe_inputs.numpy(),
            item_indices=item_indices.cpu().numpy(),
            scores=probe_scores,
        )
        write_json_atomic(staging / "config.json", c)
        steady_epochs = records[1:] or records
        median_epoch = float(
            np.median(
                [r["train_seconds"] + r["preparation_seconds"] for r in steady_epochs]
            )
        )
        data_seconds = read_json(work / "preparation.json")["seconds"]
        metrics = {
            "kind": "sasrec_training_benchmark",
            "run_id": run_id,
            "mode": c["mode"],
            "fold": "rolling_1",
            "validation_labels_read": False,
            "precision_at_20_all_targets": None,
            "precision_at_20_labeled_users": None,
            "candidate_recall": None,
            "epochs": records,
            "sequence_metadata": store.metadata,
            "initial_preparation_seconds": data_seconds,
            "checkpoint_io_seconds": checkpoint_seconds,
            "measured_completed_training_seconds": sum(
                r["train_seconds"] + r["preparation_seconds"] for r in records
            ),
            "steady_epoch_seconds_median": median_epoch,
            "estimated_20_epoch_training_seconds_same_shape": median_epoch * 20
            if c["mode"] == "benchmark"
            else None,
            "estimate_note": "Training+epoch preparation only; excludes retrieval/checkpoint I/O and changes in Optuna shapes. Smoke does not measure a full epoch.",
            "retrieval_probe": {
                "users": targets.height,
                "seconds": retrieval_seconds,
                "users_per_second": targets.height / retrieval_seconds,
                "candidate_rows": candidates.height,
                "catalog_items": len(store.item_ids),
                "selection": "ID_hash_sample_among_targets_with_positive_history",
            },
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if c["device"] == "cuda"
            else None,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
            if c["device"] == "cuda"
            else None,
            "gpu_name": torch.cuda.get_device_name() if c["device"] == "cuda" else None,
            "model_parameters": sum(p.numel() for p in model.encoder.parameters()),
            "seen_pairs_in_probe": 0,
        }
        write_json_atomic(staging / "metrics.json", metrics)
        write_json_atomic(
            staging / "manifest.json",
            {
                "sha256": {
                    str(p.relative_to(staging)): sha256_file(p)
                    for p in sorted(staging.rglob("*"))
                    if p.is_file()
                }
            },
        )
        # Free GPU optimizer state before independent CPU restore verification.
        optimizer = None
        verification = verify_artifact(staging)
        metrics["portable_verification"] = verification
        metrics["runtime_seconds_current_session"] = (
            time.perf_counter() - session_started
        )
        metrics["peak_rss_gib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        )
        write_json_atomic(staging / "metrics.json", metrics)
        manifest = read_json(staging / "manifest.json")
        manifest["sha256"]["metrics.json"] = sha256_file(staging / "metrics.json")
        write_json_atomic(staging / "manifest.json", manifest)
        check_stop()
        publish_directory_atomic(staging, output)
        reporter.phase_finish("publish", publish_start, **context)
        event("run_finish", status="completed", output=str(output))
        tqdm.write(
            f"Saved {output}/metrics.json | steady epoch including data: {median_epoch:.1f}s | GPU peak allocated: {metrics['peak_cuda_allocated_gib']} GiB"
        )
        return output
    except BaseException as error:
        event(
            "run_finish",
            status="paused" if isinstance(error, BenchmarkPaused) else "failed",
            error=repr(error),
        )
        write_json_atomic(
            work / "failure.json",
            {
                "error": repr(error),
                "resume": "repeat the same launcher/config; unfinished epoch restarts",
            },
        )
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        reporter.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task15_sasrec_benchmark_v1.json")
    )
    parser.add_argument(
        "--verify-only", type=Path, help="Verify a published benchmark without training"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Checkpoint/pause hook for limited smoke validation",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify_artifact(args.verify_only), indent=2))
        return 0
    c = load_config(args.config)
    if args.stop_after_epoch is not None and c["mode"] != "smoke":
        raise ValueError("intentional pause is restricted to smoke mode")
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if args.worker:
        try:
            run_worker(
                resolve_config(c),
                show_progress=not args.no_progress,
                stop_after_epoch=args.stop_after_epoch,
            )
        except BenchmarkPaused as error:
            print(str(error), file=sys.stderr)
            return 75
        return 0
    # Direct CLI invocation and different benchmark IDs also share the GPU lock.
    with Path("logs/task15_sasrec_benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Task15 benchmark is already running") from error
        if (Path("artifacts") / c["run_id"]).exists():
            raise FileExistsError(
                "completed benchmark exists; use --verify-only or a new run ID"
            )
        command = [
            sys.executable,
            str(Path(__file__)),
            "--config",
            str(args.config),
            "--worker",
        ]
        if args.no_progress:
            command.append("--no-progress")
        if args.stop_after_epoch is not None:
            command.extend(("--stop-after-epoch", str(args.stop_after_epoch)))
        return supervise(command, c)


if __name__ == "__main__":
    raise SystemExit(main())

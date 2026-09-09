#!/usr/bin/env python3
"""Task13: fixed-tree ALS history-profile ablations with supervised resource limits."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
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

import numpy as np
import polars as pl

from experiment_utils import (
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from history_profiles import (
    PROFILE_FEATURES,
    VARIANTS,
    HistoryProfiles,
    history_pairs,
    ignored_features,
    normalize_factors,
)
from metrics import evaluate_precision_at_20
from ranker_backtest import validate_temporal_training_scope
from rankers import (
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
)
from scripts.run_catboost_ranker import _CatBoostLogBridge
from scripts.run_ranker_backtest import (
    BacktestRun,
    _file_manifest,
    _verify_files,
    validate_output_semantics,
)
from scripts.task13_resources import supervise


def load_config(path: Path) -> dict:
    c = read_json(path)
    if c["kind"] != "task13_history_profiles" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid Task13 config/run ID")
    if c["mode"] not in ("full", "smoke") or c["seed"] != c["catboost"]["random_seed"]:
        raise ValueError("invalid mode or inconsistent seeds")
    if c["training"]["target_rows"] <= 0 or c["selection"]["fixed_tree_count"] <= 0:
        raise ValueError("positive row/tree budgets required")
    if c["mode"] == "smoke" and (
        not 1 <= c["smoke"]["user_count"] <= 256
        or c["selection"]["fixed_tree_count"] > 30
    ):
        raise ValueError("smoke must be explicitly bounded")
    if c["catboost"]["ignored_features"]:
        raise ValueError("feature ablations are controlled by VARIANTS")
    if c["protocol"] != {
        "selection_train_folds": ["rolling_1", "rolling_2"],
        "selection_eval_fold": "rolling_3",
        "canonical_train_folds": ["rolling_1", "rolling_2", "rolling_3"],
        "canonical_eval_fold": "canonical",
    }:
        raise ValueError("Task13 requires the declared walk-forward protocol")
    if (
        c["quantization"]["policies"] != ["fit_training"]
        or c["inference"]["final_k"] != 20
    ):
        raise ValueError("Task13 uses train-only borders and Precision@20")
    CatBoostPointwiseConfig.from_mapping(c["catboost"])
    if c["quantization"]["border_count"] != c["catboost"]["border_count"]:
        raise ValueError("pool and model border_count differ")
    limits = c["resources"]
    for key, value in limits.items():
        if key != "windows_drive" and (
            not isinstance(value, (int, float)) or value <= 0
        ):
            raise ValueError(f"invalid resource limit: {key}")
    if limits["poll_seconds"] > 5 or c["catboost"]["task_type"] not in ("CPU", "GPU"):
        raise ValueError("invalid resource polling interval or backend")
    return c


def resolved_config(source: dict) -> dict:
    c = json.loads(json.dumps(source))
    candidate_root = Path(c["paths"]["task06_dataset"])
    original = read_json(candidate_root / "config.json")
    c["profile_inputs"] = {}
    for fold in ("rolling_1", "rolling_2", "rolling_3", "canonical"):
        source_root = candidate_root / "folds" / fold / "sources/implicit_als"
        meta = read_json(source_root / "metadata.json")
        model = Path(meta["model_path"])
        if meta["model_storage"] == "embedded":
            model = source_root / model
        spec = original["folds"][fold]
        if meta["cutoff"] != spec["cutoff"]:
            raise ValueError(f"ALS fold cutoff differs: {fold}")
        model_meta = read_json(model / "model_config.json").get("metadata", {})
        expected = spec["output_sha256"]["history_daily.parquet"]
        if model_meta.get("fit_history_sha256", expected) != expected:
            raise ValueError(f"ALS fit history differs: {fold}")
        c["profile_inputs"][fold] = {
            "model": str(model),
            "history": str(Path(spec["path"]) / "history_daily.parquet"),
            "history_sha256": expected,
            "cutoff": spec["cutoff"],
            "files": {
                name: sha256_file(model / name)
                for name in (
                    "als_model.npz",
                    "item_mapping.parquet",
                    "model_config.json",
                )
            },
            "source_metadata_sha256": sha256_file(source_root / "metadata.json"),
        }
    c["task07_manifest_sha256"] = sha256_file(
        Path(c["paths"]["task07_dataset"]) / "dataset_manifest.json"
    )
    c["reference_metrics_sha256"] = sha256_file(
        Path(c["paths"]["task12_reference"]) / "metrics.json"
    )
    c["implementation_sha256"] = {
        str(p): sha256_file(p)
        for p in map(
            Path,
            (
                "history_profiles.py",
                "scripts/run_history_profiles.py",
                "scripts/task13_resources.py",
                "scripts/run_ranker_backtest.py",
                "ranker_backtest.py",
                "rankers.py",
                "ranker_data.py",
                "metrics.py",
                "experiment_utils.py",
                "scripts/run_catboost_ranker.py",
            ),
        )
    }
    return c


class ProfileRun(BacktestRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_features = ()
        self.loaded_profile = None
        self.loaded_fold = None

    def load_folds(self, names):
        self.features = self.base_features
        super().load_folds(names)
        self.base_features = self.features
        self.features = self.base_features + PROFILE_FEATURES

    def release_profiles(self):
        self.loaded_profile = self.loaded_fold = None
        gc.collect()

    def check_resources(self, stage):
        super().check_resources(stage)
        if self.peak_memory_mb / 1024 > self.config["resources"]["maximum_rss_gib"]:
            raise RuntimeError(
                "measured process peak RSS exceeded budget; resume requires a new resource/row-budget config"
            )

    def prepare_profiles(self, fold: str):
        context = self.contexts[fold]
        spec = self.config["profile_inputs"][fold]
        output = self.work / "profiles" / fold

        def build(destination):
            if (
                context.history_path.resolve() != Path(spec["history"]).resolve()
                or sha256_file(context.history_path) != spec["history_sha256"]
            ):
                raise ValueError(f"immutable profile history differs: {fold}")
            pairs = history_pairs(
                pl.scan_parquet(context.history_path), context.target_users
            )
            model = Path(spec["model"])
            with np.load(model / "als_model.npz", allow_pickle=False) as data:
                factors = normalize_factors(data["item_factors"])
            mapping = pl.read_parquet(model / "item_mapping.parquet")
            batches = max(1, (pairs.height + 16383) // 16384)
            with self.units(
                "history_profiles", "shared", fold, batches, "batch"
            ) as advance:
                profiles = HistoryProfiles.build(
                    pairs, context.target_users, mapping, factors, progress=advance
                )
            profiles.save(destination)
            result = {
                "fold": fold,
                "target_users": context.target_users.height,
                "history_pairs": pairs.height,
                "diagnostics": profiles.diagnostics(),
                "input": spec,
                "centroid": "mean of unit ALS item vectors, unique items per event profile",
            }
            write_json_atomic(destination / "diagnostics.json", result)
            return result

        self.operation("profiles", "shared", fold, output, build)

    def profile(self, fold):
        if fold != self.loaded_fold:
            self.release_profiles()
            self.loaded_profile = HistoryProfiles.load(self.work / "profiles" / fold)
            self.loaded_fold = fold
        return self.loaded_profile

    def read_part(self, part, *, columns=None, users=None):
        wanted = (
            list(PROFILE_FEATURES)
            if columns is None
            else [c for c in columns if c in PROFILE_FEATURES]
        )
        base = (
            None
            if columns is None
            else [c for c in columns if c not in PROFILE_FEATURES]
        )
        frame = super().read_part(part, columns=base, users=users)
        if not wanted:
            return frame
        fold = next(
            name for name, context in self.contexts.items() if part in context.parts
        )
        destination = self.work / "profile_features" / fold / part.stem

        def build(path):
            pairs = super(ProfileRun, self).read_part(
                part, columns=["user_id", "item_id"]
            )
            extra = self.profile(fold).features(pairs)
            extra.write_parquet(
                path / "features.parquet", compression="zstd", compression_level=3
            )
            return {"rows": pairs.height, "users": pairs["user_id"].n_unique()}

        self.operation(
            "profile_features", "shared", f"{fold}:{part.stem}", destination, build
        )
        extra = pl.read_parquet(
            destination / "features.parquet", columns=["user_id", "item_id", *wanted]
        )
        result = frame.join(
            extra,
            on=["user_id", "item_id"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        if any(result[c].null_count() for c in wanted):
            raise ValueError("profile feature join lost candidate rows")
        return result

    def training_pool(self, phase, policy, parts):
        self.release_profiles()
        return super().training_pool(phase, policy, parts)

    def fit(self, phase, variant, pool_path, *, tree_count):
        self.release_profiles()
        output = self.work / "models" / phase / variant

        def build(destination):
            params = dict(
                self.config["catboost"],
                iterations=tree_count,
                ignored_features=ignored_features(variant),
                config_id=f"task13_{phase}_{variant}",
            )
            model = CatBoostPointwiseModel(
                CatBoostPointwiseConfig.from_mapping(params),
                feature_columns=self.features,
            )
            loader = (
                CatBoostRankerDataLoader(feature_columns=self.features)
                .load_fit_data(train_pool=pool_path)
                .prepare_fit_data()
            )
            train_dir = self.work / "snapshots" / phase / variant
            train_dir.mkdir(parents=True, exist_ok=True)
            with self.units("fit", variant, phase, tree_count, "tree") as advance:
                bridge = _CatBoostLogBridge(advance)
                model.fit(
                    loader,
                    fixed_tree_budget=True,
                    train_dir=train_dir,
                    snapshot_file=(train_dir / "snapshot.cbsnapshot").resolve(),
                    log_cout=bridge,
                    log_cerr=bridge,
                )
            model.save(destination / "model")
            del loader, model
            gc.collect()
            return {
                "tree_count": tree_count,
                "ignored_features": ignored_features(variant),
                "training_pool_sha256": sha256_file(pool_path),
                "eval_pool": None,
            }

        self.operation("fit", variant, phase, output, build)
        return CatBoostPointwiseModel.from_artifact(output / "model"), output / "model"


def verify_precision_summary(saved, recs, truth, users) -> dict:
    """Check exact hit counts/denominators and tolerate only Float64 rounding."""
    metrics = evaluate_precision_at_20(recs, truth, users)
    hits = (
        recs.explode("item_ids", empty_as_null=True)
        .rename({"item_ids": "item_id"})
        .join(truth, on=["user_id", "item_id"], how="semi")
        .height
    )
    counts = {
        "final_hits": hits,
        "target_users": users.height,
        "labeled_users": truth["user_id"].n_unique(),
    }
    for key, value in counts.items():
        if saved[key] != value:
            raise ValueError(f"stored count differs: {key}: {saved[key]} != {value}")
    for key, denominator in (
        ("precision_at_20_all_targets", counts["target_users"]),
        ("precision_at_20_labeled_users", counts["labeled_users"]),
    ):
        exact_ratio = hits / (20 * denominator) if denominator else 0.0
        # The macro mean reduces per-user Float64 scores in parallel. Its final
        # bits can depend on group ordering. Integer counts above remain exact.
        tolerance = 32 * math.ulp(exact_ratio)
        for origin, value in (("stored", saved[key]), ("recomputed", metrics[key])):
            if not math.isfinite(value) or not math.isclose(
                value, exact_ratio, rel_tol=0.0, abs_tol=tolerance
            ):
                raise ValueError(
                    f"{origin} metric differs: {key}: {value!r} != "
                    f"{exact_ratio!r} (tolerance={tolerance!r})"
                )
    return {**metrics, **counts}


def verify_artifact(root: Path) -> dict:
    manifest = read_json(root / "artifact_manifest.json")
    if manifest["kind"] != "task13_history_profiles":
        raise ValueError("not a Task13 artifact")
    _verify_files(root, manifest["files"])
    model = CatBoostPointwiseModel.from_artifact(root / "model")
    frame = pl.read_parquet(root / "verification_features.parquet")
    expected = pl.read_parquet(root / "verification_scores.parquet")
    loader = (
        CatBoostRankerDataLoader(feature_columns=model.feature_columns)
        .load_predict_data(frame=frame)
        .prepare_predict_data()
    )
    first = model.predict(loader)
    second = model.predict(loader)
    if not first.equals(second) or not first.equals(expected):
        raise ValueError("portable repeated inference differs")
    recs = pl.read_parquet(root / "recommendations.parquet")
    truth = pl.read_parquet(root / "target_ground_truth.parquet")
    users = pl.read_parquet(root / "target_users.parquet")
    saved = read_json(root / "metrics.json")
    metrics = verify_precision_summary(saved, recs, truth, users)
    control_recs = pl.read_parquet(root / "control_recommendations.parquet")
    verify_precision_summary(
        saved["canonical_matched_control"], control_recs, truth, users
    )
    # Full history anti-join is required here too, not just in the original run.
    from types import SimpleNamespace

    context = SimpleNamespace(
        target_users=users,
        history_path=Path(
            read_json(root / "config.json")["profile_inputs"]["canonical"]["history"]
        ),
    )
    semantic = validate_output_semantics(recs, context=context)
    validate_output_semantics(control_recs, context=context)
    return {
        "checksums_valid": True,
        "portable_repeat_equal": True,
        "matched_control_verified": True,
        **semantic,
        **metrics,
    }


def append_experiment(config, metrics, output):
    if config["mode"] != "full":
        return
    path = Path("experiments/results.csv")
    row = {
        "run_id": config["run_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": "canonical_24h_known_diagnostic",
        "seed": config["seed"],
        "candidate_config": config["paths"]["task07_dataset"],
        "ranker_config": json.dumps(
            {
                "variant": metrics["winner"],
                "training": config["training"],
                "trees": config["selection"]["fixed_tree_count"],
            }
        ),
        "p20_all_targets": metrics["precision_at_20_all_targets"],
        "p20_labeled_users": metrics["precision_at_20_labeled_users"],
        "candidate_recall": metrics["candidate_recall"],
        "coverage": metrics["coverage"],
        "runtime": metrics["runtime_seconds"],
        "peak_memory": metrics["peak_memory_mb"],
        "final_hits": metrics["final_hits"],
        "artifact_path": str(output),
        "notes": "Task13 profile ablations; shared pool; r3 winner frozen before canonical; see metrics.json for matched A control",
    }
    path.parent.mkdir(exist_ok=True)
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or list(row)
        if any(r["run_id"] == config["run_id"] for r in reader):
            return
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def publish(
    run,
    winner,
    comparisons,
    canonical,
    evaluation_root,
    model_dir,
    training,
    control,
    control_root,
):
    destination = run.output.parent / f".{run.run_id}.publish"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copytree(model_dir, destination / "model")
    for name in ("recommendations.parquet", "metrics.json"):
        shutil.copy2(evaluation_root / "result" / name, destination / name)
    first = min(
        (evaluation_root / "parts").glob("*/verification_features.parquet")
    ).parent
    for name in ("verification_features.parquet", "verification_scores.parquet"):
        shutil.copy2(first / name, destination / name)
    context = run.contexts["canonical"]
    context.target_users.write_parquet(destination / "target_users.parquet")
    context.ground_truth.write_parquet(destination / "target_ground_truth.parquet")
    shutil.copy2(
        control_root / "result/recommendations.parquet",
        destination / "control_recommendations.parquet",
    )
    write_json_atomic(destination / "config.json", run.config)
    reference = read_json(
        Path(run.config["paths"]["task12_reference"]) / "metrics.json"
    )
    metrics = {
        **canonical,
        "kind": "task13_history_profiles",
        "run_id": run.run_id,
        "winner": winner,
        "selection_comparisons": comparisons,
        "canonical_used_for_selection": False,
        "canonical_previously_opened": True,
        "training": training,
        "mode": run.config["mode"],
        "runtime_seconds": run.active_runtime_seconds,
        "peak_memory_mb": run.peak_memory_mb,
        "task12_reference": {
            k: reference[k]
            for k in ("precision_at_20_all_targets", "precision_at_20_labeled_users")
        },
        "canonical_matched_control": control,
        "canonical_profile_delta_all_targets": canonical["precision_at_20_all_targets"]
        - control["precision_at_20_all_targets"],
        "canonical_profile_delta_labeled_users": canonical[
            "precision_at_20_labeled_users"
        ]
        - control["precision_at_20_labeled_users"],
        "comparison_note": "profile effect: matched A control in r3 and canonical; vs Task12 also changes row budget; canonical cannot change frozen winner",
    }
    resource_path = Path(f"logs/{run.run_id}.resources.json")
    if resource_path.exists():
        metrics["resource_monitor"] = read_json(resource_path)
    write_json_atomic(destination / "metrics.json", metrics)
    for phase in ("selection", "canonical"):
        target = destination / "quantization" / phase
        target.mkdir(parents=True)
        for source in (run.work / "pools" / phase / "fit_training").iterdir():
            if (
                source.name != "operation.json"
                and source.is_file()
                and source.stat().st_size < 10_000_000
                and source.name != "train.quantized"
            ):
                shutil.copy2(source, target / source.name)
    for fold in run.contexts:
        path = destination / "profile_diagnostics" / fold
        path.mkdir(parents=True)
        shutil.copy2(
            run.work / "profiles" / fold / "diagnostics.json", path / "diagnostics.json"
        )
    CatBoostPointwiseModel.from_artifact(
        model_dir
    ).get_feature_importance().write_parquet(destination / "feature_importance.parquet")
    write_json_atomic(
        destination / "artifact_manifest.json",
        {"kind": "task13_history_profiles", "files": _file_manifest(destination)},
    )
    verify_artifact(destination)
    publish_directory_atomic(destination, run.output)
    append_experiment(run.config, metrics, run.output)
    return metrics


def publish_only(config_path: Path, *, show_progress=True):
    """Finish a fully staged publication; this path cannot fit or score shards."""
    source = load_config(config_path)
    run_id = source["run_id"]
    output = Path("artifacts") / run_id
    staging = Path("artifacts") / f".{run_id}.publish"
    work = Path("artifacts") / f".{run_id}.work"
    if output.exists():
        raise FileExistsError(f"already published: {output}; use --verify-only")
    if not (staging / "artifact_manifest.json").is_file():
        raise ValueError(
            "publication staging is incomplete; --publish-only cannot rebuild it"
        )
    config = read_json(staging / "config.json")
    if any(config.get(k) != v for k, v in source.items()):
        raise ValueError("source config differs from staged training config")
    digest = config_sha256(config)
    checkpoint = read_json(work / "checkpoint.json")
    if (
        config != read_json(work / "config.json")
        or checkpoint["config_sha256"] != digest
        or checkpoint["run_id"] != run_id
    ):
        raise ValueError("publication and checkpoint ownership differ")
    reporter = EventProgressReporter(
        task_name=run_id,
        total_phases=1,
        log_file=f"logs/{run_id}.log",
        show_progress=show_progress,
        log_max_bytes=5_000_000,
        log_backup_count=3,
    )
    started = reporter.phase_start("publication_recovery")
    try:
        reporter.event(
            "publication_recovery_start",
            stage="publication_recovery",
            training_started=False,
        )
        original_manifest = read_json(staging / "artifact_manifest.json")
        _verify_files(staging, original_manifest["files"])
        metrics = read_json(staging / "metrics.json")
        winner = metrics["winner"]
        if winner not in VARIANTS or metrics["run_id"] != run_id:
            raise ValueError("invalid staged winner or run ID")

        def completed(stage, variant, fold, expected_path):
            record = checkpoint["completed"][f"{stage}::{variant}::{fold}"]
            if record["path"] != expected_path:
                raise ValueError("unexpected checkpoint operation path")
            root = work / expected_path
            if sha256_file(root / "operation.json") != record["manifest_sha256"]:
                raise ValueError("completed operation manifest differs")
            manifest = read_json(root / "operation.json")
            if manifest["config_sha256"] != digest:
                raise ValueError("completed operation belongs to another config")
            _verify_files(root, manifest["files"])
            return manifest

        completed("selection", "winner", "rolling_3", "frozen_selection")
        frozen = read_json(work / "frozen_selection/selection.json")
        if frozen != {
            "winner": winner,
            "comparisons": metrics["selection_comparisons"],
        }:
            raise ValueError("staged selection differs from frozen selection")
        fitted = completed("fit", winner, "canonical", f"models/canonical/{winner}")
        _verify_files(staging, fitted["files"])
        for variant, filename, summary in (
            (winner, "recommendations.parquet", metrics),
            (
                "A_base",
                "control_recommendations.parquet",
                metrics["canonical_matched_control"],
            ),
        ):
            result = completed(
                "validate_output",
                variant,
                "canonical",
                f"evaluation/canonical/{variant}/result",
            )
            _verify_files(
                staging, {filename: result["files"]["recommendations.parquet"]}
            )
            if any(summary.get(k) != v for k, v in result["result"].items()):
                raise ValueError("staged metrics differ from completed evaluation")
        verification = verify_artifact(staging)
        report_path = staging / "publication_recovery.json"
        recovery = {
            "kind": "task13_publication_recovery",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_config_sha256": digest,
            "original_implementation_sha256": config["implementation_sha256"],
            "publication_implementation_sha256": sha256_file(Path(__file__)),
            "training_started": False,
            "predictions_regenerated": False,
            "verification": verification,
            "duration_seconds": time.perf_counter() - started,
            "reason": "strict floating-point equality replaced with exact count checks and 32-ULP metric tolerance",
        }
        write_json_atomic(report_path, recovery)
        files = dict(original_manifest["files"])
        files[report_path.name] = {
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
        }
        write_json_atomic(
            staging / "artifact_manifest.json", {**original_manifest, "files": files}
        )
        publish_directory_atomic(staging, output)
        append_experiment(config, metrics, output)
        reporter.phase_finish("publication_recovery", started)
        reporter.event(
            "publication_recovery_finish",
            status="completed",
            stage="publication_recovery",
            current_metric=metrics["precision_at_20_all_targets"],
            best_config=winner,
        )
        return {
            "status": "published",
            "artifact": str(output),
            "winner": winner,
            **verification,
        }
    except BaseException as error:
        reporter.event(
            "publication_recovery_failure",
            stage="publication_recovery",
            error=repr(error),
        )
        raise
    finally:
        reporter.close()


def cleanup_recoverable(output: Path):
    """Delete only a verified completed run's owned work cache; never referenced ALS models."""
    config = read_json(output / "config.json")
    run_id = config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("invalid artifact run ID")
    if output.resolve() != (Path("artifacts") / run_id).resolve():
        raise ValueError("cleanup requires the published artifact at its original path")
    work = Path("artifacts") / f".{run_id}.work"
    if not work.exists():
        return {"cleaned": False, "reason": "no work cache"}
    if work.is_symlink() or read_json(work / "config.json") != config:
        raise ValueError("work ownership differs from published config")
    checkpoint = read_json(work / "checkpoint.json")
    if (
        checkpoint["config_sha256"] != config_sha256(config)
        or checkpoint["run_id"] != run_id
    ):
        raise ValueError("checkpoint ownership differs")
    verify_artifact(output)
    total = sum(p.stat().st_blocks * 512 for p in work.rglob("*") if p.is_file())
    shutil.rmtree(work)
    return {"cleaned": True, "allocated_bytes_removed": total, "work": str(work)}


def run_experiment(
    config_path: Path, *, show_progress=True, stop_after_selection=False
):
    source = load_config(config_path)
    output = Path("artifacts") / source["run_id"]
    work = Path("artifacts") / f".{source['run_id']}.work"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}; use --verify-only")
    if (
        work.exists()
        and any(work.iterdir())
        and not (work / "checkpoint.json").exists()
    ):
        raise ValueError("work directory lacks ownership checkpoint")
    reporter = EventProgressReporter(
        task_name=source["run_id"],
        total_phases=7,
        log_file=f"logs/{source['run_id']}.log",
        show_progress=show_progress,
        log_max_bytes=5_000_000,
        log_backup_count=3,
    )
    run = None
    try:
        reporter.event("run_start", status="running", mode=source["mode"])
        run = ProfileRun(
            resolved_config(source), output=output, work=work, reporter=reporter
        )
        with run.phase("rolling_inputs_and_profiles"):
            run.load_folds(["rolling_1", "rolling_2", "rolling_3"])
            validate_temporal_training_scope(
                ["rolling_1", "rolling_2"],
                "rolling_3",
                {k: v.manifest["cutoff"] for k, v in run.contexts.items()},
            )
            for fold in run.contexts:
                run.prepare_profiles(fold)
        with run.phase("selection_shared_pool"):
            parts, selection_training = run.training_data(
                "selection", ["rolling_1", "rolling_2"]
            )
            pool, _ = run.training_pool("selection", "fit_training", parts)
        comparisons = {}
        with run.phase("rolling_ablation"):
            stage = reporter.stage_start(
                stage="rolling_ablation", total=len(VARIANTS), unit="config"
            )
            for order, variant in enumerate(VARIANTS):
                model, model_dir = run.fit(
                    "selection",
                    variant,
                    pool,
                    tree_count=source["selection"]["fixed_tree_count"],
                )
                metric, _ = run.full_evaluation(
                    model, context=run.contexts["rolling_3"], config_id=variant
                )
                comparisons[variant] = metric
                # Each model directory is already immutable and atomically published.
                run.best.update(
                    score=(metric["precision_at_20_all_targets"], -order),
                    payload={
                        "variant": variant,
                        "model": str(model_dir),
                        "metrics": metric,
                    },
                )
                reporter.event(
                    "ablation_result",
                    stage="rolling_ablation",
                    config=variant,
                    fold="rolling_3",
                    current_metric=metric["precision_at_20_all_targets"],
                    best_metric=run.best.read()["score"][0],
                    best_config=run.best.read()["payload"]["variant"],
                )
                del model
                gc.collect()
                reporter.stage_advance()
            reporter.stage_finish(stage="rolling_ablation", started=stage)
        with run.phase("freeze_winner"):
            winner = run.best.read()["payload"]["variant"]

            def freeze(path):
                write_json_atomic(
                    path / "selection.json",
                    {"winner": winner, "comparisons": comparisons},
                )
                return {"winner": winner}

            frozen = run.operation(
                "selection", "winner", "rolling_3", work / "frozen_selection", freeze
            )
            if frozen["winner"] != winner:
                raise ValueError("frozen winner changed")
        if stop_after_selection:
            reporter.event(
                "run_finish", status="paused_after_selection", best_config=winner
            )
            return {
                "status": "paused_after_selection",
                "winner": winner,
                "work": str(work),
            }
        with run.phase("canonical_refit"):
            run.load_folds(["canonical"])
            validate_temporal_training_scope(
                ["rolling_1", "rolling_2", "rolling_3"],
                "canonical",
                {k: v.manifest["cutoff"] for k, v in run.contexts.items()},
            )
            run.prepare_profiles("canonical")
            parts, canonical_training = run.training_data(
                "canonical", ["rolling_1", "rolling_2", "rolling_3"]
            )
            pool, _ = run.training_pool("canonical", "fit_training", parts)
            model, model_dir = run.fit(
                "canonical",
                winner,
                pool,
                tree_count=source["selection"]["fixed_tree_count"],
            )
        with run.phase("canonical_once"):
            canonical, evaluation_root = run.full_evaluation(
                model, context=run.contexts["canonical"], config_id=winner
            )
            del model
            if winner != "A_base":
                control_model, _ = run.fit(
                    "canonical",
                    "A_base",
                    pool,
                    tree_count=source["selection"]["fixed_tree_count"],
                )
                control, control_root = run.full_evaluation(
                    control_model, context=run.contexts["canonical"], config_id="A_base"
                )
                del control_model
            else:
                control, control_root = canonical, evaluation_root
            run.release_profiles()
        with run.phase("publish_and_verify"):
            metrics = publish(
                run,
                winner,
                comparisons,
                canonical,
                evaluation_root,
                model_dir,
                {"selection": selection_training, "canonical": canonical_training},
                control,
                control_root,
            )
        reporter.event(
            "run_finish",
            status="completed",
            best_config=winner,
            current_metric=metrics["precision_at_20_all_targets"],
            runtime_seconds=run.active_runtime_seconds,
        )
        return metrics
    except BaseException as error:
        reporter.event("run_finish", status="failed", error=repr(error))
        raise
    finally:
        if run is not None:
            write_json_atomic(
                work / "timing.json",
                {
                    "active_runtime_seconds": run.active_runtime_seconds,
                    "peak_memory_mb": run.peak_memory_mb,
                },
            )
        reporter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task13_history_profiles_v1.json")
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="Verify checksums, top20 semantics, and repeat portable inference; no fit",
    )
    parser.add_argument(
        "--stop-after-selection",
        action="store_true",
        help="Pause with frozen r3 winner; rerun identical config to resume",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Verify and publish an already complete staging directory; never train or regenerate predictions",
    )
    parser.add_argument(
        "--cleanup-recoverable",
        action="store_true",
        help="With --verify-only: delete only this verified completed run's owned work cache",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.publish_only:
        if (
            args.verify_only
            or args.worker
            or args.stop_after_selection
            or args.cleanup_recoverable
        ):
            parser.error("--publish-only cannot be combined with other action flags")
        Path("logs").mkdir(exist_ok=True)
        with Path("logs/task13_history_profiles.lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(
                json.dumps(
                    publish_only(args.config, show_progress=not args.no_progress),
                    indent=2,
                )
            )
        return
    if args.verify_only:
        result = verify_artifact(args.verify_only)
        config = read_json(args.verify_only / "config.json")
        append_experiment(
            config, read_json(args.verify_only / "metrics.json"), args.verify_only
        )
        if args.cleanup_recoverable:
            Path("logs").mkdir(exist_ok=True)
            with Path("logs/task13_history_profiles.lock").open("w") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result["cleanup"] = cleanup_recoverable(args.verify_only)
        print(json.dumps(result, indent=2))
        return
    if args.cleanup_recoverable:
        parser.error("--cleanup-recoverable requires --verify-only")
    c = load_config(args.config)
    if args.worker:
        if os.environ.get("TASK13_SUPERVISED") != "1":
            raise ValueError("worker requires resource supervision")

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}; rerun same config to resume")

        signal.signal(signal.SIGTERM, interrupt)
        print(
            json.dumps(
                run_experiment(
                    args.config,
                    show_progress=not args.no_progress,
                    stop_after_selection=args.stop_after_selection,
                ),
                indent=2,
            )
        )
        return
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if (Path("artifacts") / c["run_id"]).exists():
        raise FileExistsError("completed output exists; use --verify-only")
    with Path("logs/task13_history_profiles.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(
            TASK13_SUPERVISED="1",
            OMP_NUM_THREADS=str(c["catboost"]["thread_count"]),
            POLARS_MAX_THREADS=str(c["catboost"]["thread_count"]),
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            "--config",
            str(args.config),
        ]
        if args.no_progress:
            command.append("--no-progress")
        if args.stop_after_selection:
            command.append("--stop-after-selection")
        raise SystemExit(supervise(command, c))


if __name__ == "__main__":
    main()

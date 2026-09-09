#!/usr/bin/env python3
"""Task 12: bounded multi-fold training, explicit quantization, and P@20 selection."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment_utils import (
    AtomicBestConfig,
    CheckpointStore,
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from metrics import evaluate_candidate_metrics_lazy, evaluate_precision_at_20
from popularity import candidates_to_recommendations
from ranker_backtest import (
    QUANTIZATION_POLICIES,
    allocate_training_probabilities,
    load_backtest_config,
    metrics_from_hits,
    quantize_training_parts,
    sample_training_rows,
    select_evaluation_users,
    select_tree_checkpoint,
    validate_temporal_training_scope,
)
from rankers import (
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
    ranker_scores_to_candidates,
)
from scripts.run_catboost_ranker import (
    FoldContext,
    _CatBoostLogBridge,
    _validate_task07,
)
from validation import ContractValidationError, validate_final_recommendations


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "operation.json"
    }


def _verify_files(root: Path, files: Mapping[str, Any]) -> None:
    for relative, record in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ContractValidationError("manifest path escapes its artifact")
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ContractValidationError(f"checksum mismatch: {path}")


class BacktestRun:
    """Checkpoint ownership, bounded logging, and atomic operation publication."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        output: Path,
        work: Path,
        reporter: EventProgressReporter,
    ):
        self.config, self.output, self.work, self.reporter = (
            config,
            output,
            work,
            reporter,
        )
        self.run_id = config["run_id"]
        self.started = time.perf_counter()
        self.features: tuple[str, ...] = ()
        self.contexts: dict[str, FoldContext] = {}
        self.smoke_users: pl.DataFrame | None = None
        self.digest = config_sha256(config)
        self.checkpoint = CheckpointStore(
            work, run_id=self.run_id, config_digest=self.digest
        )
        self.best = AtomicBestConfig(
            work / "best", run_id=self.run_id, config_digest=self.digest
        )
        timing = (
            read_json(work / "timing.json") if (work / "timing.json").exists() else {}
        )
        self.previous_runtime_seconds = float(timing.get("active_runtime_seconds", 0.0))
        self.previous_peak_memory_mb = float(timing.get("peak_memory_mb", 0.0))
        write_json_atomic(work / "config.json", config)

    @property
    def active_runtime_seconds(self) -> float:
        return self.previous_runtime_seconds + time.perf_counter() - self.started

    @property
    def peak_memory_mb(self) -> float:
        return max(
            self.previous_peak_memory_mb,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )

    @contextmanager
    def phase(self, name: str):
        started = self.reporter.phase_start(name)
        try:
            yield
        except BaseException as error:
            self.reporter.event(
                "phase_failure",
                stage=name,
                status="failed",
                error=repr(error),
                duration_seconds=time.perf_counter() - started,
            )
            raise
        else:
            self.reporter.phase_finish(name, started)

    @contextmanager
    def units(self, stage: str, config: str, fold: str, total: int, unit: str = "part"):
        started, callback = self.reporter.iteration_start(
            stage=stage, config=config, fold=fold, total=total, unit=unit
        )
        try:
            yield callback
        except BaseException:
            self.reporter.iteration_finish(
                stage=stage, config=config, fold=fold, started=started, status="failed"
            )
            raise
        else:
            self.reporter.iteration_finish(
                stage=stage,
                config=config,
                fold=fold,
                started=started,
                status="completed",
            )

    def operation(
        self,
        stage: str,
        config: str,
        fold: str,
        destination: Path,
        build: Callable[[Path], dict[str, Any]],
    ) -> dict[str, Any]:
        """Adopt a completed atomic directory after interruption before checkpoint."""
        started = self.reporter.operation_start(
            stage=stage, config=config, fold=fold, operation=destination.name
        )
        record = self.checkpoint.get(stage=stage, config=config, fold=fold)
        try:
            if destination.exists():
                manifest = read_json(destination / "operation.json")
                if manifest.get("config_sha256") != self.digest:
                    raise ContractValidationError(
                        f"incompatible operation: {destination}"
                    )
                _verify_files(destination, manifest["files"])
                if record is not None and record["manifest_sha256"] != sha256_file(
                    destination / "operation.json"
                ):
                    raise ContractValidationError(
                        f"operation manifest differs: {destination}"
                    )
                self.reporter.event(
                    "operation_resume_skip",
                    stage=stage,
                    config=config,
                    fold=fold,
                    operation=destination.name,
                )
            else:
                if record is not None:
                    raise ContractValidationError(
                        f"completed operation is missing: {destination}"
                    )
                self.check_resources(stage)
                temporary = destination.parent / f".{destination.name}.incomplete"
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir(parents=True)
                result = build(temporary)
                manifest = {
                    "config_sha256": self.digest,
                    "result": result,
                    "files": _file_manifest(temporary),
                }
                write_json_atomic(temporary / "operation.json", manifest)
                publish_directory_atomic(temporary, destination)
            self.checkpoint.complete(
                stage=stage,
                config=config,
                fold=fold,
                metadata={
                    "path": destination.relative_to(self.work).as_posix(),
                    "manifest_sha256": sha256_file(destination / "operation.json"),
                },
            )
            self.reporter.operation_finish(
                stage=stage,
                config=config,
                fold=fold,
                operation=destination.name,
                started=started,
                status="completed",
            )
            return manifest["result"]
        except BaseException as error:
            self.reporter.event(
                "operation_failure",
                stage=stage,
                config=config,
                fold=fold,
                operation=destination.name,
                error=repr(error),
                duration_seconds=time.perf_counter() - started,
            )
            raise

    def check_resources(self, stage: str) -> None:
        free = shutil.disk_usage(self.work).free / 2**30
        # Subsequent phases may consume the preflight reserve, but retain 10 GiB.
        floor = min(10.0, self.config["resources"]["minimum_free_disk_gib"])
        self.reporter.event(
            "resources",
            stage=stage,
            free_disk_gib=free,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        )
        if free < floor:
            raise RuntimeError(f"free disk dropped below {floor} GiB")

    def read_part(
        self,
        part: Path,
        *,
        columns: list[str] | None = None,
        users: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        frame = pl.scan_parquet(part)
        if columns is not None:
            frame = frame.select(columns)
        selected = users if users is not None else self.smoke_users
        if selected is not None:
            frame = frame.join(selected.lazy(), on="user_id", how="semi")
        return frame.collect(engine="streaming").sort(["user_id", "item_id"])

    def load_folds(self, names: list[str]) -> None:
        contexts, features, _, _ = _validate_task07(
            Path(self.config["paths"]["task07_dataset"]),
            folds=names,
            part_limit=1 if self.config["mode"] == "smoke" else None,
            verify_checksums=False,
        )
        if self.features and self.features != features:
            raise ContractValidationError("feature order changed across folds")
        self.features = features
        if self.config["mode"] == "smoke" and self.smoke_users is None:
            self.smoke_users = select_evaluation_users(
                next(iter(contexts.values())).target_users,
                count=self.config["smoke"]["user_count"],
                seed=self.config["seed"],
            )
        all_parts = [
            (fold, p) for fold, context in contexts.items() for p in context.parts
        ]
        with self.units(
            "verify_inputs", "task07", "+".join(names), len(all_parts)
        ) as advance:
            for index, (fold, part) in enumerate(all_parts):
                if sha256_file(part) != contexts[fold].manifest["parts"][part.name]:
                    raise ContractValidationError(
                        f"immutable Task07 shard differs: {part}"
                    )
                advance(index)
        for context in contexts.values():
            if self.smoke_users is not None:
                context.target_users = context.target_users.join(
                    self.smoke_users, on="user_id", how="semi"
                )
                context.ground_truth = context.ground_truth.join(
                    self.smoke_users, on="user_id", how="semi"
                )
                if context.target_users.height != self.smoke_users.height:
                    raise ContractValidationError(
                        "smoke users are missing from a temporal fold"
                    )
        self.contexts.update(contexts)

    def training_data(
        self, phase: str, folds: list[str]
    ) -> tuple[list[Path], dict[str, Any]]:
        output = self.work / "training_data" / phase
        counts = {}
        for fold in folds:

            def count_fold(destination: Path, fold: str = fold) -> dict[str, Any]:
                total = positive = 0
                with self.units(
                    "training_counts", phase, fold, len(self.contexts[fold].parts)
                ) as advance:
                    for index, part in enumerate(self.contexts[fold].parts):
                        df = self.read_part(
                            part,
                            columns=[
                                "user_id",
                                "item_id",
                                "label",
                                "is_training_sample",
                            ],
                        )
                        sampled = df.filter(pl.col("is_training_sample"))
                        total += sampled.height
                        positive += int(sampled["label"].sum())
                        if int(df["label"].sum()) != int(sampled["label"].sum()):
                            raise ContractValidationError(
                                "upstream training mask dropped a positive"
                            )
                        advance(index)
                result = {"rows": total, "positives": positive}
                write_json_atomic(destination / "counts.json", result)
                return result

            counts[fold] = self.operation(
                "training_counts",
                "task07",
                fold,
                self.work / "training_counts" / fold,
                count_fold,
            )
        plan = allocate_training_probabilities(
            counts, target_rows=self.config["training"]["target_rows"]
        )
        totals = {"rows": 0, "positive_rows": 0, "sample_weight_sum": 0.0}
        for fold in folds:

            def sample_fold(destination: Path, fold: str = fold) -> dict[str, Any]:
                folded = {"rows": 0, "positive_rows": 0, "sample_weight_sum": 0.0}
                context = self.contexts[fold]
                with self.units(
                    "sample_training", phase, fold, len(context.parts)
                ) as advance:
                    for index, part in enumerate(context.parts):
                        frame = self.read_part(part)
                        rows = sample_training_rows(
                            frame,
                            features=self.features,
                            fold=fold,
                            seed=self.config["seed"],
                            negative_probability=plan[fold][
                                "secondary_negative_probability"
                            ],
                        )
                        rows.write_parquet(
                            destination / part.name,
                            compression="zstd",
                            compression_level=3,
                        )
                        folded["rows"] += rows.height
                        folded["positive_rows"] += int(rows["label"].sum())
                        folded["sample_weight_sum"] += float(
                            rows["sample_weight"].cast(pl.Float64).sum()
                        )
                        advance(index)
                if folded["positive_rows"] != counts[fold]["positives"]:
                    raise ContractValidationError(
                        "secondary sampling dropped positives"
                    )
                write_json_atomic(
                    destination / "sampling.json", {**plan[fold], "actual": folded}
                )
                return folded

            folded = self.operation("sampling", phase, fold, output / fold, sample_fold)
            plan[fold]["actual"] = folded
            for key in totals:
                totals[key] += folded[key]
        result = {
            "folds": folds,
            "sampling_plan": plan,
            **totals,
            "sampling_salt": "task12:train:<fold>",
            "weighting": "inverse_full_sampling_probability",
        }
        write_json_atomic(output / "sampling.json", result)
        return sorted(output.glob("*/part-*.parquet")), result

    def training_pool(
        self, phase: str, policy: str, parts: list[Path]
    ) -> tuple[Path, dict[str, Any]]:
        output = self.work / "pools" / phase / policy

        def build(destination: Path) -> dict[str, Any]:
            def quantization_status(operation: str) -> None:
                self.reporter.stage_status(
                    stage="quantization", config=policy, fold=phase, operation=operation
                )
                self.reporter.event(
                    "quantization_progress",
                    stage="quantization",
                    config=policy,
                    fold=phase,
                    operation=operation,
                )

            with self.units("quantization", policy, phase, len(parts)) as advance:
                return quantize_training_parts(
                    parts,
                    destination=destination,
                    features=self.features,
                    policy=policy,
                    frozen_borders=Path(self.config["paths"]["task08_borders"]),
                    border_count=self.config["quantization"]["border_count"],
                    feature_border_type=self.config["quantization"][
                        "feature_border_type"
                    ],
                    seed=self.config["seed"],
                    thread_count=self.config["catboost"]["thread_count"],
                    maximum_dsv_bytes=int(
                        self.config["resources"]["maximum_dsv_gib"] * 2**30
                    ),
                    progress=advance,
                    operation_progress=quantization_status,
                )

        value = self.operation("quantization", policy, phase, output, build)
        return output / "train.quantized", value

    def fit(
        self, phase: str, policy: str, pool_path: Path, *, tree_count: int
    ) -> tuple[CatBoostPointwiseModel, Path]:
        output = self.work / "models" / phase / policy

        def build(destination: Path) -> dict[str, Any]:
            params = dict(
                self.config["catboost"],
                iterations=tree_count,
                config_id=f"task12_{phase}_{policy}",
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
            train_dir = self.work / "snapshots" / phase / policy
            train_dir.mkdir(parents=True, exist_ok=True)
            with self.units("fit", policy, phase, tree_count, "tree") as advance:
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
                "eval_pool": None,
                "fixed_tree_budget": True,
                "training_pool_sha256": sha256_file(pool_path),
            }

        self.operation("fit", policy, phase, output, build)
        return CatBoostPointwiseModel.from_artifact(output / "model"), output / "model"

    def evaluation_cache(self, context: FoldContext, users: pl.DataFrame) -> list[Path]:
        output = self.work / "evaluation_cache" / context.fold
        output.mkdir(parents=True, exist_ok=True)
        users.write_parquet(output / "users.parquet")
        parts = []
        with self.units(
            "evaluation_cache", "complete_users", context.fold, len(context.parts)
        ) as advance:
            for index, source in enumerate(context.parts):
                destination = output / source.stem

                def build(path: Path, source: Path = source) -> dict[str, Any]:
                    frame = self.read_part(
                        source,
                        columns=["user_id", "item_id", "label", *self.features],
                        users=users,
                    )
                    frame.write_parquet(
                        path / "features.parquet",
                        compression="zstd",
                        compression_level=3,
                    )
                    return {"rows": frame.height, "users": frame["user_id"].n_unique()}

                result = self.operation(
                    "evaluation_cache",
                    "complete_users",
                    f"{context.fold}:{source.stem}",
                    destination,
                    build,
                )
                if result["rows"]:
                    parts.append(destination / "features.parquet")
                advance(index)
        actual = (
            pl.scan_parquet(parts).select("user_id").unique().collect().sort("user_id")
        )
        if not actual.equals(users.sort("user_id")):
            raise ContractValidationError(
                "evaluation cache does not cover exactly the selected users"
            )
        if (
            pl.scan_parquet(parts)
            .group_by("user_id")
            .len()
            .filter(pl.col("len") < 20)
            .collect()
            .height
        ):
            raise ContractValidationError(
                "evaluation users must retain full candidate lists with at least 20 items"
            )
        return parts

    def score_curve(
        self,
        model: CatBoostPointwiseModel,
        *,
        config_id: str,
        parts: list[Path],
        users: pl.DataFrame,
        ground_truth: pl.DataFrame,
        tree_counts: list[int],
    ) -> list[dict[str, Any]]:
        root = self.work / "curves" / config_id
        totals = {
            count: {"hits": 0, "logloss_sum": 0.0, "rows": 0} for count in tree_counts
        }
        with self.units(
            "checkpoint_scores", config_id, "rolling_3", len(parts)
        ) as advance:
            for index, part in enumerate(parts):
                destination = root / part.parent.name

                def build(path: Path, part: Path = part) -> dict[str, Any]:
                    import numpy as np

                    frame = pl.read_parquet(part).sort(["user_id", "item_id"])
                    loader = (
                        CatBoostRankerDataLoader(feature_columns=self.features)
                        .load_predict_data(frame=frame)
                        .prepare_predict_data()
                    )
                    predictions = model.predict_checkpoints(
                        loader,
                        tree_counts=tree_counts,
                        batch_size=self.config["inference"]["batch_size"],
                    )
                    labels = frame["label"].to_numpy()
                    truth = frame.filter(pl.col("label") == 1).select(
                        "user_id", "item_id"
                    )
                    values = {}
                    for count, scores in predictions.items():
                        top = ranker_scores_to_candidates(scores)
                        hits = top.join(
                            truth, on=["user_id", "item_id"], how="semi"
                        ).height
                        raw = scores["ranker_score"].to_numpy()
                        values[str(count)] = {
                            "hits": hits,
                            "rows": frame.height,
                            "logloss_sum": float(
                                np.logaddexp(0, raw).sum() - np.dot(labels, raw)
                            ),
                        }
                    write_json_atomic(path / "scores.json", values)
                    return values

                result = self.operation(
                    "checkpoint_scores", config_id, part.parent.name, destination, build
                )
                for count in tree_counts:
                    for key in totals[count]:
                        totals[count][key] += result[str(count)][key]
                advance(index)
        labeled = ground_truth.join(users, on="user_id", how="semi")[
            "user_id"
        ].n_unique()
        curve = [
            {
                "config_id": config_id,
                "tree_count": count,
                **metrics_from_hits(
                    hits=values["hits"], targets=users.height, labeled=labeled
                ),
                "full_candidate_logloss": values["logloss_sum"] / values["rows"],
                "candidate_rows": values["rows"],
            }
            for count, values in totals.items()
        ]
        write_json_atomic(
            root / "curve.json",
            {
                "curve": curve,
                "selection_users_sha256": sha256_file(
                    self.work / "evaluation_cache/rolling_3/users.parquet"
                ),
            },
        )
        return curve

    def full_evaluation(
        self, model: CatBoostPointwiseModel, *, context: FoldContext, config_id: str
    ) -> tuple[dict[str, Any], Path]:
        root = self.work / "evaluation" / context.fold / config_id
        with self.units(
            "full_inference", config_id, context.fold, len(context.parts)
        ) as advance:
            for index, source in enumerate(context.parts):
                destination = root / "parts" / source.stem

                def build(
                    path: Path, source: Path = source, index: int = index
                ) -> dict[str, Any]:
                    frame = self.read_part(
                        source, columns=["user_id", "item_id", *self.features]
                    )
                    users = frame.select("user_id").unique().sort("user_id")
                    loader = (
                        CatBoostRankerDataLoader(feature_columns=self.features)
                        .load_predict_data(frame=frame)
                        .prepare_predict_data()
                    )
                    scores = model.predict(
                        loader, batch_size=self.config["inference"]["batch_size"]
                    )
                    recommendations = candidates_to_recommendations(
                        ranker_scores_to_candidates(scores), users
                    )
                    validate_final_recommendations(recommendations, expected_k=20)
                    recommendations.write_parquet(
                        path / "recommendations.parquet", compression="zstd"
                    )
                    # Retain a small complete-query probe for deterministic portable restore.
                    if index == 0:
                        probe_users = users.head(min(8, users.height))
                        frame.join(probe_users, on="user_id", how="semi").write_parquet(
                            path / "verification_features.parquet"
                        )
                        scores.join(
                            probe_users, on="user_id", how="semi"
                        ).write_parquet(path / "verification_scores.parquet")
                    return {"users": users.height, "candidate_rows": frame.height}

                self.operation(
                    "full_inference",
                    config_id,
                    f"{context.fold}:{source.stem}",
                    destination,
                    build,
                )
                advance(index)

        def combine(destination: Path) -> dict[str, Any]:
            recommendations = pl.concat(
                [
                    pl.read_parquet(path)
                    for path in sorted(
                        (root / "parts").glob("*/recommendations.parquet")
                    )
                ]
            ).sort("user_id")
            semantic = validate_output_semantics(recommendations, context=context)
            metrics = evaluate_precision_at_20(
                recommendations, context.ground_truth, context.target_users
            )
            hit_count = (
                recommendations.explode("item_ids", empty_as_null=True)
                .rename({"item_ids": "item_id"})
                .join(context.ground_truth, on=["user_id", "item_id"], how="semi")
                .height
            )
            candidates = pl.scan_parquet(context.parts).select("user_id", "item_id")
            if self.smoke_users is not None:
                candidates = candidates.join(
                    self.smoke_users.lazy(), on="user_id", how="semi"
                )
            candidate_metrics = evaluate_candidate_metrics_lazy(
                candidates, context.ground_truth, context.target_users
            )
            result = {
                **metrics,
                **candidate_metrics,
                "final_hits": hit_count,
                "target_users": context.target_users.height,
                "labeled_users": context.ground_truth["user_id"].n_unique(),
                "tree_count": model.tree_count,
                "validation": semantic,
                "fold": context.fold,
                "config_id": config_id,
            }
            recommendations.write_parquet(
                destination / "recommendations.parquet", compression="zstd"
            )
            write_json_atomic(destination / "metrics.json", result)
            return result

        result = self.operation(
            "validate_output", config_id, context.fold, root / "result", combine
        )
        return result, root


def validate_output_semantics(
    recommendations: pl.DataFrame, *, context: FoldContext
) -> dict[str, int]:
    """Validate predictions with projected history joins, avoiding a full seen-set cache."""
    validate_final_recommendations(recommendations, expected_k=20)
    if (
        not recommendations.select("user_id")
        .sort("user_id")
        .equals(context.target_users.sort("user_id"))
    ):
        raise ContractValidationError(
            "recommendations do not match the complete target universe"
        )
    pairs = (
        recommendations.explode("item_ids", empty_as_null=True)
        .rename({"item_ids": "item_id"})
        .lazy()
    )
    history = pl.scan_parquet(context.history_path).select("user_id", "item_id")
    seen = (
        history.join(pairs, on=["user_id", "item_id"], how="semi")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    items = pairs.select("item_id").unique()
    known = history.select("item_id").join(items, on="item_id", how="semi").unique()
    unknown = (
        items.join(known, on="item_id", how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if seen or unknown:
        raise ContractValidationError(
            f"invalid recommendations: seen={seen}, unknown={unknown}"
        )
    return {
        "seen_pairs": 0,
        "unknown_items": 0,
        "missing_users": 0,
        "extra_users": 0,
        "duplicate_users": 0,
        "duplicate_items": 0,
        "null_values": 0,
        "items_per_user": 20,
    }


def _preflight(
    config: dict[str, Any], work: Path, reporter: EventProgressReporter
) -> None:
    if sys.version_info[:2] != (3, 12) or catboost.__version__ != "1.2.10":
        raise RuntimeError("use repository Python 3.12 and CatBoost 1.2.10")
    limits = config["resources"]
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        memory[name] = int(value.strip().split()[0]) * 1024
    # On resume, reserved bytes already occupy the workspace and are reusable.
    allocated = (
        sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
        if work.exists()
        else 0
    )
    free = shutil.disk_usage(work.parent).free / 2**30
    required = max(
        10.0 if config["mode"] == "full" else 0.1,
        limits["minimum_free_disk_gib"] - allocated / 2**30,
    )
    available = memory["MemAvailable"] / 2**30
    reporter.event(
        "preflight",
        stage="preflight",
        free_disk_gib=free,
        required_disk_gib=required,
        available_ram_gib=available,
        required_ram_gib=limits["minimum_available_ram_gib"],
    )
    if free < required or available < limits["minimum_available_ram_gib"]:
        raise RuntimeError("insufficient disk or available RAM; see the preflight log")
    if config["catboost"]["task_type"] == "GPU":
        device = config["catboost"]["devices"]
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                f"--id={device}",
            ],
            text=True,
        )
        if int(row.strip()) < limits["minimum_free_vram_mib"]:
            raise RuntimeError("insufficient free GPU memory")


def _resolved_config(source: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    inputs = {
        key: sha256_file(value)
        for key, value in source["paths"].items()
        if key in {"task08_borders"}
    }
    for key, name in [
        ("task07_dataset", "dataset_manifest.json"),
        ("task08_model", "model_config.json"),
    ]:
        inputs[key] = sha256_file(Path(source["paths"][key]) / name)
    result["input_provenance"] = inputs
    result["implementation_sha256"] = {
        name: sha256_file(REPOSITORY_ROOT / name)
        for name in (
            "ranker_backtest.py",
            "rankers.py",
            "ranker_data.py",
            "metrics.py",
            "experiment_utils.py",
            "scripts/run_ranker_backtest.py",
            "scripts/run_catboost_ranker.py",
        )
    }
    return result


def _append_experiment(
    config: dict[str, Any], metrics: dict[str, Any], output: Path
) -> None:
    if config["mode"] != "full":
        return
    path = Path("experiments/results.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": config["run_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": "canonical_24h_known_diagnostic",
        "seed": config["seed"],
        "candidate_config": json.dumps(
            {"dataset": config["paths"]["task07_dataset"], "unchanged_union": True}
        ),
        "ranker_config": json.dumps(metrics["winner"]),
        "p20_all_targets": metrics["precision_at_20_all_targets"],
        "p20_labeled_users": metrics["precision_at_20_labeled_users"],
        "runtime": metrics["runtime_seconds"],
        "peak_memory": metrics["peak_memory_mb"],
        "artifact_path": str(output),
        "notes": "multi_fold_backtest; selection=rolling_3_complete_queries; canonical_not_used_for_selection; no_candidate_refit",
    }
    for key in (
        "candidate_recall",
        "coverage",
        "candidate_user_hit_rate",
        "candidate_oracle_p20_all_targets",
        "candidate_oracle_p20_labeled_users",
        "mean_candidate_count",
        "final_hits",
    ):
        row[key] = metrics[key]
    with path.open("a+", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or list(row)
        if any(item["run_id"] == config["run_id"] for item in rows):
            return
        stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def verify_artifact(path: Path) -> dict[str, Any]:
    config = read_json(path / "config.json")
    if config.get("kind") != "task12_ranker_backtest":
        raise ValueError("not a Task 12 artifact")
    _verify_files(path, read_json(path / "checksums.json")["files"])
    model = CatBoostPointwiseModel.from_artifact(path / "model")
    frame = pl.read_parquet(path / "verification_features.parquet")
    loader = (
        CatBoostRankerDataLoader(feature_columns=model.feature_columns)
        .load_predict_data(frame=frame)
        .prepare_predict_data()
    )
    first = model.predict(loader, batch_size=config["inference"]["batch_size"])
    second = model.predict(loader, batch_size=config["inference"]["batch_size"])
    expected = pl.read_parquet(path / "verification_scores.parquet").sort(
        ["user_id", "item_id"]
    )
    if not first.equals(second) or not first.equals(expected):
        raise ContractValidationError(
            "portable inference differs from the saved deterministic probe"
        )
    recommendations = pl.read_parquet(path / "canonical/recommendations.parquet")
    targets = pl.read_parquet(path / "canonical/target_users.parquet")
    truth = pl.read_parquet(path / "canonical/ground_truth.parquet")
    validate_final_recommendations(recommendations, expected_k=20)
    if (
        not recommendations.select("user_id")
        .sort("user_id")
        .equals(targets.sort("user_id"))
    ):
        raise ContractValidationError("published user universe differs")
    measured = evaluate_precision_at_20(recommendations, truth, targets)
    metrics = read_json(path / "metrics.json")
    if any(abs(metrics[key] - value) > 1e-15 for key, value in measured.items()):
        raise ContractValidationError("published Precision@20 differs from predictions")
    return {
        "status": "verified",
        "run_id": config["run_id"],
        "portable_repeat_equal": True,
        "tree_count": model.tree_count,
        **measured,
    }


def _publish(
    run: BacktestRun,
    *,
    winner: dict[str, Any],
    model_dir: Path,
    canonical: dict[str, Any],
    canonical_root: Path,
    rolling: dict[str, Any],
    rolling_root: Path,
    curves: dict[str, list[dict[str, Any]]],
    training: dict[str, Any],
) -> dict[str, Any]:
    config = run.config
    context = run.contexts["canonical"]
    staging = run.work / "publication.incomplete"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    shutil.copytree(model_dir, staging / "model")
    shutil.copytree(
        run.work / "selected_models" / winner["policy"] / "model",
        staging / "selection_model",
    )
    for fold, root in [("canonical", canonical_root), ("rolling_3", rolling_root)]:
        (staging / fold).mkdir()
        for name in ["metrics.json", "recommendations.parquet"]:
            shutil.copyfile(root / "result" / name, staging / fold / name)
    context.target_users.write_parquet(staging / "canonical/target_users.parquet")
    context.ground_truth.write_parquet(staging / "canonical/ground_truth.parquet")
    first = canonical_root / "parts" / context.parts[0].stem
    for name in ("verification_features.parquet", "verification_scores.parquet"):
        shutil.copyfile(first / name, staging / name)
    (staging / "quantization").mkdir()
    for phase, policy in [("selection", p) for p in QUANTIZATION_POLICIES] + [
        ("canonical", winner["policy"])
    ]:
        target = staging / "quantization" / f"{phase}_{policy}"
        target.mkdir()
        source = run.work / "pools" / phase / policy
        for name in ("borders.tsv", "quantization.json", "columns.cd"):
            shutil.copyfile(source / name, target / name)
    write_json_atomic(staging / "selection_curves.json", {"curves": curves})
    rows = [row for curve in curves.values() for row in curve]
    pl.DataFrame(rows).write_csv(staging / "selection_curves.csv")
    write_json_atomic(staging / "winner.json", winner)
    write_json_atomic(staging / "training.json", training)
    write_json_atomic(staging / "config.json", config)
    baseline = None
    if config["mode"] == "full":
        baseline = read_json(
            Path(config["paths"]["task08_model"]).parent / "metrics.json"
        )
    metrics = {
        **canonical,
        "kind": "task12_ranker_backtest",
        "run_id": config["run_id"],
        "mode": config["mode"],
        "winner": winner,
        "selection": {
            policy: select_tree_checkpoint(curve) for policy, curve in curves.items()
        },
        "rolling_3_full": rolling,
        "canonical_used_for_selection": False,
        "canonical_previously_opened": True,
        "portable_repeat_equal": True,
        "runtime_seconds": run.active_runtime_seconds,
        "peak_memory_mb": run.peak_memory_mb,
        "comparisons": {
            "task08_canonical": (
                {
                    key: baseline[key]
                    for key in [
                        "precision_at_20_all_targets",
                        "precision_at_20_labeled_users",
                        "final_hits",
                    ]
                }
                if baseline
                else None
            ),
            "delta_p20_all_targets": (
                canonical["precision_at_20_all_targets"]
                - baseline["precision_at_20_all_targets"]
                if baseline
                else None
            ),
            "delta_p20_labeled_users": (
                canonical["precision_at_20_labeled_users"]
                - baseline["precision_at_20_labeled_users"]
                if baseline
                else None
            ),
            "delta_hits": canonical["final_hits"] - baseline["final_hits"]
            if baseline
            else None,
            "smoke_scores_not_comparable_to_full_runs": config["mode"] == "smoke",
        },
    }
    write_json_atomic(staging / "metrics.json", metrics)
    write_json_atomic(staging / "checksums.json", {"files": _file_manifest(staging)})
    verify_artifact(staging)
    publish_directory_atomic(staging, run.output)
    _append_experiment(config, metrics, run.output)
    if config["resources"]["cleanup_training_cache_on_success"]:
        # Only this verified completed run's known temporary subdirectories.
        if read_json(run.work / "config.json") != config:
            raise ContractValidationError("cleanup ownership changed")
        for name in ("training_data", "pools", "evaluation_cache", "snapshots"):
            path = run.work / name
            if path.exists():
                shutil.rmtree(path)
    return metrics


def run_backtest(
    config_path: Path,
    *,
    output: Path | None = None,
    work: Path | None = None,
    log_file: Path | None = None,
    show_progress: bool = True,
    stop_after_selection: bool = False,
) -> dict[str, Any]:
    source = load_backtest_config(config_path)
    output = output or Path("artifacts") / source["run_id"]
    work = work or Path("artifacts") / f".{source['run_id']}.work"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a completed artifact: {output}")
    protected = [
        Path("data").resolve(),
        Path(source["paths"]["task07_dataset"]).resolve(),
        Path(source["paths"]["task08_model"]).resolve().parent,
    ]
    for destination in (work.resolve(), output.resolve()):
        if any(
            destination == p
            or destination.is_relative_to(p)
            or p.is_relative_to(destination)
            for p in protected
        ):
            raise ValueError("run directories overlap immutable inputs")
    if (
        work.resolve() == output.resolve()
        or work.resolve().is_relative_to(output.resolve())
        or output.resolve().is_relative_to(work.resolve())
    ):
        raise ValueError("work and output directories must be separate")
    if (
        work.exists()
        and any(work.iterdir())
        and not (work / "checkpoint.json").is_file()
    ):
        raise ValueError("existing work directory lacks run ownership metadata")
    work.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_file or Path("logs") / f"{source['run_id']}.log"
    reporter = EventProgressReporter(
        task_name=source["run_id"],
        total_phases=8,
        log_file=log_file,
        show_progress=show_progress,
        log_max_bytes=5_000_000,
        log_backup_count=3,
    )
    started = time.perf_counter()
    run: BacktestRun | None = None
    try:
        reporter.event("run_start", status="running", mode=source["mode"])
        _preflight(source, work, reporter)
        run = BacktestRun(
            _resolved_config(source), output=output, work=work, reporter=reporter
        )
        with run.phase("rolling_inputs"):
            run.load_folds(["rolling_1", "rolling_2", "rolling_3"])
            validate_temporal_training_scope(
                ["rolling_1", "rolling_2"],
                "rolling_3",
                {k: v.manifest["cutoff"] for k, v in run.contexts.items()},
            )
            context = run.contexts["rolling_3"]
            count = min(source["selection"]["user_count"], context.target_users.height)
            users = select_evaluation_users(
                context.target_users, count=count, seed=source["seed"]
            )
            evaluation_parts = run.evaluation_cache(context, users)
        with run.phase("multi_fold_training_rows"):
            selection_parts, selection_training = run.training_data(
                "selection", ["rolling_1", "rolling_2"]
            )
        curves = {}
        with run.phase("rolling_comparison"):
            stage = reporter.stage_start(
                stage="rolling_comparison", total=3, unit="config"
            )
            baseline = CatBoostPointwiseModel.from_artifact(
                Path(source["paths"]["task08_model"])
            )
            curves["task08_baseline"] = run.score_curve(
                baseline,
                config_id="task08_baseline",
                parts=evaluation_parts,
                users=users,
                ground_truth=context.ground_truth,
                tree_counts=[baseline.tree_count],
            )
            del baseline
            reporter.stage_advance()
            for order, policy in enumerate(QUANTIZATION_POLICIES):
                pool, _ = run.training_pool("selection", policy, selection_parts)
                model, _ = run.fit(
                    "selection",
                    policy,
                    pool,
                    tree_count=source["catboost"]["iterations"],
                )
                curves[policy] = run.score_curve(
                    model,
                    config_id=policy,
                    parts=evaluation_parts,
                    users=users,
                    ground_truth=context.ground_truth,
                    tree_counts=source["selection"]["tree_counts"],
                )
                best_point = select_tree_checkpoint(curves[policy])

                def save_prefix(
                    destination: Path,
                    fitted_model: CatBoostPointwiseModel = model,
                    point: dict[str, Any] = best_point,
                ) -> dict[str, Any]:
                    fitted_model.with_tree_count(point["tree_count"]).save(
                        destination / "model"
                    )
                    write_json_atomic(destination / "selection.json", point)
                    return point

                selected_dir = run.work / "selected_models" / policy
                run.operation(
                    "select_prefix", policy, "rolling_3", selected_dir, save_prefix
                )
                run.best.update(
                    score=(
                        best_point["precision_at_20_all_targets"],
                        -best_point["tree_count"],
                        -order,
                    ),
                    payload={
                        "policy": policy,
                        "tree_count": best_point["tree_count"],
                        "selection_metrics": best_point,
                        "model_path": str(selected_dir / "model"),
                        "model_sha256": sha256_file(selected_dir / "model/model.cbm"),
                    },
                )
                reporter.event(
                    "config_finish",
                    stage="rolling_comparison",
                    config=policy,
                    fold="rolling_3",
                    current_metric=best_point["precision_at_20_all_targets"],
                    best_metric=run.best.read()["score"][0],
                    best_config=run.best.read()["payload"]["policy"],
                    tree_count=best_point["tree_count"],
                )
                del model, save_prefix
                gc.collect()
                reporter.stage_advance()
            reporter.stage_finish(stage="rolling_comparison", started=stage)
        with run.phase("freeze_winner"):
            payload = run.best.read()["payload"]
            winner = {
                key: value for key, value in payload.items() if key != "model_path"
            }
            winner.update(
                selection_fold="rolling_3",
                selection_users=users.height,
                selection_user_sampling="ID_only_complete_candidate_lists",
                canonical_used_for_selection=False,
                selection_beats_task08=winner["selection_metrics"]["final_hits"]
                > curves["task08_baseline"][0]["final_hits"],
            )
            frozen_path = run.work / "winner.json"
            if frozen_path.exists() and read_json(frozen_path) != winner:
                raise ContractValidationError("frozen winner changed during resume")
            write_json_atomic(frozen_path, winner)
            write_json_atomic(run.work / "selection_curves.json", {"curves": curves})
            reporter.event(
                "winner_frozen",
                stage="freeze_winner",
                config=winner["policy"],
                fold="rolling_3",
                current_metric=winner["selection_metrics"][
                    "precision_at_20_all_targets"
                ],
                tree_count=winner["tree_count"],
            )
        if stop_after_selection:
            reporter.event(
                "run_finish",
                status="paused_after_selection",
                runtime_seconds=time.perf_counter() - started,
            )
            return {
                "status": "paused_after_selection",
                "work_dir": str(work),
                "winner": winner,
            }
        with run.phase("rolling_full_report"):
            selected_model = CatBoostPointwiseModel.from_artifact(
                run.work / "selected_models" / winner["policy"] / "model"
            )
            rolling, rolling_root = run.full_evaluation(
                selected_model, context=context, config_id=winner["policy"]
            )
            del selected_model
            gc.collect()
        with run.phase("canonical_refit"):
            # Only now open canonical targets/GT and features; the decision is immutable.
            run.load_folds(["canonical"])
            validate_temporal_training_scope(
                ["rolling_1", "rolling_2", "rolling_3"],
                "canonical",
                {k: v.manifest["cutoff"] for k, v in run.contexts.items()},
            )
            canonical_parts, canonical_training = run.training_data(
                "canonical", ["rolling_1", "rolling_2", "rolling_3"]
            )
            pool, _ = run.training_pool("canonical", winner["policy"], canonical_parts)
            model, model_dir = run.fit(
                "canonical", winner["policy"], pool, tree_count=winner["tree_count"]
            )
        with run.phase("canonical_once"):
            canonical, canonical_root = run.full_evaluation(
                model, context=run.contexts["canonical"], config_id=winner["policy"]
            )
            del model
            gc.collect()
        with run.phase("publish_and_verify"):
            metrics = _publish(
                run,
                winner=winner,
                model_dir=model_dir,
                canonical=canonical,
                canonical_root=canonical_root,
                rolling=rolling,
                rolling_root=rolling_root,
                curves=curves,
                training={
                    "selection": selection_training,
                    "canonical": canonical_training,
                },
            )
        reporter.event(
            "run_finish",
            status="completed",
            current_metric=metrics["precision_at_20_all_targets"],
            best_config=winner["policy"],
            runtime_seconds=time.perf_counter() - started,
        )
        return metrics
    except BaseException as error:
        reporter.event(
            "run_finish",
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            runtime_seconds=time.perf_counter() - started,
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task12_ranker_backtest_v1.json")
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="Verify a published artifact and deterministic portable inference; no fit.",
    )
    parser.add_argument(
        "--stop-after-selection",
        action="store_true",
        help="Pause after freezing the rolling winner; rerun without this flag to resume.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_artifact(args.verify_only), indent=2))
        return
    Path("logs").mkdir(exist_ok=True)
    # One lock also covers direct CLI calls with different run IDs on this GPU.
    with Path("logs/task12_ranker_backtest.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "another Task 12 process already owns the run lock"
            ) from error

        def interrupt(signum, frame):
            raise KeyboardInterrupt(
                f"received signal {signum}; rerun the same config to resume"
            )

        signal.signal(signal.SIGTERM, interrupt)
        result = run_backtest(
            args.config,
            show_progress=not args.no_progress,
            stop_after_selection=args.stop_after_selection,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

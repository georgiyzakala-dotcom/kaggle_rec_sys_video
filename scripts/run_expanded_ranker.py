#!/usr/bin/env python3
"""Validate/refit CatBoost on SASRec600/ALS600 candidates and build a submission."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import importlib.metadata
import json
import os
import re
import shutil
import signal
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from expanded_ranker import (
    CAPS,
    EVALUATION_FOLDS,
    EXTRA_FEATURES,
    PHASE_FOLDS,
    cross_scored_union,
    materialize_features,
    sample_union,
    validate_phase_horizon,
)
from experiment_utils import (
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
    read_json,
    sha256_file,
    write_json_atomic,
)
from history_profiles import HistoryProfiles, history_pairs, normalize_factors
from implicit_model import ImplicitALSDataLoader, ImplicitALSModel
from item2item import Item2ItemConfig, Item2ItemDataLoader, Item2ItemModel
from metrics import evaluate_precision_at_20
from pipeline import (
    SOURCE_ORDER,
    _recency_grids,
    load_feature_lookups,
    recommendations_with_fallback,
)
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
)
from ranker_backtest import allocate_training_probabilities
from rankers import CatBoostPointwiseConfig, CatBoostPointwiseModel
from sasrec_data import SASRecDataLoader, SequenceStore, prepare_sequence_store
from sasrec_model import SASRecCandidateModel
from sasrec_selection import validate_unseen
from sasrec_top300 import predict_vectorized_seen
from scripts.run_profile_full_fit import FullProfileRun, score
from scripts.run_ranker_backtest import (
    BacktestRun,
    _file_manifest,
    _verify_files,
    validate_output_semantics,
)
from scripts.run_sasrec_benchmark import BenchmarkPaused, configure_torch
from scripts.run_sasrec_folds import FoldsRunner, read_users
from scripts.run_sasrec_optuna import verify_hashes
from scripts.task13_resources import supervise
from submission import read_submission, submission_schema, write_submission

FOLDS = tuple(PHASE_FOLDS["final"])
IMPLEMENTATION = (
    "expanded_ranker.py",
    "scripts/run_expanded_ranker.py",
    "scripts/run_ranker_backtest.py",
    "scripts/run_profile_full_fit.py",
    "scripts/run_history_profiles.py",
    "ranker_backtest.py",
    "rankers.py",
    "ranker_data.py",
    "candidate_pipeline.py",
    "features.py",
    "history_profiles.py",
    "full_history_profiles.py",
    "pipeline.py",
    "submission.py",
    "validation.py",
    "metrics.py",
    "sasrec_model.py",
    "sasrec_data.py",
    "sasrec_top300.py",
    "sasrec_selection.py",
    "scripts/run_sasrec_folds.py",
    "scripts/run_sasrec_benchmark.py",
    "scripts/run_sasrec_optuna.py",
    "scripts/task13_resources.py",
    "experiment_utils.py",
    "implicit_model.py",
    "item2item.py",
    "popularity.py",
    "interfaces.py",
    "data_utils.py",
    "scripts/run_catboost_ranker.py",
)


def load_config(path):
    c = read_json(path)
    if c["kind"] != "expanded_sasrec_ranker_submission" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid kind/run ID")
    if (
        c["mode"] not in ("smoke", "full")
        or c["candidate_caps"] != CAPS
        or c["seed"] != 42
    ):
        raise ValueError("invalid mode, frozen candidate caps or seed")
    if (
        c["catboost"]["random_seed"] != c["seed"]
        or not 1 <= c["catboost"]["thread_count"] <= 8
    ):
        raise ValueError("invalid random seed or CPU limit")
    if (
        not 0 < c["catboost"]["gpu_ram_part"] <= 0.7
        or not 0 < c["resources"]["maximum_rss_gib"] <= 40
    ):
        raise ValueError("GPU/RAM ceiling exceeded")
    if (
        not 1 <= c["inference"]["users_per_shard"] <= 512
        or c["inference"]["final_k"] != 20
    ):
        raise ValueError("use bounded user shards and exact top20")
    if c["mode"] == "full":
        if (
            c["smoke"]
            or c["catboost"]["task_type"] != "GPU"
            or c["training"]["target_rows"] != 30_000_000
        ):
            raise ValueError(
                "full run requires GPU, all users and the validated 30M row budget"
            )
    elif not (
        1 <= c["smoke"]["user_count"] <= 64
        and 1 <= c["smoke"]["sasrec_epochs"] <= 2
        and 1 <= c["smoke"]["sasrec_training_users"] <= 512
        and 1 <= c["catboost"]["iterations"] <= 30
        and 1 <= c["training"]["target_rows"] <= 100_000
    ):
        raise ValueError("smoke must be explicitly bounded")
    CatBoostPointwiseConfig.from_mapping(c["catboost"])
    return c


def resolve_config(c):
    c = json.loads(json.dumps(c))
    deep = Path(c["paths"]["top600"])
    full = Path(c["paths"]["full_reference"])
    ref = Path(c["paths"]["ranker_reference"])
    for name, directory, filename in (
        ("top600", deep, "manifest.json"),
        ("full_reference", full, "manifest.json"),
        ("ranker_reference", ref, "artifact_manifest.json"),
    ):
        if sha256_file(directory / filename) != c["pinned_manifests"][name]:
            raise ValueError(f"pinned artifact changed: {name}")
    d = read_json(deep / "config.json")
    c["sasrec_parent"] = {
        "path": d["parent_artifact"],
        "manifest_sha256": d["parent_manifest_sha256"],
    }
    c["parent_users_per_shard"] = d["evaluation"]["users_per_shard"]
    if c["parent_users_per_shard"] % c["inference"]["users_per_shard"]:
        raise ValueError("user shard size must divide the frozen prediction shard size")
    old = read_json(full / "config.json")
    reference = read_json(ref / "model/model_config.json")
    if read_json(deep / "metrics.json")["mode"] != "full" or old["mode"] != "full":
        raise ValueError("full input artifacts required")
    if read_json(full / "submission_schema.json") != submission_schema():
        raise ValueError("submission serialization differs from the established format")
    c["feature_columns"] = reference["feature_columns"] + list(EXTRA_FEATURES)
    if len(c["feature_columns"]) != len(set(c["feature_columns"])):
        raise ValueError("duplicate feature names")
    frozen = CatBoostPointwiseConfig.from_mapping(reference["catboost"]).to_dict()
    actual = CatBoostPointwiseConfig.from_mapping(c["catboost"]).to_dict()
    allowed = {
        "config_id",
        "thread_count",
        "devices",
        "metric_period",
        "snapshot_interval_seconds",
        "gpu_ram_part",
    }
    if c["mode"] == "smoke":
        allowed |= {"iterations", "depth", "task_type"}
    if any(actual[k] != frozen[k] for k in frozen.keys() - allowed):
        raise ValueError("keep the validated Task13 statistical recipe")
    c["prediction_times"] = old["prediction_times"]
    c["original_raw_sha256"] = old["input_sha256"]
    c["fold_specs"] = {}
    for f in d["folds"]:
        name = f["name"]
        info = d["input"][name]
        lookup = Path(c["paths"]["task07_dataset"]) / "folds" / name / "lookups"
        lm = read_json(lookup / "lookup_manifest.json")
        if lm["history_sha256"] != info["history_sha256"]:
            raise ValueError(f"history/lookup mismatch: {name}")
        models = {}
        for source in SOURCE_ORDER:
            root = Path(info["source_root"]) / "sources" / source
            meta = read_json(root / "metadata.json")
            models[source] = str(
                root / meta["model_path"]
                if meta["model_storage"] == "embedded"
                else Path(meta["model_path"])
            )
        profile = Path(c["paths"]["profile_cache_root"]) / name
        c["fold_specs"][name] = {
            "history": str(Path(f["artifact"]) / "history_daily.parquet"),
            "history_sha256": info["history_sha256"],
            "cutoff": info["split"]["cutoff"],
            "source_root": info["source_root"],
            "models": models,
            "sasrec_model": str(
                Path(d["parent_artifact"]) / "folds" / name / "trained/model"
            ),
            "sequence_cache": d["sequence_caches"][name],
            "lookup": str(lookup),
            "lookup_manifest_sha256": sha256_file(lookup / "lookup_manifest.json"),
            "profile_cache": str(profile) if profile.exists() else None,
            "profile_manifest_sha256": sha256_file(profile / "operation.json")
            if profile.exists()
            else None,
            "frozen_input_sha256": info["file_sha256"],
            "cross_score_model_sha256": {
                str(file): sha256_file(file)
                for source in SOURCE_ORDER[:3]
                for file in Path(models[source]).iterdir()
                if file.name
                in (
                    "model_config.json",
                    "config.json",
                    "item_ranking.parquet",
                    "neighbor_table.parquet",
                )
            },
        }
    c["recipe"] = d["recipe"]
    c["sasrec_runtime"] = {
        "device": "cuda" if c["catboost"]["task_type"] == "GPU" else "cpu",
        "gpu_id": int(c["catboost"]["devices"]),
        "cpu_threads": c["catboost"]["thread_count"],
        "seed": c["seed"],
        "precision": "bfloat16" if c["catboost"]["task_type"] == "GPU" else "float32",
        "resources": {"cuda_memory_fraction": 0.70, "epoch_arrays_gib": 8},
        "recipe": d["recipe"],
        "training": dict(d["recipe"]["training"]),
        "reuse_selection_fold_model": False,
        "selection_artifact": None,
    }
    if c["mode"] == "smoke":
        c["sasrec_runtime"]["training"].update(
            epochs=c["smoke"]["sasrec_epochs"],
            max_users=c["smoke"]["sasrec_training_users"],
        )
    for phase in PHASE_FOLDS:
        validate_phase_horizon(
            phase,
            c["fold_specs"],
            datetime.fromisoformat(c["prediction_times"]["prediction_start"]),
        )
    c["implementation_sha256"] = {p: sha256_file(ROOT / p) for p in IMPLEMENTATION}
    c["library_versions"] = {
        p: importlib.metadata.version(p)
        for p in ("torch", "catboost", "polars", "numpy", "scipy", "implicit")
    }
    c["reference_metrics"] = read_json(ref / "metrics.json")
    return c


def load_models(paths):
    models = {}
    for name, root in paths.items():
        root = Path(root)
        meta = (
            read_json(root / "model_config.json")
            if (root / "model_config.json").exists()
            else read_json(root / "config.json")["model_config"]
        )
        if name == "global_popularity":
            models[name] = GlobalPopularityModel.from_fitted_ranking(
                PopularityScore(meta["score_type"]),
                pl.read_parquet(root / "item_ranking.parquet"),
            )
        elif name == "recency_popularity":
            models[name] = RecencyPopularityModel.from_fitted_ranking(
                RecencyPopularityConfig.from_dict(meta["recency_config"]),
                pl.read_parquet(root / "item_ranking.parquet"),
            )
        elif name == "item2item":
            models[name] = Item2ItemModel.from_fitted_neighbors(
                Item2ItemConfig.from_dict(meta["item2item_config"]),
                pl.read_parquet(root / "neighbor_table.parquet"),
            )
        else:
            models[name] = ImplicitALSModel.from_artifact(root)
    return models


class ProductionSAS(FoldsRunner):
    """Reuse the tested atomic epoch/RNG training implementation with the outer run log."""

    def __init__(self, run):
        self.c, self.work, self.digest = (
            run.config["sasrec_runtime"],
            run.work / "sasrec",
            run.digest,
        )
        self.work.mkdir(exist_ok=True)
        self.reporter, self.show_progress, self.fold = (
            run.reporter,
            run.show_progress,
            "full_history",
        )
        self.check_stop, self.save_timing = run.check_stop, run.save_timing
        self.stop_after_epoch = None


class ExpandedRun(BacktestRun):
    heartbeat = FullProfileRun.heartbeat

    def __init__(self, c, *, show_progress=True):
        output, work = (
            Path("artifacts") / c["run_id"],
            Path("artifacts") / f".{c['run_id']}.work",
        )
        if output.exists():
            raise FileExistsError(output)
        work.mkdir(parents=True, exist_ok=True)
        reporter = EventProgressReporter(
            task_name=c["run_id"],
            total_phases=7,
            log_file=f"logs/{c['run_id']}.log",
            show_progress=show_progress,
        )
        super().__init__(c, output=output, work=work, reporter=reporter)
        self.features = tuple(c["feature_columns"])
        self.show_progress, self.stopped = show_progress, False
        self.handlers = {
            sig: signal.signal(sig, self.interrupt)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        self.all_users = pl.read_parquet(c["paths"]["target_users"]).sort("user_id")
        self.users = (
            self.all_users.head(c["smoke"]["user_count"])
            if c["mode"] == "smoke"
            else self.all_users
        )

    def interrupt(self, signum, frame):
        self.stopped = True

    def check_stop(self):
        if self.stopped:
            raise BenchmarkPaused("stop requested; repeat the same command to resume")

    def save_timing(self):
        write_json_atomic(
            self.work / "timing.json",
            {
                "active_runtime_seconds": self.active_runtime_seconds,
                "peak_memory_mb": self.peak_memory_mb,
            },
        )

    def check_resources(self, stage):
        self.check_stop()
        super().check_resources(stage)

    def preflight(self):
        def build(path):
            deep = Path(self.config["paths"]["top600"])
            verify_hashes(deep)
            parent = Path(self.config["sasrec_parent"]["path"])
            if (
                sha256_file(parent / "manifest.json")
                != self.config["sasrec_parent"]["manifest_sha256"]
            ):
                raise ValueError("SASRec parent manifest changed")
            verify_hashes(parent)
            full = Path(self.config["paths"]["full_reference"])
            _verify_files(full, read_json(full / "manifest.json")["files"])
            ref = Path(self.config["paths"]["ranker_reference"])
            _verify_files(ref, read_json(ref / "artifact_manifest.json")["files"])
            for key in ("train", "target_users"):
                if (
                    sha256_file(self.config["paths"][key])
                    != self.config["original_raw_sha256"][key]
                ):
                    raise ValueError(
                        f"raw input differs from full-history reference: {key}"
                    )
            for name, spec in self.config["fold_specs"].items():
                self.reporter.stage_status(
                    stage="preflight",
                    config="expanded",
                    fold=name,
                    operation="hash_inputs",
                )
                for file, digest in spec["frozen_input_sha256"].items():
                    self.check_stop()
                    if sha256_file(file) != digest:
                        raise ValueError(f"frozen input changed: {file}")
                for file, digest in spec["cross_score_model_sha256"].items():
                    if sha256_file(file) != digest:
                        raise ValueError(f"cross-score model changed: {file}")
                lookup = Path(spec["lookup"])
                if (
                    sha256_file(lookup / "lookup_manifest.json")
                    != spec["lookup_manifest_sha256"]
                ):
                    raise ValueError("lookup manifest changed")
                lm = read_json(lookup / "lookup_manifest.json")
                for filename, digest in lm["files"].items():
                    if sha256_file(lookup / filename) != digest:
                        raise ValueError("lookup checksum differs")
                if (
                    sha256_file(
                        Path(spec["models"]["item2item"]) / "neighbor_table.parquet"
                    )
                    != lm["input_models"]["sha256"]["item2item_neighbor_table"]
                ):
                    raise ValueError("co-visitation model differs from the frozen fold")
                if spec["profile_cache"]:
                    p = Path(spec["profile_cache"])
                    if (
                        sha256_file(p / "operation.json")
                        != spec["profile_manifest_sha256"]
                    ):
                        raise ValueError("profile manifest changed")
                    manifest = read_json(p / "operation.json")
                    if (
                        manifest["result"]["input"]["history_sha256"]
                        != spec["history_sha256"]
                    ):
                        raise ValueError("profile cache belongs to another history")
                    _verify_files(p, manifest["files"])
            result = {
                "input_artifacts_verified": True,
                "target_users": self.users.height,
                "feature_count": len(self.features),
                "candidate_caps": CAPS,
            }
            write_json_atomic(path / "inputs.json", result)
            return result

        return self.operation(
            "preflight", "expanded", "all", self.work / "inputs", build
        )

    def load_context(self, fold):
        spec = self.config["fold_specs"][fold]
        models = load_models(spec["models"])
        store = SequenceStore.load(Path(spec["sequence_cache"]))
        if store.metadata["history_sha256"] != spec["history_sha256"]:
            raise ValueError("sequence cache belongs to another fold")
        configure_torch(self.config["sasrec_runtime"])
        sasrec = SASRecCandidateModel.from_artifact(
            Path(spec["sasrec_model"]), device=self.config["sasrec_runtime"]["device"]
        )
        if spec["profile_cache"]:
            profiles = HistoryProfiles.load(Path(spec["profile_cache"]))
        else:
            destination = self.work / "profiles" / fold

            def build(path):
                pairs = history_pairs(pl.scan_parquet(spec["history"]), self.users)
                result = HistoryProfiles.build(
                    pairs,
                    self.users,
                    models["implicit_als"].item_mapping,
                    normalize_factors(models["implicit_als"].backend.item_factors),
                )
                result.save(path)
                return {
                    "history_sha256": spec["history_sha256"],
                    "users": self.users.height,
                }

            self.operation("profiles", "expanded", fold, destination, build)
            profiles = HistoryProfiles.load(destination)
        parent = Path(
            read_json(Path(self.config["paths"]["top600"]) / "config.json")[
                "parent_artifact"
            ]
        )
        truth = pl.read_parquet(
            parent / "folds" / fold / "target_ground_truth.parquet"
        ).join(self.users, on="user_id", how="semi")
        return SimpleNamespace(
            fold=fold,
            spec=spec,
            models=models,
            sasrec=sasrec,
            store=store,
            profiles=profiles,
            lookups=load_feature_lookups(Path(spec["lookup"])),
            truth=truth,
            cutoff=datetime.fromisoformat(spec["cutoff"]),
        )

    def release(self, context=None):
        gc.collect()
        if self.config["sasrec_runtime"]["device"] == "cuda":
            torch.cuda.empty_cache()

    def read_sources(self, context, users):
        root = Path(self.config["paths"]["top600"]) / "folds" / context.fold / "parts"
        # Parent shards and IDs are in the same sorted target order for every fold.
        start = int(
            np.searchsorted(self.all_users["user_id"].to_numpy(), users["user_id"][0])
        )
        index = start // self.config["parent_users_per_shard"]
        directory = root / f"part-{index:05d}"
        if getattr(context, "part_index", None) != index:
            part_users = self.all_users.slice(
                index * self.config["parent_users_per_shard"],
                self.config["parent_users_per_shard"],
            )
            context.part_sources = {
                name: read_users(
                    Path(context.spec["source_root"])
                    / "sources"
                    / name
                    / "candidates.parquet",
                    part_users,
                )
                for name in SOURCE_ORDER[:3]
            }
            for name, filename in (
                ("implicit_als", "als600.parquet"),
                ("sasrec", "sasrec600.parquet"),
            ):
                context.part_sources[name] = pl.read_parquet(directory / filename)
            context.part_history = read_users(Path(context.spec["history"]), part_users)
            context.part_index = index
        return {
            name: frame.join(users, on="user_id", how="semi")
            for name, frame in context.part_sources.items()
        }

    def feature_query(self, context, users, *, probability=None):
        self.check_stop()
        sources = self.read_sources(context, users)
        history = context.part_history.join(users, on="user_id", how="semi")
        union, seeds = cross_scored_union(
            sources,
            history=history,
            users=users,
            models=context.models,
            sasrec=context.sasrec,
            store=context.store,
            cutoff=context.cutoff,
            seed=self.config["seed"],
        )
        validate_unseen(union.select("user_id", "item_id"), users, context.store)
        truth = context.truth.join(users, on="user_id", how="semi")
        selected = (
            union
            if probability is None
            else sample_union(
                union,
                truth,
                fold=context.fold,
                seed=self.config["seed"],
                negative_probability=probability,
            )
        )
        frame = materialize_features(
            selected,
            seeds=seeds,
            lookups=context.lookups,
            models=context.models,
            profiles=context.profiles,
            cutoff=context.cutoff,
            features=self.features,
        )
        return frame, union

    def batches(self):
        size = self.config["inference"]["users_per_shard"]
        return list(self.users.iter_slices(size))

    def counts(self, fold):
        root = Path(self.config["paths"]["top600"])
        stats = pl.read_parquet(root / "folds" / fold / "per_user.parquet").join(
            self.users, on="user_id", how="semi"
        )
        key = "m__als600_sasrec600_1800__"
        return {
            "rows": int(stats[key + "count"].sum()),
            "positives": int(stats[key + "hits"].sum()),
        }

    def samples(self, phase):
        counts = {f: self.counts(f) for f in PHASE_FOLDS[phase]}
        allocation = allocate_training_probabilities(
            counts, target_rows=self.config["training"]["target_rows"]
        )
        parts, summaries = [], {}
        for fold in PHASE_FOLDS[phase]:
            context = None
            total = {"rows": 0, "positives": 0, "candidate_rows": 0}
            try:
                with self.units(
                    "training_features", phase, fold, len(self.batches()), "shard"
                ) as advance:
                    for index, users in enumerate(self.batches()):
                        destination = (
                            self.work
                            / "training_data"
                            / phase
                            / fold
                            / f"part-{index:05d}"
                        )
                        if not destination.exists() and context is None:
                            with self.heartbeat("training_features", f"load_{fold}"):
                                context = self.load_context(fold)

                        def build(path, context=context, users=users, fold=fold):
                            frame, union = self.feature_query(
                                context,
                                users,
                                probability=allocation[fold][
                                    "secondary_negative_probability"
                                ],
                            )
                            frame.write_parquet(
                                path / "features.parquet", compression="zstd"
                            )
                            return {
                                "rows": frame.height,
                                "positives": int(frame["label"].sum()),
                                "candidate_rows": union.height,
                            }

                        result = self.operation(
                            "training_features",
                            phase,
                            f"{fold}:{index}",
                            destination,
                            build,
                        )
                        del build
                        for key in total:
                            total[key] += result[key]
                        if result["rows"]:
                            parts.append(destination / "features.parquet")
                        advance(index, None, result)
                if (
                    total["positives"] != counts[fold]["positives"]
                    or total["candidate_rows"] != counts[fold]["rows"]
                ):
                    raise ValueError(
                        f"candidate union differs from verified top600 counts: {fold}"
                    )
                summaries[fold] = {**allocation[fold], **total}
            finally:
                del context
                self.release()
        return parts, {
            "folds": summaries,
            "rows": sum(v["rows"] for v in summaries.values()),
            "sampler": "all_positive_hits_plus_uniform_negatives",
            "weight": "1/inclusion_probability",
            "ranks_computed_before_sampling": True,
            "labels_used_for_candidate_fit": False,
        }

    def evaluate(self, phase, model):
        fold = EVALUATION_FOLDS[phase]
        context = None
        parts, counts = [], {"candidate_rows": 0, "candidate_hits": 0}
        try:
            with self.units(
                "evaluation", phase, fold, len(self.batches()), "shard"
            ) as advance:
                for index, users in enumerate(self.batches()):
                    destination = self.work / "evaluation" / phase / f"part-{index:05d}"
                    if not destination.exists() and context is None:
                        context = self.load_context(fold)

                    def build(path, context=context, users=users):
                        frame, union = self.feature_query(context, users)
                        scores = score(
                            model, frame, self.config["inference"]["batch_size"]
                        )
                        recs, _, _ = recommendations_with_fallback(
                            scores, union=union, target_users=users
                        )
                        recs.write_parquet(path / "recommendations.parquet")
                        truth = context.truth.join(users, on="user_id", how="semi")
                        return {
                            "candidate_rows": union.height,
                            "candidate_hits": union.join(
                                truth, on=["user_id", "item_id"], how="semi"
                            ).height,
                        }

                    result = self.operation(
                        "evaluation", phase, f"{fold}:{index}", destination, build
                    )
                    del build
                    for key in counts:
                        counts[key] += result[key]
                    parts.append(destination / "recommendations.parquet")
                    advance(index, None, result)
            spec = self.config["fold_specs"][fold]
            parent = Path(
                read_json(Path(self.config["paths"]["top600"]) / "config.json")[
                    "parent_artifact"
                ]
            )
            truth = read_users(
                parent / "folds" / fold / "target_ground_truth.parquet", self.users
            )
            recs = pl.concat([pl.read_parquet(p) for p in parts]).sort("user_id")
            validation = validate_output_semantics(
                recs,
                context=SimpleNamespace(
                    target_users=self.users, history_path=Path(spec["history"])
                ),
            )
            expected = self.counts(fold)
            if counts != {
                "candidate_rows": expected["rows"],
                "candidate_hits": expected["positives"],
            }:
                raise ValueError("evaluation union differs from verified top600 counts")
            metrics = {
                **evaluate_precision_at_20(recs, truth, self.users),
                **counts,
                "fold": fold,
                "target_users": self.users.height,
                "labeled_users": truth["user_id"].n_unique(),
                "candidate_recall": counts["candidate_hits"] / truth.height
                if truth.height
                else 0.0,
                "mean_candidate_count": counts["candidate_rows"] / self.users.height,
                "coverage": 1.0,
                "validation": validation,
                "training_folds": list(PHASE_FOLDS[phase]),
            }
            metrics["final_hits"] = (
                recs.explode("item_ids", empty_as_null=True)
                .rename({"item_ids": "item_id"})
                .join(truth, on=["user_id", "item_id"], how="semi")
                .height
            )
            reference = self.config["reference_metrics"]
            if fold == "rolling_3":
                reference = reference["selection_comparisons"]["D_all"]
            metrics["reference_comparable"] = self.config["mode"] == "full"
            metrics["reference_p20_all_targets"] = reference[
                "precision_at_20_all_targets"
            ]
            metrics["delta_p20_all_targets"] = (
                metrics["precision_at_20_all_targets"]
                - reference["precision_at_20_all_targets"]
                if metrics["reference_comparable"]
                else None
            )
            self.reporter.event(
                "validation_result",
                stage="evaluation",
                config=phase,
                fold=fold,
                operation="full_queries",
                current_metric=metrics["precision_at_20_all_targets"],
                best_metric=reference["precision_at_20_all_targets"],
                **counts,
            )
            return recs, metrics
        finally:
            del context
            self.release()

    def cleanup_phase(self, phase):
        # Called only after a portable, checksummed phase result is committed.
        for name in ("training_data", "pools", "models", "snapshots", "evaluation"):
            path = self.work / name / phase
            if path.exists():
                shutil.rmtree(path)

    def ranker_phase(self, phase):
        complete = self.work / "completed" / phase

        def build(path):
            parts, training = self.samples(phase)
            with self.heartbeat("quantization", phase):
                pool, quantization = self.training_pool(phase, "fit_training", parts)
            model, model_dir = self.fit(
                phase,
                "fit_training",
                pool,
                tree_count=self.config["catboost"]["iterations"],
            )
            # Reload a saved model and compare predictions on unquantized features.
            probe = (
                pl.read_parquet(parts[0])
                .head(256)
                .select("user_id", "item_id", *self.features)
            )
            scores = score(model, probe, 256)
            repeated = score(
                CatBoostPointwiseModel.from_artifact(model_dir), probe, 256
            )
            if not scores.equals(repeated):
                raise ValueError("portable ranker predictions differ")
            probe.write_parquet(path / "verification_features.parquet")
            scores.write_parquet(path / "verification_scores.parquet")
            metrics = None
            if phase in EVALUATION_FOLDS:
                recs, metrics = self.evaluate(phase, model)
                recs.write_parquet(path / "recommendations.parquet")
                fold = EVALUATION_FOLDS[phase]
                truth_path = (
                    Path(self.config["sasrec_parent"]["path"])
                    / "folds"
                    / fold
                    / "target_ground_truth.parquet"
                )
                read_users(truth_path, self.users).write_parquet(
                    path / "target_ground_truth.parquet"
                )
            shutil.copytree(model_dir, path / "model", copy_function=os.link)
            os.link(pool.parent / "borders.tsv", path / "borders.tsv")
            write_json_atomic(
                path / "training.json",
                {"sampling": training, "quantization": quantization},
            )
            write_json_atomic(
                path / "metrics.json", metrics or {"future_labels_available": False}
            )
            return {
                "phase": phase,
                "training_rows": training["rows"],
                "metrics": metrics,
            }

        result = self.operation("completed_phase", "expanded", phase, complete, build)
        self.cleanup_phase(phase)
        self.release()
        return result

    def production_sasrec(self):
        full = Path(self.config["paths"]["full_reference"])
        digest = read_json(full / "manifest.json")["files"]["history_daily.parquet"][
            "sha256"
        ]
        cutoff = datetime.fromisoformat(
            self.config["prediction_times"]["prediction_start"]
        )

        def build(path):
            prepare_sequence_store(
                full / "history_daily.parquet",
                path / "store",
                cutoff=cutoff,
                expected_sha256=digest,
                seed=self.config["seed"],
                event=lambda name, **kw: self.reporter.event(
                    name,
                    stage="sequences",
                    config="expanded",
                    fold="full_history",
                    **kw,
                ),
            )
            return {"history_sha256": digest}

        self.operation(
            "sequences", "expanded", "full_history", self.work / "full_sequences", build
        )
        store = SequenceStore.load(self.work / "full_sequences/store")
        runner = ProductionSAS(self)
        model = runner.train({"name": "full_history"}, runner.work, store)
        if model.history_sha256 != digest:
            raise ValueError("production SASRec was not fitted on complete raw history")
        del model, store, runner
        self.release()

    def production_sources(self, context, users, history):
        seed, models = self.config["seed"], context.models
        windows, half_lives = _recency_grids(models["recency_popularity"].config)
        item = models["item2item"].config
        als = models["implicit_als"]
        loaders = {
            "global_popularity": PopularityDataLoader(seed=seed),
            "recency_popularity": RecencyPopularityDataLoader(
                reference_time=context.cutoff,
                windows_hours=windows,
                half_lives_hours=half_lives,
                seed=seed,
            ),
            "item2item": Item2ItemDataLoader(
                reference_time=context.cutoff,
                max_history_items=item.history_cap,
                max_seed_items=item.seed_k,
                seed=seed,
            ),
            "implicit_als": ImplicitALSDataLoader(
                config=als.config,
                reference_time=context.cutoff,
                user_mapping=als.user_mapping,
                item_mapping=als.item_mapping,
                seed=seed,
            ),
        }
        sources = {}
        for name, loader in loaders.items():
            self.check_stop()
            loader.load_predict_data(
                history=history, target_users=users
            ).prepare_predict_data()
            sources[name] = models[name].predict(loader, k=CAPS[name], batch_size=128)
        loader = SASRecDataLoader(
            context.store, max_length=context.sasrec.config.max_length, seed=seed
        )
        loader.load_predict_data(
            user_ids=users["user_id"].to_numpy()
        ).prepare_predict_data()
        with tqdm(
            total=(users.height + 63) // 64,
            desc="SASRec600 | retrieval",
            unit="batch",
            position=3,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_progress,
        ) as bar:

            def advance(index):
                bar.update(index - bar.n)

            sources["sasrec"] = predict_vectorized_seen(
                context.sasrec,
                loader,
                k=600,
                batch_size=64,
                item_chunk_size=32768,
                check_stop=self.check_stop,
                callback=advance,
            )
        return sources

    def production_inference(self):
        full = Path(self.config["paths"]["full_reference"])
        context = None
        model = CatBoostPointwiseModel.from_artifact(
            self.work / "completed/final/model"
        )
        try:
            with self.units(
                "inference", "expanded", "full_history", len(self.batches()), "shard"
            ) as advance:
                for index, users in enumerate(self.batches()):
                    destination = self.work / "inference" / f"part-{index:05d}"
                    if not destination.exists() and context is None:
                        configure_torch(self.config["sasrec_runtime"])
                        context = SimpleNamespace(
                            models=load_models(
                                {
                                    n: str(full / "candidate_models" / n / "model")
                                    for n in SOURCE_ORDER
                                }
                            ),
                            sasrec=SASRecCandidateModel.from_artifact(
                                self.work / "sasrec/trained/model",
                                device=self.config["sasrec_runtime"]["device"],
                            ),
                            store=SequenceStore.load(
                                self.work / "full_sequences/store"
                            ),
                            cutoff=datetime.fromisoformat(
                                self.config["prediction_times"]["prediction_start"]
                            ),
                            profiles=HistoryProfiles.load(full / "profiles"),
                            lookups=load_feature_lookups(full / "lookups"),
                        )

                    def build(
                        path, context=context, users=users, index=index, model=model
                    ):
                        history = read_users(full / "history_daily.parquet", users)
                        sources = self.production_sources(context, users, history)
                        union, seeds = cross_scored_union(
                            sources,
                            history=history,
                            users=users,
                            models=context.models,
                            sasrec=context.sasrec,
                            store=context.store,
                            cutoff=context.cutoff,
                            seed=self.config["seed"],
                        )
                        validate_unseen(
                            union.select("user_id", "item_id"), users, context.store
                        )
                        frame = materialize_features(
                            union,
                            seeds=seeds,
                            lookups=context.lookups,
                            models=context.models,
                            profiles=context.profiles,
                            cutoff=context.cutoff,
                            features=self.features,
                        )
                        scores = score(
                            model, frame, self.config["inference"]["batch_size"]
                        )
                        recs, _, fallback = recommendations_with_fallback(
                            scores, union=union, target_users=users
                        )
                        recs.write_parquet(path / "recommendations.parquet")
                        if index == 0:
                            probe = frame.head(256)
                            probe.write_parquet(path / "verification_features.parquet")
                            score(model, probe, 256).write_parquet(
                                path / "verification_scores.parquet"
                            )
                        return {
                            "users": users.height,
                            "candidate_rows": union.height,
                            "fallback": fallback,
                        }

                    result = self.operation(
                        "inference",
                        "expanded",
                        f"full_history:{index}",
                        destination,
                        build,
                    )
                    del build
                    advance(index, None, result)
        finally:
            del context, model
            self.release()

    def publish(self):
        full = Path(self.config["paths"]["full_reference"])

        def build(path):
            root = path / "artifact"
            root.mkdir()
            for filename in ("history_daily.parquet",):
                os.link(full / filename, root / filename)
            self.users.write_parquet(root / "target_users.parquet")
            for folder in ("candidate_models", "profiles", "lookups"):
                shutil.copytree(full / folder, root / folder, copy_function=os.link)
            shutil.copytree(
                self.work / "sasrec/trained",
                root / "candidate_models/sasrec",
                copy_function=os.link,
            )
            for phase in PHASE_FOLDS:
                shutil.copytree(
                    self.work / "completed" / phase,
                    root / "ranker_phases" / phase,
                    copy_function=os.link,
                )
            shutil.copytree(
                self.work / "completed/final/model",
                root / "model",
                copy_function=os.link,
            )
            for filename in (
                "verification_features.parquet",
                "verification_scores.parquet",
            ):
                os.link(self.work / "inference/part-00000" / filename, root / filename)
            recs = pl.concat(
                [
                    pl.read_parquet(p / "recommendations.parquet")
                    for p in sorted((self.work / "inference").glob("part-*"))
                ]
            ).sort("user_id")
            validation = validate_output_semantics(
                recs,
                context=SimpleNamespace(
                    target_users=self.users, history_path=full / "history_daily.parquet"
                ),
            )
            recs.write_parquet(root / "recommendations.parquet")
            csv_info = write_submission(recs, root / "submission.csv")
            validations = {
                p: read_json(root / "ranker_phases" / p / "metrics.json")
                for p in EVALUATION_FOLDS
            }
            improved = (
                all(
                    v["delta_p20_all_targets"] is not None
                    and v["delta_p20_all_targets"] > 0
                    for v in validations.values()
                )
                if self.config["mode"] == "full"
                else None
            )
            metrics = {
                "run_id": self.run_id,
                "mode": self.config["mode"],
                "kind": self.config["kind"],
                "precision_at_20_all_targets": None,
                "precision_at_20_labeled_users": None,
                "target_users": recs.height,
                "coverage": 1.0,
                "feature_count": len(self.features),
                "candidate_caps": CAPS,
                "validation_folds": validations,
                "improved_on_both_validation_folds": improved,
                "submission": csv_info,
                "validation": validation,
                "runtime_seconds": self.active_runtime_seconds,
                "peak_memory_mb": self.peak_memory_mb,
                "canonical_previously_opened": True,
                "full_sasrec_training": read_json(
                    self.work / "sasrec/trained/metrics.json"
                ),
            }
            write_json_atomic(root / "config.json", self.config)
            write_json_atomic(root / "metrics.json", metrics)
            write_json_atomic(root / "submission_schema.json", submission_schema())
            write_json_atomic(
                root / "manifest.json",
                {"config_sha256": self.digest, "files": _file_manifest(root)},
            )
            verify_artifact(root)
            return {
                "submission_rows": recs.height,
                "improved_on_both_validation_folds": improved,
            }

        staging = self.work / "publication"
        self.operation("publication", "expanded", "full_history", staging, build)
        # Copy hard links so the committed operation remains valid after publication.
        temporary = self.output.parent / f".{self.run_id}.publish"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(staging / "artifact", temporary, copy_function=os.link)
        publish_directory_atomic(temporary, self.output)
        append_experiment(self.output)
        return {
            "artifact": str(self.output),
            "submission": str(self.output / "submission.csv"),
            "metrics": str(self.output / "metrics.json"),
        }

    def run(self, stop_after_phase=None):
        try:
            with self.phase("preflight"), self.heartbeat("preflight", "verify_inputs"):
                self.preflight()
            for phase in PHASE_FOLDS:
                with self.phase(phase):
                    self.ranker_phase(phase)
                if stop_after_phase == phase:
                    raise BenchmarkPaused(
                        f"paused after {phase}; rerun without --stop-after-phase"
                    )
            with (
                self.phase("full_sasrec"),
                self.heartbeat("full_sasrec", "prepare_and_train"),
            ):
                self.production_sasrec()
            with self.phase("full_inference"):
                self.production_inference()
            with (
                self.phase("publication"),
                self.heartbeat("publication", "validate_and_publish"),
            ):
                return self.publish()
        finally:
            self.save_timing()
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
            self.reporter.close()


def verify_artifact(root):
    root = Path(root)
    c, m = read_json(root / "config.json"), read_json(root / "metrics.json")
    manifest = read_json(root / "manifest.json")
    if manifest["config_sha256"] != config_sha256(c):
        raise ValueError("artifact configuration differs")
    _verify_files(root, manifest["files"])
    users = pl.read_parquet(root / "target_users.parquet")
    if c["mode"] == "full" and not users.equals(
        pl.read_parquet(c["paths"]["target_users"]).sort("user_id")
    ):
        raise ValueError("full submission target universe differs")
    recs = read_submission(root / "submission.csv")
    if (
        not recs.equals(pl.read_parquet(root / "recommendations.parquet"))
        or read_json(root / "submission_schema.json") != submission_schema()
    ):
        raise ValueError("submission round trip differs")
    validate_output_semantics(
        recs,
        context=SimpleNamespace(
            target_users=users, history_path=root / "history_daily.parquet"
        ),
    )
    for phase, metrics in m["validation_folds"].items():
        directory = root / "ranker_phases" / phase
        truth = pl.read_parquet(directory / "target_ground_truth.parquet")
        recs_fold = pl.read_parquet(directory / "recommendations.parquet")
        repeated = evaluate_precision_at_20(recs_fold, truth, users)
        if any(abs(repeated[k] - metrics[k]) > 1e-15 for k in repeated):
            raise ValueError("saved validation precision differs from recommendations")
        validate_phase_horizon(
            phase,
            c["fold_specs"],
            datetime.fromisoformat(c["prediction_times"]["prediction_start"]),
        )
    history_sha = sha256_file(root / "history_daily.parquet")
    for name in SOURCE_ORDER:
        if (
            read_json(root / "candidate_models" / name / "manifest.json")[
                "history_sha256"
            ]
            != history_sha
        ):
            raise ValueError("candidate model history differs")
    if (
        read_json(root / "candidate_models/sasrec/model/model_config.json")[
            "history_sha256"
        ]
        != history_sha
    ):
        raise ValueError("SASRec history differs")
    model = CatBoostPointwiseModel.from_artifact(root / "model")
    if list(model.feature_columns) != c["feature_columns"]:
        raise ValueError("ranker feature schema differs")
    probe = pl.read_parquet(root / "verification_features.parquet")
    if not score(model, probe, 256).equals(
        pl.read_parquet(root / "verification_scores.parquet")
    ):
        raise ValueError("portable ranker prediction differs")
    return {
        "verified": True,
        "target_users": users.height,
        "mode": m["mode"],
        "submission_sha256": sha256_file(root / "submission.csv"),
    }


def append_experiment(root):
    c, m = read_json(root / "config.json"), read_json(root / "metrics.json")
    if c["mode"] != "full":
        return
    path = Path("experiments/results.csv")
    with path.open("r+", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        reader = csv.DictReader(stream)
        fields, existing = reader.fieldnames, {r["run_id"] for r in reader}
        stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fieldnames=fields)
        for values in m["validation_folds"].values():
            run_id = c["run_id"] + "_" + values["fold"]
            if run_id in existing:
                continue
            row = dict.fromkeys(fields, "")
            row.update(
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                split=values["fold"],
                seed=42,
                candidate_config="ALS600+SASRec600+global200+recency200+item2item200",
                ranker_config="D_all+SASRec11",
                p20_all_targets=values["precision_at_20_all_targets"],
                p20_labeled_users=values["precision_at_20_labeled_users"],
                candidate_recall=values["candidate_recall"],
                coverage=values["coverage"],
                runtime=m["runtime_seconds"],
                artifact_path=str(root),
                notes="fixed 1030 trees; fresh training-only borders; canonical previously opened; full-user temporal evaluation",
            )
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task15_sasrec_ranker_v1.json")
    )
    parser.add_argument(
        "--verify-only", type=Path, help="Validate a completed artifact; no training"
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=tuple(PHASE_FOLDS),
        help="Pause at a committed phase boundary; rerun without this option to resume",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_artifact(args.verify_only), indent=2))
        append_experiment(args.verify_only)
        return
    c = load_config(args.config)
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if args.worker:
        if os.environ.get("EXPANDED_RANKER_SUPERVISED") != "1":
            raise ValueError("worker requires the resource supervisor")
        result = ExpandedRun(resolve_config(c), show_progress=not args.no_progress).run(
            args.stop_after_phase
        )
        print(json.dumps(result, indent=2))
        return
    with ExitStack() as stack:
        for name in (
            "task15_sasrec_benchmark",
            "task13_history_profiles",
            "task14_full_fit",
            "expanded_ranker",
        ):
            lock = stack.enter_context(Path(f"logs/{name}.lock").open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (Path("artifacts") / c["run_id"]).exists():
            raise FileExistsError("completed output exists; use --verify-only")
        os.environ.update(
            EXPANDED_RANKER_SUPERVISED="1",
            OMP_NUM_THREADS=str(c["catboost"]["thread_count"]),
            POLARS_MAX_THREADS=str(c["catboost"]["thread_count"]),
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
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
        if args.stop_after_phase:
            command += ["--stop-after-phase", args.stop_after_phase]
        # An interrupted job already owns reusable cache bytes; count these toward
        # its start reserve, while retaining the absolute stop reserve.
        work = Path("artifacts") / f".{c['run_id']}.work"
        if work.exists():
            allocated = (
                sum(p.stat().st_size for p in work.rglob("*") if p.is_file()) / 2**30
            )
            for start, stop in (
                ("minimum_free_disk_gib", "stop_free_disk_gib"),
                ("windows_minimum_start_free_gib", "windows_stop_free_gib"),
            ):
                c["resources"][start] = max(
                    c["resources"][stop], c["resources"][start] - allocated
                )
        raise SystemExit(supervise(command, c))


if __name__ == "__main__":
    try:
        main()
    except BenchmarkPaused as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(75)

#!/usr/bin/env python3
"""Task14: refit every production source and the Task13 D_all ranker; write submission.csv."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import re
import shutil
import signal
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
from features import HistoryFeatureConfig
from full_history_profiles import (
    add_profile_features,
    validate_recipe,
    validate_training_horizon,
)
from history_profiles import HistoryProfiles, history_pairs, normalize_factors
from pipeline import (
    SOURCE_ORDER,
    TRAINING_FOLDS,
    build_candidate_shard,
    build_feature_lookups,
    build_feature_shard,
    candidate_union_config,
    derive_prediction_times,
    fit_candidate_source,
    load_candidate_source,
    load_feature_lookups,
    load_frozen_candidate_configs,
    materialize_full_history,
    recommendations_with_fallback,
)
from rankers import (
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
)
from scripts.run_catboost_ranker import _CatBoostLogBridge
from scripts.run_history_profiles import ProfileRun
from scripts.run_history_profiles import resolved_config as profile_config
from scripts.run_ranker_backtest import (
    _file_manifest,
    _verify_files,
    validate_output_semantics,
)
from scripts.task13_resources import supervise
from submission import read_submission, submission_schema, write_submission


def load_config(path: Path) -> dict:
    c = read_json(path)
    if c["kind"] != "task14_profile_full_fit" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"]
    ):
        raise ValueError("invalid Task14 config/run ID")
    if c["mode"] not in ("full", "smoke") or c["seed"] != c["catboost"]["random_seed"]:
        raise ValueError("invalid mode or inconsistent seeds")
    if c["training"]["target_rows"] <= 0 or c["catboost"]["iterations"] <= 0:
        raise ValueError("positive row/tree budgets required")
    if c["mode"] == "smoke" and not (
        1 <= c["smoke"]["user_count"] <= 256
        and c["catboost"]["iterations"] <= 30
        and c["training"]["target_rows"] <= 100_000
    ):
        raise ValueError("smoke must be explicitly bounded")
    if c["mode"] == "full" and c["smoke"]["user_count"] is not None:
        raise ValueError("full fit must not restrict the history/user universe")
    if c["inference"]["final_k"] != 20 or c["inference"]["batch_size"] <= 0:
        raise ValueError("invalid inference limits")
    if set(c["features"]) != {"windows_hours", "trend_windows_hours"}:
        raise ValueError("use the shared HistoryFeatureConfig window names")
    candidate = c["candidates"]
    if (
        candidate["source_candidate_k"] != 200
        or candidate["materialized_total_cap"] != 800
    ):
        raise ValueError("keep the candidate depths used in Task13")
    if not 1 <= candidate["users_per_shard"] <= 2048:
        raise ValueError("candidate shard must contain at most 2048 users")
    if set(candidate["predict_batch_sizes"]) != set(SOURCE_ORDER) or any(
        not 1 <= x <= 2048 for x in candidate["predict_batch_sizes"].values()
    ):
        raise ValueError("invalid candidate batches")
    limits = c["resources"]
    for key, value in limits.items():
        if key != "windows_drive" and (
            not isinstance(value, (int, float)) or value <= 0
        ):
            raise ValueError(f"invalid resource limit: {key}")
    if limits["poll_seconds"] > 5 or limits["maximum_rss_gib"] > 40:
        raise ValueError("retain the RAM ceiling and active resource monitor")
    if not 0 < c["catboost"]["gpu_ram_part"] <= 0.70:
        raise ValueError("retain the GPU allocation ceiling")
    if not 1 <= c["catboost"]["thread_count"] <= 8:
        raise ValueError("use at most eight CPU threads")
    reference = Path(c["paths"]["task13_reference"])
    validate_recipe(
        c,
        read_json(reference / "config.json"),
        read_json(reference / "model/model_config.json"),
    )
    return c


def resolved_config(source: dict) -> dict:
    c = profile_config(source)
    c["source_config_sha256"] = config_sha256(source)
    reference = Path(c["paths"]["task13_reference"])
    metrics = read_json(reference / "metrics.json")
    if metrics["mode"] != "full" or metrics["winner"] != c["variant"]:
        raise ValueError("reference must be the completed full Task13 D_all winner")
    c["validation_reference"] = {
        "run_id": metrics["run_id"],
        "winner": metrics["winner"],
        "precision_at_20_all_targets": metrics["precision_at_20_all_targets"],
        "precision_at_20_labeled_users": metrics["precision_at_20_labeled_users"],
        "files": {
            p: sha256_file(reference / p)
            for p in (
                "config.json",
                "metrics.json",
                "model/model_config.json",
                "model/model.cbm",
            )
        },
    }
    source_configs, c["candidate_config_provenance"] = load_frozen_candidate_configs(
        c["paths"]["winner_artifacts"]
    )
    frozen_candidates = read_json(Path(c["paths"]["task06_dataset"]) / "config.json")[
        "winner_model_configs"
    ]
    for name, config in source_configs.items():
        actual = config.value if name == "global_popularity" else config.to_dict()
        if actual != frozen_candidates[name]:
            raise ValueError(
                f"candidate recipe differs from Task13 training inputs: {name}"
            )
    source_features = read_json(Path(c["paths"]["task07_dataset"]) / "config.json")[
        "history_features"
    ]
    feature_config = HistoryFeatureConfig.from_mapping(c["features"])
    if (
        list(feature_config.windows_hours) != source_features["windows_hours"]
        or list(feature_config.trend_windows_hours)
        != source_features["trend_windows_hours"]
    ):
        raise ValueError(
            "production history windows differ from the ranker training features"
        )
    schema_path = Path(c["paths"]["submission_schema_reference"])
    if read_json(schema_path) != submission_schema():
        raise ValueError(
            "CSV serialization differs from the previously submitted Task11 file"
        )
    c["input_sha256"] = {
        key: sha256_file(c["paths"][key])
        for key in ("train", "target_users", "submission_schema_reference")
    }
    times = derive_prediction_times(c["paths"]["train"])
    validate_training_horizon(c["profile_inputs"], times["prediction_start"])
    c["prediction_times"] = {k: v.isoformat() for k, v in times.items()}
    for name in (
        "full_history_profiles.py",
        "scripts/run_profile_full_fit.py",
        "pipeline.py",
        "features.py",
        "submission.py",
        "popularity.py",
        "data_utils.py",
        "implicit_model.py",
        "candidate_pipeline.py",
        "item2item.py",
        "validation.py",
        "interfaces.py",
        "ensemble.py",
    ):
        p = Path(name)
        if not p.is_file():
            raise FileNotFoundError(p)
        c["implementation_sha256"][name] = sha256_file(p)
    return c


def score(model, frame, batch_size):
    loader = CatBoostRankerDataLoader(feature_columns=model.feature_columns, seed=42)
    loader.load_predict_data(frame=frame).prepare_predict_data()
    return model.predict(loader, batch_size=batch_size)


class FullProfileRun(ProfileRun):
    @contextmanager
    def heartbeat(self, stage, operation):
        """Opaque native calls expose elapsed time without inventing an ETA."""
        stopped = threading.Event()

        def report():
            elapsed = 0
            while not stopped.wait(30):
                elapsed += 30
                self.reporter.event(
                    "operation_heartbeat",
                    stage=stage,
                    config="full_fit",
                    fold="full_history",
                    operation=operation,
                    elapsed_seconds=elapsed,
                )

        thread = threading.Thread(target=report, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=1)

    def fit_final_ranker(self, pool):
        output = self.work / "ranker"

        def build(path):
            model = CatBoostPointwiseModel(
                CatBoostPointwiseConfig.from_mapping(self.config["catboost"]),
                feature_columns=self.features,
            )
            loader = (
                CatBoostRankerDataLoader(feature_columns=self.features)
                .load_fit_data(train_pool=pool)
                .prepare_fit_data()
            )
            snapshots = self.work / "snapshots/final/D_all"
            snapshots.mkdir(parents=True, exist_ok=True)
            with self.units(
                "fit",
                "D_all",
                "four_folds",
                self.config["catboost"]["iterations"],
                "tree",
            ) as advance:
                bridge = _CatBoostLogBridge(advance)
                model.fit(
                    loader,
                    fixed_tree_budget=True,
                    train_dir=snapshots,
                    snapshot_file=(snapshots / "snapshot.cbsnapshot").resolve(),
                    log_cout=bridge,
                    log_cerr=bridge,
                )
            model.save(path / "model")
            result = {
                "tree_count": model.tree_count,
                "training_pool_sha256": sha256_file(pool),
                "eval_pool": None,
                "warm_start": False,
                "folds": list(TRAINING_FOLDS),
            }
            del model, loader
            gc.collect()
            return result

        self.release_profiles()
        self.operation("fit", "D_all", "four_folds", output, build)
        return output / "model"

    def prepare_full_history(self):
        output = self.work / "full_history"

        def build(path):
            users = (
                self.smoke_users
                if self.config["mode"] == "smoke"
                else pl.read_parquet(self.config["paths"]["target_users"])
            )
            users = users.select("user_id").sort("user_id")
            if (
                users.schema != {"user_id": pl.UInt64}
                or users["user_id"].null_count()
                or users["user_id"].n_unique() != users.height
            ):
                raise ValueError("invalid target user universe")
            users.write_parquet(path / "target_users.parquet")
            with self.heartbeat("full_history", "split_free_daily_aggregation"):
                result = materialize_full_history(
                    self.config["paths"]["train"],
                    path / "history_daily.parquet",
                    selected_users=users if self.config["mode"] == "smoke" else None,
                )
            result["scope"] = (
                "all_raw_events"
                if self.config["mode"] == "full"
                else "smoke_users_only"
            )
            write_json_atomic(path / "diagnostics.json", result)
            return result

        return self.operation("prepare", "full_fit", "full_history", output, build)

    def fit_sources(self, history, reference_time, configs):
        manifests = {}
        for name in SOURCE_ORDER:

            def build(path, name=name):
                with self.heartbeat("candidate_fit", name):
                    if name == "implicit_als":
                        with self.units(
                            "candidate_fit",
                            name,
                            "full_history",
                            configs[name].iterations,
                            "iteration",
                        ) as advance:
                            return fit_candidate_source(
                                name,
                                config=configs[name],
                                history_path=history,
                                reference_time=reference_time,
                                destination=path / "artifact",
                                seed=self.config["seed"],
                                als_callback=advance,
                            )
                    return fit_candidate_source(
                        name,
                        config=configs[name],
                        history_path=history,
                        reference_time=reference_time,
                        destination=path / "artifact",
                        seed=self.config["seed"],
                    )

            manifests[name] = self.operation(
                "candidate_fit",
                name,
                "full_history",
                self.work / "candidate_models" / name,
                build,
            )
            gc.collect()
        return manifests

    def prepare_full_features(self, history, users, reference_time, als_model):
        def lookups(path):
            with self.heartbeat("features", "full_history_lookups"):
                return build_feature_lookups(
                    history_path=history,
                    als_model=als_model,
                    reference_time=reference_time,
                    config=HistoryFeatureConfig.from_mapping(self.config["features"]),
                    destination=path / "artifact",
                )

        self.operation(
            "features", "base201", "full_history", self.work / "lookups", lookups
        )

        def profiles(path):
            model = self.work / "candidate_models/implicit_als/artifact/model"
            pairs = history_pairs(pl.scan_parquet(history), users)
            with np.load(model / "als_model.npz", allow_pickle=False) as arrays:
                factors = normalize_factors(arrays["item_factors"])
            mapping = pl.read_parquet(model / "item_mapping.parquet")
            with self.units(
                "profiles",
                "D_all",
                "full_history",
                max(1, (pairs.height + 16383) // 16384),
                "batch",
            ) as advance:
                result = HistoryProfiles.build(
                    pairs, users, mapping, factors, progress=advance
                )
            result.save(path)
            metadata = {
                "history_sha256": sha256_file(history),
                "als_sha256": sha256_file(model / "als_model.npz"),
                "reference_time": reference_time.isoformat(),
                "diagnostics": result.diagnostics(),
            }
            write_json_atomic(path / "diagnostics.json", metadata)
            return metadata

        self.operation(
            "features",
            "profiles20",
            "full_history",
            self.work / "full_profiles",
            profiles,
        )

    def infer(
        self,
        history,
        users,
        reference_time,
        models,
        model_dir,
        *,
        stop_after_shards=None,
    ):
        c = self.config["candidates"]
        union_config = candidate_union_config(
            source_candidate_k=c["source_candidate_k"],
            total_cap=c["materialized_total_cap"],
        )
        lookups = load_feature_lookups(self.work / "lookups/artifact")
        profiles = HistoryProfiles.load(self.work / "full_profiles")
        schema = read_json(
            Path(self.config["paths"]["task07_dataset"]) / "feature_schema.json"
        )
        base_dtypes = {
            entry["name"]: entry["dtype"]
            for entry in schema["columns"]
            if entry["name"] in self.base_features
        }
        neighbors = models["item2item"].neighbor_table
        known = (
            pl.scan_parquet(history)
            .select("item_id")
            .unique()
            .collect(engine="streaming")
        )
        target_history = (
            pl.scan_parquet(history)
            .join(users.lazy(), on="user_id", how="semi")
            .collect(engine="streaming")
        )
        model = CatBoostPointwiseModel.from_artifact(model_dir)
        repeat_model = CatBoostPointwiseModel.from_artifact(model_dir)
        count = (users.height + c["users_per_shard"] - 1) // c["users_per_shard"]
        with self.units(
            "inference", "D_all", "full_history", count, "shard"
        ) as advance:
            for index in range(count):
                shard_users = users.slice(
                    index * c["users_per_shard"], c["users_per_shard"]
                )

                def build(path, shard_users=shard_users, index=index):
                    union, seeds, diagnostics = build_candidate_shard(
                        models=models,
                        history_shard=target_history.join(
                            shard_users, on="user_id", how="semi"
                        ),
                        target_users=shard_users,
                        reference_time=reference_time,
                        union_config=union_config,
                        predict_batch_sizes=c["predict_batch_sizes"],
                        known_items=known,
                        neighbor_table=neighbors,
                        seed=self.config["seed"],
                    )
                    base = build_feature_shard(
                        union,
                        seeds=seeds,
                        lookups=lookups,
                        neighbor_table=neighbors,
                        item2item_config=models["item2item"].config,
                        reference_time=reference_time,
                        expected_features=self.base_features,
                        expected_feature_dtypes=base_dtypes,
                    )
                    frame = add_profile_features(base, profiles)
                    if tuple(frame.columns[2:]) != self.features:
                        raise ValueError(
                            "production feature order differs from training"
                        )
                    scores = score(model, frame, self.config["inference"]["batch_size"])
                    repeated = score(
                        repeat_model, frame, self.config["inference"]["batch_size"]
                    )
                    if not scores.equals(repeated):
                        raise ValueError(
                            "portable inference differs on a complete shard"
                        )
                    recs, _, fallback = recommendations_with_fallback(
                        scores, union=union, target_users=shard_users
                    )
                    recs.write_parquet(
                        path / "recommendations.parquet", compression="zstd"
                    )
                    if index == 0:
                        probe = frame.head(512)
                        probe.write_parquet(path / "verification_features.parquet")
                        score(model, probe, 256).write_parquet(
                            path / "verification_scores.parquet"
                        )
                    result = {
                        "candidate_rows": union.height,
                        "users": shard_users.height,
                        "source_diagnostics": diagnostics,
                        "fallback": fallback,
                        "portable_repeat_equal": True,
                    }
                    write_json_atomic(path / "diagnostics.json", result)
                    return result

                self.operation(
                    "inference",
                    "D_all",
                    f"shard_{index:05d}",
                    self.work / "inference" / f"shard_{index:05d}",
                    build,
                )
                advance(index)
                if stop_after_shards is not None and index + 1 >= stop_after_shards:
                    return False
        return True


def verify_artifact(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    _verify_files(root, manifest["files"])
    c, metrics = read_json(root / "config.json"), read_json(root / "metrics.json")
    if config_sha256(c) != manifest["config_sha256"]:
        raise ValueError("artifact config differs")
    users = pl.read_parquet(root / "target_users.parquet")
    if c["mode"] == "full":
        source = Path(c["paths"]["target_users"])
        if sha256_file(source) != c["input_sha256"]["target_users"] or not users.equals(
            pl.read_parquet(source).select("user_id").sort("user_id")
        ):
            raise ValueError(
                "full submission user universe differs from the raw target file"
            )
    recs = pl.read_parquet(root / "recommendations.parquet")
    csv_recs = read_submission(root / "submission.csv")
    if not csv_recs.equals(recs):
        raise ValueError("CSV differs from ranked recommendations")
    if read_json(root / "submission_schema.json") != submission_schema():
        raise ValueError("unexpected submission serialization")
    validation = validate_output_semantics(
        recs,
        context=SimpleNamespace(
            target_users=users, history_path=root / "history_daily.parquet"
        ),
    )
    history_sha = sha256_file(root / "history_daily.parquet")
    if metrics["full_history"]["sha256"] != history_sha:
        raise ValueError("full history checksum differs")
    expected_scope = "all_raw_events" if c["mode"] == "full" else "smoke_users_only"
    if metrics["full_history"]["scope"] != expected_scope:
        raise ValueError("full fit history scope differs")
    for name in SOURCE_ORDER:
        source = read_json(root / "candidate_models" / name / "manifest.json")
        if (
            source["source"] != name
            or source["history_sha256"] != history_sha
            or source["reference_time"] != c["prediction_times"]["prediction_start"]
        ):
            raise ValueError(
                f"candidate source was not fitted on this full history: {name}"
            )
        for relative, checksum in source["files"].items():
            if (
                sha256_file(root / "candidate_models" / name / "model" / relative)
                != checksum
            ):
                raise ValueError(f"candidate source file differs: {name}/{relative}")
    profile_meta = read_json(root / "profiles/diagnostics.json")
    if profile_meta["history_sha256"] != history_sha or profile_meta[
        "als_sha256"
    ] != sha256_file(root / "candidate_models/implicit_als/model/als_model.npz"):
        raise ValueError("production profiles do not use the newly fitted ALS")
    lookup = read_json(root / "lookups/manifest.json")
    if lookup["history_sha256"] != history_sha:
        raise ValueError("production statistics use a different history")
    if metrics["training"]["folds"] != list(TRAINING_FOLDS):
        raise ValueError("ranker did not use all four training folds")
    if any(
        metrics[k] is not None
        for k in ("precision_at_20_all_targets", "precision_at_20_labeled_users")
    ):
        raise ValueError("unobserved next-day labels cannot have a measured P20")
    portable = CatBoostPointwiseModel.from_artifact(root / "model")
    if portable.tree_count != c["catboost"]["iterations"]:
        raise ValueError("portable ranker has an incorrect tree budget")
    probe = pl.read_parquet(root / "verification_features.parquet")
    expected = pl.read_parquet(root / "verification_scores.parquet")
    if not score(portable, probe, 256).equals(expected) or not score(
        portable, probe, 256
    ).equals(expected):
        raise ValueError("portable ranker probe differs")
    actual_profiles = HistoryProfiles.load(root / "profiles").features(
        probe.select("user_id", "item_id")
    )
    if not actual_profiles.equals(probe.select(actual_profiles.columns)):
        raise ValueError("saved profile vectors do not reproduce prediction features")
    return {
        "status": "verified",
        "users": users.height,
        "submission_sha256": sha256_file(root / "submission.csv"),
        "validation": validation,
        "all_sources_full_refit": True,
        "portable_repeat_equal": True,
    }


def append_experiment(c, metrics, output):
    if c["mode"] == "smoke":
        return
    path = Path("experiments/results.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("existing experiment log has no header")
        if any(row["run_id"] == c["run_id"] for row in reader):
            return
        row = {
            "run_id": c["run_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split": "full_history_submission",
            "seed": c["seed"],
            "candidate_config": json.dumps(c["candidates"]),
            "ranker_config": json.dumps(
                {
                    "variant": "D_all",
                    "trees": c["catboost"]["iterations"],
                    "folds": list(TRAINING_FOLDS),
                }
            ),
            "coverage": metrics["coverage"],
            "runtime": metrics["runtime_seconds"],
            "peak_memory": metrics["peak_memory_mb"],
            "artifact_path": str(output),
            "notes": "all production sources refitted; fresh borders; no observed future labels; validation_reference=Task13; no Kaggle upload",
        }
        handle.seek(0, os.SEEK_END)
        csv.DictWriter(handle, fieldnames=fields).writerow(
            {key: row.get(key, "") for key in fields}
        )
        handle.flush()
        os.fsync(handle.fileno())


def prepare_publication(run, model_dir, history_metrics, training, quantization):
    def build(path):
        root = path / "artifact"
        root.mkdir()
        recs = pl.concat(
            [
                pl.read_parquet(p)
                for p in sorted(
                    (run.work / "inference").glob("shard_*/recommendations.parquet")
                )
            ]
        ).sort("user_id")
        recs.write_parquet(root / "recommendations.parquet", compression="zstd")
        for name in ("target_users.parquet", "history_daily.parquet"):
            shutil.copy2(run.work / "full_history" / name, root / name)
        shutil.copytree(model_dir, root / "model")
        for name in SOURCE_ORDER:
            shutil.copytree(
                run.work / "candidate_models" / name / "artifact",
                root / "candidate_models" / name,
            )
        shutil.copytree(run.work / "lookups/artifact", root / "lookups")
        shutil.copytree(run.work / "full_profiles", root / "profiles")
        shutil.copy2(
            run.work / "pools/final/fit_training/borders.tsv", root / "borders.tsv"
        )
        for name in ("verification_features.parquet", "verification_scores.parquet"):
            shutil.copy2(run.work / "inference/shard_00000" / name, root / name)
        write_json_atomic(root / "config.json", run.config)
        write_json_atomic(root / "submission_schema.json", submission_schema())
        write_json_atomic(
            root / "training.json", {"sampling": training, "quantization": quantization}
        )
        write_json_atomic(
            root / "shards.json",
            {
                "shards": [
                    read_json(p)
                    for p in sorted(
                        (run.work / "inference").glob("shard_*/diagnostics.json")
                    )
                ]
            },
        )
        csv_info = write_submission(recs, root / "submission.csv")
        metrics = {
            "run_id": run.run_id,
            "kind": run.config["kind"],
            "mode": run.config["mode"],
            "precision_at_20_all_targets": None,
            "precision_at_20_labeled_users": None,
            "candidate_recall": None,
            "validation_reference": run.config["validation_reference"],
            "full_history": history_metrics,
            "training": training,
            "feature_count": len(run.features),
            "tree_count": run.config["catboost"]["iterations"],
            "target_users": recs.height,
            "coverage": 1.0,
            "submission": csv_info,
            "runtime_seconds": run.active_runtime_seconds,
            "peak_memory_mb": run.peak_memory_mb,
        }
        write_json_atomic(root / "metrics.json", metrics)
        write_json_atomic(
            root / "manifest.json",
            {"config_sha256": run.digest, "files": _file_manifest(root)},
        )
        return metrics

    return run.operation(
        "publish", "D_all", "full_history", run.work / "publication", build
    )


def publish_only(config_path: Path) -> dict:
    """Finish a completely staged submission without entering any fitting code."""
    source = load_config(config_path)
    work = Path("artifacts") / f".{source['run_id']}.work"
    output = Path("artifacts") / source["run_id"]
    saved = read_json(work / "config.json")
    if saved["source_config_sha256"] != config_sha256(source):
        raise ValueError("publication source config differs from the completed run")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}; use --verify-only")
    operation = read_json(work / "publication/operation.json")
    if operation["config_sha256"] != config_sha256(saved):
        raise ValueError("publication operation belongs to another config")
    _verify_files(work / "publication", operation["files"])
    staging = work / "publication/artifact"
    if read_json(staging / "config.json") != saved:
        raise ValueError("staged artifact config differs from the fitting run")
    verification = verify_artifact(staging)
    metrics = read_json(staging / "metrics.json")
    publish_directory_atomic(staging, output)
    write_json_atomic(work / "verification.json", verification)
    append_experiment(saved, metrics, output)
    return {
        "output": str(output),
        "submission": str(output / "submission.csv"),
        **verification,
    }


def run_experiment(
    config_path: Path,
    *,
    show_progress=True,
    stop_after_training=False,
    stop_after_shards=None,
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
    if stop_after_shards is not None and (
        source["mode"] != "smoke" or stop_after_shards <= 0
    ):
        raise ValueError("stop-after-shards is a bounded smoke recovery check")
    reporter = EventProgressReporter(
        task_name=source["run_id"],
        total_phases=8,
        log_file=f"logs/{source['run_id']}.log",
        show_progress=show_progress,
        log_max_bytes=5_000_000,
        log_backup_count=3,
    )
    run = None
    try:
        reporter.event("run_start", status="running", mode=source["mode"])
        run = FullProfileRun(
            resolved_config(source), output=output, work=work, reporter=reporter
        )
        with run.phase("training_inputs_and_profiles"):
            run.load_folds(list(TRAINING_FOLDS))
            expected = read_json(
                Path(source["paths"]["task13_reference"]) / "model/model_config.json"
            )["feature_columns"]
            if list(run.features) != expected:
                raise ValueError("four-fold feature schema differs from Task13 winner")
            for fold in TRAINING_FOLDS:
                run.prepare_profiles(fold)
        with run.phase("full_history"):
            history_metrics = run.prepare_full_history()
        with run.phase("training_rows_and_fresh_borders"):
            parts, training = run.training_data("final", list(TRAINING_FOLDS))
            pool, quantization = run.training_pool("final", "fit_training", parts)
        with run.phase("fit_ranker"):
            model_dir = run.fit_final_ranker(pool)
        if stop_after_training:
            reporter.event("run_pause", operation="after_training", status="paused")
            return {"status": "paused", "checkpoint": str(work / "checkpoint.json")}
        history = work / "full_history/history_daily.parquet"
        users = pl.read_parquet(work / "full_history/target_users.parquet")
        reference_time = datetime.fromisoformat(
            run.config["prediction_times"]["prediction_start"]
        )
        configs, _ = load_frozen_candidate_configs(source["paths"]["winner_artifacts"])
        with run.phase("fit_all_candidate_sources"):
            run.fit_sources(history, reference_time, configs)
        models = {
            name: load_candidate_source(
                name, work / "candidate_models" / name / "artifact", configs[name]
            )
            for name in SOURCE_ORDER
        }
        with run.phase("full_history_features_and_profiles"):
            run.prepare_full_features(
                history, users, reference_time, models["implicit_als"]
            )
        with run.phase("candidate_generation_and_ranking"):
            complete = run.infer(
                history,
                users,
                reference_time,
                models,
                model_dir,
                stop_after_shards=stop_after_shards,
            )
        del models
        gc.collect()
        if not complete:
            reporter.event(
                "run_pause", operation="after_inference_shard", status="paused"
            )
            return {"status": "paused", "checkpoint": str(work / "checkpoint.json")}
        with run.phase("publish_and_verify"):
            prepare_publication(run, model_dir, history_metrics, training, quantization)
            result = publish_only(config_path)
        reporter.event(
            "run_finish",
            status="completed",
            runtime_seconds=run.active_runtime_seconds,
            best_config="D_all",
            current_metric=None,
            validation_reference=run.config["validation_reference"][
                "precision_at_20_all_targets"
            ],
        )
        return result
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
        "--config", type=Path, default=Path("configs/task14_full_fit_v1.json")
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--verify-only",
        type=Path,
        help="Check completed CSV, full-history fits and portable prediction; no fit",
    )
    actions.add_argument(
        "--publish-only",
        action="store_true",
        help="Verify and publish a complete staging artifact; no fit or inference",
    )
    actions.add_argument(
        "--stop-after-training",
        action="store_true",
        help="Pause after fitting; rerun identical command without this flag to resume",
    )
    actions.add_argument(
        "--stop-after-shards",
        type=int,
        help="Smoke only: pause after N inference shards to exercise recovery",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_only:
        result = verify_artifact(args.verify_only)
        append_experiment(
            read_json(args.verify_only / "config.json"),
            read_json(args.verify_only / "metrics.json"),
            args.verify_only,
        )
        print(json.dumps(result, indent=2))
        return
    c = load_config(args.config)
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if args.worker:
        if os.environ.get("TASK14_SUPERVISED") != "1":
            raise ValueError("worker requires resource supervision")

        def interrupt(signum, frame):
            raise KeyboardInterrupt(
                f"signal {signum}; rerun identical config to resume"
            )

        signal.signal(signal.SIGTERM, interrupt)
        print(
            json.dumps(
                run_experiment(
                    args.config,
                    show_progress=not args.no_progress,
                    stop_after_training=args.stop_after_training,
                    stop_after_shards=args.stop_after_shards,
                ),
                indent=2,
            )
        )
        return
    with Path("logs/task14_full_fit.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.publish_only:
            print(json.dumps(publish_only(args.config), indent=2))
            return
        if (Path("artifacts") / c["run_id"]).exists():
            raise FileExistsError("completed output exists; use --verify-only")
        os.environ.update(
            TASK14_SUPERVISED="1",
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
        if args.stop_after_training:
            command.append("--stop-after-training")
        if args.stop_after_shards is not None:
            command += ["--stop-after-shards", str(args.stop_after_shards)]
        raise SystemExit(supervise(command, c))


if __name__ == "__main__":
    main()

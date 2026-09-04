#!/usr/bin/env python3
"""Select temporal popularity on rolling folds and evaluate one canonical model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from interfaces import FINAL_RECOMMENDATION_SCHEMA
from metrics import evaluate_candidate_metrics, evaluate_precision_at_20
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
    RecencyScoreKind,
    candidates_to_recommendations,
    fill_with_global_popularity,
)
from scripts.run_global_popularity import (
    _final_hit_count,
    _load_evaluation_frames,
    _peak_memory_mb,
    _read_json,
    _sha256,
    _validate_fold,
    _write_json,
)
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_json_config,
    validate_loader,
    validate_model_config,
    validate_recommendations_against_history,
)


class RecencyExperimentError(ValueError):
    """Raised when task-03 experiment configuration is invalid."""


def _checked_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecencyExperimentError(f"{name} must be a positive integer")
    return value


def _load_model_configs(source_config: dict[str, Any]) -> list[RecencyPopularityConfig]:
    values = source_config.get("model_configs")
    if not isinstance(values, list) or not values:
        raise RecencyExperimentError("model_configs must be a non-empty list")
    configs = [RecencyPopularityConfig.from_dict(value) for value in values]
    config_ids = [config.config_id for config in configs]
    if len(set(config_ids)) != len(config_ids):
        raise RecencyExperimentError("model config_id values must be unique")
    return configs


def _validate_temporal_grid(
    model_configs: list[RecencyPopularityConfig],
    *,
    windows_hours: list[float],
    half_lives_hours: list[float],
) -> None:
    windows = {float(value) for value in windows_hours}
    half_lives = {float(value) for value in half_lives_hours}
    missing_windows: set[float] = set()
    missing_half_lives: set[float] = set()
    for config in model_configs:
        if config.score_kind is RecencyScoreKind.WINDOW:
            if config.window_hours is not None:
                missing_windows.add(config.window_hours)
        elif config.score_kind is RecencyScoreKind.DECAY:
            assert config.half_life_hours is not None
            if config.half_life_hours not in half_lives:
                missing_half_lives.add(config.half_life_hours)
        elif config.score_kind is RecencyScoreKind.TRENDING:
            assert config.short_window_hours is not None
            assert config.long_window_hours is not None
            missing_windows.update(
                (config.short_window_hours, config.long_window_hours)
            )
        else:
            missing_windows.update(
                window
                for window, _ in config.window_weights
                if window is not None
            )
    missing_windows.difference_update(windows)
    if missing_windows or missing_half_lives:
        raise RecencyExperimentError(
            "temporal grid does not cover model configs: "
            f"windows={sorted(missing_windows)}, "
            f"half_lives={sorted(missing_half_lives)}"
        )


def _prepare_recency_loader(
    *,
    history_path: Path,
    target_users: pl.DataFrame,
    reference_time: datetime,
    windows_hours: list[float],
    half_lives_hours: list[float],
    seed: int,
) -> RecencyPopularityDataLoader:
    loader = (
        RecencyPopularityDataLoader(
            reference_time=reference_time,
            windows_hours=windows_hours,
            half_lives_hours=half_lives_hours,
            seed=seed,
        )
        .load_fit_data(history=history_path)
        .prepare_fit_data()
        .load_predict_data(history=history_path, target_users=target_users)
        .prepare_predict_data()
    )
    validate_loader(loader)
    return loader


def _prepare_fallback(
    *,
    history_path: Path,
    target_users: pl.DataFrame,
    final_k: int,
    predict_batch_size: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    loader = (
        PopularityDataLoader(seed=seed)
        .load_fit_data(history=history_path)
        .prepare_fit_data()
        .load_predict_data(history=history_path, target_users=target_users)
        .prepare_predict_data()
    )
    model = GlobalPopularityModel(PopularityScore.RELEVANT_INTERACTION_COUNT)
    model.fit(loader)
    candidates = model.predict(
        loader, k=final_k, batch_size=predict_batch_size
    )
    validate_candidate_output(
        candidates, k=final_k, source_name=model.source_name
    )
    recommendations = candidates_to_recommendations(
        candidates, target_users, k=final_k
    )
    return candidates, recommendations


def _fallback_usage(
    recommendations: pl.DataFrame, primary_candidates: pl.DataFrame
) -> dict[str, int]:
    recommendation_pairs = recommendations.explode(
        "item_ids", empty_as_null=True
    ).select(
        "user_id", pl.col("item_ids").cast(pl.Int32).alias("item_id")
    )
    primary_pairs = primary_candidates.select("user_id", "item_id")
    fallback_pairs = recommendation_pairs.join(
        primary_pairs, on=["user_id", "item_id"], how="anti"
    )
    return {
        "fallback_positions": fallback_pairs.height,
        "fallback_users": fallback_pairs.get_column("user_id").n_unique(),
    }


def _evaluate_config(
    *,
    loader: RecencyPopularityDataLoader,
    model_config: RecencyPopularityConfig,
    fallback_candidates: pl.DataFrame,
    history_path: Path,
    target_users: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
    keep_outputs: bool,
) -> tuple[
    dict[str, Any],
    RecencyPopularityModel | None,
    pl.DataFrame | None,
]:
    started = time.perf_counter()
    model = RecencyPopularityModel(model_config)
    validate_model_config(model)
    model.fit(loader)
    candidates = model.predict(
        loader, k=candidate_k, batch_size=predict_batch_size
    )
    validate_candidate_output(
        candidates, k=candidate_k, source_name=model.source_name
    )
    candidate_metrics = evaluate_candidate_metrics(
        candidates, target_ground_truth, target_users
    )
    recommendations = fill_with_global_popularity(
        candidates,
        fallback_candidates,
        target_users,
        pl.scan_parquet(history_path),
        k=final_k,
    )
    validate_recommendations_against_history(
        recommendations,
        target_users=target_users,
        history_daily=pl.scan_parquet(history_path),
        expected_k=final_k,
    )
    precision = evaluate_precision_at_20(
        recommendations, target_ground_truth, target_users
    )
    fallback_usage = _fallback_usage(recommendations, candidates)
    metrics = {
        "config_id": model_config.config_id,
        "model_config": model_config.to_dict(),
        "target_users": target_users.height,
        "target_labeled_users": target_ground_truth.get_column(
            "user_id"
        ).n_unique(),
        "target_ground_truth_pairs": target_ground_truth.height,
        "candidate_rows": candidates.height,
        "ranked_items": model.item_ranking.height,
        "final_recommendation_rows": recommendations.height,
        **candidate_metrics,
        **precision,
        "final_hits": _final_hit_count(recommendations, target_ground_truth),
        **fallback_usage,
        "runtime_seconds": time.perf_counter() - started,
    }
    if keep_outputs:
        return metrics, model, recommendations
    return metrics, None, None


def _selection_summary(
    fold_results: list[dict[str, Any]],
    model_configs: list[RecencyPopularityConfig],
) -> dict[str, dict[str, float]]:
    keys = (
        "precision_at_20_all_targets",
        "precision_at_20_labeled_users",
        "candidate_recall",
        "candidate_user_hit_rate",
        "candidate_oracle_p20_all_targets",
        "candidate_oracle_p20_labeled_users",
        "coverage",
        "mean_candidate_count",
        "fallback_positions",
        "fallback_users",
        "runtime_seconds",
    )
    summary: dict[str, dict[str, float]] = {}
    for model_config in model_configs:
        results = [
            config_result
            for fold in fold_results
            for config_result in fold["configs"]
            if config_result["config_id"] == model_config.config_id
        ]
        summary[model_config.config_id] = {
            f"mean_{key}": float(fmean(result[key] for result in results))
            for key in keys
        }
    return summary


def _choose_config(
    summary: dict[str, dict[str, float]],
    model_configs: list[RecencyPopularityConfig],
) -> RecencyPopularityConfig:
    priority = {
        model_config.config_id: index
        for index, model_config in enumerate(model_configs)
    }

    def selection_key(
        model_config: RecencyPopularityConfig,
    ) -> tuple[float, float, float, int]:
        values = summary[model_config.config_id]
        return (
            -values["mean_precision_at_20_all_targets"],
            -values["mean_precision_at_20_labeled_users"],
            -values["mean_candidate_oracle_p20_all_targets"],
            priority[model_config.config_id],
        )

    return min(model_configs, key=selection_key)


def _load_baseline_artifact(
    artifact_dir: Path,
    *,
    target_users: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame]:
    required = {
        "config": artifact_dir / "config.json",
        "metrics": artifact_dir / "metrics.json",
        "recommendations": artifact_dir / "recommendations.parquet",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"baseline artifact {artifact_dir} is missing: {missing}"
        )
    config = _read_json(required["config"])
    metrics = _read_json(required["metrics"])
    recommendations = (
        pl.scan_parquet(required["recommendations"])
        .join(target_users.lazy(), on="user_id", how="semi")
        .sort("user_id")
        .collect(engine="streaming")
    )
    if recommendations.schema != FINAL_RECOMMENDATION_SCHEMA:
        raise ContractValidationError("baseline recommendations schema is invalid")
    return {"config": config, "metrics": metrics}, recommendations


def run_experiment(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    smoke_user_limit: int | None = None,
) -> dict[str, Any]:
    """Run rolling ablations, freeze one config, then open canonical once."""

    started = time.perf_counter()
    config_source = Path(config_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if smoke_user_limit is not None:
        _checked_positive_int(smoke_user_limit, name="smoke_user_limit")
    source_config = _read_json(config_source)
    validate_json_config(source_config, name="experiment config")

    seed = int(source_config.get("seed", 42))
    candidate_k = _checked_positive_int(
        source_config.get("candidate_k"), name="candidate_k"
    )
    final_k = _checked_positive_int(
        source_config.get("final_k"), name="final_k"
    )
    predict_batch_size = _checked_positive_int(
        source_config.get("predict_batch_size"), name="predict_batch_size"
    )
    if candidate_k < final_k or final_k != 20:
        raise RecencyExperimentError(
            "candidate_k must be at least final_k and final_k must be 20"
        )
    if source_config.get("fallback_score_type") != (
        PopularityScore.RELEVANT_INTERACTION_COUNT.value
    ):
        raise RecencyExperimentError(
            "fallback_score_type must preserve task02 relevant_interaction_count"
        )

    windows_value = source_config.get("windows_hours")
    half_lives_value = source_config.get("half_lives_hours")
    if not isinstance(windows_value, list) or not isinstance(
        half_lives_value, list
    ):
        raise RecencyExperimentError(
            "windows_hours and half_lives_hours must be lists"
        )
    windows_hours = [float(value) for value in windows_value]
    half_lives_hours = [float(value) for value in half_lives_value]
    model_configs = _load_model_configs(source_config)
    _validate_temporal_grid(
        model_configs,
        windows_hours=windows_hours,
        half_lives_hours=half_lives_hours,
    )

    folds = source_config.get("folds")
    if not isinstance(folds, dict):
        raise RecencyExperimentError("config folds must be an object")
    selection_paths = folds.get("selection")
    if not isinstance(selection_paths, list) or len(selection_paths) != 3:
        raise RecencyExperimentError("exactly three selection folds are required")
    canonical_path = folds.get("canonical")
    if not isinstance(canonical_path, str):
        raise RecencyExperimentError("canonical fold path must be a string")
    baseline_path = source_config.get("baseline_artifact")
    if not isinstance(baseline_path, str):
        raise RecencyExperimentError("baseline_artifact must be a string")

    fold_results: list[dict[str, Any]] = []
    fold_manifests: list[dict[str, Any]] = []
    previous_cutoff: datetime | None = None
    for path_value in selection_paths:
        if not isinstance(path_value, str):
            raise RecencyExperimentError("selection fold paths must be strings")
        fold_dir = Path(path_value)
        paths, manifest = _validate_fold(fold_dir, selection=True)
        cutoff = datetime.fromisoformat(manifest["cutoff"])
        if previous_cutoff is not None and cutoff <= previous_cutoff:
            raise RecencyExperimentError(
                "selection folds must be ordered by increasing cutoff"
            )
        previous_cutoff = cutoff
        target_users, target_ground_truth = _load_evaluation_frames(
            paths, smoke_user_limit=smoke_user_limit
        )
        recency_loader = _prepare_recency_loader(
            history_path=paths["history_daily.parquet"],
            target_users=target_users,
            reference_time=cutoff,
            windows_hours=windows_hours,
            half_lives_hours=half_lives_hours,
            seed=seed,
        )
        fallback_candidates, _ = _prepare_fallback(
            history_path=paths["history_daily.parquet"],
            target_users=target_users,
            final_k=final_k,
            predict_batch_size=predict_batch_size,
            seed=seed,
        )
        config_results: list[dict[str, Any]] = []
        for model_config in model_configs:
            values, _, _ = _evaluate_config(
                loader=recency_loader,
                model_config=model_config,
                fallback_candidates=fallback_candidates,
                history_path=paths["history_daily.parquet"],
                target_users=target_users,
                target_ground_truth=target_ground_truth,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
                keep_outputs=False,
            )
            config_results.append(values)
            gc.collect()
        fold_results.append(
            {
                "fold": manifest,
                "loader_config": recency_loader.get_config(),
                "configs": config_results,
            }
        )
        fold_manifests.append(manifest)

    selection_summary = _selection_summary(fold_results, model_configs)
    selected_config = _choose_config(selection_summary, model_configs)

    canonical_dir = Path(canonical_path)
    canonical_files, canonical_manifest = _validate_fold(
        canonical_dir, selection=False
    )
    canonical_cutoff = datetime.fromisoformat(canonical_manifest["cutoff"])
    if previous_cutoff is not None and canonical_cutoff <= previous_cutoff:
        raise RecencyExperimentError(
            "canonical cutoff must be later than every selection cutoff"
        )
    target_users, target_ground_truth = _load_evaluation_frames(
        canonical_files, smoke_user_limit=smoke_user_limit
    )
    canonical_loader = _prepare_recency_loader(
        history_path=canonical_files["history_daily.parquet"],
        target_users=target_users,
        reference_time=canonical_cutoff,
        windows_hours=windows_hours,
        half_lives_hours=half_lives_hours,
        seed=seed,
    )
    fallback_candidates, baseline_recommendations = _prepare_fallback(
        history_path=canonical_files["history_daily.parquet"],
        target_users=target_users,
        final_k=final_k,
        predict_batch_size=predict_batch_size,
        seed=seed,
    )
    validate_recommendations_against_history(
        baseline_recommendations,
        target_users=target_users,
        history_daily=pl.scan_parquet(canonical_files["history_daily.parquet"]),
        expected_k=final_k,
    )
    baseline_precision = evaluate_precision_at_20(
        baseline_recommendations, target_ground_truth, target_users
    )
    baseline_final_hits = _final_hit_count(
        baseline_recommendations, target_ground_truth
    )
    baseline_artifact, stored_baseline_recommendations = _load_baseline_artifact(
        Path(baseline_path), target_users=target_users
    )
    baseline_recommendations_match = baseline_recommendations.sort(
        "user_id"
    ).equals(stored_baseline_recommendations.sort("user_id"))
    if not baseline_recommendations_match:
        raise RuntimeError(
            "fixed global fallback does not reproduce task02 recommendations"
        )
    if smoke_user_limit is None:
        stored_p20 = baseline_artifact["metrics"].get(
            "precision_at_20_all_targets"
        )
        if stored_p20 != baseline_precision["precision_at_20_all_targets"]:
            raise RuntimeError("computed baseline metric does not match task02")

    canonical_metrics, selected_model, recommendations = _evaluate_config(
        loader=canonical_loader,
        model_config=selected_config,
        fallback_candidates=fallback_candidates,
        history_path=canonical_files["history_daily.parquet"],
        target_users=target_users,
        target_ground_truth=target_ground_truth,
        candidate_k=candidate_k,
        final_k=final_k,
        predict_batch_size=predict_batch_size,
        keep_outputs=True,
    )
    assert selected_model is not None
    assert recommendations is not None
    hit_delta = canonical_metrics["final_hits"] - baseline_final_hits

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        ranking_path = staging / "item_ranking.parquet"
        recommendations_path = staging / "recommendations.parquet"
        model_config_path = staging / "model_config.json"
        selected_model.item_ranking.write_parquet(
            ranking_path,
            compression="zstd",
            statistics=True,
            row_group_size=262_144,
        )
        recommendations.write_parquet(
            recommendations_path,
            compression="zstd",
            statistics=True,
            row_group_size=262_144,
        )
        portable_model_config = {
            "artifact_version": 1,
            "model": "recency_popularity",
            "source_name": selected_model.source_name,
            "recency_config": selected_config.to_dict(),
            "candidate_schema": {
                "user_id": "UInt64",
                "item_id": "Int32",
                "score": "Float64",
                "rank": "UInt32",
                "source": "String",
            },
            "timestamp_semantics": "daily_row_first_timestamp",
            "fit_reference_time": canonical_cutoff.isoformat(),
            "fit_history_sha256": canonical_manifest["output_sha256"][
                "history_daily.parquet"
            ],
            "reuse_policy": (
                "reuse ranking only with the recorded history; refit the same "
                "config for a new fold or full-history fit"
            ),
        }
        _write_json(model_config_path, portable_model_config)

        restored_config = RecencyPopularityConfig.from_dict(
            _read_json(model_config_path)["recency_config"]
        )
        restored_model = RecencyPopularityModel.from_fitted_ranking(
            restored_config, pl.read_parquet(ranking_path)
        )
        repeated_candidates = restored_model.predict(
            canonical_loader, k=final_k, batch_size=predict_batch_size
        )
        repeated_recommendations = fill_with_global_popularity(
            repeated_candidates,
            fallback_candidates,
            target_users,
            pl.scan_parquet(canonical_files["history_daily.parquet"]),
            k=final_k,
        )
        deterministic_match = recommendations.equals(repeated_recommendations)
        if not deterministic_match:
            raise RuntimeError(
                "artifact-restored canonical inference is not deterministic"
            )

        output_sha256 = {
            "model_config.json": _sha256(model_config_path),
            "item_ranking.parquet": _sha256(ranking_path),
            "recommendations.parquet": _sha256(recommendations_path),
        }
        mode = "limited_smoke" if smoke_user_limit is not None else "full"
        resolved_config = {
            "run_id": run_id,
            "artifact_version": 1,
            "mode": mode,
            "seed": seed,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "predict_batch_size": predict_batch_size,
            "windows_hours": windows_hours,
            "half_lives_hours": half_lives_hours,
            "fallback_score_type": source_config["fallback_score_type"],
            "model_configs": [config.to_dict() for config in model_configs],
            "selection": {
                "metric": "mean_precision_at_20_all_targets",
                "tie_break": [
                    "mean_precision_at_20_labeled_users",
                    "mean_candidate_oracle_p20_all_targets",
                    "configured model order",
                ],
                "selected_config_id": selected_config.config_id,
            },
            "folds": {
                "selection": fold_manifests,
                "canonical": canonical_manifest,
            },
            "baseline_artifact": baseline_path,
            "model_config": selected_model.get_config(),
            "data_loader_config": canonical_loader.get_config(),
            "smoke_user_limit": smoke_user_limit,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": _sha256(config_source),
        }
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "selected_config_id": selected_config.config_id,
            "selection_folds": fold_results,
            "selection_summary": selection_summary,
            "canonical": canonical_metrics,
            "baseline_comparison": {
                "run_id": baseline_artifact["config"].get("run_id"),
                **baseline_precision,
                "final_hits": baseline_final_hits,
                "recommendations_match_stored_artifact": (
                    baseline_recommendations_match
                ),
                "selected_minus_baseline_p20_all_targets": (
                    canonical_metrics["precision_at_20_all_targets"]
                    - baseline_precision["precision_at_20_all_targets"]
                ),
                "selected_minus_baseline_p20_labeled_users": (
                    canonical_metrics["precision_at_20_labeled_users"]
                    - baseline_precision["precision_at_20_labeled_users"]
                ),
                "selected_minus_baseline_final_hits": hit_delta,
            },
            **{
                key: canonical_metrics[key]
                for key in (
                    "precision_at_20_all_targets",
                    "precision_at_20_labeled_users",
                    "candidate_recall",
                    "candidate_user_hit_rate",
                    "candidate_oracle_p20_all_targets",
                    "candidate_oracle_p20_labeled_users",
                    "coverage",
                    "mean_candidate_count",
                    "p50_candidate_count",
                    "p90_candidate_count",
                    "p95_candidate_count",
                    "p99_candidate_count",
                    "final_hits",
                    "fallback_positions",
                    "fallback_users",
                )
            },
            "exclusive_hits": canonical_metrics["final_hits"],
            "exclusive_hits_semantics": "standalone_no_ensemble",
            "deterministic_recommendations_match": deterministic_match,
            "output_sha256": output_sha256,
            "runtime_seconds": time.perf_counter() - started,
            "canonical_runtime_seconds": canonical_metrics["runtime_seconds"],
            "peak_memory_mb": _peak_memory_mb(),
        }
        _write_json(staging / "config.json", resolved_config)
        _write_json(staging / "metrics.json", metrics)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select recency/trending popularity on three early folds and "
            "evaluate only the selected configuration on canonical holdout."
        )
    )
    parser.add_argument(
        "--config", default="configs/task03_recency_popularity_v1.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/task03_recency_popularity_v1"
    )
    parser.add_argument("--run-id", default="task03_recency_popularity_v1")
    parser.add_argument("--smoke-user-limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        smoke_user_limit=args.smoke_user_limit,
    )
    summary = {
        "run_id": metrics["run_id"],
        "mode": metrics["mode"],
        "selected_config_id": metrics["selected_config_id"],
        "precision_at_20_all_targets": metrics[
            "precision_at_20_all_targets"
        ],
        "precision_at_20_labeled_users": metrics[
            "precision_at_20_labeled_users"
        ],
        "candidate_recall": metrics["candidate_recall"],
        "fallback_positions": metrics["fallback_positions"],
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_memory_mb": metrics["peak_memory_mb"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

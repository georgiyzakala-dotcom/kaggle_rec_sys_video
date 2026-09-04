#!/usr/bin/env python3
"""Select and evaluate the task-02 global-popularity baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_utils import GROUND_TRUTH_SCHEMA, TARGET_USER_SCHEMA
from metrics import evaluate_candidate_metrics, evaluate_precision_at_20
from popularity import (
    POPULARITY_SCORE_ORDER,
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    candidates_to_recommendations,
)
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_json_config,
    validate_loader,
    validate_model_config,
    validate_recommendations_against_history,
)


class PopularityExperimentError(ValueError):
    """Raised when an experiment config or fold artifact is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PopularityExperimentError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PopularityExperimentError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _fold_files(fold_dir: Path) -> dict[str, Path]:
    names = (
        "config.json",
        "metrics.json",
        "history_daily.parquet",
        "target_ground_truth.parquet",
        "target_users.parquet",
    )
    paths = {name: fold_dir / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"fold artifact {fold_dir} is missing files: {missing}"
        )
    return paths


def _validate_fold(
    fold_dir: Path, *, selection: bool
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = _fold_files(fold_dir)
    config = _read_json(paths["config.json"])
    metrics = _read_json(paths["metrics.json"])
    if config.get("mode") != "full" or metrics.get("mode") != "full":
        raise PopularityExperimentError(
            f"model selection requires a full fold artifact: {fold_dir}"
        )
    split = config.get("split")
    if not isinstance(split, dict) or not isinstance(split.get("cutoff"), str):
        raise PopularityExperimentError(f"invalid split config in {fold_dir}")
    cutoff = datetime.fromisoformat(split["cutoff"])
    end_value = split.get("validation_end_exclusive")
    if selection:
        if not isinstance(end_value, str):
            raise PopularityExperimentError(
                f"selection fold must have an exclusive validation end: {fold_dir}"
            )
        if datetime.fromisoformat(end_value) - cutoff != timedelta(days=1):
            raise PopularityExperimentError(
                f"selection fold must cover exactly 24 hours: {fold_dir}"
            )
    elif end_value is not None:
        raise PopularityExperimentError(
            "canonical fold must use the complete raw tail without an explicit end"
        )
    diagnostics = metrics.get("deterministic_diagnostics")
    if not isinstance(diagnostics, dict):
        raise PopularityExperimentError(
            f"fold metrics lack deterministic diagnostics: {fold_dir}"
        )
    return paths, {
        "path": fold_dir.as_posix(),
        "run_id": config.get("run_id"),
        "cutoff": split["cutoff"],
        "validation_end_exclusive": end_value,
        "output_sha256": diagnostics.get("output_sha256"),
    }


def _load_evaluation_frames(
    paths: dict[str, Path], *, smoke_user_limit: int | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    target_users = pl.read_parquet(paths["target_users.parquet"])
    if target_users.schema != TARGET_USER_SCHEMA:
        raise ContractValidationError("fold target-user schema is invalid")
    if smoke_user_limit is not None:
        target_users = target_users.sort("user_id").head(smoke_user_limit)
    ground_truth = (
        pl.scan_parquet(paths["target_ground_truth.parquet"])
        .join(target_users.lazy(), on="user_id", how="semi")
        .select(GROUND_TRUTH_SCHEMA.names())
        .cast(GROUND_TRUTH_SCHEMA)
        .sort(("user_id", "item_id"))
        .collect(engine="streaming")
    )
    return target_users, ground_truth


def _final_hit_count(
    recommendations: pl.DataFrame, target_ground_truth: pl.DataFrame
) -> int:
    return (
        recommendations.explode("item_ids", empty_as_null=True)
        .drop_nulls("item_ids")
        .select(
            "user_id", pl.col("item_ids").cast(pl.Int32).alias("item_id")
        )
        .join(target_ground_truth, on=["user_id", "item_id"], how="semi")
        .height
    )


def _evaluate_score(
    *,
    loader: PopularityDataLoader,
    history_path: Path,
    target_users: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    score_type: PopularityScore,
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
    keep_outputs: bool,
) -> tuple[
    dict[str, Any],
    GlobalPopularityModel | None,
    pl.DataFrame | None,
]:
    started = time.perf_counter()
    model = GlobalPopularityModel(score_type)
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
    recommendations = candidates_to_recommendations(
        candidates, target_users, k=final_k
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
    final_hits = _final_hit_count(recommendations, target_ground_truth)
    metrics = {
        "score_type": score_type.value,
        "target_users": target_users.height,
        "target_labeled_users": target_ground_truth.get_column(
            "user_id"
        ).n_unique(),
        "target_ground_truth_pairs": target_ground_truth.height,
        "candidate_rows": candidates.height,
        "final_recommendation_rows": recommendations.height,
        **candidate_metrics,
        **precision,
        "final_hits": final_hits,
        "exclusive_hits": final_hits,
        "runtime_seconds": time.perf_counter() - started,
    }
    if keep_outputs:
        return metrics, model, recommendations
    return metrics, None, None


def _selection_summary(
    fold_results: list[dict[str, Any]], score_types: list[PopularityScore]
) -> dict[str, dict[str, float]]:
    keys = (
        "precision_at_20_all_targets",
        "precision_at_20_labeled_users",
        "candidate_recall",
        "candidate_user_hit_rate",
        "candidate_oracle_p20_all_targets",
        "candidate_oracle_p20_labeled_users",
        "runtime_seconds",
    )
    summary: dict[str, dict[str, float]] = {}
    for score_type in score_types:
        results = [
            score_result
            for fold in fold_results
            for score_result in fold["scores"]
            if score_result["score_type"] == score_type.value
        ]
        summary[score_type.value] = {
            f"mean_{key}": float(fmean(result[key] for result in results))
            for key in keys
        }
    return summary


def _choose_score(
    summary: dict[str, dict[str, float]],
    score_types: list[PopularityScore],
) -> PopularityScore:
    priority = {score.value: index for index, score in enumerate(score_types)}

    def selection_key(score: PopularityScore) -> tuple[float, float, float, int]:
        values = summary[score.value]
        return (
            -values["mean_precision_at_20_all_targets"],
            -values["mean_precision_at_20_labeled_users"],
            -values["mean_candidate_oracle_p20_all_targets"],
            priority[score.value],
        )

    return min(score_types, key=selection_key)


def _prepare_loader(
    *,
    history_path: Path,
    target_users: pl.DataFrame,
    seed: int,
) -> PopularityDataLoader:
    loader = (
        PopularityDataLoader(seed=seed)
        .load_fit_data(history=history_path)
        .prepare_fit_data()
        .load_predict_data(history=history_path, target_users=target_users)
        .prepare_predict_data()
    )
    validate_loader(loader)
    return loader


def run_experiment(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    smoke_user_limit: int | None = None,
) -> dict[str, Any]:
    """Run rolling selection, then evaluate only the winner on canonical."""

    started = time.perf_counter()
    config_source = Path(config_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if smoke_user_limit is not None and (
        isinstance(smoke_user_limit, bool) or smoke_user_limit <= 0
    ):
        raise ValueError("smoke_user_limit must be a positive integer")
    source_config = _read_json(config_source)
    validate_json_config(source_config, name="experiment config")

    seed = int(source_config.get("seed", 42))
    candidate_k = int(source_config["candidate_k"])
    final_k = int(source_config["final_k"])
    predict_batch_size = int(source_config["predict_batch_size"])
    if candidate_k < final_k or final_k != 20:
        raise PopularityExperimentError(
            "candidate_k must be at least final_k and task 02 final_k must be 20"
        )
    configured_scores = source_config.get("score_types")
    if configured_scores != list(POPULARITY_SCORE_ORDER):
        raise PopularityExperimentError(
            "score_types must contain the four task-02 variants in fixed order"
        )
    score_types = [PopularityScore(value) for value in configured_scores]

    folds = source_config.get("folds")
    if not isinstance(folds, dict):
        raise PopularityExperimentError("config folds must be an object")
    selection_paths = folds.get("selection")
    if not isinstance(selection_paths, list) or len(selection_paths) != 3:
        raise PopularityExperimentError("exactly three selection folds are required")
    canonical_path = folds.get("canonical")
    if not isinstance(canonical_path, str):
        raise PopularityExperimentError("canonical fold path must be a string")

    fold_results: list[dict[str, Any]] = []
    fold_manifests: list[dict[str, Any]] = []
    previous_cutoff: datetime | None = None
    for value in selection_paths:
        if not isinstance(value, str):
            raise PopularityExperimentError("selection fold paths must be strings")
        fold_dir = Path(value)
        paths, manifest = _validate_fold(fold_dir, selection=True)
        cutoff = datetime.fromisoformat(manifest["cutoff"])
        if previous_cutoff is not None and cutoff <= previous_cutoff:
            raise PopularityExperimentError(
                "selection folds must be ordered by increasing cutoff"
            )
        previous_cutoff = cutoff
        target_users, target_ground_truth = _load_evaluation_frames(
            paths, smoke_user_limit=smoke_user_limit
        )
        loader = _prepare_loader(
            history_path=paths["history_daily.parquet"],
            target_users=target_users,
            seed=seed,
        )
        scores: list[dict[str, Any]] = []
        for score_type in score_types:
            values, _, _ = _evaluate_score(
                loader=loader,
                history_path=paths["history_daily.parquet"],
                target_users=target_users,
                target_ground_truth=target_ground_truth,
                score_type=score_type,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
                keep_outputs=False,
            )
            scores.append(values)
        fold_results.append({"fold": manifest, "scores": scores})
        fold_manifests.append(manifest)

    selection_summary = _selection_summary(fold_results, score_types)
    selected_score = _choose_score(selection_summary, score_types)

    canonical_dir = Path(canonical_path)
    canonical_files, canonical_manifest = _validate_fold(
        canonical_dir, selection=False
    )
    if previous_cutoff is not None and datetime.fromisoformat(
        canonical_manifest["cutoff"]
    ) <= previous_cutoff:
        raise PopularityExperimentError(
            "canonical cutoff must be later than every selection cutoff"
        )
    target_users, target_ground_truth = _load_evaluation_frames(
        canonical_files, smoke_user_limit=smoke_user_limit
    )
    canonical_loader = _prepare_loader(
        history_path=canonical_files["history_daily.parquet"],
        target_users=target_users,
        seed=seed,
    )
    canonical_metrics, selected_model, recommendations = _evaluate_score(
        loader=canonical_loader,
        history_path=canonical_files["history_daily.parquet"],
        target_users=target_users,
        target_ground_truth=target_ground_truth,
        score_type=selected_score,
        candidate_k=candidate_k,
        final_k=final_k,
        predict_batch_size=predict_batch_size,
        keep_outputs=True,
    )
    assert selected_model is not None
    assert recommendations is not None

    repeated_candidates = selected_model.predict(
        canonical_loader, k=final_k, batch_size=predict_batch_size
    )
    repeated_recommendations = candidates_to_recommendations(
        repeated_candidates, target_users, k=final_k
    )
    deterministic_match = recommendations.equals(repeated_recommendations)
    if not deterministic_match:
        raise RuntimeError("repeated canonical inference is not deterministic")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        ranking_path = staging / "item_ranking.parquet"
        recommendations_path = staging / "recommendations.parquet"
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
        output_sha256 = {
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
            "score_types": configured_scores,
            "positive_interaction_unit": "daily_user_item_row",
            "selection": {
                "metric": "mean_precision_at_20_all_targets",
                "tie_break": [
                    "mean_precision_at_20_labeled_users",
                    "mean_candidate_oracle_p20_all_targets",
                    "configured score order",
                ],
                "selected_score_type": selected_score.value,
            },
            "folds": {
                "selection": fold_manifests,
                "canonical": canonical_manifest,
            },
            "model_config": selected_model.get_config(),
            "data_loader_config": canonical_loader.get_config(),
            "smoke_user_limit": smoke_user_limit,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": _sha256(config_source),
        }
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "selected_score_type": selected_score.value,
            "selection_folds": fold_results,
            "selection_summary": selection_summary,
            "canonical": canonical_metrics,
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
                    "exclusive_hits",
                )
            },
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
            "Select a global-popularity score on three early folds and evaluate "
            "only the selected score on the canonical holdout."
        )
    )
    parser.add_argument(
        "--config", default="configs/task02_global_popularity_v1.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/task02_global_popularity_v1"
    )
    parser.add_argument("--run-id", default="task02_global_popularity_v1")
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
        "selected_score_type": metrics["selected_score_type"],
        "precision_at_20_all_targets": metrics[
            "precision_at_20_all_targets"
        ],
        "precision_at_20_labeled_users": metrics[
            "precision_at_20_labeled_users"
        ],
        "candidate_recall": metrics["candidate_recall"],
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_memory_mb": metrics["peak_memory_mb"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

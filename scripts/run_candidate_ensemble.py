#!/usr/bin/env python3
"""Select RRF on precomputed task-06 datasets and evaluate one canonical winner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from candidate_pipeline import CandidateUnionConfig, generator_columns
from ensemble import RRFConfig, RRFEnsembleModel
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
from interfaces import CANDIDATE_SCHEMA
from metrics import (
    evaluate_candidate_metrics_lazy,
    evaluate_precision_at_20,
)
from popularity import candidates_to_recommendations
from scripts.run_global_popularity import _final_hit_count, _peak_memory_mb
from validation import validate_recommendations_against_history

SOURCE_ORDER = (
    "global_popularity",
    "recency_popularity",
    "item2item",
    "implicit_als",
)
STAGE_PRIORITY = {
    "stage1_caps": 1.0,
    "stage2_constant": 2.0,
    "stage3_weights": 3.0,
}


class CandidateEnsembleExperimentError(ValueError):
    """Raised for incompatible offline data or RRF experiment config."""


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateEnsembleExperimentError(f"{name} must be positive")
    return value


def _materialized_config(dataset_config: Mapping[str, Any]) -> CandidateUnionConfig:
    value = dataset_config.get("materialized_union")
    if not isinstance(value, Mapping):
        raise CandidateEnsembleExperimentError(
            "offline dataset lacks materialized_union"
        )
    source_caps = value.get("source_caps")
    if not isinstance(source_caps, Mapping):
        raise CandidateEnsembleExperimentError(
            "offline dataset lacks materialized source caps"
        )
    source_order = value.get("source_order", list(SOURCE_ORDER))
    if (
        not isinstance(source_order, list)
        or set(source_order) != set(source_caps)
        or len(source_order) != len(source_caps)
    ):
        raise CandidateEnsembleExperimentError(
            "offline source_order must cover materialized source caps"
        )
    return CandidateUnionConfig.from_mapping(
        {str(name): int(source_caps[name]) for name in source_order},
        total_cap=int(value["total_cap"]),
    )


def _load_cap_profiles(
    source: Mapping[str, Any], *, materialized: CandidateUnionConfig
) -> list[dict[str, Any]]:
    ablation = source.get("rrf_ablation")
    if not isinstance(ablation, Mapping):
        raise CandidateEnsembleExperimentError("rrf_ablation must be an object")
    values = ablation.get("cap_profiles")
    if not isinstance(values, list) or not values:
        raise CandidateEnsembleExperimentError(
            "rrf_ablation.cap_profiles must be non-empty"
        )
    profiles: list[dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise CandidateEnsembleExperimentError("invalid cap profile")
        caps = {name: int(raw[name]) for name in SOURCE_ORDER}
        config = CandidateUnionConfig.from_mapping(
            caps, total_cap=int(raw["total_cap"])
        )
        for name in SOURCE_ORDER:
            if config.cap_for(name) > materialized.cap_for(name):
                raise CandidateEnsembleExperimentError(
                    f"cap profile {raw['name']} exceeds materialized {name} cap"
                )
        profiles.append({"name": raw["name"], "config": config})
    if len({profile["name"] for profile in profiles}) != len(profiles):
        raise CandidateEnsembleExperimentError("cap profile names must be unique")
    return profiles


def _weight_profiles(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    ablation = source["rrf_ablation"]
    values = ablation.get("weight_profiles")
    if not isinstance(values, list) or not values:
        raise CandidateEnsembleExperimentError("weight_profiles must be non-empty")
    profiles: list[dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise CandidateEnsembleExperimentError("invalid weight profile")
        profiles.append(
            {
                "name": raw["name"],
                "weights": tuple(
                    (source_name, float(raw[source_name]))
                    for source_name in SOURCE_ORDER
                ),
            }
        )
    if len({profile["name"] for profile in profiles}) != len(profiles):
        raise CandidateEnsembleExperimentError("weight profile names must be unique")
    return profiles


def _fold_frames(
    dataset_root: Path, fold_label: str
) -> tuple[pl.DataFrame, pl.DataFrame, Path, list[Path]]:
    fold_root = dataset_root / "folds" / fold_label
    manifest = read_json(fold_root / "dataset_manifest.json")
    parts = sorted((fold_root / "union_features").glob("part-*.parquet"))
    expected_parts = manifest.get("parts")
    if not isinstance(expected_parts, Mapping) or len(parts) != len(expected_parts):
        raise CandidateEnsembleExperimentError(
            f"offline dataset parts differ for {fold_label}"
        )
    for path in parts:
        if sha256_file(path) != expected_parts.get(path.name):
            raise CandidateEnsembleExperimentError(
                f"offline dataset checksum differs: {path}"
            )
    target_users_path = fold_root / "target_users.parquet"
    ground_truth_path = fold_root / "target_ground_truth.parquet"
    if sha256_file(target_users_path) != manifest.get(
        "target_users_sha256"
    ) or sha256_file(ground_truth_path) != manifest.get("target_ground_truth_sha256"):
        raise CandidateEnsembleExperimentError(
            f"offline evaluation-frame checksum differs for {fold_label}"
        )
    target_users = pl.read_parquet(target_users_path)
    ground_truth = pl.read_parquet(ground_truth_path)
    history_value = manifest.get("history_path")
    if not isinstance(history_value, str):
        raise CandidateEnsembleExperimentError(
            f"offline dataset lacks history reference for {fold_label}"
        )
    history = Path(history_value)
    if not history.is_file():
        raise FileNotFoundError(history)
    if target_users.height != int(manifest["target_users"]):
        raise CandidateEnsembleExperimentError(
            f"offline target count differs for {fold_label}"
        )
    return target_users, ground_truth, history, parts


def _eligible_expression(config: CandidateUnionConfig) -> pl.Expr:
    return pl.any_horizontal(
        pl.col(generator_columns(spec.source)["generated"])
        & (pl.col(generator_columns(spec.source)["rank"]) <= spec.cap)
        for spec in config.sources
    )


def _contribution_metrics_lazy(
    features: pl.LazyFrame,
    *,
    ground_truth: pl.DataFrame,
    config: CandidateUnionConfig,
) -> dict[str, Any]:
    active_names: dict[str, str] = {}
    active_exprs: list[pl.Expr] = []
    for spec in config.sources:
        columns = generator_columns(spec.source)
        name = f"__active_{spec.source}"
        active_names[spec.source] = name
        active_exprs.append(
            (
                pl.col(columns["generated"]) & (pl.col(columns["rank"]) <= spec.cap)
            ).alias(name)
        )
    hits = (
        features.with_columns(active_exprs)
        .join(ground_truth.lazy(), on=["user_id", "item_id"], how="semi")
        .with_columns(
            pl.sum_horizontal(
                pl.col(name).cast(pl.UInt8) for name in active_names.values()
            ).alias("__active_count")
        )
    )
    expressions: list[pl.Expr] = [
        (pl.col("__active_count") > 0).sum().alias("union_hits")
    ]
    for source, name in active_names.items():
        expressions.extend(
            (
                pl.col(name).sum().alias(f"source__{source}"),
                (pl.col(name) & (pl.col("__active_count") == 1))
                .sum()
                .alias(f"exclusive__{source}"),
            )
        )
    sources = config.source_names
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            expressions.append(
                (pl.col(active_names[left]) & pl.col(active_names[right]))
                .sum()
                .alias(f"overlap__{left}__{right}")
            )
    values = hits.select(expressions).collect(engine="streaming").row(0, named=True)
    return {
        "union_relevant_hits": int(values["union_hits"] or 0),
        "source_relevant_hits": {
            source: int(values[f"source__{source}"] or 0) for source in sources
        },
        "exclusive_relevant_hits": {
            source: int(values[f"exclusive__{source}"] or 0) for source in sources
        },
        "pairwise_relevant_hit_overlap": {
            key.removeprefix("overlap__"): int(value or 0)
            for key, value in values.items()
            if key.startswith("overlap__")
        },
    }


def _top_rrf_candidates(
    parts: Sequence[Path],
    *,
    model: RRFEnsembleModel,
    materialized: CandidateUnionConfig,
) -> pl.DataFrame:
    projection = ["user_id", "item_id"]
    for spec in materialized.sources:
        columns = generator_columns(spec.source)
        projection.extend((columns["generated"], columns["rank"]))
    outputs: list[pl.DataFrame] = []
    for path in parts:
        features = pl.read_parquet(path, columns=projection)
        ranked = model.rank_candidates(
            features,
            materialized_config=materialized,
            validate_features=False,
        ).filter(pl.col("rank") <= model.config.final_k)
        if ranked.height:
            outputs.append(ranked)
    return (
        pl.concat(outputs, rechunk=True).cast(CANDIDATE_SCHEMA)
        if outputs
        else pl.DataFrame(schema=CANDIDATE_SCHEMA)
    )


def _evaluate_config_fold(
    *,
    dataset_root: Path,
    fold_label: str,
    materialized: CandidateUnionConfig,
    config: RRFConfig,
) -> tuple[dict[str, Any], pl.DataFrame]:
    started = time.perf_counter()
    target_users, ground_truth, _history, parts = _fold_frames(dataset_root, fold_label)
    features = pl.scan_parquet([path.as_posix() for path in parts])
    eligible = features.filter(_eligible_expression(config.candidate_config))
    candidate_metrics = evaluate_candidate_metrics_lazy(
        eligible, ground_truth, target_users
    )
    model = RRFEnsembleModel(config)
    top_candidates = _top_rrf_candidates(parts, model=model, materialized=materialized)
    recommendations = candidates_to_recommendations(
        top_candidates, target_users, k=config.final_k
    )
    precision = evaluate_precision_at_20(recommendations, ground_truth, target_users)
    metrics = {
        "config_id": config.config_id,
        "rrf_config": config.to_dict(),
        "fold": fold_label,
        "target_users": target_users.height,
        "target_labeled_users": ground_truth.get_column("user_id").n_unique(),
        "target_ground_truth_pairs": ground_truth.height,
        **candidate_metrics,
        **precision,
        **_contribution_metrics_lazy(
            features, ground_truth=ground_truth, config=config.candidate_config
        ),
        "final_hits": _final_hit_count(recommendations, ground_truth),
        "fallback_positions": 0,
        "fallback_users": 0,
        "oracle_minus_rrf_p20_all_targets": (
            candidate_metrics["candidate_oracle_p20_all_targets"]
            - precision["precision_at_20_all_targets"]
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    return metrics, recommendations


def _summary(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    if len(results) != 3:
        raise RuntimeError("RRF selection requires exactly three fold results")
    keys = (
        "precision_at_20_all_targets",
        "precision_at_20_labeled_users",
        "candidate_recall",
        "candidate_oracle_p20_all_targets",
        "candidate_oracle_p20_labeled_users",
        "mean_candidate_count",
        "oracle_minus_rrf_p20_all_targets",
        "runtime_seconds",
    )
    return {
        f"mean_{key}": float(fmean(result[key] for result in results)) for key in keys
    }


def _selection_key(summary: Mapping[str, float]) -> tuple[float, ...]:
    return (
        float(summary["mean_precision_at_20_all_targets"]),
        float(summary["mean_precision_at_20_labeled_users"]),
        float(summary["mean_candidate_oracle_p20_all_targets"]),
        -float(summary["mean_mean_candidate_count"]),
    )


def _stage_configs(
    *,
    source: Mapping[str, Any],
    materialized: CandidateUnionConfig,
    retained_cap: CandidateUnionConfig | None,
    retained_constant: float | None,
    stage: str,
) -> list[RRFConfig]:
    cap_profiles = _load_cap_profiles(source, materialized=materialized)
    equal_weights = tuple((name, 1.0) for name in SOURCE_ORDER)
    if stage == "stage1_caps":
        return [
            RRFConfig(
                config_id=f"caps_{profile['name']}",
                candidate_config=profile["config"],
                weights=equal_weights,
                rrf_constant=60.0,
            )
            for profile in cap_profiles
        ]
    if retained_cap is None:
        raise RuntimeError("retained cap config is required")
    if stage == "stage2_constant":
        values = source["rrf_ablation"].get("rrf_constants")
        if not isinstance(values, list) or not values:
            raise CandidateEnsembleExperimentError("rrf_constants must be non-empty")
        return [
            RRFConfig(
                config_id=f"constant_{value}",
                candidate_config=retained_cap,
                weights=equal_weights,
                rrf_constant=float(value),
            )
            for value in values
        ]
    if retained_constant is None:
        raise RuntimeError("retained RRF constant is required")
    return [
        RRFConfig(
            config_id=f"weights_{profile['name']}",
            candidate_config=retained_cap,
            weights=profile["weights"],
            rrf_constant=retained_constant,
        )
        for profile in _weight_profiles(source)
    ]


def _evaluate_stage(
    *,
    name: str,
    configs: Sequence[RRFConfig],
    dataset_root: Path,
    materialized: CandidateUnionConfig,
    checkpoint: CheckpointStore,
    metrics_root: Path,
    reporter: EventProgressReporter,
    best: AtomicBestConfig,
    eligible_max_cap: int | None = None,
) -> tuple[dict[str, Any], RRFConfig]:
    stage_started = reporter.stage_start(
        stage=name, total=len(configs) * 3, unit="config-fold"
    )
    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, float]] = {}
    best_so_far: RRFConfig | None = None
    for config in configs:
        fold_results: list[dict[str, Any]] = []
        for fold_label in ("rolling_1", "rolling_2", "rolling_3"):
            result_path = metrics_root / name / config.config_id / f"{fold_label}.json"
            record = checkpoint.get(
                stage=name, config=config.config_id, fold=fold_label
            )
            if record is not None:
                if not result_path.is_file() or sha256_file(result_path) != record.get(
                    "sha256"
                ):
                    raise CandidateEnsembleExperimentError(
                        f"completed RRF checkpoint is corrupt: {result_path}"
                    )
                result = read_json(result_path)
                reporter.event(
                    "operation_resume_skip",
                    stage=name,
                    config=config.config_id,
                    fold=fold_label,
                )
            else:
                operation_started = reporter.operation_start(
                    stage=name,
                    config=config.config_id,
                    fold=fold_label,
                    operation="offline_rrf_evaluation",
                )
                result, _recommendations = _evaluate_config_fold(
                    dataset_root=dataset_root,
                    fold_label=fold_label,
                    materialized=materialized,
                    config=config,
                )
                write_json_atomic(result_path, result)
                checkpoint.complete(
                    stage=name,
                    config=config.config_id,
                    fold=fold_label,
                    metadata={
                        "sha256": sha256_file(result_path),
                        "precision_at_20_all_targets": result[
                            "precision_at_20_all_targets"
                        ],
                    },
                )
                reporter.operation_finish(
                    stage=name,
                    config=config.config_id,
                    fold=fold_label,
                    operation="offline_rrf_evaluation",
                    started=operation_started,
                    precision_at_20_all_targets=result["precision_at_20_all_targets"],
                    oracle_p20=result["candidate_oracle_p20_all_targets"],
                )
            fold_results.append(result)
            reporter.stage_advance()
        all_results[config.config_id] = fold_results
        summary = _summary(fold_results)
        summaries[config.config_id] = summary
        is_eligible = (
            eligible_max_cap is None
            or config.candidate_config.total_cap <= eligible_max_cap
        )
        if is_eligible and (
            best_so_far is None
            or _selection_key(summary)
            > _selection_key(summaries[best_so_far.config_id])
        ):
            best_so_far = config
        reporter.event(
            "config_finish",
            stage=name,
            config=config.config_id,
            fold="rolling_mean",
            current_metric=summary["mean_precision_at_20_all_targets"],
            best_metric=(
                summaries[best_so_far.config_id]["mean_precision_at_20_all_targets"]
                if best_so_far is not None
                else None
            ),
            best_config=(best_so_far.config_id if best_so_far is not None else None),
            eligible=is_eligible,
        )
    eligible = [
        config
        for config in configs
        if eligible_max_cap is None
        or config.candidate_config.total_cap <= eligible_max_cap
    ]
    if not eligible:
        raise CandidateEnsembleExperimentError(
            f"stage {name} has no eligible configurations"
        )
    order = {config.config_id: index for index, config in enumerate(configs)}
    winner = max(
        eligible,
        key=lambda config: (
            _selection_key(summaries[config.config_id]),
            -order[config.config_id],
        ),
    )
    winner_summary = summaries[winner.config_id]
    best.update(
        score=(STAGE_PRIORITY[name], *_selection_key(winner_summary)),
        payload={
            "stage": name,
            "config": winner.to_dict(),
            "summary": winner_summary,
        },
    )
    reporter.stage_finish(stage=name, started=stage_started, winner=winner.config_id)
    return {
        "stage": name,
        "configs": [config.to_dict() for config in configs],
        "fold_results": all_results,
        "summary": summaries,
        "winner": winner.config_id,
    }, winner


def run_experiment(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    checkpoint_dir: str | Path,
    best_model_dir: str | Path,
    log_file: str | Path | None,
    show_progress: bool,
    allow_smoke_dataset: bool = False,
) -> dict[str, Any]:
    """Select RRF only on rolling datasets, then evaluate canonical once."""

    started = time.perf_counter()
    source = read_json(config_path)
    digest = config_sha256(source)
    dataset_root = Path(dataset_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    dataset_config = read_json(dataset_root / "config.json")
    if dataset_config.get("kind") != "task06_offline_candidate_datasets":
        raise CandidateEnsembleExperimentError("invalid offline candidate artifact")
    if dataset_config.get("mode") != "full" and not allow_smoke_dataset:
        raise CandidateEnsembleExperimentError(
            "limited-smoke candidate data requires --allow-smoke-dataset"
        )
    materialized = _materialized_config(dataset_config)
    if materialized.source_names != SOURCE_ORDER:
        raise CandidateEnsembleExperimentError("offline source order differs")
    checkpoint = CheckpointStore(checkpoint_dir, run_id=run_id, config_digest=digest)
    best = AtomicBestConfig(best_model_dir, run_id=run_id, config_digest=digest)
    work = Path(checkpoint_dir) / "artifact"
    work.mkdir(parents=True, exist_ok=True)
    metrics_root = Path(checkpoint_dir) / "rrf_metrics"
    reporter = EventProgressReporter(
        task_name="task06-rrf",
        total_phases=5,
        log_file=log_file,
        show_progress=show_progress,
    )
    reporter.event(
        "run_start",
        run_id=run_id,
        config=Path(config_path).as_posix(),
        dataset=dataset_root.as_posix(),
        output=output.as_posix(),
        pid=os.getpid(),
    )
    try:
        stages: list[dict[str, Any]] = []
        phase = reporter.phase_start("stage1_caps")
        cap_configs = _stage_configs(
            source=source,
            materialized=materialized,
            retained_cap=None,
            retained_constant=None,
            stage="stage1_caps",
        )
        eligible_max = _positive_int(
            source["rrf_ablation"].get("eligible_winner_max_cap"),
            name="eligible_winner_max_cap",
        )
        stage, cap_winner = _evaluate_stage(
            name="stage1_caps",
            configs=cap_configs,
            dataset_root=dataset_root,
            materialized=materialized,
            checkpoint=checkpoint,
            metrics_root=metrics_root,
            reporter=reporter,
            best=best,
            eligible_max_cap=eligible_max,
        )
        stages.append(stage)
        reporter.phase_finish("stage1_caps", phase, winner=cap_winner.config_id)

        phase = reporter.phase_start("stage2_constant")
        constant_configs = _stage_configs(
            source=source,
            materialized=materialized,
            retained_cap=cap_winner.candidate_config,
            retained_constant=None,
            stage="stage2_constant",
        )
        stage, constant_winner = _evaluate_stage(
            name="stage2_constant",
            configs=constant_configs,
            dataset_root=dataset_root,
            materialized=materialized,
            checkpoint=checkpoint,
            metrics_root=metrics_root,
            reporter=reporter,
            best=best,
        )
        stages.append(stage)
        reporter.phase_finish(
            "stage2_constant", phase, winner=constant_winner.config_id
        )

        phase = reporter.phase_start("stage3_weights")
        weight_configs = _stage_configs(
            source=source,
            materialized=materialized,
            retained_cap=cap_winner.candidate_config,
            retained_constant=constant_winner.rrf_constant,
            stage="stage3_weights",
        )
        stage, winner = _evaluate_stage(
            name="stage3_weights",
            configs=weight_configs,
            dataset_root=dataset_root,
            materialized=materialized,
            checkpoint=checkpoint,
            metrics_root=metrics_root,
            reporter=reporter,
            best=best,
        )
        stages.append(stage)
        reporter.phase_finish("stage3_weights", phase, winner=winner.config_id)

        canonical_phase = reporter.phase_start("canonical_winner")
        canonical_path = work / "canonical_metrics.json"
        recommendations_path = work / "recommendations.parquet"
        canonical_record = checkpoint.get(
            stage="canonical", config=winner.config_id, fold="canonical"
        )
        if canonical_record is not None:
            if (
                not canonical_path.is_file()
                or not recommendations_path.is_file()
                or sha256_file(canonical_path) != canonical_record.get("metrics_sha256")
                or sha256_file(recommendations_path)
                != canonical_record.get("recommendations_sha256")
            ):
                raise CandidateEnsembleExperimentError(
                    "canonical checkpoint is corrupt"
                )
            canonical_metrics = read_json(canonical_path)
            recommendations = pl.read_parquet(recommendations_path)
            reporter.event(
                "operation_resume_skip",
                stage="canonical",
                config=winner.config_id,
                fold="canonical",
            )
        else:
            operation_started = reporter.operation_start(
                stage="canonical",
                config=winner.config_id,
                fold="canonical",
                operation="evaluate_once",
            )
            canonical_metrics, recommendations = _evaluate_config_fold(
                dataset_root=dataset_root,
                fold_label="canonical",
                materialized=materialized,
                config=winner,
            )
            write_json_atomic(canonical_path, canonical_metrics)
            recommendations.write_parquet(
                recommendations_path, compression="zstd", statistics=True
            )
            checkpoint.complete(
                stage="canonical",
                config=winner.config_id,
                fold="canonical",
                metadata={
                    "metrics_sha256": sha256_file(canonical_path),
                    "recommendations_sha256": sha256_file(recommendations_path),
                },
            )
            reporter.operation_finish(
                stage="canonical",
                config=winner.config_id,
                fold="canonical",
                operation="evaluate_once",
                started=operation_started,
                precision_at_20_all_targets=canonical_metrics[
                    "precision_at_20_all_targets"
                ],
            )
        canonical_targets, _canonical_gt, canonical_history, _canonical_parts = (
            _fold_frames(dataset_root, "canonical")
        )
        validate_recommendations_against_history(
            recommendations,
            target_users=canonical_targets,
            history_daily=pl.scan_parquet(canonical_history),
            expected_k=winner.final_k,
        )
        reporter.phase_finish(
            "canonical_winner", canonical_phase, winner=winner.config_id
        )

        publish_phase = reporter.phase_start("portable_restore_publish")
        model_dir = work / "model"
        if not model_dir.exists():
            RRFEnsembleModel(winner).save(model_dir)
        restored = RRFEnsembleModel.from_artifact(model_dir)
        if restored.config != winner:
            raise RuntimeError("restored RRF config differs from winner")
        resolved = {
            "run_id": run_id,
            "artifact_version": 1,
            "kind": "task06_rrf_ensemble",
            "seed": int(source.get("seed", 42)),
            "offline_dataset": dataset_root.as_posix(),
            "offline_dataset_config_sha256": sha256_file(dataset_root / "config.json"),
            "materialized_union": materialized.to_dict(),
            "selection": {
                "folds": ["rolling_1", "rolling_2", "rolling_3"],
                "canonical_isolation": True,
                "primary_metric": "mean_precision_at_20_all_targets",
                "selected_config": winner.to_dict(),
            },
            "source_config_path": Path(config_path).as_posix(),
            "source_config_sha256": digest,
            "library_versions": {
                "python": platform.python_version(),
                "polars": pl.__version__,
            },
        }
        metrics = {
            "run_id": run_id,
            "selected_config_id": winner.config_id,
            "selected_config": winner.to_dict(),
            "selection_stages": stages,
            "canonical_evaluated_config_count": 1,
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
                    "oracle_minus_rrf_p20_all_targets",
                )
            },
            "recommendations_sha256": sha256_file(recommendations_path),
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        write_json_atomic(work / "config.json", resolved)
        write_json_atomic(work / "metrics.json", metrics)
        publish_directory_atomic(work, output)
        reporter.phase_finish(
            "portable_restore_publish", publish_phase, output=output.as_posix()
        )
        reporter.event(
            "run_finish",
            run_id=run_id,
            status="completed",
            duration_seconds=time.perf_counter() - started,
            output=output.as_posix(),
        )
        return metrics
    except BaseException as error:
        reporter.event(
            "run_finish",
            run_id=run_id,
            status="failed",
            duration_seconds=time.perf_counter() - started,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        reporter.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select RRF on three precomputed rolling candidate datasets and "
            "evaluate one canonical winner. Candidate models are never run."
        )
    )
    parser.add_argument("--config", default="configs/task06_candidate_ensemble_v1.json")
    parser.add_argument(
        "--dataset-dir", default="artifacts/task06_candidate_datasets_v1"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/task06_candidate_ensemble_v1"
    )
    parser.add_argument("--run-id", default="task06_candidate_ensemble_v1")
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/.task06_candidate_ensemble_v1.checkpoint",
    )
    parser.add_argument(
        "--best-model-dir",
        default="artifacts/.task06_candidate_ensemble_v1.best-model",
    )
    parser.add_argument("--log-file", default="logs/task06_candidate_ensemble_v1.log")
    parser.add_argument(
        "--allow-smoke-dataset",
        action="store_true",
        help="allow a limited-smoke offline dataset for engineering validation",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run_experiment(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
        best_model_dir=args.best_model_dir,
        log_file=args.log_file or None,
        show_progress=not args.no_progress,
        allow_smoke_dataset=args.allow_smoke_dataset,
    )
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "run_id",
                    "selected_config_id",
                    "precision_at_20_all_targets",
                    "precision_at_20_labeled_users",
                    "candidate_recall",
                    "candidate_oracle_p20_all_targets",
                    "oracle_minus_rrf_p20_all_targets",
                    "runtime_seconds",
                    "peak_memory_mb",
                )
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

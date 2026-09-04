#!/usr/bin/env python3
"""Select item-to-item co-visitation on rolling folds and evaluate canonical."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import fmean
from typing import Any

import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from item2item import (
    Item2ItemConfig,
    Item2ItemDataLoader,
    Item2ItemModel,
    _build_pair_statistics,
    _neighbor_table_from_pair_statistics,
)
from metrics import evaluate_candidate_metrics, evaluate_precision_at_20
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
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
from scripts.run_recency_popularity import _load_baseline_artifact
from validation import (
    validate_candidate_output,
    validate_json_config,
    validate_loader,
    validate_model_config,
    validate_recommendations_against_history,
)


class Item2ItemExperimentError(ValueError):
    """Raised when task-04 experiment configuration is invalid."""


@dataclass
class _FoldContext:
    files: dict[str, Path]
    manifest: dict[str, Any]
    history_source: Path | pl.DataFrame
    target_users: pl.DataFrame
    target_ground_truth: pl.DataFrame
    loader: Item2ItemDataLoader
    fallback_candidates: pl.DataFrame


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Item2ItemExperimentError(f"{name} must be a positive integer")
    return value


def _grid(source: dict[str, Any], name: str) -> list[Any]:
    value = source.get(name)
    if not isinstance(value, list) or not value:
        raise Item2ItemExperimentError(f"ablation.{name} must be a non-empty list")
    if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
        raise Item2ItemExperimentError(f"ablation.{name} must not contain duplicates")
    return value


def _prepare_item2item_loader(
    *,
    history_path: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    reference_time: datetime,
    max_history_items: int,
    max_seed_items: int,
    seed: int,
) -> Item2ItemDataLoader:
    loader = (
        Item2ItemDataLoader(
            reference_time=reference_time,
            max_history_items=max_history_items,
            max_seed_items=max_seed_items,
            seed=seed,
        )
        .load_fit_data(history=history_path)
        .prepare_fit_data()
        .load_predict_data(history=history_path, target_users=target_users)
        .prepare_predict_data()
    )
    validate_loader(loader)
    return loader


def _prepare_global_candidates(
    *,
    history_path: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
) -> pl.DataFrame:
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
        loader, k=candidate_k, batch_size=predict_batch_size
    )
    validate_candidate_output(
        candidates, k=candidate_k, source_name=model.source_name
    )
    return candidates


def _fallback_usage(
    recommendations: pl.DataFrame, primary_candidates: pl.DataFrame
) -> dict[str, int]:
    pairs = recommendations.explode("item_ids", empty_as_null=True).select(
        "user_id", pl.col("item_ids").cast(pl.Int32).alias("item_id")
    )
    fallback = pairs.join(
        primary_candidates.select("user_id", "item_id"),
        on=["user_id", "item_id"],
        how="anti",
    )
    return {
        "fallback_positions": fallback.height,
        "fallback_users": fallback.get_column("user_id").n_unique(),
    }


def _evaluate_model(
    *,
    context: _FoldContext,
    model: Item2ItemModel,
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
    keep_outputs: bool,
) -> tuple[dict[str, Any], pl.DataFrame | None, pl.DataFrame | None]:
    started = time.perf_counter()
    validate_model_config(model)
    candidates = model.predict(
        context.loader, k=candidate_k, batch_size=predict_batch_size
    )
    validate_candidate_output(
        candidates, k=candidate_k, source_name=model.source_name
    )
    candidate_metrics = evaluate_candidate_metrics(
        candidates, context.target_ground_truth, context.target_users
    )
    recommendations = fill_with_global_popularity(
        candidates,
        context.fallback_candidates,
        context.target_users,
        (
            context.history_source.lazy()
            if isinstance(context.history_source, pl.DataFrame)
            else pl.scan_parquet(context.history_source)
        ),
        k=final_k,
    )
    validate_recommendations_against_history(
        recommendations,
        target_users=context.target_users,
        history_daily=(
            context.history_source.lazy()
            if isinstance(context.history_source, pl.DataFrame)
            else pl.scan_parquet(context.history_source)
        ),
        expected_k=final_k,
    )
    precision = evaluate_precision_at_20(
        recommendations, context.target_ground_truth, context.target_users
    )
    usage = _fallback_usage(recommendations, candidates)
    metrics = {
        "config_id": model.config.config_id,
        "model_config": model.config.to_dict(),
        "target_users": context.target_users.height,
        "target_labeled_users": context.target_ground_truth.get_column(
            "user_id"
        ).n_unique(),
        "target_ground_truth_pairs": context.target_ground_truth.height,
        "candidate_rows": candidates.height,
        "neighbor_rows": model.neighbor_table.height,
        "neighbor_items": model.neighbor_table.get_column("item_id").n_unique(),
        "final_recommendation_rows": recommendations.height,
        **candidate_metrics,
        **precision,
        "final_hits": _final_hit_count(
            recommendations, context.target_ground_truth
        ),
        **usage,
        "runtime_seconds": time.perf_counter() - started,
    }
    if keep_outputs:
        return metrics, candidates, recommendations
    return metrics, None, None


def _pair_base_key(config: Item2ItemConfig) -> tuple[Any, ...]:
    return (
        config.profile.value,
        config.direction.value,
        config.pair_weight.value,
        config.history_cap,
        config.pair_half_life_hours,
    )


def _evaluate_fitted_stage(
    contexts: Sequence[_FoldContext],
    configs: Sequence[Item2ItemConfig],
    *,
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
) -> list[dict[str, Any]]:
    fold_results: list[dict[str, Any]] = []
    for context in contexts:
        config_results: list[dict[str, Any]] = []
        grouped: dict[tuple[Any, ...], list[Item2ItemConfig]] = {}
        for config in configs:
            grouped.setdefault(_pair_base_key(config), []).append(config)
        for group in grouped.values():
            statistics = _build_pair_statistics(
                context.loader.fit_profiles, group[0]
            )
            for config in group:
                neighbors = _neighbor_table_from_pair_statistics(
                    statistics, config
                )
                model = Item2ItemModel.from_fitted_neighbors(config, neighbors)
                values, _, _ = _evaluate_model(
                    context=context,
                    model=model,
                    candidate_k=candidate_k,
                    final_k=final_k,
                    predict_batch_size=predict_batch_size,
                    keep_outputs=False,
                )
                config_results.append(values)
                del model, neighbors
                gc.collect()
            del statistics
            gc.collect()
        order = {config.config_id: index for index, config in enumerate(configs)}
        config_results.sort(key=lambda value: order[value["config_id"]])
        fold_results.append(
            {
                "fold": context.manifest,
                "loader_config": context.loader.get_config(),
                "configs": config_results,
            }
        )
    return fold_results


def _evaluate_cached_stage(
    contexts: Sequence[_FoldContext],
    configs: Sequence[Item2ItemConfig],
    cache_paths: Sequence[Path],
    *,
    candidate_k: int,
    final_k: int,
    predict_batch_size: int,
) -> list[dict[str, Any]]:
    fold_results: list[dict[str, Any]] = []
    for context, cache_path in zip(contexts, cache_paths, strict=True):
        neighbors = pl.read_parquet(cache_path)
        base_model = Item2ItemModel.from_fitted_neighbors(configs[0], neighbors)
        config_results: list[dict[str, Any]] = []
        for config in configs:
            model = base_model.with_inference_config(config)
            values, _, _ = _evaluate_model(
                context=context,
                model=model,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
                keep_outputs=False,
            )
            config_results.append(values)
            del model
            gc.collect()
        fold_results.append(
            {
                "fold": context.manifest,
                "loader_config": context.loader.get_config(),
                "configs": config_results,
            }
        )
        del base_model, neighbors
        gc.collect()
    return fold_results


def _selection_summary(
    fold_results: Sequence[dict[str, Any]],
    configs: Sequence[Item2ItemConfig],
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
    result: dict[str, dict[str, float]] = {}
    for config in configs:
        values = [
            candidate
            for fold in fold_results
            for candidate in fold["configs"]
            if candidate["config_id"] == config.config_id
        ]
        if len(values) != len(fold_results):
            raise RuntimeError("selection stage has incomplete fold results")
        result[config.config_id] = {
            f"mean_{key}": float(fmean(value[key] for value in values))
            for key in keys
        }
    return result


def _choose_config(
    summary: dict[str, dict[str, float]],
    configs: Sequence[Item2ItemConfig],
) -> Item2ItemConfig:
    priority = {config.config_id: index for index, config in enumerate(configs)}

    def key(config: Item2ItemConfig) -> tuple[float, float, float, int]:
        values = summary[config.config_id]
        return (
            -values["mean_precision_at_20_all_targets"],
            -values["mean_precision_at_20_labeled_users"],
            -values["mean_candidate_oracle_p20_all_targets"],
            priority[config.config_id],
        )

    return min(configs, key=key)


def _stage_record(
    *,
    name: str,
    changed_block: str,
    configs: Sequence[Item2ItemConfig],
    fold_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], Item2ItemConfig]:
    summary = _selection_summary(fold_results, configs)
    winner = _choose_config(summary, configs)
    return (
        {
            "stage": name,
            "changed_block": changed_block,
            "config_order": [config.config_id for config in configs],
            "winner_config_id": winner.config_id,
            "winner_config": winner.to_dict(),
            "summary": summary,
            "folds": fold_results,
        },
        winner,
    )


def _replace_variants(
    winner: Item2ItemConfig,
    *,
    stage: str,
    field: str,
    values: Sequence[Any],
    token: Callable[[Any], str] = str,
) -> list[Item2ItemConfig]:
    return [
        replace(
            winner,
            config_id=f"{stage}_{field}_{token(value)}",
            **{field: value},
        )
        for value in values
    ]


def _write_neighbor_cache(
    contexts: Sequence[_FoldContext],
    relation_config: Item2ItemConfig,
    cache_dir: Path,
) -> list[Path]:
    cache_dir.mkdir()
    paths: list[Path] = []
    for index, context in enumerate(contexts):
        statistics = _build_pair_statistics(
            context.loader.fit_profiles, relation_config
        )
        neighbors = _neighbor_table_from_pair_statistics(
            statistics, relation_config
        )
        path = cache_dir / f"fold_{index}.parquet"
        neighbors.write_parquet(
            path,
            compression="zstd",
            statistics=True,
            row_group_size=262_144,
        )
        paths.append(path)
        del statistics, neighbors
        gc.collect()
    return paths


def _candidate_hit_pairs(
    candidates: pl.DataFrame, ground_truth: pl.DataFrame
) -> pl.DataFrame:
    return (
        candidates.select("user_id", "item_id")
        .unique()
        .join(ground_truth, on=["user_id", "item_id"], how="semi")
        .sort(("user_id", "item_id"))
    )


def _limited_history(
    history_path: Path,
    target_users: pl.DataFrame,
    *,
    context_user_limit: int,
) -> pl.DataFrame:
    """Create a deterministic smoke-only history scope in memory."""

    history = pl.scan_parquet(history_path)
    context_users = (
        history.select("user_id")
        .unique()
        .sort("user_id")
        .head(context_user_limit)
        .collect(engine="streaming")
    )
    context_users = pl.concat((context_users, target_users)).unique().sort("user_id")
    return (
        history.join(context_users.lazy(), on="user_id", how="semi")
        .sort(("user_id", "item_id", "date"))
        .collect(engine="streaming")
    )


def _candidate_overlap_metrics(
    item2item: pl.DataFrame,
    global_candidates: pl.DataFrame,
    recency_candidates: pl.DataFrame,
    ground_truth: pl.DataFrame,
) -> dict[str, int]:
    item_hits = _candidate_hit_pairs(item2item, ground_truth)
    global_hits = _candidate_hit_pairs(global_candidates, ground_truth)
    recency_hits = _candidate_hit_pairs(recency_candidates, ground_truth)
    popularity_union = pl.concat((global_hits, recency_hits)).unique()
    keys = ["user_id", "item_id"]
    return {
        "item2item_candidate_hits": item_hits.height,
        "global_candidate_hits": global_hits.height,
        "recency_candidate_hits": recency_hits.height,
        "item2item_overlap_hits_with_global": item_hits.join(
            global_hits, on=keys, how="semi"
        ).height,
        "item2item_exclusive_hits_vs_global": item_hits.join(
            global_hits, on=keys, how="anti"
        ).height,
        "item2item_overlap_hits_with_recency": item_hits.join(
            recency_hits, on=keys, how="semi"
        ).height,
        "item2item_exclusive_hits_vs_recency": item_hits.join(
            recency_hits, on=keys, how="anti"
        ).height,
        "item2item_exclusive_hits_vs_popularity_union": item_hits.join(
            popularity_union, on=keys, how="anti"
        ).height,
        "popularity_union_candidate_hits": popularity_union.height,
    }


def _load_recency_candidates(
    artifact_dir: Path,
    *,
    history_path: Path | pl.DataFrame,
    target_users: pl.DataFrame,
    cutoff: datetime,
    candidate_k: int,
    predict_batch_size: int,
    seed: int,
) -> pl.DataFrame:
    config_path = artifact_dir / "model_config.json"
    ranking_path = artifact_dir / "item_ranking.parquet"
    if not config_path.is_file() or not ranking_path.is_file():
        raise FileNotFoundError(
            f"recency artifact is incomplete: {artifact_dir}"
        )
    portable = _read_json(config_path)
    config = RecencyPopularityConfig.from_dict(portable["recency_config"])
    model = RecencyPopularityModel.from_fitted_ranking(
        config, pl.read_parquet(ranking_path)
    )
    loader = (
        RecencyPopularityDataLoader(
            reference_time=cutoff,
            windows_hours=[],
            half_lives_hours=[],
            seed=seed,
        )
        .load_predict_data(history=history_path, target_users=target_users)
        .prepare_predict_data()
    )
    candidates = model.predict(
        loader, k=candidate_k, batch_size=predict_batch_size
    )
    validate_candidate_output(
        candidates, k=candidate_k, source_name=model.source_name
    )
    return candidates


def _baseline_comparison(
    artifact_dir: Path,
    *,
    target_users: pl.DataFrame,
    target_ground_truth: pl.DataFrame,
    selected_metrics: dict[str, Any],
) -> tuple[dict[str, Any], pl.DataFrame]:
    artifact, recommendations = _load_baseline_artifact(
        artifact_dir, target_users=target_users
    )
    precision = evaluate_precision_at_20(
        recommendations, target_ground_truth, target_users
    )
    hits = _final_hit_count(recommendations, target_ground_truth)
    return (
        {
            "run_id": artifact["config"].get("run_id"),
            **precision,
            "final_hits": hits,
            "selected_minus_baseline_p20_all_targets": (
                selected_metrics["precision_at_20_all_targets"]
                - precision["precision_at_20_all_targets"]
            ),
            "selected_minus_baseline_p20_labeled_users": (
                selected_metrics["precision_at_20_labeled_users"]
                - precision["precision_at_20_labeled_users"]
            ),
            "selected_minus_baseline_final_hits": (
                selected_metrics["final_hits"] - hits
            ),
        },
        recommendations,
    )


def run_experiment(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    smoke_user_limit: int | None = None,
    smoke_context_user_limit: int | None = None,
) -> dict[str, Any]:
    """Run seven rolling-only stages, then evaluate one canonical winner."""

    started = time.perf_counter()
    config_source = Path(config_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if smoke_user_limit is not None:
        _positive_int(smoke_user_limit, name="smoke_user_limit")
    if smoke_context_user_limit is not None:
        _positive_int(
            smoke_context_user_limit, name="smoke_context_user_limit"
        )
        if smoke_user_limit is None:
            raise Item2ItemExperimentError(
                "smoke_context_user_limit requires smoke_user_limit"
            )
    elif smoke_user_limit is not None:
        smoke_context_user_limit = max(1000, smoke_user_limit * 20)
    source = _read_json(config_source)
    validate_json_config(source, name="item2item experiment config")
    seed = int(source.get("seed", 42))
    candidate_k = _positive_int(source.get("candidate_k"), name="candidate_k")
    final_k = _positive_int(source.get("final_k"), name="final_k")
    predict_batch_size = _positive_int(
        source.get("predict_batch_size"), name="predict_batch_size"
    )
    if candidate_k < final_k or final_k != 20:
        raise Item2ItemExperimentError(
            "candidate_k must be at least final_k and final_k must be 20"
        )
    if source.get("fallback_score_type") != (
        PopularityScore.RELEVANT_INTERACTION_COUNT.value
    ):
        raise Item2ItemExperimentError(
            "fallback_score_type must be relevant_interaction_count"
        )
    base_value = source.get("base_config")
    if not isinstance(base_value, dict):
        raise Item2ItemExperimentError("base_config must be an object")
    base = Item2ItemConfig.from_dict(base_value)
    ablation = source.get("ablation")
    if not isinstance(ablation, dict):
        raise Item2ItemExperimentError("ablation must be an object")
    profiles = _grid(ablation, "profiles")
    directions = _grid(ablation, "directions")
    pair_weights = _grid(ablation, "pair_weights")
    normalizations = _grid(ablation, "normalizations")
    min_pair_users = _grid(ablation, "min_pair_users")
    neighbor_k_values = _grid(ablation, "neighbor_k")
    seed_k_values = _grid(ablation, "seed_k")
    seed_recency_values = _grid(ablation, "seed_recency_half_life_hours")
    seed_strengths = _grid(ablation, "seed_strengths")

    folds = source.get("folds")
    if not isinstance(folds, dict):
        raise Item2ItemExperimentError("folds must be an object")
    selection_paths = folds.get("selection")
    canonical_value = folds.get("canonical")
    if not isinstance(selection_paths, list) or len(selection_paths) != 3:
        raise Item2ItemExperimentError("exactly three selection folds are required")
    if not isinstance(canonical_value, str):
        raise Item2ItemExperimentError("canonical fold path must be a string")
    task02_value = source.get("task02_artifact")
    task03_value = source.get("task03_artifact")
    if not isinstance(task02_value, str) or not isinstance(task03_value, str):
        raise Item2ItemExperimentError(
            "task02_artifact and task03_artifact must be strings"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        contexts: list[_FoldContext] = []
        previous_cutoff: datetime | None = None
        for path_value in selection_paths:
            if not isinstance(path_value, str):
                raise Item2ItemExperimentError(
                    "selection fold paths must be strings"
                )
            files, manifest = _validate_fold(Path(path_value), selection=True)
            cutoff = datetime.fromisoformat(manifest["cutoff"])
            if previous_cutoff is not None and cutoff <= previous_cutoff:
                raise Item2ItemExperimentError(
                    "selection folds must be ordered by increasing cutoff"
                )
            previous_cutoff = cutoff
            target_users, ground_truth = _load_evaluation_frames(
                files, smoke_user_limit=smoke_user_limit
            )
            history_source: Path | pl.DataFrame = files[
                "history_daily.parquet"
            ]
            if smoke_context_user_limit is not None:
                history_source = _limited_history(
                    files["history_daily.parquet"],
                    target_users,
                    context_user_limit=smoke_context_user_limit,
                )
                known_items = history_source.select("item_id").unique()
                ground_truth = ground_truth.join(
                    known_items, on="item_id", how="semi"
                )
            loader = _prepare_item2item_loader(
                history_path=history_source,
                target_users=target_users,
                reference_time=cutoff,
                max_history_items=base.history_cap,
                max_seed_items=max(int(value) for value in seed_k_values),
                seed=seed,
            )
            fallback = _prepare_global_candidates(
                history_path=history_source,
                target_users=target_users,
                candidate_k=candidate_k,
                predict_batch_size=predict_batch_size,
                seed=seed,
            )
            contexts.append(
                _FoldContext(
                    files=files,
                    manifest=manifest,
                    history_source=history_source,
                    target_users=target_users,
                    target_ground_truth=ground_truth,
                    loader=loader,
                    fallback_candidates=fallback,
                )
            )

        stages: list[dict[str, Any]] = []
        stage1_configs = [
            replace(
                base,
                config_id=f"s1_{profile}_{direction}_{weight}",
                profile=profile,
                direction=direction,
                pair_weight=weight,
            )
            for profile, direction, weight in product(
                profiles, directions, pair_weights
            )
        ]
        stage1_results = _evaluate_fitted_stage(
            contexts,
            stage1_configs,
            candidate_k=candidate_k,
            final_k=final_k,
            predict_batch_size=predict_batch_size,
        )
        stage, winner = _stage_record(
            name="stage1_pair_definition",
            changed_block="profile_direction_pair_weight",
            configs=stage1_configs,
            fold_results=stage1_results,
        )
        stages.append(stage)

        stage_specs = (
            ("stage2_normalization", "normalization", normalizations, str),
            (
                "stage3_min_pair_users",
                "min_pair_users",
                min_pair_users,
                str,
            ),
            ("stage4_neighbor_k", "neighbor_k", neighbor_k_values, str),
        )
        for stage_name, field, values, token_function in stage_specs:
            configs = _replace_variants(
                winner,
                stage=stage_name,
                field=field,
                values=values,
                token=token_function,
            )
            fold_results = _evaluate_fitted_stage(
                contexts,
                configs,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
            )
            stage, winner = _stage_record(
                name=stage_name,
                changed_block=field,
                configs=configs,
                fold_results=fold_results,
            )
            stages.append(stage)

        cache_paths = _write_neighbor_cache(
            contexts, winner, staging / "selection_neighbor_cache"
        )
        inference_specs = (
            ("stage5_seed_k", "seed_k", seed_k_values, str),
            (
                "stage6_seed_recency",
                "seed_recency_half_life_hours",
                seed_recency_values,
                lambda value: "none" if value is None else f"{value}h",
            ),
            (
                "stage7_seed_strength",
                "seed_strength",
                seed_strengths,
                str,
            ),
        )
        for stage_name, field, values, token_function in inference_specs:
            configs = _replace_variants(
                winner,
                stage=stage_name,
                field=field,
                values=values,
                token=token_function,
            )
            fold_results = _evaluate_cached_stage(
                contexts,
                configs,
                cache_paths,
                candidate_k=candidate_k,
                final_k=final_k,
                predict_batch_size=predict_batch_size,
            )
            stage, winner = _stage_record(
                name=stage_name,
                changed_block=field,
                configs=configs,
                fold_results=fold_results,
            )
            stages.append(stage)
        shutil.rmtree(staging / "selection_neighbor_cache")
        del contexts
        gc.collect()

        # Canonical isolation boundary: no canonical file is opened above.
        canonical_files, canonical_manifest = _validate_fold(
            Path(canonical_value), selection=False
        )
        canonical_cutoff = datetime.fromisoformat(canonical_manifest["cutoff"])
        if previous_cutoff is not None and canonical_cutoff <= previous_cutoff:
            raise Item2ItemExperimentError(
                "canonical cutoff must be later than selection cutoffs"
            )
        target_users, ground_truth = _load_evaluation_frames(
            canonical_files, smoke_user_limit=smoke_user_limit
        )
        canonical_history: Path | pl.DataFrame = canonical_files[
            "history_daily.parquet"
        ]
        if smoke_context_user_limit is not None:
            canonical_history = _limited_history(
                canonical_files["history_daily.parquet"],
                target_users,
                context_user_limit=smoke_context_user_limit,
            )
            ground_truth = ground_truth.join(
                canonical_history.select("item_id").unique(),
                on="item_id",
                how="semi",
            )
        canonical_loader = _prepare_item2item_loader(
            history_path=canonical_history,
            target_users=target_users,
            reference_time=canonical_cutoff,
            max_history_items=winner.history_cap,
            max_seed_items=winner.seed_k,
            seed=seed,
        )
        global_candidates = _prepare_global_candidates(
            history_path=canonical_history,
            target_users=target_users,
            candidate_k=candidate_k,
            predict_batch_size=predict_batch_size,
            seed=seed,
        )
        canonical_context = _FoldContext(
            files=canonical_files,
            manifest=canonical_manifest,
            history_source=canonical_history,
            target_users=target_users,
            target_ground_truth=ground_truth,
            loader=canonical_loader,
            fallback_candidates=global_candidates,
        )
        selected_model = Item2ItemModel(winner).fit(canonical_loader)
        canonical_metrics, candidates, recommendations = _evaluate_model(
            context=canonical_context,
            model=selected_model,
            candidate_k=candidate_k,
            final_k=final_k,
            predict_batch_size=predict_batch_size,
            keep_outputs=True,
        )
        assert candidates is not None
        assert recommendations is not None

        recency_candidates = _load_recency_candidates(
            Path(task03_value),
            history_path=canonical_history,
            target_users=target_users,
            cutoff=canonical_cutoff,
            candidate_k=candidate_k,
            predict_batch_size=predict_batch_size,
            seed=seed,
        )
        overlap = _candidate_overlap_metrics(
            candidates, global_candidates, recency_candidates, ground_truth
        )
        candidate_union = pl.concat(
            (candidates, global_candidates, recency_candidates), rechunk=True
        )
        union_metrics = evaluate_candidate_metrics(
            candidate_union, ground_truth, target_users
        )
        task02_comparison, task02_recommendations = _baseline_comparison(
            Path(task02_value),
            target_users=target_users,
            target_ground_truth=ground_truth,
            selected_metrics=canonical_metrics,
        )
        task03_comparison, task03_recommendations = _baseline_comparison(
            Path(task03_value),
            target_users=target_users,
            target_ground_truth=ground_truth,
            selected_metrics=canonical_metrics,
        )
        final_hit_pairs = _candidate_hit_pairs(
            recommendations.explode("item_ids", empty_as_null=True)
            .select(
                "user_id",
                pl.col("item_ids").cast(pl.Int32).alias("item_id"),
            )
            .with_columns(
                pl.lit(0.0).alias("score"),
                pl.lit(1, dtype=pl.UInt32).alias("rank"),
                pl.lit("final").alias("source"),
            )
            .select(candidates.columns)
            .cast(candidates.schema),
            ground_truth,
        )
        source_contribution = {
            "final_hits": final_hit_pairs.height,
            "final_hits_present_in_item2item_candidates": final_hit_pairs.join(
                candidates.select("user_id", "item_id").unique(),
                on=["user_id", "item_id"],
                how="semi",
            ).height,
            "final_hits_present_in_global_candidates": final_hit_pairs.join(
                global_candidates.select("user_id", "item_id").unique(),
                on=["user_id", "item_id"],
                how="semi",
            ).height,
            "final_hits_present_in_recency_candidates": final_hit_pairs.join(
                recency_candidates.select("user_id", "item_id").unique(),
                on=["user_id", "item_id"],
                how="semi",
            ).height,
        }

        neighbor_path = staging / "neighbor_table.parquet"
        recommendations_path = staging / "recommendations.parquet"
        model_config_path = staging / "model_config.json"
        selected_model.neighbor_table.write_parquet(
            neighbor_path,
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
        portable = {
            "artifact_version": 1,
            "model": "item2item",
            "source_name": selected_model.source_name,
            "item2item_config": winner.to_dict(),
            "neighbor_schema": {
                "item_id": "Int32",
                "neighbor_item_id": "Int32",
                "score": "Float64",
                "co_user_count": "UInt32",
                "rank": "UInt32",
            },
            "fit_reference_time": canonical_cutoff.isoformat(),
            "fit_history_sha256": canonical_manifest["output_sha256"][
                "history_daily.parquet"
            ]
            if smoke_context_user_limit is None
            else None,
            "fit_history_scope": (
                "full_fold_history"
                if smoke_context_user_limit is None
                else "limited_smoke_context"
            ),
            "reuse_policy": (
                "reuse neighbors only with the recorded history; refit the same "
                "config for a new fold or full-history fit"
            ),
        }
        _write_json(model_config_path, portable)

        restored = Item2ItemModel.from_fitted_neighbors(
            Item2ItemConfig.from_dict(
                _read_json(model_config_path)["item2item_config"]
            ),
            pl.read_parquet(neighbor_path),
        )
        restored_candidates = restored.predict(
            canonical_loader, k=final_k, batch_size=predict_batch_size
        )
        restored_recommendations = fill_with_global_popularity(
            restored_candidates,
            global_candidates,
            target_users,
            (
                canonical_history.lazy()
                if isinstance(canonical_history, pl.DataFrame)
                else pl.scan_parquet(canonical_history)
            ),
            k=final_k,
        )
        deterministic_match = recommendations.equals(restored_recommendations)
        if not deterministic_match:
            raise RuntimeError("artifact-restored item2item top-20 differs")

        output_sha256 = {
            "model_config.json": _sha256(model_config_path),
            "neighbor_table.parquet": _sha256(neighbor_path),
            "recommendations.parquet": _sha256(recommendations_path),
        }
        mode = "limited_smoke" if smoke_user_limit is not None else "full"
        resolved = {
            "run_id": run_id,
            "artifact_version": 1,
            "mode": mode,
            "seed": seed,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "predict_batch_size": predict_batch_size,
            "fallback_score_type": source["fallback_score_type"],
            "base_config": base.to_dict(),
            "ablation": ablation,
            "selection": {
                "fold_count": 3,
                "canonical_isolation": True,
                "metric": "mean_precision_at_20_all_targets",
                "tie_break": [
                    "mean_precision_at_20_labeled_users",
                    "mean_candidate_oracle_p20_all_targets",
                    "configured order",
                ],
                "selected_config_id": winner.config_id,
                "selected_config": winner.to_dict(),
            },
            "folds": {
                "selection": [stage1["fold"] for stage1 in stage1_results],
                "canonical": canonical_manifest,
            },
            "task02_artifact": task02_value,
            "task03_artifact": task03_value,
            "model_config": selected_model.get_config(),
            "data_loader_config": canonical_loader.get_config(),
            "smoke_user_limit": smoke_user_limit,
            "smoke_context_user_limit": smoke_context_user_limit,
            "source_config_path": config_source.as_posix(),
            "source_config_sha256": _sha256(config_source),
        }
        metrics = {
            "run_id": run_id,
            "mode": mode,
            "selected_config_id": winner.config_id,
            "selected_config": winner.to_dict(),
            "selection_stages": stages,
            "canonical_evaluated_config_count": 1,
            "canonical": canonical_metrics,
            "candidate_hit_overlap": overlap,
            "candidate_union_metrics": union_metrics,
            "source_contribution_to_final_hits": source_contribution,
            "task02_comparison": task02_comparison,
            "task03_comparison": task03_comparison,
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
            "exclusive_hits": overlap[
                "item2item_exclusive_hits_vs_popularity_union"
            ],
            "exclusive_hits_semantics": "candidate_hits_vs_task02_task03_union",
            "deterministic_recommendations_match": deterministic_match,
            "baseline_recommendation_rows": {
                "task02": task02_recommendations.height,
                "task03": task03_recommendations.height,
            },
            "output_sha256": output_sha256,
            "runtime_seconds": time.perf_counter() - started,
            "canonical_runtime_seconds": canonical_metrics["runtime_seconds"],
            "peak_memory_mb": _peak_memory_mb(),
        }
        _write_json(staging / "config.json", resolved)
        _write_json(staging / "metrics.json", metrics)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select item-to-item co-visitation on three rolling folds and "
            "evaluate exactly one winner on canonical holdout."
        )
    )
    parser.add_argument("--config", default="configs/task04_item2item_v1.json")
    parser.add_argument("--output-dir", default="artifacts/task04_item2item_v1")
    parser.add_argument("--run-id", default="task04_item2item_v1")
    parser.add_argument("--smoke-user-limit", type=int)
    parser.add_argument("--smoke-context-user-limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        smoke_user_limit=args.smoke_user_limit,
        smoke_context_user_limit=args.smoke_context_user_limit,
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

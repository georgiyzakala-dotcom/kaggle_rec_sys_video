"""Resource-bounded full-history fitting and inference primitives."""

from __future__ import annotations

import gc
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from catboost import Pool
from catboost.utils import quantize

from candidate_pipeline import (
    CandidateSourceSpec,
    CandidateUnionConfig,
    attach_implicit_als_cross_scores,
    attach_item2item_cross_scores,
    attach_ranking_cross_scores,
    build_candidate_union,
    run_candidate_model,
    validate_union_features,
)
from data_utils import (
    DAILY_INTERACTION_SCHEMA,
    DAILY_KEYS,
    TARGET_USER_SCHEMA,
    aggregate_daily_interactions,
    scan_raw_interactions,
)
from experiment_utils import read_json, sha256_file, write_json_atomic
from features import (
    HistoryFeatureConfig,
    build_als_factor_norm_lookups,
    build_item_features,
    build_user_features,
    validate_history_cutoff,
)
from implicit_model import ImplicitALSConfig, ImplicitALSDataLoader, ImplicitALSModel
from interfaces import CANDIDATE_SCHEMA, FINAL_RECOMMENDATION_SCHEMA
from item2item import Item2ItemConfig, Item2ItemDataLoader, Item2ItemModel
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
)
from ranker_data import (
    build_inference_ranker_shard,
    deterministic_sampling_uniform,
    feature_column_names,
)
from rankers import ranker_scores_to_candidates, write_column_description
from validation import (
    ContractValidationError,
    validate_final_recommendations,
)

SOURCE_ORDER = (
    "global_popularity",
    "recency_popularity",
    "item2item",
    "implicit_als",
)
TRAINING_FOLDS = ("rolling_1", "rolling_2", "rolling_3", "canonical")
TRAINING_STRATA = (
    "positive",
    "multi_collaborative_top50",
    "multi_other",
    "single_item2item_top50",
    "single_als_top50",
    "easy_negative",
)


class FullFitPipelineError(ValueError):
    """Raised when a Task 11 production contract is violated."""


def atomic_write_parquet(
    frame: pl.DataFrame,
    destination: str | Path,
    *,
    compression_level: int = 3,
) -> None:
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        frame.write_parquet(
            temporary,
            compression="zstd",
            compression_level=compression_level,
            statistics=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sink_parquet_atomic(frame: pl.LazyFrame, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        frame.sink_parquet(
            temporary,
            compression="zstd",
            compression_level=3,
            statistics=True,
            row_group_size=262_144,
            maintain_order=True,
            engine="streaming",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def derive_prediction_times(train_path: str | Path) -> dict[str, datetime]:
    maximum = scan_raw_interactions(train_path).select(pl.col("date").max()).collect().item()
    if not isinstance(maximum, datetime):
        raise FullFitPipelineError("raw train has no valid maximum timestamp")
    prediction_start = maximum + timedelta(microseconds=1)
    return {
        "history_end": maximum,
        "prediction_start": prediction_start,
        "prediction_end_exclusive": prediction_start + timedelta(days=1),
    }


def _ordered_daily_diagnostics(path: Path) -> dict[str, Any]:
    frame = pl.scan_parquet(path)
    if frame.collect_schema() != DAILY_INTERACTION_SCHEMA:
        raise ContractValidationError("full history has invalid daily schema")
    first, second, third = (pl.col(name) for name in DAILY_KEYS)
    prev_first, prev_second, prev_third = first.shift(1), second.shift(1), third.shift(1)
    duplicate = (
        (first == prev_first) & (second == prev_second) & (third == prev_third)
    ).fill_null(False)
    out_of_order = (
        (first < prev_first)
        | ((first == prev_first) & (second < prev_second))
        | (
            (first == prev_first)
            & (second == prev_second)
            & (third < prev_third)
        )
    ).fill_null(False)
    row = frame.select(
        rows=pl.len(),
        users=pl.col("user_id").n_unique(),
        items=pl.col("item_id").n_unique(),
        min_date=pl.col("date").min(),
        max_date=pl.col("date").max(),
        min_dt=pl.col("dt").min(),
        max_dt=pl.col("dt").max(),
        null_cells=pl.sum_horizontal(pl.all().null_count()),
        duplicate_rows=duplicate.sum(),
        out_of_order_rows=out_of_order.sum(),
    ).collect(engine="streaming").row(0, named=True)
    if row["null_cells"] or row["duplicate_rows"] or row["out_of_order_rows"]:
        raise ContractValidationError(f"invalid full-history aggregate: {row}")
    return {
        name: value.isoformat() if isinstance(value, (date, datetime)) else int(value)
        for name, value in row.items()
    }


def materialize_full_history(
    train_path: str | Path,
    destination: str | Path,
    *,
    selected_users: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Aggregate all supplied raw events to the shared daily contract."""

    output = Path(destination)
    raw = scan_raw_interactions(train_path)
    if selected_users is not None:
        if selected_users.schema != TARGET_USER_SCHEMA:
            raise ContractValidationError("selected users have invalid schema")
        raw = raw.join(selected_users.lazy(), on="user_id", how="semi")
    raw_summary = raw.select(
        rows=pl.len(),
        users=pl.col("user_id").n_unique(),
        items=pl.col("item_id").n_unique(),
        min_date=pl.col("date").min(),
        max_date=pl.col("date").max(),
        null_cells=pl.sum_horizontal(pl.all().null_count()),
    ).collect(engine="streaming").row(0, named=True)
    if not raw_summary["rows"] or raw_summary["null_cells"]:
        raise ContractValidationError("raw source for full history is empty or null")
    _sink_parquet_atomic(aggregate_daily_interactions(raw), output)
    diagnostics = _ordered_daily_diagnostics(output)
    diagnostics.update(
        raw={
            name: value.isoformat() if isinstance(value, datetime) else int(value)
            for name, value in raw_summary.items()
        },
        sha256=sha256_file(output),
        schema=[
            {"name": name, "dtype": str(dtype)}
            for name, dtype in DAILY_INTERACTION_SCHEMA.items()
        ],
    )
    return diagnostics


def load_frozen_candidate_configs(
    winner_artifacts: Mapping[str, str | Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {"task02", "task03", "task04", "task05"}
    if set(winner_artifacts) != expected:
        raise FullFitPipelineError("winner_artifacts must contain task02 through task05")
    roots = {name: Path(path) for name, path in winner_artifacts.items()}
    task02 = read_json(roots["task02"] / "config.json")
    task03 = read_json(roots["task03"] / "model_config.json")
    task04 = read_json(roots["task04"] / "model_config.json")
    task05 = read_json(roots["task05"] / "model_config.json")
    configs = {
        "global_popularity": PopularityScore(task02["model_config"]["score_type"]),
        "recency_popularity": RecencyPopularityConfig.from_dict(
            task03["recency_config"]
        ),
        "item2item": Item2ItemConfig.from_dict(task04["item2item_config"]),
        "implicit_als": ImplicitALSConfig.from_dict(
            task05["implicit_als_config"]
        ),
    }
    provenance_files = {
        "task02": roots["task02"] / "config.json",
        "task03": roots["task03"] / "model_config.json",
        "task04": roots["task04"] / "model_config.json",
        "task05": roots["task05"] / "model_config.json",
    }
    provenance = {
        name: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for name, path in provenance_files.items()
    }
    return configs, provenance


def _recency_grids(config: RecencyPopularityConfig) -> tuple[list[float], list[float]]:
    source = config.to_dict()
    windows: list[float] = []
    half_lives: list[float] = []
    for key in ("window_hours", "short_window_hours", "long_window_hours"):
        if source.get(key) is not None:
            windows.append(float(source[key]))
    for entry in source.get("window_weights", []):
        if entry.get("window_hours") is not None:
            windows.append(float(entry["window_hours"]))
    if source.get("half_life_hours") is not None:
        half_lives.append(float(source["half_life_hours"]))
    return sorted(set(windows)), sorted(set(half_lives))


def _directory_checksums(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    ignored = set(exclude)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in ignored
    }


def fit_candidate_source(
    source_name: str,
    *,
    config: Any,
    history_path: str | Path,
    reference_time: datetime,
    destination: str | Path,
    seed: int = 42,
    als_callback: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fit and atomically publish one frozen full-history source model."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        model_dir = temporary / "model"
        if source_name == "global_popularity":
            loader = PopularityDataLoader(seed=seed)
            loader.load_fit_data(history=history_path).prepare_fit_data()
            model = GlobalPopularityModel(config).fit(loader)
            model_dir.mkdir()
            model.item_ranking.write_parquet(
                model_dir / "item_ranking.parquet", compression="zstd", statistics=True
            )
            write_json_atomic(model_dir / "model_config.json", model.get_config())
        elif source_name == "recency_popularity":
            windows, half_lives = _recency_grids(config)
            loader = RecencyPopularityDataLoader(
                reference_time=reference_time,
                windows_hours=windows,
                half_lives_hours=half_lives,
                seed=seed,
            )
            loader.load_fit_data(history=history_path).prepare_fit_data()
            model = RecencyPopularityModel(config).fit(loader)
            model_dir.mkdir()
            model.item_ranking.write_parquet(
                model_dir / "item_ranking.parquet", compression="zstd", statistics=True
            )
            write_json_atomic(model_dir / "model_config.json", model.get_config())
        elif source_name == "item2item":
            loader = Item2ItemDataLoader(
                reference_time=reference_time,
                max_history_items=config.history_cap,
                max_seed_items=config.seed_k,
                seed=seed,
            )
            loader.load_fit_data(history=history_path).prepare_fit_data()
            model = Item2ItemModel(config).fit(loader)
            model_dir.mkdir()
            model.neighbor_table.write_parquet(
                model_dir / "neighbor_table.parquet", compression="zstd", statistics=True
            )
            write_json_atomic(model_dir / "model_config.json", model.get_config())
        elif source_name == "implicit_als":
            loader = ImplicitALSDataLoader(
                config=config, reference_time=reference_time, seed=seed
            )
            loader.load_fit_data(history=history_path).prepare_fit_data()
            model = ImplicitALSModel(config).fit(
                loader, show_progress=False, callback=als_callback
            )
            model.save(
                model_dir,
                metadata={
                    "fit_history_scope": "full_history",
                    "fit_reference_time": reference_time.isoformat(),
                    "fit_history_sha256": sha256_file(history_path),
                },
            )
        else:
            raise FullFitPipelineError(f"unknown candidate source: {source_name}")
        manifest = {
            "artifact_version": 1,
            "source": source_name,
            "reference_time": reference_time.isoformat(),
            "history_sha256": sha256_file(history_path),
            "model_config": model.get_config(),
            "files": _directory_checksums(model_dir),
        }
        write_json_atomic(temporary / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_candidate_source(source_name: str, directory: str | Path, config: Any) -> Any:
    root = Path(directory) / "model"
    if source_name == "global_popularity":
        return GlobalPopularityModel.from_fitted_ranking(
            config, pl.read_parquet(root / "item_ranking.parquet")
        )
    if source_name == "recency_popularity":
        return RecencyPopularityModel.from_fitted_ranking(
            config, pl.read_parquet(root / "item_ranking.parquet")
        )
    if source_name == "item2item":
        return Item2ItemModel.from_fitted_neighbors(
            config, pl.read_parquet(root / "neighbor_table.parquet")
        )
    if source_name == "implicit_als":
        return ImplicitALSModel.from_artifact(root)
    raise FullFitPipelineError(f"unknown candidate source: {source_name}")


def candidate_union_config(*, source_candidate_k: int, total_cap: int) -> CandidateUnionConfig:
    return CandidateUnionConfig(
        sources=tuple(
            CandidateSourceSpec(source=name, cap=source_candidate_k)
            for name in SOURCE_ORDER
        ),
        total_cap=total_cap,
    )


def _item2item_seeds(loader: Item2ItemDataLoader) -> pl.DataFrame:
    batches = list(loader.iter_predict_batches(batch_size=None))
    if not batches:
        raise ContractValidationError("item2item prediction loader yielded no batches")
    return pl.concat([batch.seeds for batch in batches], rechunk=True)


def build_candidate_shard(
    *,
    models: Mapping[str, Any],
    history_shard: pl.DataFrame,
    target_users: pl.DataFrame,
    reference_time: datetime,
    union_config: CandidateUnionConfig,
    predict_batch_sizes: Mapping[str, int],
    known_items: pl.DataFrame,
    neighbor_table: pl.DataFrame,
    seed: int = 42,
    minimum_fallback_k: int = 20,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Generate four sources and their cross-scored union for one user shard."""

    if set(models) != set(SOURCE_ORDER):
        raise FullFitPipelineError("candidate model set is incomplete")
    global_loader = PopularityDataLoader(seed=seed)
    global_loader.load_predict_data(
        history=history_shard, target_users=target_users
    ).prepare_predict_data()
    recency_config = models["recency_popularity"].config
    windows, half_lives = _recency_grids(recency_config)
    recency_loader = RecencyPopularityDataLoader(
        reference_time=reference_time,
        windows_hours=windows,
        half_lives_hours=half_lives,
        seed=seed,
    )
    recency_loader.load_predict_data(
        history=history_shard, target_users=target_users
    ).prepare_predict_data()
    item_config = models["item2item"].config
    item_loader = Item2ItemDataLoader(
        reference_time=reference_time,
        max_history_items=item_config.history_cap,
        max_seed_items=item_config.seed_k,
        seed=seed,
    )
    item_loader.load_predict_data(
        history=history_shard, target_users=target_users
    ).prepare_predict_data()
    als_model = models["implicit_als"]
    als_loader = ImplicitALSDataLoader(
        config=als_model.config,
        reference_time=reference_time,
        user_mapping=als_model.user_mapping,
        item_mapping=als_model.item_mapping,
        seed=seed,
    )
    als_loader.load_predict_data(
        history=history_shard, target_users=target_users
    ).prepare_predict_data()
    loaders = {
        "global_popularity": global_loader,
        "recency_popularity": recency_loader,
        "item2item": item_loader,
        "implicit_als": als_loader,
    }
    sources = {
        name: run_candidate_model(
            models[name],
            loaders[name],
            k=next(spec.cap for spec in union_config.sources if spec.source == name),
            batch_size=int(predict_batch_sizes[name]),
        )
        for name in SOURCE_ORDER
    }
    union = build_candidate_union(sources, union_config)
    union = attach_ranking_cross_scores(
        union,
        source="global_popularity",
        item_ranking=models["global_popularity"].item_ranking,
    )
    union = attach_ranking_cross_scores(
        union,
        source="recency_popularity",
        item_ranking=models["recency_popularity"].item_ranking,
    )
    seeds = _item2item_seeds(item_loader)
    union = attach_item2item_cross_scores(
        union,
        seeds=seeds,
        neighbor_table=neighbor_table,
        config=item_config,
        reference_time=reference_time,
    )
    union = attach_implicit_als_cross_scores(union, model=als_model)
    validate_union_features(
        union, config=union_config, require_cross_scores=SOURCE_ORDER
    )
    unknown = union.join(known_items, on="item_id", how="anti").height
    seen = union.join(
        history_shard.select("user_id", "item_id").unique(),
        on=("user_id", "item_id"),
        how="semi",
    ).height
    actual_users = union.select("user_id").unique().sort("user_id")
    if unknown or seen or not actual_users.equals(target_users.sort("user_id")):
        raise ContractValidationError(
            f"invalid candidate shard: unknown={unknown}, seen={seen}, "
            f"user_match={actual_users.equals(target_users.sort('user_id'))}"
        )
    source_rows = {name: sources[name].height for name in SOURCE_ORDER}
    source_users = {
        name: sources[name].get_column("user_id").n_unique() for name in SOURCE_ORDER
    }
    fallback_counts = target_users.join(
        sources["global_popularity"].group_by("user_id").len(name="count"),
        on="user_id",
        how="left",
    ).with_columns(pl.col("count").fill_null(0))
    if fallback_counts.filter(pl.col("count") < minimum_fallback_k).height:
        raise ContractValidationError(
            "global popularity cannot provide the required deterministic fallback"
        )
    contribution = {
        name: int(union.get_column(f"generated_by_{name}").sum())
        for name in SOURCE_ORDER
    }
    return union, seeds, {
        "rows": union.height,
        "users": target_users.height,
        "source_rows": source_rows,
        "source_users": source_users,
        "source_contribution_rows": contribution,
        "fallback_min_candidates": int(fallback_counts.get_column("count").min()),
        "min_candidates": int(union.group_by("user_id").len()["len"].min()),
        "max_candidates": int(union.group_by("user_id").len()["len"].max()),
    }


def build_feature_lookups(
    *,
    history_path: str | Path,
    als_model: ImplicitALSModel,
    reference_time: datetime,
    config: HistoryFeatureConfig,
    destination: str | Path,
) -> dict[str, Any]:
    output = Path(destination)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        validate_history_cutoff(history_path, cutoff=reference_time)
        history = pl.scan_parquet(history_path)
        frames = {
            "user_features.parquet": build_user_features(
                history, cutoff=reference_time, config=config
            ),
            "item_features.parquet": build_item_features(
                history, cutoff=reference_time, config=config
            ),
        }
        als_users, als_items = build_als_factor_norm_lookups(als_model)
        frames["als_user_norms.parquet"] = als_users
        frames["als_item_norms.parquet"] = als_items
        for name, frame in frames.items():
            frame.write_parquet(
                temporary / name, compression="zstd", compression_level=3, statistics=True
            )
        manifest = {
            "artifact_version": 1,
            "history_sha256": sha256_file(history_path),
            "reference_time": reference_time.isoformat(),
            "history_feature_config": config.to_dict(),
            "rows": {name: frame.height for name, frame in frames.items()},
            "files": _directory_checksums(temporary),
        }
        write_json_atomic(temporary / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_feature_lookups(directory: str | Path) -> dict[str, pl.DataFrame]:
    root = Path(directory)
    return {
        "user_features": pl.read_parquet(root / "user_features.parquet"),
        "item_features": pl.read_parquet(root / "item_features.parquet"),
        "als_user_norms": pl.read_parquet(root / "als_user_norms.parquet"),
        "als_item_norms": pl.read_parquet(root / "als_item_norms.parquet"),
    }


def build_feature_shard(
    union: pl.DataFrame,
    *,
    seeds: pl.DataFrame,
    lookups: Mapping[str, pl.DataFrame],
    neighbor_table: pl.DataFrame | pl.LazyFrame | str | Path,
    item2item_config: Item2ItemConfig,
    reference_time: datetime,
    expected_features: Sequence[str],
    expected_feature_dtypes: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    result = build_inference_ranker_shard(
        union,
        user_features=lookups["user_features"],
        item_features=lookups["item_features"],
        item2item_seeds=seeds,
        item2item_neighbors=neighbor_table,
        item2item_config=item2item_config,
        als_user_norms=lookups["als_user_norms"],
        als_item_norms=lookups["als_item_norms"],
        cutoff=reference_time,
    )
    actual = feature_column_names(result)
    if actual != list(expected_features):
        raise ContractValidationError("full-history feature order differs from Task 08")
    if expected_feature_dtypes is not None:
        actual_dtypes = {name: str(result.schema[name]) for name in actual}
        if actual_dtypes != dict(expected_feature_dtypes):
            differences = {
                name: {
                    "expected": expected_feature_dtypes.get(name),
                    "actual": actual_dtypes.get(name),
                }
                for name in actual
                if actual_dtypes.get(name) != expected_feature_dtypes.get(name)
            }
            raise ContractValidationError(
                f"full-history feature dtypes differ from Task 07: {differences}"
            )
    for name in actual:
        column = result.get_column(name)
        if column.dtype.is_float() and not column.is_finite().all():
            raise ContractValidationError(f"feature {name} is not finite")
    return result.select("user_id", "item_id", *actual)


def _training_stratum_expression() -> pl.Expr:
    negative = pl.col("label") == 0
    multi = negative & (pl.col("source_count") >= 2)
    i2i = (
        negative
        & pl.col("generated_by_item2item")
        & (pl.col("generator_rank_item2item") <= 50)
    )
    als = (
        negative
        & pl.col("generated_by_implicit_als")
        & (pl.col("generator_rank_implicit_als") <= 50)
    )
    collaborative = i2i | als
    return (
        pl.when(pl.col("label") == 1)
        .then(pl.lit("positive"))
        .when(multi & collaborative)
        .then(pl.lit("multi_collaborative_top50"))
        .when(multi)
        .then(pl.lit("multi_other"))
        .when(i2i)
        .then(pl.lit("single_item2item_top50"))
        .when(als)
        .then(pl.lit("single_als_top50"))
        .otherwise(pl.lit("easy_negative"))
        .alias("sampling_stratum")
    )


def count_training_strata(parts: Sequence[str | Path]) -> dict[str, int]:
    if not parts:
        raise FullFitPipelineError("training fold has no Parquet parts")
    frame = pl.scan_parquet([Path(path) for path in parts]).select(
        "label",
        "is_training_sample",
        "source_count",
        "generated_by_item2item",
        "generator_rank_item2item",
        "generated_by_implicit_als",
        "generator_rank_implicit_als",
    )
    selected = frame.filter(pl.col("is_training_sample")).with_columns(
        _training_stratum_expression()
    )
    counts = selected.group_by("sampling_stratum").len().collect(engine="streaming")
    result = {name: 0 for name in TRAINING_STRATA}
    result.update(
        {str(row["sampling_stratum"]): int(row["len"]) for row in counts.iter_rows(named=True)}
    )
    result["training_rows"] = sum(result[name] for name in TRAINING_STRATA)
    return result


def build_sampling_plan(
    fold_parts: Mapping[str, Sequence[str | Path]], *, target_rows: int
) -> dict[str, Any]:
    if set(fold_parts) != set(TRAINING_FOLDS):
        raise FullFitPipelineError("training parts must cover all four folds")
    if target_rows <= 0:
        raise FullFitPipelineError("target_rows must be positive")
    base, remainder = divmod(target_rows, len(TRAINING_FOLDS))
    folds: dict[str, Any] = {}
    for index, fold in enumerate(TRAINING_FOLDS):
        target = base + (1 if index < remainder else 0)
        counts = count_training_strata(fold_parts[fold])
        positives = counts["positive"]
        available_negatives = counts["training_rows"] - positives
        if target <= positives or not available_negatives:
            raise FullFitPipelineError(
                f"target allocation for {fold} cannot retain positives and negatives"
            )
        probability = (target - positives) / available_negatives
        if probability <= 0 or probability > 1:
            raise FullFitPipelineError(
                f"invalid secondary sampling probability for {fold}: {probability}"
            )
        strata: dict[str, Any] = {}
        for stratum in TRAINING_STRATA:
            first_stage_probability = 0.05 if stratum == "easy_negative" else 1.0
            second_stage_probability = (
                1.0 if stratum == "positive" else probability
            )
            full_probability = first_stage_probability * second_stage_probability
            strata[stratum] = {
                "source_rows": counts[stratum],
                "task07_inclusion_probability": first_stage_probability,
                "task11_inclusion_probability": second_stage_probability,
                "full_inclusion_probability": full_probability,
                "object_weight": 1.0 / full_probability,
            }
        folds[fold] = {
            "target_rows": target,
            "source_counts": counts,
            "secondary_negative_probability": probability,
            "full_probability_hard_negative": probability,
            "full_probability_easy_negative": 0.05 * probability,
            "weight_hard_negative": 1.0 / probability,
            "weight_easy_negative": 1.0 / (0.05 * probability),
            "strata": strata,
        }
    return {
        "artifact_version": 1,
        "algorithm": "task07_sample_then_splitmix64_user_item_fold",
        "seed": 42,
        "target_rows": target_rows,
        "fold_order": list(TRAINING_FOLDS),
        "folds": folds,
    }


def sample_training_part(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold: str,
    fold_id: int,
    seed: int,
    secondary_negative_probability: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply deterministic second-stage sampling and complete IPW weights."""

    if fold not in TRAINING_FOLDS:
        raise FullFitPipelineError(f"unknown training fold: {fold}")
    if not 0 < secondary_negative_probability <= 1:
        raise FullFitPipelineError("secondary sampling probability must be in (0, 1]")
    required = {
        "user_id",
        "item_id",
        "label",
        "is_training_sample",
        "sampling_probability",
        "source_count",
        "generated_by_item2item",
        "generator_rank_item2item",
        "generated_by_implicit_als",
        "generator_rank_implicit_als",
        *feature_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ContractValidationError(f"training part lacks columns: {sorted(missing)}")
    selected = frame.filter(pl.col("is_training_sample"))
    uniform = deterministic_sampling_uniform(
        selected, seed=seed, fold=f"task11:{fold}"
    )
    selected = (
        selected.with_columns(pl.Series("__uniform", uniform, dtype=pl.Float64))
        .filter(
            (pl.col("label") == 1)
            | (pl.col("__uniform") < secondary_negative_probability)
        )
        .drop("__uniform")
        .with_columns(_training_stratum_expression())
        .with_columns(
            pl.lit(fold_id, dtype=pl.UInt8).alias("fold_id"),
            (
                pl.col("sampling_probability").cast(pl.Float64)
                * pl.when(pl.col("label") == 1)
                .then(1.0)
                .otherwise(secondary_negative_probability)
            ).alias("full_sampling_probability"),
        )
        .with_columns(
            (1.0 / pl.col("full_sampling_probability"))
            .cast(pl.Float32)
            .alias("sample_weight")
        )
        .select(
            "user_id",
            "item_id",
            "fold_id",
            "sampling_stratum",
            pl.col("label").cast(pl.UInt8),
            "full_sampling_probability",
            "sample_weight",
            *(pl.col(name).cast(pl.Float32) for name in feature_columns),
        )
        .sort(("user_id", "item_id"))
    )
    if selected.is_empty():
        raise ContractValidationError("sampled training part is empty")
    weights = selected.get_column("sample_weight")
    if not weights.is_finite().all() or (weights <= 0).any():
        raise ContractValidationError("sample weights must be finite and positive")
    source_positives = frame.filter(pl.col("label") == 1).height
    output_positives = int(selected.get_column("label").sum())
    if output_positives != source_positives:
        raise ContractValidationError("second-stage sampling dropped positive rows")
    strata = selected.group_by("sampling_stratum").len()
    return selected, {
        "rows": selected.height,
        "positive_rows": output_positives,
        "negative_rows": selected.height - output_positives,
        "sample_weight_sum": float(weights.sum()),
        "strata": {
            str(row["sampling_stratum"]): int(row["len"])
            for row in strata.iter_rows(named=True)
        },
    }


def quantize_parquet_parts_via_bounded_dsv(
    parts: Sequence[str | Path],
    *,
    feature_columns: Sequence[str],
    borders_path: str | Path,
    column_description_path: str | Path,
    dsv_path: str | Path,
    output_path: str | Path,
    maximum_dsv_bytes: int,
    thread_count: int = 8,
    random_seed: int = 42,
    progress_callback: Callable[[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Create one Pool through a size-capped, resumable temporary DSV."""

    sources = [Path(path) for path in parts]
    if not sources or any(not path.is_file() for path in sources):
        raise FileNotFoundError("one or more sampled training parts are missing")
    borders = Path(borders_path)
    if not borders.is_file():
        raise FileNotFoundError(borders)
    if maximum_dsv_bytes <= 0:
        raise ValueError("maximum_dsv_bytes must be positive")
    cd = Path(column_description_path)
    write_column_description(cd, feature_columns=feature_columns)
    dsv = Path(dsv_path)
    dsv.parent.mkdir(parents=True, exist_ok=True)
    if not dsv.exists():
        temporary_dsv = dsv.parent / f".{dsv.name}.tmp-{uuid.uuid4().hex}"
        try:
            with temporary_dsv.open(
                "w", encoding="utf-8", buffering=8 * 1024 * 1024
            ) as stream:
                for index, source in enumerate(sources):
                    rows = pl.read_parquet(
                        source, columns=["label", "sample_weight", *feature_columns]
                    ).select(
                        pl.col("label").cast(pl.UInt8),
                        pl.col("sample_weight").cast(pl.Float32),
                        *(pl.col(name).cast(pl.Float32) for name in feature_columns),
                    )
                    rows.write_csv(
                        stream,
                        separator="\t",
                        include_header=False,
                        float_scientific=True,
                    )
                    stream.flush()
                    if temporary_dsv.stat().st_size > maximum_dsv_bytes:
                        raise FullFitPipelineError(
                            "temporary CatBoost DSV exceeded its configured byte cap"
                        )
                    if progress_callback is not None:
                        progress_callback(index, temporary_dsv.stat().st_size)
            os.replace(temporary_dsv, dsv)
        finally:
            temporary_dsv.unlink(missing_ok=True)
    if dsv.stat().st_size > maximum_dsv_bytes:
        raise FullFitPipelineError(
            "existing temporary CatBoost DSV exceeds its configured byte cap"
        )
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        pool = quantize(
            data_path=dsv.as_posix(),
            column_description=cd.as_posix(),
            delimiter="\t",
            has_header=False,
            thread_count=thread_count,
            input_borders=borders.as_posix(),
            task_type="CPU",
            random_seed=random_seed,
        )
        pool.save(temporary.as_posix())
        rows = pool.num_row()
        columns = pool.num_col()
        del pool
        gc.collect()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    restored = Pool(f"quantized://{output.resolve().as_posix()}")
    if restored.num_row() != rows or restored.num_col() != len(feature_columns):
        raise ContractValidationError("restored quantized Pool differs")
    return {
        "rows": rows,
        "feature_count": columns,
        "sha256": sha256_file(output),
        "borders_sha256": sha256_file(borders),
        "columns_sha256": sha256_file(cd),
        "temporary_dsv_bytes": dsv.stat().st_size,
        "temporary_dsv_sha256": sha256_file(dsv),
        "temporary_dsv_byte_cap": maximum_dsv_bytes,
        "transport": "size_capped_regular_dsv",
    }


def recommendations_with_fallback(
    scores: pl.DataFrame,
    *,
    union: pl.DataFrame,
    target_users: pl.DataFrame,
    k: int = 20,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Rank scores and deterministically fill from frozen global popularity."""

    primary = ranker_scores_to_candidates(scores, k=k)
    fallback = (
        union.filter(pl.col("generated_by_global_popularity"))
        .select(
            "user_id",
            "item_id",
            pl.col("generator_score_global_popularity").alias("score"),
            pl.col("generator_rank_global_popularity").alias("rank"),
            pl.lit("global_popularity_fallback").alias("source"),
        )
        .cast(CANDIDATE_SCHEMA)
        .sort(("user_id", "source", "rank"))
    )
    primary_counts = primary.group_by("user_id").len(name="primary_count")
    combined = (
        pl.concat(
            [
                primary.with_columns(pl.lit(0, dtype=pl.UInt8).alias("priority")),
                fallback.with_columns(pl.lit(1, dtype=pl.UInt8).alias("priority")),
            ]
        )
        .sort(
            ("user_id", "priority", "rank", "score", "item_id"),
            descending=(False, False, False, True, False),
        )
        .unique(subset=("user_id", "item_id"), keep="first", maintain_order=True)
        .with_columns(
            pl.col("item_id").cum_count().over("user_id").cast(pl.UInt32).alias("final_rank")
        )
        .filter(pl.col("final_rank") <= k)
        .join(target_users, on="user_id", how="semi")
        .sort(("user_id", "final_rank"))
    )
    counts = target_users.join(
        combined.group_by("user_id").len(name="count"), on="user_id", how="left"
    ).with_columns(pl.col("count").fill_null(0))
    if counts.filter(pl.col("count") != k).height:
        raise ContractValidationError("fallback could not produce exact top-k")
    recommendations = (
        combined.group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").alias("item_ids"))
        .select(FINAL_RECOMMENDATION_SCHEMA.names())
        .cast(FINAL_RECOMMENDATION_SCHEMA)
    )
    validate_final_recommendations(recommendations, expected_k=k)
    long = combined.select(
        "user_id",
        "item_id",
        pl.col("final_rank").alias("rank"),
        "score",
        "source",
    )
    usage = target_users.join(primary_counts, on="user_id", how="left").with_columns(
        pl.col("primary_count").fill_null(0)
    )
    return recommendations, long, {
        "users": target_users.height,
        "rows": long.height,
        "fallback_users": usage.filter(pl.col("primary_count") < k).height,
        "fallback_positions": int(
            long.filter(pl.col("source") == "global_popularity_fallback").height
        ),
    }


def write_checksums_manifest(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    manifest_path = directory / "checksums.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    files = _directory_checksums(directory, exclude=("checksums.json",))
    manifest = {
        "artifact_version": 1,
        "algorithm": "sha256",
        "self_excluded": "checksums.json",
        "files": files,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def verify_checksums_manifest(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    manifest = read_json(directory / "checksums.json")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ContractValidationError("checksum manifest lacks files")
    actual_names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() != "checksums.json"
    }
    if actual_names != set(expected):
        raise ContractValidationError(
            "published file set differs from checksum manifest"
        )
    invalid = [
        name for name, digest in expected.items() if sha256_file(directory / name) != digest
    ]
    if invalid:
        raise ContractValidationError(f"checksum mismatch: {invalid}")
    return {"files": len(expected), "valid": True}


__all__ = [
    "SOURCE_ORDER",
    "TRAINING_FOLDS",
    "TRAINING_STRATA",
    "FullFitPipelineError",
    "atomic_write_parquet",
    "build_candidate_shard",
    "build_feature_lookups",
    "build_feature_shard",
    "build_sampling_plan",
    "candidate_union_config",
    "count_training_strata",
    "derive_prediction_times",
    "fit_candidate_source",
    "load_candidate_source",
    "load_feature_lookups",
    "load_frozen_candidate_configs",
    "materialize_full_history",
    "quantize_parquet_parts_via_bounded_dsv",
    "recommendations_with_fallback",
    "sample_training_part",
    "verify_checksums_manifest",
    "write_checksums_manifest",
]

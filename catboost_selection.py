"""Configuration and deterministic selection logic for Task 09 CatBoost runs."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from experiment_utils import config_sha256, read_json
from rankers import CatBoostPointwiseConfig, validate_feature_columns

SELECTION_ARTIFACT_KIND: Final = "task09_catboost_selection"
SELECTION_ARTIFACT_VERSION: Final = 1


class CatBoostSelectionError(ValueError):
    """Raised for an invalid Task 09 protocol or search configuration."""


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatBoostSelectionError(f"{name} must be a positive integer")
    return value


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatBoostSelectionError(f"{name} must be a number")
    result = float(value)
    if not 0 < result <= 1:
        raise CatBoostSelectionError(f"{name} must be in (0, 1]")
    return result


@dataclass(frozen=True)
class FoldPair:
    pair_id: str
    train_fold: str
    eval_fold: str
    pool_cache: Path

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> FoldPair:
        required = {"pair_id", "train_fold", "eval_fold", "pool_cache"}
        if set(source) != required:
            raise CatBoostSelectionError(
                f"fold pair must contain exactly {sorted(required)}"
            )
        values = {name: source[name] for name in required}
        for name in ("pair_id", "train_fold", "eval_fold", "pool_cache"):
            if not isinstance(values[name], str) or not values[name]:
                raise CatBoostSelectionError(f"fold pair {name} must be non-empty")
        if values["train_fold"] == values["eval_fold"]:
            raise CatBoostSelectionError("train and eval folds must differ")
        if "canonical" in {values["train_fold"], values["eval_fold"]}:
            raise CatBoostSelectionError("canonical fold is forbidden in selection")
        return cls(
            pair_id=values["pair_id"],
            train_fold=values["train_fold"],
            eval_fold=values["eval_fold"],
            pool_cache=Path(values["pool_cache"]),
        )


def _validate_fold_pairs(value: object) -> tuple[FoldPair, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise CatBoostSelectionError("Task 09 requires exactly two fold pairs")
    pairs = tuple(FoldPair.from_mapping(item) for item in value)
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise CatBoostSelectionError("fold pair IDs must be unique")
    if pairs[0].eval_fold != pairs[1].train_fold:
        raise CatBoostSelectionError("fold pairs must form a walk-forward chain")
    if len({pairs[0].train_fold, pairs[0].eval_fold, pairs[1].eval_fold}) != 3:
        raise CatBoostSelectionError("walk-forward folds must be distinct")
    return pairs


def _validate_feature_sets(
    value: object, *, feature_columns: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or "all" not in value:
        raise CatBoostSelectionError("feature_sets must contain all")
    features = validate_feature_columns(feature_columns)
    known = set(features)
    result: dict[str, tuple[str, ...]] = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not name or not isinstance(spec, Mapping):
            raise CatBoostSelectionError("invalid feature-set definition")
        ignored = spec.get("ignored_features")
        if not isinstance(ignored, list):
            raise CatBoostSelectionError(
                f"feature set {name} must define ignored_features"
            )
        checked = tuple(ignored)
        if any(not isinstance(column, str) or not column for column in checked):
            raise CatBoostSelectionError(f"feature set {name} has invalid columns")
        if len(set(checked)) != len(checked):
            raise CatBoostSelectionError(f"feature set {name} has duplicate columns")
        unknown = set(checked).difference(known)
        if unknown:
            raise CatBoostSelectionError(
                f"feature set {name} contains unknown columns: {sorted(unknown)}"
            )
        if len(checked) == len(features):
            raise CatBoostSelectionError(f"feature set {name} removes every feature")
        result[name] = checked
    if result["all"]:
        raise CatBoostSelectionError("feature set all cannot ignore features")
    return result


def _merge_mapping(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_mapping(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_profile(
    base: Mapping[str, Any],
    challenger: Mapping[str, Any] | None,
    *,
    feature_sets: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Resolve one immutable profile from an incumbent and optional overrides."""

    profile = copy.deepcopy(dict(base))
    if challenger is not None:
        if set(challenger) != {"config_id", "overrides"}:
            raise CatBoostSelectionError(
                "challenger must contain exactly config_id and overrides"
            )
        overrides = challenger["overrides"]
        if not isinstance(overrides, Mapping):
            raise CatBoostSelectionError("challenger overrides must be an object")
        profile = _merge_mapping(profile, overrides)
        profile["config_id"] = challenger["config_id"]
    required = {
        "config_id",
        "catboost",
        "train_negative_keep_probability",
        "feature_set",
    }
    if set(profile) != required:
        raise CatBoostSelectionError(
            f"resolved profile must contain exactly {sorted(required)}"
        )
    config_id = profile["config_id"]
    if not isinstance(config_id, str) or not config_id:
        raise CatBoostSelectionError("profile config_id must be non-empty")
    feature_set = profile["feature_set"]
    if feature_set not in feature_sets:
        raise CatBoostSelectionError(f"unknown feature set: {feature_set}")
    probability = _probability(
        profile["train_negative_keep_probability"],
        name=f"{config_id}.train_negative_keep_probability",
    )
    catboost_source = copy.deepcopy(dict(profile["catboost"]))
    catboost_source["config_id"] = config_id
    catboost_source["ignored_features"] = list(feature_sets[feature_set])
    catboost_config = CatBoostPointwiseConfig.from_mapping(catboost_source)
    return {
        "config_id": config_id,
        "catboost": catboost_config.to_dict(),
        "train_negative_keep_probability": probability,
        "feature_set": feature_set,
        "ignored_features": list(feature_sets[feature_set]),
        "profile_sha256": config_sha256(
            {
                "catboost": catboost_config.to_dict(),
                "train_negative_keep_probability": probability,
                "feature_set": feature_set,
            }
        ),
    }


def validate_selection_config(
    source: Mapping[str, Any], *, feature_columns: Sequence[str]
) -> dict[str, Any]:
    required = {
        "task07_artifact",
        "task06_rrf_artifact",
        "task08_artifact",
        "fold_pairs",
        "negative_pool_caches",
        "row_sampling",
        "pool",
        "feature_sets",
        "baseline",
        "search",
        "canonical",
        "inference",
        "seed",
    }
    missing = required.difference(source)
    if missing:
        raise CatBoostSelectionError(f"config lacks fields: {sorted(missing)}")
    unknown = set(source).difference(required)
    if unknown:
        raise CatBoostSelectionError(f"config has unknown fields: {sorted(unknown)}")
    pairs = _validate_fold_pairs(source["fold_pairs"])
    if isinstance(source["seed"], bool) or not isinstance(source["seed"], int):
        raise CatBoostSelectionError("seed must be an integer")
    sampling = source["row_sampling"]
    if not isinstance(sampling, Mapping):
        raise CatBoostSelectionError("row_sampling must be an object")
    if set(sampling) != {"eval_negative_keep_probability", "weighting"}:
        raise CatBoostSelectionError("invalid row_sampling fields")
    _probability(
        sampling["eval_negative_keep_probability"],
        name="eval_negative_keep_probability",
    )
    if sampling["weighting"] != "inverse_sampling_probability":
        raise CatBoostSelectionError("inverse sampling weights are required")
    pool = source["pool"]
    if not isinstance(pool, Mapping):
        raise CatBoostSelectionError("pool must be an object")
    for name in ("border_count", "thread_count"):
        _positive_int(pool.get(name), name=f"pool.{name}")
    if pool.get("feature_border_type") not in {
        "Median",
        "Uniform",
        "UniformAndQuantiles",
        "MaxLogSum",
        "MinEntropy",
        "GreedyLogSum",
    }:
        raise CatBoostSelectionError("unsupported pool.feature_border_type")
    if pool.get("quantization_task_type") not in {"CPU", "GPU"}:
        raise CatBoostSelectionError("pool.quantization_task_type must be CPU or GPU")
    feature_sets = _validate_feature_sets(
        source["feature_sets"], feature_columns=feature_columns
    )
    baseline = resolve_profile(source["baseline"], None, feature_sets=feature_sets)
    if baseline["catboost"]["random_seed"] != source["seed"]:
        raise CatBoostSelectionError("baseline and experiment seeds must match")
    if baseline["catboost"]["border_count"] != pool["border_count"]:
        raise CatBoostSelectionError("CatBoost and pool border_count must match")
    search = source["search"]
    if not isinstance(search, Mapping):
        raise CatBoostSelectionError("search must be an object")
    if set(search) != {
        "primary_metric",
        "secondary_metric",
        "tie_epsilon",
        "max_unique_configs",
        "stages",
    }:
        raise CatBoostSelectionError("invalid search fields")
    if search["primary_metric"] != "precision_at_20_labeled_users":
        raise CatBoostSelectionError("labeled-user P@20 must be the primary metric")
    if search["secondary_metric"] != "precision_at_20_all_targets":
        raise CatBoostSelectionError("all-target P@20 must remain a separate metric")
    tie_epsilon = search["tie_epsilon"]
    if (
        isinstance(tie_epsilon, bool)
        or not isinstance(tie_epsilon, (int, float))
        or not 0 <= float(tie_epsilon) < 1
    ):
        raise CatBoostSelectionError("tie_epsilon must be in [0, 1)")
    stages = search["stages"]
    if not isinstance(stages, list) or not stages:
        raise CatBoostSelectionError("search stages must be a non-empty list")
    config_ids = {baseline["config_id"]}
    challenger_count = 0
    for stage in stages:
        if not isinstance(stage, Mapping) or set(stage) != {"stage_id", "challengers"}:
            raise CatBoostSelectionError("invalid search stage")
        if not isinstance(stage["stage_id"], str) or not stage["stage_id"]:
            raise CatBoostSelectionError("stage_id must be non-empty")
        challengers = stage["challengers"]
        if not isinstance(challengers, list) or not challengers:
            raise CatBoostSelectionError("each stage requires challengers")
        for challenger in challengers:
            resolved = resolve_profile(
                source["baseline"], challenger, feature_sets=feature_sets
            )
            if resolved["config_id"] in config_ids:
                raise CatBoostSelectionError("config IDs must be globally unique")
            config_ids.add(resolved["config_id"])
            challenger_count += 1
    maximum = _positive_int(
        search["max_unique_configs"], name="search.max_unique_configs"
    )
    if 1 + challenger_count > maximum:
        raise CatBoostSelectionError("configured search exceeds max_unique_configs")
    negative_caches = source["negative_pool_caches"]
    if not isinstance(negative_caches, Mapping):
        raise CatBoostSelectionError("negative_pool_caches must be an object")
    for pair in pairs:
        path = negative_caches.get(pair.pair_id)
        if not isinstance(path, str) or not path:
            raise CatBoostSelectionError(
                f"negative pool cache is missing for {pair.pair_id}"
            )
    canonical = source["canonical"]
    if not isinstance(canonical, Mapping) or set(canonical) != {
        "fold",
        "latest_pair_id",
        "evaluate_config_count",
    }:
        raise CatBoostSelectionError("invalid canonical section")
    if (
        canonical["fold"] != "canonical"
        or canonical["latest_pair_id"] != pairs[-1].pair_id
    ):
        raise CatBoostSelectionError("canonical must use the latest walk-forward model")
    if canonical["evaluate_config_count"] != 1:
        raise CatBoostSelectionError("exactly one canonical config is allowed")
    inference = source["inference"]
    if not isinstance(inference, Mapping) or set(inference) != {
        "batch_size",
        "final_k",
    }:
        raise CatBoostSelectionError("invalid inference section")
    _positive_int(inference["batch_size"], name="inference.batch_size")
    if _positive_int(inference["final_k"], name="inference.final_k") != 20:
        raise CatBoostSelectionError("final_k must equal 20")
    result = copy.deepcopy(dict(source))
    result["fold_pairs"] = pairs
    result["feature_sets"] = feature_sets
    result["baseline_resolved"] = baseline
    return result


def load_selection_config(
    path: str | Path, *, feature_columns: Sequence[str]
) -> dict[str, Any]:
    return validate_selection_config(read_json(path), feature_columns=feature_columns)


def stage_challengers(
    incumbent: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    feature_sets: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Apply every stage mutation independently to the current incumbent."""

    base = {
        "config_id": incumbent["config_id"],
        "catboost": incumbent["catboost"],
        "train_negative_keep_probability": incumbent["train_negative_keep_probability"],
        "feature_set": incumbent["feature_set"],
    }
    return [
        resolve_profile(base, challenger, feature_sets=feature_sets)
        for challenger in stage["challengers"]
    ]


def aggregate_fold_results(
    config_id: str, fold_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(fold_results) != 2:
        raise CatBoostSelectionError("exactly two fold results are required")
    labeled = [float(row["precision_at_20_labeled_users"]) for row in fold_results]
    all_targets = [float(row["precision_at_20_all_targets"]) for row in fold_results]
    values = (*labeled, *all_targets)
    if not all(math.isfinite(value) for value in values):
        raise CatBoostSelectionError("fold metrics must be finite")
    trees = [int(row["tree_count"]) for row in fold_results]
    return {
        "config_id": config_id,
        "fold_results": [dict(row) for row in fold_results],
        "mean_precision_at_20_labeled_users": sum(labeled) / len(labeled),
        "min_precision_at_20_labeled_users": min(labeled),
        "fold_spread_precision_at_20_labeled_users": max(labeled) - min(labeled),
        "mean_precision_at_20_all_targets": sum(all_targets) / len(all_targets),
        "mean_tree_count": sum(trees) / len(trees),
        "total_hits": sum(int(row["final_hits"]) for row in fold_results),
        "runtime_seconds": sum(float(row["runtime_seconds"]) for row in fold_results),
        "physical_runtime_seconds": sum(
            float(row.get("physical_runtime_seconds", row["runtime_seconds"]))
            for row in fold_results
        ),
    }


def is_better_result(
    challenger: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    tie_epsilon: float,
    order: Mapping[str, int],
) -> bool:
    """Apply the predeclared metric hierarchy without mixing denominators."""

    higher = (
        "mean_precision_at_20_labeled_users",
        "min_precision_at_20_labeled_users",
    )
    for name in higher:
        delta = float(challenger[name]) - float(incumbent[name])
        if delta > tie_epsilon:
            return True
        if delta < -tie_epsilon:
            return False
    spread_delta = float(
        challenger["fold_spread_precision_at_20_labeled_users"]
    ) - float(incumbent["fold_spread_precision_at_20_labeled_users"])
    if spread_delta < -tie_epsilon:
        return True
    if spread_delta > tie_epsilon:
        return False
    all_delta = float(challenger["mean_precision_at_20_all_targets"]) - float(
        incumbent["mean_precision_at_20_all_targets"]
    )
    if all_delta > tie_epsilon:
        return True
    if all_delta < -tie_epsilon:
        return False
    tree_delta = float(challenger["mean_tree_count"]) - float(
        incumbent["mean_tree_count"]
    )
    if tree_delta < 0:
        return True
    if tree_delta > 0:
        return False
    return order[challenger["config_id"]] < order[incumbent["config_id"]]


def select_best_result(
    results: Sequence[Mapping[str, Any]],
    *,
    tie_epsilon: float,
    order: Sequence[str],
) -> dict[str, Any]:
    if not results:
        raise CatBoostSelectionError("cannot select from empty results")
    order_map = {config_id: index for index, config_id in enumerate(order)}
    if len(order_map) != len(order):
        raise CatBoostSelectionError("selection order must contain unique IDs")
    if any(result["config_id"] not in order_map for result in results):
        raise CatBoostSelectionError("selection order lacks a result config")
    best = dict(results[0])
    for result in results[1:]:
        if is_better_result(
            result,
            best,
            tie_epsilon=tie_epsilon,
            order=order_map,
        ):
            best = dict(result)
    return best


__all__ = [
    "SELECTION_ARTIFACT_KIND",
    "SELECTION_ARTIFACT_VERSION",
    "CatBoostSelectionError",
    "FoldPair",
    "aggregate_fold_results",
    "is_better_result",
    "load_selection_config",
    "resolve_profile",
    "select_best_result",
    "stage_challengers",
    "validate_selection_config",
]

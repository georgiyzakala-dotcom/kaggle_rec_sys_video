"""Group construction and bounded Task 10 CatBoost LTR protocol."""

from __future__ import annotations

import copy
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from experiment_utils import config_sha256, read_json, sha256_file
from rankers import CatBoostLTRConfig, validate_feature_columns
from validation import ContractValidationError

LTR_SELECTION_ARTIFACT_KIND: Final = "task10_catboost_ltr_selection"
LTR_SELECTION_ARTIFACT_VERSION: Final = 1
LTR_POOL_KIND: Final = "task10_grouped_quantized_pools"
LTR_POOL_ARTIFACT_VERSION: Final = 1
SUPPORTED_OBJECTIVES: Final = frozenset(
    {"QuerySoftMax", "YetiRankPairwise", "QueryCrossEntropy"}
)
TASK10_SEARCH_OBJECTIVES: Final = frozenset({"QuerySoftMax", "YetiRankPairwise"})


class CatBoostLTRError(ValueError):
    """Raised for an invalid Task 10 contract or search configuration."""


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatBoostLTRError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatBoostLTRError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CatBoostLTRError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class LTRFoldPair:
    pair_id: str
    train_fold: str
    eval_fold: str
    pool_cache: Path
    borders_source: Path

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> LTRFoldPair:
        required = {
            "pair_id",
            "train_fold",
            "eval_fold",
            "pool_cache",
            "borders_source",
        }
        if not isinstance(source, Mapping) or set(source) != required:
            raise CatBoostLTRError(f"fold pair must contain exactly {sorted(required)}")
        for name in required:
            if not isinstance(source[name], str) or not source[name]:
                raise CatBoostLTRError(f"fold pair {name} must be non-empty")
        if source["train_fold"] == source["eval_fold"]:
            raise CatBoostLTRError("train and eval folds must differ")
        if "canonical" in {source["train_fold"], source["eval_fold"]}:
            raise CatBoostLTRError("canonical fold is forbidden in selection")
        return cls(
            pair_id=source["pair_id"],
            train_fold=source["train_fold"],
            eval_fold=source["eval_fold"],
            pool_cache=Path(source["pool_cache"]),
            borders_source=Path(source["borders_source"]),
        )


def _validate_fold_pairs(value: object) -> tuple[LTRFoldPair, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise CatBoostLTRError("Task 10 requires exactly two fold pairs")
    pairs = tuple(LTRFoldPair.from_mapping(item) for item in value)
    if pairs[0].eval_fold != pairs[1].train_fold:
        raise CatBoostLTRError("fold pairs must form a walk-forward chain")
    if len({pair.pair_id for pair in pairs}) != 2:
        raise CatBoostLTRError("fold pair IDs must be unique")
    if len({pairs[0].train_fold, pairs[0].eval_fold, pairs[1].eval_fold}) != 3:
        raise CatBoostLTRError("walk-forward folds must be distinct")
    return pairs


def build_dense_group_mapping(target_users: pl.DataFrame) -> pl.DataFrame:
    """Map exact UInt64 target IDs to stable dense Int64 group IDs."""

    if (
        target_users.columns != ["user_id"]
        or target_users.schema["user_id"] != pl.UInt64
    ):
        raise ContractValidationError("target users must contain one UInt64 user_id")
    if (
        target_users.get_column("user_id").null_count()
        or target_users.get_column("user_id").n_unique() != target_users.height
    ):
        raise ContractValidationError("target users must be unique and non-null")
    ordered = target_users.sort("user_id")
    return ordered.with_row_index("group_id").select(
        pl.col("user_id"), pl.col("group_id").cast(pl.Int64)
    )


def splitmix64(values: np.ndarray, *, seed: int) -> np.ndarray:
    """Return stable UInt64 hashes without converting IDs through float."""

    source = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        result = source + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return result ^ (result >> np.uint64(31))


def select_complete_eval_users(
    labeled_users: pl.DataFrame, *, count: int, seed: int
) -> pl.DataFrame:
    """Select complete evaluation queries by a stable user-level hash."""

    if (
        labeled_users.columns != ["user_id"]
        or labeled_users.schema["user_id"] != pl.UInt64
    ):
        raise ContractValidationError("labeled users must contain one UInt64 user_id")
    if labeled_users.get_column("user_id").n_unique() != labeled_users.height:
        raise ContractValidationError("labeled users must be unique")
    checked_count = _positive_int(count, name="eval_group_count")
    if checked_count > labeled_users.height:
        raise ContractValidationError("eval_group_count exceeds labeled user count")
    users = labeled_users.get_column("user_id").to_numpy()
    hashes = splitmix64(users, seed=seed)
    order = np.lexsort((users, hashes))[:checked_count]
    return pl.DataFrame({"user_id": pl.Series(users[order], dtype=pl.UInt64)}).sort(
        "user_id"
    )


def validate_grouped_source_rows(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    require_training_flag: bool,
) -> None:
    """Validate immutable Task 07 row identity, order, labels and features."""

    features = validate_feature_columns(feature_columns)
    required = {"user_id", "item_id", "label", *features}
    if require_training_flag:
        required.add("is_training_sample")
    missing = required.difference(frame.columns)
    if missing:
        raise ContractValidationError(f"ranker shard lacks columns: {sorted(missing)}")
    if frame.schema["user_id"] != pl.UInt64 or frame.schema["item_id"] != pl.Int32:
        raise ContractValidationError("ranker shard ID dtypes differ")
    if frame.schema["label"] != pl.UInt8:
        raise ContractValidationError("ranker shard label must be UInt8")
    if frame.select(pl.struct("user_id", "item_id").is_duplicated().any()).item():
        raise ContractValidationError("ranker shard contains duplicate pairs")
    identities = frame.select("user_id", "item_id")
    if not identities.equals(identities.sort(("user_id", "item_id"))):
        raise ContractValidationError("ranker shard rows are not ordered by IDs")
    if frame.select(
        pl.any_horizontal(pl.col(name).is_null() for name in required).any()
    ).item():
        raise ContractValidationError("ranker shard contains null values")
    if frame.filter(~pl.col("label").is_in([0, 1])).height:
        raise ContractValidationError("ranker labels must be binary")
    for name in features:
        values = frame.get_column(name).cast(pl.Float32).to_numpy()
        if not np.isfinite(values).all():
            raise ContractValidationError(f"feature {name} is not finite")


def prepare_grouped_rows(
    frame: pl.DataFrame,
    *,
    group_mapping: pl.DataFrame,
    feature_columns: Sequence[str],
    training_only: bool,
    selected_users: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Prepare one complete, ordered GroupId DSV part without object weights."""

    features = validate_feature_columns(feature_columns)
    validate_grouped_source_rows(
        frame,
        feature_columns=features,
        require_training_flag=training_only,
    )
    source = frame.filter(pl.col("is_training_sample")) if training_only else frame
    if selected_users is not None:
        source = source.join(selected_users, on="user_id", how="semi")
    if source.height == 0:
        raise ContractValidationError("grouped rows must not be empty")
    identities = source.select("user_id", "item_id", "label")
    joined = source.join(group_mapping, on="user_id", how="left", validate="m:1")
    if joined.get_column("group_id").null_count():
        raise ContractValidationError("group mapping does not cover every user")
    joined = joined.sort(("group_id", "item_id"))
    if not joined.select("user_id", "item_id", "label").equals(identities):
        raise ContractValidationError("group mapping changed immutable row order")
    groups = joined.group_by("group_id", maintain_order=True).agg(
        pl.len().alias("row_count"),
        pl.col("label").sum().cast(pl.Int64).alias("positive_count"),
    )
    result = joined.select(
        pl.col("label").cast(pl.UInt8),
        pl.col("group_id").cast(pl.Int64),
        *(pl.col(name).cast(pl.Float32) for name in features),
    )
    diagnostics = {
        "rows": result.height,
        "groups": groups.height,
        "positive_rows": int(result.get_column("label").sum()),
        "positive_groups": int(groups.filter(pl.col("positive_count") > 0).height),
        "zero_positive_groups": int(
            groups.filter(pl.col("positive_count") == 0).height
        ),
        "min_group_size": int(groups.get_column("row_count").min()),
        "max_group_size": int(groups.get_column("row_count").max()),
    }
    return result, diagnostics


def write_grouped_dsv_part(
    frame: pl.DataFrame,
    destination: str | Path,
    *,
    group_mapping: pl.DataFrame,
    feature_columns: Sequence[str],
    training_only: bool,
    selected_users: pl.DataFrame | None = None,
) -> dict[str, Any]:
    output = Path(destination)
    if output.exists():
        raise FileExistsError(output)
    rows, diagnostics = prepare_grouped_rows(
        frame,
        group_mapping=group_mapping,
        feature_columns=feature_columns,
        training_only=training_only,
        selected_users=selected_users,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        rows.write_csv(
            temporary,
            separator="\t",
            include_header=False,
            float_scientific=True,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **diagnostics,
        "size_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }


def write_ltr_column_description(
    destination: str | Path, *, feature_columns: Sequence[str]
) -> None:
    features = validate_feature_columns(feature_columns)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    lines = ["0\tLabel", "1\tGroupId"]
    lines.extend(f"{index + 2}\tNum\t{name}" for index, name in enumerate(features))
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def resolve_ltr_profile(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != {"config_id", "catboost"}:
        raise CatBoostLTRError("LTR profile requires config_id and catboost")
    if not isinstance(source["config_id"], str) or not source["config_id"]:
        raise CatBoostLTRError("profile config_id must be non-empty")
    catboost_source = copy.deepcopy(dict(source["catboost"]))
    catboost_source["config_id"] = source["config_id"]
    try:
        config = CatBoostLTRConfig.from_mapping(catboost_source)
    except (TypeError, ValueError) as error:
        raise CatBoostLTRError(
            f"invalid profile {source['config_id']}: {error}"
        ) from error
    payload = {"config_id": source["config_id"], "catboost": config.to_dict()}
    return {**payload, "profile_sha256": config_sha256(payload)}


def resolve_inherited_challenger(
    incumbent: Mapping[str, Any], challenger: Mapping[str, Any]
) -> dict[str, Any]:
    required = {"config_id", "family_overrides"}
    if not isinstance(challenger, Mapping) or set(challenger) != required:
        raise CatBoostLTRError(
            "inherited challenger requires config_id and family_overrides"
        )
    current = CatBoostLTRConfig.from_mapping(incumbent["catboost"])
    overrides = challenger["family_overrides"]
    if not isinstance(overrides, Mapping) or set(overrides) != TASK10_SEARCH_OBJECTIVES:
        raise CatBoostLTRError(
            "family_overrides must define both production Task 10 objectives"
        )
    selected = overrides[current.objective]
    if not isinstance(selected, Mapping):
        raise CatBoostLTRError("family override must be an object")
    source = {
        "config_id": challenger["config_id"],
        "catboost": _merge_mapping(current.to_dict(), selected),
    }
    return resolve_ltr_profile(source)


def validate_ltr_selection_config(
    source: Mapping[str, Any], *, feature_columns: Sequence[str]
) -> dict[str, Any]:
    required = {
        "task07_artifact",
        "task06_rrf_artifact",
        "task08_artifact",
        "task09_artifact",
        "fold_pairs",
        "pool",
        "objective_stage",
        "objective_exclusions",
        "inherited_stages",
        "search",
        "pointwise_comparator",
        "canonical",
        "inference",
        "resources",
        "seed",
    }
    optional = {"recovery"}
    if (
        not isinstance(source, Mapping)
        or not required.issubset(source)
        or set(source).difference(required | optional)
    ):
        missing = (
            required.difference(source) if isinstance(source, Mapping) else required
        )
        unknown = (
            set(source).difference(required | optional)
            if isinstance(source, Mapping)
            else set()
        )
        raise CatBoostLTRError(
            f"invalid config fields; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    features = validate_feature_columns(feature_columns)
    if len(features) != 201:
        raise CatBoostLTRError("Task 10 requires all 201 Task 07 features")
    if isinstance(source["seed"], bool) or not isinstance(source["seed"], int):
        raise CatBoostLTRError("seed must be an integer")
    pairs = _validate_fold_pairs(source["fold_pairs"])
    pool = source["pool"]
    pool_fields = {
        "border_count",
        "quantization_task_type",
        "thread_count",
        "eval_group_count",
        "eval_group_seed",
    }
    if not isinstance(pool, Mapping) or set(pool) != pool_fields:
        raise CatBoostLTRError("invalid pool fields")
    for name in ("border_count", "thread_count", "eval_group_count"):
        _positive_int(pool[name], name=f"pool.{name}")
    if pool["quantization_task_type"] != "CPU":
        raise CatBoostLTRError("grouped pools must reuse borders on CPU")
    if not isinstance(pool["eval_group_seed"], int):
        raise CatBoostLTRError("pool.eval_group_seed must be an integer")
    objective_stage = source["objective_stage"]
    if not isinstance(objective_stage, Mapping) or set(objective_stage) != {
        "stage_id",
        "profiles",
    }:
        raise CatBoostLTRError("invalid objective_stage")
    profiles = objective_stage["profiles"]
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise CatBoostLTRError("objective_stage requires exactly two profiles")
    resolved_profiles = [resolve_ltr_profile(profile) for profile in profiles]
    objectives = {
        CatBoostLTRConfig.from_mapping(profile["catboost"]).objective
        for profile in resolved_profiles
    }
    if objectives != TASK10_SEARCH_OBJECTIVES:
        raise CatBoostLTRError(
            "objective_stage must cover QuerySoftMax and YetiRankPairwise"
        )
    exclusions = source["objective_exclusions"]
    if exclusions != [
        {
            "objective": "QueryCrossEntropy",
            "status": "excluded_before_full_run",
            "reason": "catboost_1_2_10_gpu_max_query_size_256",
            "smoke_train_max_group_size": 273,
            "smoke_eval_max_group_size": 709,
        }
    ]:
        raise CatBoostLTRError("QueryCrossEntropy GPU exclusion must be recorded")
    for profile in resolved_profiles:
        config = CatBoostLTRConfig.from_mapping(profile["catboost"])
        if config.random_seed != source["seed"]:
            raise CatBoostLTRError("profile seed differs from experiment seed")
        if config.border_count != pool["border_count"]:
            raise CatBoostLTRError("profile border_count differs from pool")
    inherited = source["inherited_stages"]
    if not isinstance(inherited, list) or len(inherited) != 3:
        raise CatBoostLTRError("exactly three inherited stages are required")
    stage_ids = {objective_stage["stage_id"]}
    config_ids = {profile["config_id"] for profile in resolved_profiles}
    if len(config_ids) != 2:
        raise CatBoostLTRError("profile config IDs must be unique")
    for stage in inherited:
        if not isinstance(stage, Mapping) or set(stage) != {"stage_id", "challengers"}:
            raise CatBoostLTRError("invalid inherited stage")
        if not isinstance(stage["stage_id"], str) or not stage["stage_id"]:
            raise CatBoostLTRError("stage_id must be non-empty")
        if stage["stage_id"] in stage_ids:
            raise CatBoostLTRError("stage IDs must be unique")
        stage_ids.add(stage["stage_id"])
        challengers = stage["challengers"]
        if not isinstance(challengers, list) or len(challengers) != 2:
            raise CatBoostLTRError("each inherited stage requires two challengers")
        for challenger in challengers:
            config_id = (
                challenger.get("config_id") if isinstance(challenger, Mapping) else None
            )
            if (
                not isinstance(config_id, str)
                or not config_id
                or config_id in config_ids
            ):
                raise CatBoostLTRError("challenger config IDs must be unique")
            config_ids.add(config_id)
            for base in resolved_profiles:
                resolve_inherited_challenger(base, challenger)
    search = source["search"]
    if not isinstance(search, Mapping) or set(search) != {
        "primary_metric",
        "secondary_metric",
        "tie_epsilon",
        "max_unique_configs",
    }:
        raise CatBoostLTRError("invalid search fields")
    if search["primary_metric"] != "precision_at_20_labeled_users":
        raise CatBoostLTRError("labeled-user P@20 must be primary")
    if search["secondary_metric"] != "precision_at_20_all_targets":
        raise CatBoostLTRError("all-target P@20 must remain separate")
    epsilon = search["tie_epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not 0 <= float(epsilon) < 1
    ):
        raise CatBoostLTRError("tie_epsilon must be in [0, 1)")
    maximum = _positive_int(search["max_unique_configs"], name="max_unique_configs")
    if len(config_ids) != 8 or len(config_ids) > maximum:
        raise CatBoostLTRError("Task 10 must define exactly eight bounded profiles")
    comparator = source["pointwise_comparator"]
    comparator_fields = {
        "config_id",
        "artifact",
        "profile_sha256",
        "fold_results",
    }
    if not isinstance(comparator, Mapping) or set(comparator) != comparator_fields:
        raise CatBoostLTRError("invalid pointwise comparator")
    if comparator["config_id"] != "s12_pos_weight_16":
        raise CatBoostLTRError("pointwise comparator must be Task 09 s12")
    fold_results = comparator["fold_results"]
    if not isinstance(fold_results, list) or [
        row.get("pair_id") for row in fold_results
    ] != [pair.pair_id for pair in pairs]:
        raise CatBoostLTRError("pointwise comparator fold order differs")
    canonical = source["canonical"]
    if not isinstance(canonical, Mapping) or set(canonical) != {
        "fold",
        "latest_pair_id",
        "evaluate_config_count",
    }:
        raise CatBoostLTRError("invalid canonical section")
    if canonical != {
        "fold": "canonical",
        "latest_pair_id": pairs[-1].pair_id,
        "evaluate_config_count": 1,
    }:
        raise CatBoostLTRError("canonical must evaluate the latest model exactly once")
    inference = source["inference"]
    if not isinstance(inference, Mapping) or set(inference) != {
        "batch_size",
        "final_k",
    }:
        raise CatBoostLTRError("invalid inference section")
    _positive_int(inference["batch_size"], name="inference.batch_size")
    if _positive_int(inference["final_k"], name="inference.final_k") != 20:
        raise CatBoostLTRError("final_k must equal 20")
    resources = source["resources"]
    resource_fields = {
        "catboost_version",
        "cpu_thread_limit",
        "max_rss_budget_gib",
        "min_available_ram_gib",
        "min_free_disk_gib",
        "min_gpu_count",
        "min_gpu_memory_mib",
    }
    host_resource_fields = {
        "windows_host_drive",
        "min_windows_host_free_gib",
        "stop_windows_host_free_gib",
    }
    if (
        not isinstance(resources, Mapping)
        or not resource_fields.issubset(resources)
        or set(resources).difference(resource_fields | host_resource_fields)
        or bool(set(resources).intersection(host_resource_fields))
        != host_resource_fields.issubset(resources)
    ):
        raise CatBoostLTRError("invalid resources section")
    if resources["catboost_version"] != "1.2.10":
        raise CatBoostLTRError("Task 10 is pinned to catboost==1.2.10")
    for name in (
        "cpu_thread_limit",
        "max_rss_budget_gib",
        "min_available_ram_gib",
        "min_free_disk_gib",
        "min_gpu_count",
        "min_gpu_memory_mib",
    ):
        _positive_int(resources[name], name=f"resources.{name}")
    if resources["max_rss_budget_gib"] != 50:
        raise CatBoostLTRError("Task 10 max RSS budget must remain 50 GiB")
    if resources["min_available_ram_gib"] >= resources["max_rss_budget_gib"]:
        raise CatBoostLTRError("RAM preflight must remain below the 50 GiB budget")
    if host_resource_fields.issubset(resources):
        drive = resources["windows_host_drive"]
        if (
            not isinstance(drive, str)
            or len(drive) != 1
            or not drive.isascii()
            or not drive.isalpha()
            or drive != drive.upper()
        ):
            raise CatBoostLTRError("windows_host_drive must be one uppercase letter")
        for name in (
            "min_windows_host_free_gib",
            "stop_windows_host_free_gib",
        ):
            _positive_int(resources[name], name=f"resources.{name}")
        if (
            resources["stop_windows_host_free_gib"]
            >= resources["min_windows_host_free_gib"]
        ):
            raise CatBoostLTRError(
                "Windows host stop threshold must be below launch threshold"
            )
    if pool["thread_count"] > resources["cpu_thread_limit"]:
        raise CatBoostLTRError("pool threads exceed the CPU thread limit")
    for profile in resolved_profiles:
        if (
            CatBoostLTRConfig.from_mapping(profile["catboost"]).thread_count
            > resources["cpu_thread_limit"]
        ):
            raise CatBoostLTRError("profile threads exceed the CPU thread limit")
    recovery = source.get("recovery")
    if recovery is not None:
        recovery_fields = {
            "source_run_id",
            "source_config",
            "source_config_sha256",
            "source_checkpoint_dir",
            "source_checkpoint_sha256",
            "protocol_amendment",
            "protocol_amendment_sha256",
            "import_completed_profiles",
            "resource_probe",
        }
        if not isinstance(recovery, Mapping) or set(recovery) != recovery_fields:
            raise CatBoostLTRError("invalid recovery section")
        if (
            not isinstance(recovery["source_run_id"], str)
            or not recovery["source_run_id"]
        ):
            raise CatBoostLTRError("recovery source_run_id must be non-empty")
        for name in (
            "source_config",
            "source_checkpoint_dir",
            "protocol_amendment",
        ):
            value = recovery[name]
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                raise CatBoostLTRError(f"recovery {name} must be a relative path")
        for name in (
            "source_config_sha256",
            "source_checkpoint_sha256",
            "protocol_amendment_sha256",
        ):
            value = recovery[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise CatBoostLTRError(f"recovery {name} must be a SHA-256 digest")
        imported = recovery["import_completed_profiles"]
        if (
            not isinstance(imported, list)
            or not imported
            or len(set(imported)) != len(imported)
            or any(config_id not in config_ids for config_id in imported)
        ):
            raise CatBoostLTRError("invalid recovery import_completed_profiles")
        probe = recovery["resource_probe"]
        probe_fields = {"profile_id", "fold_pair", "iterations", "on_gpu_oom"}
        if not isinstance(probe, Mapping) or set(probe) != probe_fields:
            raise CatBoostLTRError("invalid recovery resource_probe")
        by_id = {profile["config_id"]: profile for profile in resolved_profiles}
        probe_id = probe["profile_id"]
        if (
            probe_id not in by_id
            or probe_id in imported
            or CatBoostLTRConfig.from_mapping(by_id[probe_id]["catboost"]).objective
            != "YetiRankPairwise"
        ):
            raise CatBoostLTRError(
                "resource probe must target the non-imported YetiRankPairwise profile"
            )
        if probe["fold_pair"] not in {pair.pair_id for pair in pairs}:
            raise CatBoostLTRError("resource probe fold_pair is unknown")
        if probe["iterations"] != 1:
            raise CatBoostLTRError("resource probe must use exactly one iteration")
        if probe["on_gpu_oom"] != "exclude_profile_and_continue":
            raise CatBoostLTRError("unsupported resource probe OOM policy")
    result = copy.deepcopy(dict(source))
    result["fold_pairs"] = pairs
    result["objective_profiles"] = resolved_profiles
    return result


def load_ltr_selection_config(
    path: str | Path, *, feature_columns: Sequence[str]
) -> dict[str, Any]:
    return validate_ltr_selection_config(
        read_json(path), feature_columns=feature_columns
    )


def inherited_stage_challengers(
    incumbent: Mapping[str, Any], stage: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        resolve_inherited_challenger(incumbent, challenger)
        for challenger in stage["challengers"]
    ]


__all__ = [
    "LTR_POOL_ARTIFACT_VERSION",
    "LTR_POOL_KIND",
    "LTR_SELECTION_ARTIFACT_KIND",
    "LTR_SELECTION_ARTIFACT_VERSION",
    "SUPPORTED_OBJECTIVES",
    "TASK10_SEARCH_OBJECTIVES",
    "CatBoostLTRError",
    "LTRFoldPair",
    "build_dense_group_mapping",
    "inherited_stage_challengers",
    "load_ltr_selection_config",
    "prepare_grouped_rows",
    "resolve_inherited_challenger",
    "resolve_ltr_profile",
    "select_complete_eval_users",
    "splitmix64",
    "validate_grouped_source_rows",
    "validate_ltr_selection_config",
    "write_grouped_dsv_part",
    "write_ltr_column_description",
]

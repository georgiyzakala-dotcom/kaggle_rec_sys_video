"""Contracts for the Task13 D_all recipe refitted for next-day inference."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from history_profiles import PROFILE_FEATURES, HistoryProfiles
from pipeline import TRAINING_FOLDS
from rankers import CatBoostPointwiseConfig


def validate_recipe(
    config: dict, reference_config: dict, reference_model: dict
) -> None:
    """Keep statistical choices frozen; allow bounded smoke and operational knobs."""
    if config["training"]["fold_order"] != list(TRAINING_FOLDS):
        raise ValueError("final ranker must use all four supervised temporal folds")
    if config["quantization"] != reference_config["quantization"]:
        raise ValueError("quantization must match the Task13 train-only recipe")
    if config["variant"] != "D_all" or config["catboost"]["ignored_features"]:
        raise ValueError("final model must use every D_all feature")
    actual = CatBoostPointwiseConfig.from_mapping(config["catboost"]).to_dict()
    expected = CatBoostPointwiseConfig.from_mapping(
        reference_model["catboost"]
    ).to_dict()
    operational = {
        "config_id",
        "thread_count",
        "devices",
        "metric_period",
        "snapshot_interval_seconds",
        "gpu_ram_part",
    }
    if config["mode"] == "smoke":
        operational |= {"iterations", "depth", "task_type"}
    for key in expected.keys() - operational:
        if actual[key] != expected[key]:
            raise ValueError(f"Task13 statistical parameter differs: {key}")
    if config["mode"] == "full":
        if (
            config["training"]["target_rows"]
            != reference_config["training"]["target_rows"]
        ):
            raise ValueError("retain the validated Task13 row budget for the RAM limit")
        if config["catboost"]["iterations"] != reference_model["tree_count"]:
            raise ValueError("fixed tree budget differs from the selected model")
    if (
        tuple(reference_model["feature_columns"][-len(PROFILE_FEATURES) :])
        != PROFILE_FEATURES
    ):
        raise ValueError("Task13 profile feature schema differs")


def validate_training_horizon(inputs: dict, prediction_start: datetime) -> None:
    """Require chronological historical folds whose label windows already ended."""
    from datetime import timedelta

    cutoffs = [datetime.fromisoformat(inputs[f]["cutoff"]) for f in TRAINING_FOLDS]
    if cutoffs != sorted(set(cutoffs)):
        raise ValueError("training fold cutoffs must be strictly increasing")
    if any(cutoff + timedelta(days=1) > prediction_start for cutoff in cutoffs):
        raise ValueError("training labels extend into the prediction period")


def add_profile_features(base: pl.DataFrame, profiles: HistoryProfiles) -> pl.DataFrame:
    """Preserve exact IDs and row order while joining the same 20 training scalars."""
    extra = profiles.features(base.select("user_id", "item_id"))
    result = base.join(
        extra,
        on=["user_id", "item_id"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if any(result[c].null_count() for c in PROFILE_FEATURES):
        raise ValueError("production profile join lost candidates")
    return result

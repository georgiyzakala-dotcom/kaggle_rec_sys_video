"""Deterministic reciprocal-rank-fusion baseline for offline candidates."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import polars as pl

from candidate_pipeline import (
    CandidateUnionConfig,
    generator_columns,
    validate_union_features,
)
from data_utils import GROUND_TRUTH_SCHEMA
from interfaces import CANDIDATE_SCHEMA, RANKER_OUTPUT_SCHEMA
from popularity import fill_with_global_popularity
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_no_nulls,
    validate_ranker_output,
    validate_unique_keys,
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _weight(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


@dataclass(frozen=True)
class RRFConfig:
    """Portable source caps, weights, and reciprocal-rank constant."""

    config_id: str
    candidate_config: CandidateUnionConfig
    weights: tuple[tuple[str, float], ...]
    rrf_constant: float = 60.0
    final_k: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be a non-empty string")
        if not isinstance(self.candidate_config, CandidateUnionConfig):
            raise TypeError("candidate_config must be CandidateUnionConfig")
        weights = tuple(
            (source, _weight(value, name=f"weight[{source}]"))
            for source, value in self.weights
        )
        names = [source for source, _ in weights]
        if len(set(names)) != len(names):
            raise ValueError("RRF weight source names must be unique")
        if set(names) != set(self.candidate_config.source_names):
            raise ValueError("RRF weights must cover candidate sources exactly")
        if not any(value > 0 for _, value in weights):
            raise ValueError("at least one RRF weight must be positive")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "rrf_constant",
            _positive_float(self.rrf_constant, name="rrf_constant"),
        )
        object.__setattr__(self, "final_k", _positive_int(self.final_k, name="final_k"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RRFConfig:
        if not isinstance(value, Mapping):
            raise TypeError("RRF config must be a mapping")
        source_caps = value.get("source_caps")
        weights = value.get("weights")
        if not isinstance(source_caps, Mapping) or not isinstance(weights, Mapping):
            raise TypeError("RRF config requires source_caps and weights objects")
        source_order = value.get("source_order", list(source_caps))
        if (
            not isinstance(source_order, list)
            or set(source_order) != set(source_caps)
            or len(source_order) != len(source_caps)
        ):
            raise ValueError("RRF source_order must cover source_caps exactly")
        candidate_config = CandidateUnionConfig.from_mapping(
            {str(name): int(source_caps[name]) for name in source_order},
            total_cap=int(value["total_cap"]),
        )
        return cls(
            config_id=str(value["config_id"]),
            candidate_config=candidate_config,
            weights=tuple(
                (source, float(weights[source]))
                for source in candidate_config.source_names
            ),
            rrf_constant=float(value.get("rrf_constant", 60.0)),
            final_k=int(value.get("final_k", 20)),
        )

    def weight_for(self, source: str) -> float:
        for name, value in self.weights:
            if name == source:
                return value
        raise KeyError(source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            **self.candidate_config.to_dict(),
            "weights": dict(self.weights),
            "rrf_constant": self.rrf_constant,
            "final_k": self.final_k,
        }


class RRFEnsembleModel:
    """Score an existing candidate union without fitting source models."""

    source_name = "rrf_ensemble"

    def __init__(self, config: RRFConfig | Mapping[str, Any]) -> None:
        self._config = (
            config if isinstance(config, RRFConfig) else RRFConfig.from_dict(config)
        )

    @property
    def config(self) -> RRFConfig:
        return self._config

    def _validate_materialized_config(
        self, materialized_config: CandidateUnionConfig
    ) -> None:
        if set(materialized_config.source_names) != set(
            self._config.candidate_config.source_names
        ):
            raise ValueError(
                "materialized union and RRF config must contain identical sources"
            )
        for source in self._config.candidate_config.source_names:
            if self._config.candidate_config.cap_for(
                source
            ) > materialized_config.cap_for(source):
                raise ValueError(
                    f"RRF cap for {source} exceeds materialized source cap"
                )

    def score(
        self,
        features: pl.DataFrame,
        *,
        materialized_config: CandidateUnionConfig,
        validate_features: bool = True,
    ) -> pl.DataFrame:
        """Return one finite RRF score for each eligible union pair."""

        self._validate_materialized_config(materialized_config)
        if validate_features:
            validate_union_features(features, config=materialized_config)
        eligibility: list[pl.Expr] = []
        contributions: list[pl.Expr] = []
        for spec in self._config.candidate_config.sources:
            columns = generator_columns(spec.source)
            active = pl.col(columns["generated"]) & (
                pl.col(columns["rank"]) <= spec.cap
            )
            eligibility.append(active)
            contributions.append(
                pl.when(active)
                .then(
                    self._config.weight_for(spec.source)
                    / (
                        self._config.rrf_constant
                        + pl.col(columns["rank"]).cast(pl.Float64)
                    )
                )
                .otherwise(0.0)
            )
        eligible = pl.any_horizontal(eligibility)
        result = (
            features.filter(eligible)
            .select(
                "user_id",
                "item_id",
                pl.sum_horizontal(contributions).cast(pl.Float64).alias("ranker_score"),
            )
            .sort(
                ("user_id", "ranker_score", "item_id"),
                descending=(False, True, False),
            )
            .cast(RANKER_OUTPUT_SCHEMA)
        )
        validate_ranker_output(result)
        return result

    def rank_candidates(
        self,
        features: pl.DataFrame,
        *,
        materialized_config: CandidateUnionConfig,
        validate_features: bool = True,
    ) -> pl.DataFrame:
        """Convert RRF scores to a typed single-source candidate table."""

        scored = self.score(
            features,
            materialized_config=materialized_config,
            validate_features=validate_features,
        )
        ranked = (
            scored.with_columns(
                pl.col("item_id")
                .cum_count()
                .over("user_id")
                .cast(pl.UInt32)
                .alias("rank"),
                pl.lit(self.source_name).alias("source"),
            )
            .rename({"ranker_score": "score"})
            .select(CANDIDATE_SCHEMA.names())
            .cast(CANDIDATE_SCHEMA)
        )
        validate_candidate_output(
            ranked,
            k=self._config.candidate_config.total_cap,
            source_name=self.source_name,
        )
        return ranked

    def recommend(
        self,
        features: pl.DataFrame,
        *,
        materialized_config: CandidateUnionConfig,
        popularity_candidates: pl.DataFrame,
        target_users: pl.DataFrame,
        history_daily: pl.DataFrame | pl.LazyFrame,
    ) -> pl.DataFrame:
        """Return deterministic top-k with the established global fallback."""

        ranked = self.rank_candidates(features, materialized_config=materialized_config)
        return fill_with_global_popularity(
            ranked,
            popularity_candidates,
            target_users,
            history_daily,
            k=self._config.final_k,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "model": "rrf_ensemble",
            "source_name": self.source_name,
            "rrf_config": self._config.to_dict(),
            "score": "sum(weight / (rrf_constant + generator_rank))",
            "tie_break": ["ranker_score DESC", "item_id ASC"],
        }

    def save(self, artifact_dir: str | Path) -> None:
        """Atomically save a portable parameter-only RRF artifact."""

        directory = Path(artifact_dir)
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite RRF artifact: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = directory.parent / f".{directory.name}.staging-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            value = {
                "artifact_version": 1,
                **self.get_config(),
            }
            (temporary / "model_config.json").write_text(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, directory)
        except BaseException:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
            raise

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> RRFEnsembleModel:
        path = Path(artifact_dir) / "model_config.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read RRF model config: {error}") from error
        if (
            not isinstance(value, dict)
            or value.get("artifact_version") != 1
            or value.get("model") != "rrf_ensemble"
            or value.get("source_name") != cls.source_name
        ):
            raise ContractValidationError("invalid RRF portable artifact")
        return cls(RRFConfig.from_dict(value["rrf_config"]))


def source_contribution_metrics(
    features: pl.DataFrame,
    *,
    ground_truth: pl.DataFrame,
    materialized_config: CandidateUnionConfig,
    candidate_config: CandidateUnionConfig,
) -> dict[str, Any]:
    """Count per-source, exclusive, and pairwise relevant hits under caps."""

    if ground_truth.schema != GROUND_TRUTH_SCHEMA:
        raise ContractValidationError("ground truth has invalid schema")
    validate_no_nulls(ground_truth, name="ground truth")
    validate_unique_keys(ground_truth, keys=("user_id", "item_id"), name="ground truth")
    if set(materialized_config.source_names) != set(candidate_config.source_names):
        raise ValueError("candidate configs must cover identical sources")
    validate_union_features(features, config=materialized_config)
    active_names: dict[str, str] = {}
    expressions: list[pl.Expr] = []
    for spec in candidate_config.sources:
        if spec.cap > materialized_config.cap_for(spec.source):
            raise ValueError(f"cap for {spec.source} exceeds materialized cap")
        generator = generator_columns(spec.source)
        active_name = f"__active_{spec.source}"
        active_names[spec.source] = active_name
        expressions.append(
            (
                pl.col(generator["generated"]) & (pl.col(generator["rank"]) <= spec.cap)
            ).alias(active_name)
        )
    hits = (
        features.with_columns(expressions)
        .join(ground_truth, on=["user_id", "item_id"], how="semi")
        .with_columns(
            pl.sum_horizontal(
                pl.col(name).cast(pl.UInt8) for name in active_names.values()
            )
            .cast(pl.UInt8)
            .alias("__active_count")
        )
    )
    source_hits = {
        source: hits.filter(pl.col(name)).height
        for source, name in active_names.items()
    }
    exclusive_hits = {
        source: hits.filter(pl.col(name) & (pl.col("__active_count") == 1)).height
        for source, name in active_names.items()
    }
    pairwise_overlap: dict[str, int] = {}
    sources = candidate_config.source_names
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            pairwise_overlap[f"{left}__{right}"] = hits.filter(
                pl.col(active_names[left]) & pl.col(active_names[right])
            ).height
    return {
        "union_relevant_hits": hits.filter(pl.col("__active_count") > 0).height,
        "source_relevant_hits": source_hits,
        "exclusive_relevant_hits": exclusive_hits,
        "pairwise_relevant_hit_overlap": pairwise_overlap,
    }


__all__ = [
    "RRFConfig",
    "RRFEnsembleModel",
    "source_contribution_metrics",
]

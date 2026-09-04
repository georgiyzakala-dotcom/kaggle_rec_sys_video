"""Portable pointwise and group-aware CatBoost rankers for Task 07 data."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import catboost
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRanker, FeaturesData, Pool

from experiment_utils import sha256_file, write_json_atomic
from interfaces import (
    CANDIDATE_SCHEMA,
    RANKER_OUTPUT_SCHEMA,
    RankerDataLoader,
    RankerModel,
)
from ranker_data import deterministic_sampling_uniform
from validation import ContractValidationError, validate_ranker_output

POINTWISE_MODEL_KIND: Final = "catboost_pointwise"
POINTWISE_ARTIFACT_VERSION: Final = 1
LTR_MODEL_KIND: Final = "catboost_ltr"
LTR_ARTIFACT_VERSION: Final = 1


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


def _non_negative_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _probability(value: object, *, name: str, allow_one: bool = True) -> float:
    result = _positive_float(value, name=name)
    upper_valid = result <= 1 if allow_one else result < 1
    if not upper_valid:
        operator = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{name} must be in {operator}")
    return result


def validate_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return a stable non-empty feature list after strict validation."""

    result = tuple(columns)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError("feature_columns must contain non-empty names")
    if len(set(result)) != len(result):
        raise ValueError("feature_columns must be unique")
    forbidden = {
        "user_id",
        "item_id",
        "label",
        "is_hard_negative",
        "is_training_sample",
        "sampling_probability",
        "sample_weight",
    }
    overlap = forbidden.intersection(result)
    if overlap:
        raise ValueError(
            f"metadata columns cannot be model features: {sorted(overlap)}"
        )
    return result


@dataclass(frozen=True)
class CatBoostPointwiseConfig:
    """Portable binary pointwise CatBoost profile."""

    config_id: str = "pointwise_gpu_baseline"
    loss_function: str = "Logloss"
    eval_metric: str = "Logloss"
    iterations: int = 1200
    depth: int = 7
    learning_rate: float = 0.08
    l2_leaf_reg: float = 3.0
    border_count: int = 32
    random_seed: int = 42
    task_type: str = "GPU"
    devices: str = "0"
    thread_count: int = 8
    early_stopping_rounds: int = 80
    metric_period: int = 10
    gpu_ram_part: float = 0.85
    boosting_type: str = "Plain"
    bootstrap_type: str = "Bernoulli"
    subsample: float | None = 0.8
    bagging_temperature: float | None = None
    random_strength: float = 1.0
    scale_pos_weight: float = 1.0
    ignored_features: tuple[str, ...] = ()
    snapshot_interval_seconds: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be non-empty")
        if self.loss_function not in {"Logloss", "CrossEntropy"}:
            raise ValueError("pointwise loss_function must be Logloss or CrossEntropy")
        if not isinstance(self.eval_metric, str) or not self.eval_metric:
            raise ValueError("eval_metric must be non-empty")
        for name in (
            "iterations",
            "depth",
            "border_count",
            "thread_count",
            "early_stopping_rounds",
            "metric_period",
            "snapshot_interval_seconds",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in ("learning_rate", "l2_leaf_reg", "scale_pos_weight"):
            object.__setattr__(
                self, name, _positive_float(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "random_strength",
            _non_negative_float(self.random_strength, name="random_strength"),
        )
        if self.loss_function != "Logloss" and self.scale_pos_weight != 1.0:
            raise ValueError("scale_pos_weight is supported only with Logloss")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, Integral
        ):
            raise TypeError("random_seed must be an integer")
        object.__setattr__(self, "random_seed", int(self.random_seed))
        if self.task_type not in {"CPU", "GPU"}:
            raise ValueError("task_type must be CPU or GPU")
        if not isinstance(self.devices, str) or not self.devices:
            raise ValueError("devices must be non-empty")
        object.__setattr__(
            self,
            "gpu_ram_part",
            _probability(self.gpu_ram_part, name="gpu_ram_part", allow_one=False),
        )
        if self.boosting_type not in {"Plain", "Ordered"}:
            raise ValueError("unsupported boosting_type")
        if self.bootstrap_type not in {"Bernoulli", "Bayesian"}:
            raise ValueError("bootstrap_type must be Bernoulli or Bayesian")
        if self.bootstrap_type == "Bernoulli":
            if self.subsample is None:
                raise ValueError("Bernoulli bootstrap requires subsample")
            object.__setattr__(
                self,
                "subsample",
                _probability(self.subsample, name="subsample", allow_one=True),
            )
            if self.bagging_temperature is not None:
                raise ValueError(
                    "bagging_temperature is incompatible with Bernoulli bootstrap"
                )
        else:
            if self.subsample is not None:
                raise ValueError("Bayesian bootstrap requires subsample=null")
            temperature = (
                1.0 if self.bagging_temperature is None else self.bagging_temperature
            )
            object.__setattr__(
                self,
                "bagging_temperature",
                _non_negative_float(temperature, name="bagging_temperature"),
            )
        ignored = tuple(self.ignored_features)
        if any(not isinstance(name, str) or not name for name in ignored):
            raise ValueError("ignored_features must contain non-empty names")
        if len(set(ignored)) != len(ignored):
            raise ValueError("ignored_features must be unique")
        object.__setattr__(self, "ignored_features", ignored)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> CatBoostPointwiseConfig:
        if not isinstance(source, Mapping):
            raise TypeError("catboost config must be an object")
        fields = cls.__dataclass_fields__
        unknown = set(source).difference(fields)
        if unknown:
            raise ValueError(f"unknown CatBoost parameters: {sorted(unknown)}")
        return cls(**{name: source[name] for name in source})

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["ignored_features"] = list(self.ignored_features)
        return result

    def training_params(self, *, train_dir: str | Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "loss_function": self.loss_function,
            "eval_metric": self.eval_metric,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
            "random_seed": self.random_seed,
            "task_type": self.task_type,
            "thread_count": self.thread_count,
            "boosting_type": self.boosting_type,
            "bootstrap_type": self.bootstrap_type,
            "random_strength": self.random_strength,
            "allow_writing_files": train_dir is not None,
        }
        if self.bootstrap_type == "Bernoulli":
            result["subsample"] = self.subsample
        else:
            result["bagging_temperature"] = self.bagging_temperature
        if self.scale_pos_weight != 1.0:
            result["scale_pos_weight"] = self.scale_pos_weight
        if self.ignored_features:
            result["ignored_features"] = list(self.ignored_features)
        if train_dir is not None:
            result["train_dir"] = Path(train_dir).as_posix()
        if self.task_type == "GPU":
            result.update(devices=self.devices, gpu_ram_part=self.gpu_ram_part)
        return result


def _parse_loss_description(value: str) -> tuple[str, dict[str, str]]:
    if not isinstance(value, str) or not value:
        raise ValueError("loss_function must be non-empty")
    name, separator, raw_parameters = value.partition(":")
    parameters: dict[str, str] = {}
    if separator:
        for token in raw_parameters.split(";"):
            key, equals, parameter_value = token.partition("=")
            if not equals or not key or not parameter_value or key in parameters:
                raise ValueError(f"invalid loss_function parameter: {token!r}")
            parameters[key] = parameter_value
    return name, parameters


def _loss_float(parameters: Mapping[str, str], name: str) -> float:
    try:
        value = float(parameters[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"loss parameter {name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"loss parameter {name} must be finite")
    return value


@dataclass(frozen=True)
class CatBoostLTRConfig:
    """Strict CatBoost 1.2.10 group-aware ranking profile."""

    config_id: str
    loss_function: str
    eval_metric: str = "PrecisionAt:top=20;border=0"
    custom_metric: tuple[str, ...] = (
        "NDCG:top=20;type=Base;denominator=LogPosition",
    )
    iterations: int = 1200
    depth: int = 7
    learning_rate: float = 0.08
    l2_leaf_reg: float = 3.0
    border_count: int = 32
    random_seed: int = 42
    task_type: str = "GPU"
    devices: str = "0"
    thread_count: int = 8
    early_stopping_rounds: int = 100
    metric_period: int = 20
    gpu_ram_part: float = 0.80
    boosting_type: str = "Plain"
    bootstrap_type: str = "Bernoulli"
    subsample: float | None = 0.8
    sampling_unit: str = "Object"
    random_strength: float | None = None
    group_weighting: str = "unit"
    snapshot_interval_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be non-empty")
        objective, parameters = _parse_loss_description(self.loss_function)
        allowed_parameters = {
            "QuerySoftMax": {"beta"},
            "YetiRankPairwise": {"mode", "permutations", "decay"},
            "QueryCrossEntropy": {"alpha"},
        }
        if objective not in allowed_parameters:
            raise ValueError(
                "LTR loss_function must be QuerySoftMax, YetiRankPairwise, "
                "or QueryCrossEntropy"
            )
        unknown = set(parameters).difference(allowed_parameters[objective])
        if unknown:
            raise ValueError(
                f"unsupported {objective} parameters: {sorted(unknown)}"
            )
        if "use_weights" in parameters:
            raise ValueError("use_weights cannot be an optimization parameter")
        if objective == "QuerySoftMax":
            beta = _loss_float(parameters, "beta") if parameters else 1.0
            _positive_float(beta, name="QuerySoftMax.beta")
        elif objective == "YetiRankPairwise":
            if parameters.get("mode", "Classic") != "Classic":
                raise ValueError("GPU YetiRankPairwise requires mode=Classic")
            if "permutations" in parameters:
                try:
                    permutations = int(parameters["permutations"])
                except ValueError as error:
                    raise ValueError(
                        "YetiRankPairwise.permutations must be an integer"
                    ) from error
                if str(permutations) != parameters["permutations"]:
                    raise ValueError(
                        "YetiRankPairwise.permutations must be an integer"
                    )
                _positive_int(permutations, name="YetiRankPairwise.permutations")
            if "decay" in parameters:
                decay = _loss_float(parameters, "decay")
                if not 0 < decay < 1:
                    raise ValueError("YetiRankPairwise.decay must be in (0, 1)")
        else:
            alpha = _loss_float(parameters, "alpha") if parameters else 0.95
            if not 0 <= alpha <= 1:
                raise ValueError("QueryCrossEntropy.alpha must be in [0, 1]")
        if self.eval_metric != "PrecisionAt:top=20;border=0":
            raise ValueError("LTR eval_metric must be PrecisionAt@20")
        custom = tuple(self.custom_metric)
        if custom != (
            "NDCG:top=20;type=Base;denominator=LogPosition",
        ):
            raise ValueError("LTR custom_metric must be NDCG@20")
        object.__setattr__(self, "custom_metric", custom)
        for name in (
            "iterations",
            "depth",
            "border_count",
            "thread_count",
            "early_stopping_rounds",
            "metric_period",
            "snapshot_interval_seconds",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in ("learning_rate", "l2_leaf_reg"):
            object.__setattr__(
                self, name, _positive_float(getattr(self, name), name=name)
            )
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, Integral
        ):
            raise TypeError("random_seed must be an integer")
        object.__setattr__(self, "random_seed", int(self.random_seed))
        if self.task_type not in {"CPU", "GPU"}:
            raise ValueError("task_type must be CPU or GPU")
        if not isinstance(self.devices, str) or not self.devices:
            raise ValueError("devices must be non-empty")
        object.__setattr__(
            self,
            "gpu_ram_part",
            _probability(self.gpu_ram_part, name="gpu_ram_part", allow_one=False),
        )
        if self.boosting_type != "Plain":
            raise ValueError("LTR boosting_type must be Plain")
        if self.bootstrap_type not in {"Bernoulli", "No"}:
            raise ValueError("LTR bootstrap_type must be Bernoulli or No")
        if self.bootstrap_type == "Bernoulli":
            if self.subsample is None:
                raise ValueError("Bernoulli bootstrap requires subsample")
            object.__setattr__(
                self,
                "subsample",
                _probability(self.subsample, name="subsample", allow_one=True),
            )
        elif self.subsample is not None:
            raise ValueError("No bootstrap requires subsample=null")
        if self.sampling_unit not in {"Object", "Group"}:
            raise ValueError("sampling_unit must be Object or Group")
        if (
            self.task_type == "GPU"
            and self.sampling_unit == "Group"
            and objective != "YetiRankPairwise"
        ):
            raise ValueError(
                "GPU sampling_unit=Group is supported only by YetiRankPairwise"
            )
        if self.task_type == "GPU" and objective in {
            "YetiRankPairwise",
            "QueryCrossEntropy",
        } and self.depth > 8:
            raise ValueError(f"GPU {objective} depth cannot exceed 8")
        if objective in {"YetiRankPairwise", "QueryCrossEntropy"}:
            if self.random_strength is not None:
                raise ValueError(f"random_strength is unsupported for {objective}")
        else:
            strength = 1.0 if self.random_strength is None else self.random_strength
            object.__setattr__(
                self,
                "random_strength",
                _non_negative_float(strength, name="random_strength"),
            )
        if self.group_weighting not in {"unit", "inverse_positive_count"}:
            raise ValueError(
                "group_weighting must be unit or inverse_positive_count"
            )

    @property
    def objective(self) -> str:
        return _parse_loss_description(self.loss_function)[0]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> CatBoostLTRConfig:
        if not isinstance(source, Mapping):
            raise TypeError("catboost LTR config must be an object")
        fields = cls.__dataclass_fields__
        unknown = set(source).difference(fields)
        if unknown:
            raise ValueError(f"unknown CatBoost LTR parameters: {sorted(unknown)}")
        values = {name: source[name] for name in source}
        if "custom_metric" in values:
            values["custom_metric"] = tuple(values["custom_metric"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["custom_metric"] = list(self.custom_metric)
        return result

    def training_params(self, *, train_dir: str | Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "loss_function": self.loss_function,
            "eval_metric": self.eval_metric,
            "custom_metric": list(self.custom_metric),
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
            "random_seed": self.random_seed,
            "task_type": self.task_type,
            "thread_count": self.thread_count,
            "boosting_type": self.boosting_type,
            "bootstrap_type": self.bootstrap_type,
            "sampling_unit": self.sampling_unit,
            "allow_writing_files": train_dir is not None,
        }
        if self.bootstrap_type == "Bernoulli":
            result["subsample"] = self.subsample
        if self.random_strength is not None:
            result["random_strength"] = self.random_strength
        if train_dir is not None:
            result["train_dir"] = Path(train_dir).as_posix()
        if self.task_type == "GPU":
            result.update(devices=self.devices, gpu_ram_part=self.gpu_ram_part)
        return result


@dataclass(frozen=True)
class CatBoostFitBatch:
    train_pool: Pool
    eval_pool: Pool | None


@dataclass(frozen=True)
class CatBoostPredictBatch:
    user_ids: np.ndarray
    item_ids: np.ndarray
    features: np.ndarray


def _load_pool(value: str | Path | Pool) -> Pool:
    if isinstance(value, Pool):
        return value
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return Pool(f"quantized://{path.resolve().as_posix()}")


class CatBoostRankerDataLoader(
    RankerDataLoader[CatBoostFitBatch, CatBoostPredictBatch]
):
    """Expose immutable quantized fit pools and ordered task-07 features."""

    def __init__(self, *, feature_columns: Sequence[str], seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._feature_columns = validate_feature_columns(feature_columns)
        self._train_source: str | Path | Pool | None = None
        self._eval_source: str | Path | Pool | None = None
        self._train_pool: Pool | None = None
        self._eval_pool: Pool | None = None
        self._predict_source: pl.DataFrame | None = None
        self._predict_ids: pl.DataFrame | None = None
        self._predict_matrix: np.ndarray | None = None

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    def _load_fit_data(self, **kwargs: Any) -> None:
        if set(kwargs) not in ({"train_pool"}, {"train_pool", "eval_pool"}):
            raise TypeError(
                "load_fit_data requires train_pool= and optional eval_pool="
            )
        self._train_source = kwargs["train_pool"]
        self._eval_source = kwargs.get("eval_pool")
        self._train_pool = None
        self._eval_pool = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_fit_data does not accept arguments")
        assert self._train_source is not None
        train = _load_pool(self._train_source)
        evaluation = (
            _load_pool(self._eval_source)
            if self._eval_source is not None
            else None
        )
        pools = [("train", train)]
        if evaluation is not None:
            pools.append(("eval", evaluation))
        for name, pool in pools:
            if pool.num_col() != len(self._feature_columns):
                raise ContractValidationError(
                    f"{name} pool feature count differs from task-07 schema"
                )
            names = tuple(pool.get_feature_names())
            if names and names != self._feature_columns:
                raise ContractValidationError(
                    f"{name} pool feature order differs from task-07 schema"
                )
            if pool.num_row() == 0:
                raise ContractValidationError(f"{name} pool must not be empty")
        self._train_pool = train
        self._eval_pool = evaluation

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[CatBoostFitBatch]:
        if batch_size is not None:
            raise ValueError("CatBoost fit consumes one complete quantized pool")
        assert self._train_pool is not None
        yield CatBoostFitBatch(self._train_pool, self._eval_pool)

    def _load_predict_data(self, **kwargs: Any) -> None:
        if set(kwargs) != {"frame"} or not isinstance(kwargs["frame"], pl.DataFrame):
            raise TypeError("load_predict_data requires frame=PolarsDataFrame")
        self._predict_source = kwargs["frame"]
        self._predict_ids = None
        self._predict_matrix = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("prepare_predict_data does not accept arguments")
        assert self._predict_source is not None
        frame = self._predict_source
        required = ("user_id", "item_id", *self._feature_columns)
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ContractValidationError(f"prediction frame lacks columns: {missing}")
        if frame.schema["user_id"] != pl.UInt64 or frame.schema["item_id"] != pl.Int32:
            raise ContractValidationError("prediction IDs have invalid dtypes")
        if frame.select(
            pl.any_horizontal(pl.col(name).is_null() for name in required).any()
        ).item():
            raise ContractValidationError("prediction frame contains null values")
        ordered = frame.select(required).sort(("user_id", "item_id"))
        duplicates = ordered.select(
            pl.struct("user_id", "item_id").is_duplicated().any()
        ).item()
        if duplicates:
            raise ContractValidationError("prediction frame contains duplicate pairs")
        matrix = np.empty(
            (ordered.height, len(self._feature_columns)), dtype=np.float32
        )
        for index, name in enumerate(self._feature_columns):
            dtype = ordered.schema[name]
            if dtype == pl.String or dtype == pl.Categorical or dtype == pl.Object:
                raise ContractValidationError(f"feature {name} is not numeric")
            values = ordered.get_column(name).cast(pl.Float32).to_numpy()
            if not np.isfinite(values).all():
                raise ContractValidationError(f"feature {name} is not finite")
            matrix[:, index] = values
        self._predict_ids = ordered.select("user_id", "item_id")
        self._predict_matrix = np.ascontiguousarray(matrix)

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[CatBoostPredictBatch]:
        assert self._predict_ids is not None and self._predict_matrix is not None
        size = batch_size or max(self._predict_ids.height, 1)
        users = self._predict_ids.get_column("user_id").to_numpy()
        items = self._predict_ids.get_column("item_id").to_numpy()
        for offset in range(0, self._predict_ids.height, size):
            stop = min(offset + size, self._predict_ids.height)
            yield CatBoostPredictBatch(
                user_ids=users[offset:stop],
                item_ids=items[offset:stop],
                features=np.ascontiguousarray(self._predict_matrix[offset:stop]),
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "catboost_ranker",
            "seed": self.seed,
            "feature_columns": list(self._feature_columns),
            "feature_count": len(self._feature_columns),
            "fit_input": "quantized_catboost_pool",
            "predict_input": "immutable_task07_shard",
        }


class CatBoostPointwiseModel(RankerModel):
    """Binary CatBoost model emitting candidate-level raw scores."""

    def __init__(
        self,
        config: CatBoostPointwiseConfig | Mapping[str, Any],
        *,
        feature_columns: Sequence[str],
    ) -> None:
        self._config = (
            config
            if isinstance(config, CatBoostPointwiseConfig)
            else CatBoostPointwiseConfig.from_mapping(config)
        )
        self._feature_columns = validate_feature_columns(feature_columns)
        unknown_ignored = set(self._config.ignored_features).difference(
            self._feature_columns
        )
        if unknown_ignored:
            raise ValueError(
                "ignored_features are absent from feature_columns: "
                f"{sorted(unknown_ignored)}"
            )
        if len(self._config.ignored_features) == len(self._feature_columns):
            raise ValueError("ignored_features cannot remove every feature")
        self._model: CatBoostClassifier | None = None
        self._best_iteration_override: int | None = None
        self._best_score_override: dict[str, dict[str, float]] | None = None

    @property
    def config(self) -> CatBoostPointwiseConfig:
        return self._config

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def tree_count(self) -> int:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return int(self._model.tree_count_)

    @property
    def best_iteration(self) -> int:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        if self._best_iteration_override is not None:
            return self._best_iteration_override
        value = self._model.get_best_iteration()
        return int(
            value if value is not None and int(value) >= 0 else self.tree_count - 1
        )

    @property
    def best_score(self) -> dict[str, dict[str, float]]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        if self._best_score_override is not None:
            return {
                dataset: dict(scores)
                for dataset, scores in self._best_score_override.items()
            }
        return {
            str(dataset): {
                str(metric): float(value) for metric, value in scores.items()
            }
            for dataset, scores in self._model.get_best_score().items()
        }

    def _fit(self, loader: RankerDataLoader[Any, Any], **kwargs: Any) -> None:
        if not isinstance(loader, CatBoostRankerDataLoader):
            raise TypeError("CatBoostPointwiseModel requires CatBoostRankerDataLoader")
        if loader.feature_columns != self._feature_columns:
            raise ContractValidationError("loader feature order differs from model")
        batches = list(loader.iter_fit_batches())
        if len(batches) != 1:
            raise ContractValidationError(
                "CatBoost fit requires exactly one pool batch"
            )
        train_dir = kwargs.pop("train_dir", None)
        snapshot_file = kwargs.pop("snapshot_file", None)
        log_cout = kwargs.pop("log_cout", None)
        log_cerr = kwargs.pop("log_cerr", None)
        fixed_tree_budget = kwargs.pop("fixed_tree_budget", False)
        if not isinstance(fixed_tree_budget, bool):
            raise TypeError("fixed_tree_budget must be boolean")
        if kwargs:
            raise TypeError(f"unsupported fit arguments: {sorted(kwargs)}")
        params = self._config.training_params(train_dir=train_dir)
        model = CatBoostClassifier(**params)
        fit_kwargs: dict[str, Any] = {"verbose": self._config.metric_period}
        if fixed_tree_budget:
            if batches[0].eval_pool is not None:
                raise ContractValidationError(
                    "fixed-tree fit must not receive an evaluation pool"
                )
            fit_kwargs["use_best_model"] = False
        else:
            if batches[0].eval_pool is None:
                raise ContractValidationError(
                    "early-stopped fit requires an evaluation pool"
                )
            fit_kwargs.update(
                eval_set=batches[0].eval_pool,
                use_best_model=True,
                early_stopping_rounds=self._config.early_stopping_rounds,
            )
        if snapshot_file is not None:
            fit_kwargs.update(
                save_snapshot=True,
                snapshot_file=Path(snapshot_file).as_posix(),
                snapshot_interval=self._config.snapshot_interval_seconds,
            )
        if log_cout is not None:
            fit_kwargs["log_cout"] = log_cout
        if log_cerr is not None:
            fit_kwargs["log_cerr"] = log_cerr
        model.fit(batches[0].train_pool, **fit_kwargs)
        if fixed_tree_budget and int(model.tree_count_) != self._config.iterations:
            raise ContractValidationError(
                "fixed-tree CatBoost fit did not consume the configured budget"
            )
        if tuple(model.feature_names_) != self._feature_columns:
            raise ContractValidationError("fitted CatBoost feature order changed")
        self._model = model
        self._best_iteration_override = None
        self._best_score_override = None

    def _predict(
        self, loader: RankerDataLoader[Any, Any], **kwargs: Any
    ) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        if not isinstance(loader, CatBoostRankerDataLoader):
            raise TypeError("CatBoostPointwiseModel requires CatBoostRankerDataLoader")
        if loader.feature_columns != self._feature_columns:
            raise ContractValidationError("loader feature order differs from model")
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unsupported predict arguments: {sorted(kwargs)}")
        outputs: list[pl.DataFrame] = []
        for batch in loader.iter_predict_batches(batch_size=batch_size):
            data = FeaturesData(
                num_feature_data=batch.features,
                num_feature_names=list(self._feature_columns),
            )
            score = np.asarray(
                self._model.predict(data, prediction_type="RawFormulaVal"),
                dtype=np.float64,
            ).reshape(-1)
            outputs.append(
                pl.DataFrame(
                    {
                        "user_id": pl.Series(batch.user_ids, dtype=pl.UInt64),
                        "item_id": pl.Series(batch.item_ids, dtype=pl.Int32),
                        "ranker_score": pl.Series(score, dtype=pl.Float64),
                    }
                )
            )
        result = (
            pl.concat(outputs, rechunk=True).cast(RANKER_OUTPUT_SCHEMA)
            if outputs
            else pl.DataFrame(schema=RANKER_OUTPUT_SCHEMA)
        )
        validate_ranker_output(result)
        return result

    def get_feature_importance(self) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        values = self._model.get_feature_importance(type="FeatureImportance")
        return pl.DataFrame(
            {
                "feature": list(self._feature_columns),
                "importance": pl.Series(values, dtype=pl.Float64),
            }
        ).sort(("importance", "feature"), descending=(True, False))

    def get_config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": POINTWISE_MODEL_KIND,
            "catboost": self._config.to_dict(),
            "feature_columns": list(self._feature_columns),
            "feature_count": len(self._feature_columns),
            "score": "RawFormulaVal",
            "tie_break": ["ranker_score DESC", "item_id ASC"],
        }
        if self._model is not None:
            result.update(
                tree_count=self.tree_count,
                best_iteration=self.best_iteration,
                best_score=self.best_score,
            )
        return result

    def save(self, artifact_dir: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        destination = Path(artifact_dir)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite model artifact: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = (
            destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
        )
        temporary.mkdir()
        try:
            model_path = temporary / "model.cbm"
            self._model.save_model(model_path.as_posix(), format="cbm")
            metadata = {
                "artifact_version": POINTWISE_ARTIFACT_VERSION,
                "kind": POINTWISE_MODEL_KIND,
                "catboost_version": catboost.__version__,
                **self.get_config(),
                "model_sha256": sha256_file(model_path),
            }
            write_json_atomic(temporary / "model_config.json", metadata)
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
            raise

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> CatBoostPointwiseModel:
        root = Path(artifact_dir)
        config_path = root / "model_config.json"
        model_path = root / "model.cbm"
        if not config_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(root)
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read CatBoost artifact: {error}") from error
        if (
            not isinstance(metadata, dict)
            or metadata.get("artifact_version") != POINTWISE_ARTIFACT_VERSION
            or metadata.get("kind") != POINTWISE_MODEL_KIND
            or metadata.get("model") != POINTWISE_MODEL_KIND
        ):
            raise ContractValidationError("invalid CatBoost model artifact")
        if sha256_file(model_path) != metadata.get("model_sha256"):
            raise ContractValidationError("CatBoost model checksum differs")
        model = cls(
            CatBoostPointwiseConfig.from_mapping(metadata["catboost"]),
            feature_columns=metadata["feature_columns"],
        )
        fitted = CatBoostClassifier()
        fitted.load_model(model_path.as_posix(), format="cbm")
        if tuple(fitted.feature_names_) != model.feature_columns:
            raise ContractValidationError("restored CatBoost feature order differs")
        model._model = fitted
        model._best_iteration_override = int(metadata["best_iteration"])
        raw_best_score = metadata.get("best_score", {})
        if not isinstance(raw_best_score, dict):
            raise ContractValidationError("restored CatBoost best score is invalid")
        model._best_score_override = {
            str(dataset): {
                str(metric): float(value) for metric, value in scores.items()
            }
            for dataset, scores in raw_best_score.items()
            if isinstance(scores, dict)
        }
        if model.tree_count != int(metadata["tree_count"]):
            raise ContractValidationError("restored CatBoost tree count differs")
        return model


def validate_pool_groups(pool: Pool, *, name: str) -> dict[str, int]:
    """Validate non-empty contiguous CatBoost groups without decoding their IDs."""

    hashes = pool.get_group_id_hash()
    if hashes is None:
        raise ContractValidationError(f"{name} pool lacks group_id metadata")
    values = np.asarray(hashes, dtype=np.uint64).reshape(-1)
    if values.size != pool.num_row() or values.size == 0:
        raise ContractValidationError(f"{name} pool group metadata has invalid size")
    starts = np.flatnonzero(
        np.concatenate((np.array([True]), values[1:] != values[:-1]))
    )
    unique = np.unique(values[starts])
    if unique.size != starts.size:
        raise ContractValidationError(f"{name} pool contains a split group")
    stops = np.concatenate((starts[1:], np.array([values.size])))
    sizes = stops - starts
    if np.any(sizes <= 0):
        raise ContractValidationError(f"{name} pool contains an empty group")
    return {
        "rows": int(values.size),
        "groups": int(starts.size),
        "min_group_size": int(sizes.min()),
        "max_group_size": int(sizes.max()),
    }


def apply_inverse_positive_group_weights(pool: Pool) -> dict[str, float | int]:
    """Assign mean-one inverse-positive-count weights to a grouped Pool."""

    diagnostics = validate_pool_groups(pool, name="train")
    hashes = np.asarray(pool.get_group_id_hash(), dtype=np.uint64).reshape(-1)
    labels = np.asarray(pool.get_label(), dtype=np.float64).reshape(-1)
    if labels.size != hashes.size or not np.isfinite(labels).all():
        raise ContractValidationError("train labels are invalid")
    starts = np.flatnonzero(
        np.concatenate((np.array([True]), hashes[1:] != hashes[:-1]))
    )
    stops = np.concatenate((starts[1:], np.array([hashes.size])))
    positive_counts = np.add.reduceat((labels > 0).astype(np.int64), starts)
    group_weights = 1.0 / np.maximum(positive_counts, 1)
    group_weights /= group_weights.mean()
    row_weights = np.repeat(group_weights, stops - starts).astype(np.float32)
    pool.set_group_weight(row_weights)
    return {
        **diagnostics,
        "positive_groups": int((positive_counts > 0).sum()),
        "zero_positive_groups": int((positive_counts == 0).sum()),
        "min_group_weight": float(group_weights.min()),
        "max_group_weight": float(group_weights.max()),
        "mean_group_weight": float(group_weights.mean()),
    }


class CatBoostLTRDataLoader(CatBoostRankerDataLoader):
    """Load immutable quantized pools and require valid contiguous query groups."""

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        super()._prepare_fit_data(**kwargs)
        assert self._train_pool is not None and self._eval_pool is not None
        validate_pool_groups(self._train_pool, name="train")
        validate_pool_groups(self._eval_pool, name="eval")

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "loader": "catboost_ltr",
            "group_contract": "contiguous_group_id_required",
        }


class CatBoostRankerModel(RankerModel):
    """Portable group-aware CatBoost ranker emitting raw candidate scores."""

    def __init__(
        self,
        config: CatBoostLTRConfig | Mapping[str, Any],
        *,
        feature_columns: Sequence[str],
    ) -> None:
        self._config = (
            config
            if isinstance(config, CatBoostLTRConfig)
            else CatBoostLTRConfig.from_mapping(config)
        )
        self._feature_columns = validate_feature_columns(feature_columns)
        self._model: CatBoostRanker | None = None
        self._best_iteration_override: int | None = None
        self._best_score_override: dict[str, dict[str, float]] | None = None
        self._group_weight_diagnostics: dict[str, float | int] | None = None

    @property
    def config(self) -> CatBoostLTRConfig:
        return self._config

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def tree_count(self) -> int:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return int(self._model.tree_count_)

    @property
    def best_iteration(self) -> int:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        if self._best_iteration_override is not None:
            return self._best_iteration_override
        value = self._model.get_best_iteration()
        return int(
            value if value is not None and int(value) >= 0 else self.tree_count - 1
        )

    @property
    def best_score(self) -> dict[str, dict[str, float]]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        if self._best_score_override is not None:
            return {
                dataset: dict(scores)
                for dataset, scores in self._best_score_override.items()
            }
        return {
            str(dataset): {
                str(metric): float(value) for metric, value in scores.items()
            }
            for dataset, scores in self._model.get_best_score().items()
        }

    @property
    def group_weight_diagnostics(self) -> dict[str, float | int] | None:
        return (
            None
            if self._group_weight_diagnostics is None
            else dict(self._group_weight_diagnostics)
        )

    def _fit(self, loader: RankerDataLoader[Any, Any], **kwargs: Any) -> None:
        if not isinstance(loader, CatBoostLTRDataLoader):
            raise TypeError("CatBoostRankerModel requires CatBoostLTRDataLoader")
        if loader.feature_columns != self._feature_columns:
            raise ContractValidationError("loader feature order differs from model")
        batches = list(loader.iter_fit_batches())
        if len(batches) != 1:
            raise ContractValidationError("CatBoost LTR fit requires one pool batch")
        train_dir = kwargs.pop("train_dir", None)
        snapshot_file = kwargs.pop("snapshot_file", None)
        log_cout = kwargs.pop("log_cout", None)
        log_cerr = kwargs.pop("log_cerr", None)
        if kwargs:
            raise TypeError(f"unsupported fit arguments: {sorted(kwargs)}")
        train_pool = batches[0].train_pool
        validate_pool_groups(train_pool, name="train")
        validate_pool_groups(batches[0].eval_pool, name="eval")
        if self._config.group_weighting == "inverse_positive_count":
            self._group_weight_diagnostics = apply_inverse_positive_group_weights(
                train_pool
            )
        else:
            self._group_weight_diagnostics = None
        model = CatBoostRanker(
            **self._config.training_params(train_dir=train_dir)
        )
        fit_kwargs: dict[str, Any] = {
            "eval_set": batches[0].eval_pool,
            "use_best_model": True,
            "early_stopping_rounds": self._config.early_stopping_rounds,
            "verbose": self._config.metric_period,
        }
        if snapshot_file is not None:
            fit_kwargs.update(
                save_snapshot=True,
                snapshot_file=Path(snapshot_file).as_posix(),
                snapshot_interval=self._config.snapshot_interval_seconds,
            )
        if log_cout is not None:
            fit_kwargs["log_cout"] = log_cout
        if log_cerr is not None:
            fit_kwargs["log_cerr"] = log_cerr
        model.fit(train_pool, **fit_kwargs)
        if tuple(model.feature_names_) != self._feature_columns:
            raise ContractValidationError("fitted CatBoost LTR feature order changed")
        self._model = model
        self._best_iteration_override = None
        self._best_score_override = None

    def _predict(
        self, loader: RankerDataLoader[Any, Any], **kwargs: Any
    ) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        if not isinstance(loader, CatBoostRankerDataLoader):
            raise TypeError("CatBoostRankerModel requires a CatBoost data loader")
        if loader.feature_columns != self._feature_columns:
            raise ContractValidationError("loader feature order differs from model")
        batch_size = kwargs.pop("batch_size", None)
        if kwargs:
            raise TypeError(f"unsupported predict arguments: {sorted(kwargs)}")
        outputs: list[pl.DataFrame] = []
        for batch in loader.iter_predict_batches(batch_size=batch_size):
            data = FeaturesData(
                num_feature_data=batch.features,
                num_feature_names=list(self._feature_columns),
            )
            score = np.asarray(self._model.predict(data), dtype=np.float64).reshape(-1)
            outputs.append(
                pl.DataFrame(
                    {
                        "user_id": pl.Series(batch.user_ids, dtype=pl.UInt64),
                        "item_id": pl.Series(batch.item_ids, dtype=pl.Int32),
                        "ranker_score": pl.Series(score, dtype=pl.Float64),
                    }
                )
            )
        result = (
            pl.concat(outputs, rechunk=True).cast(RANKER_OUTPUT_SCHEMA)
            if outputs
            else pl.DataFrame(schema=RANKER_OUTPUT_SCHEMA)
        )
        validate_ranker_output(result)
        return result

    def predict_pool(self, pool: Pool) -> np.ndarray:
        """Predict a CatBoost Pool for portable-model verification."""

        if self._model is None:
            raise RuntimeError("model is not fitted")
        if tuple(pool.get_feature_names()) != self._feature_columns:
            raise ContractValidationError("prediction pool feature order differs")
        return np.asarray(self._model.predict(pool), dtype=np.float64).reshape(-1)

    def get_feature_importance(self) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        values = self._model.get_feature_importance(type="PredictionValuesChange")
        return pl.DataFrame(
            {
                "feature": list(self._feature_columns),
                "importance": pl.Series(values, dtype=pl.Float64),
            }
        ).sort(("importance", "feature"), descending=(True, False))

    def get_config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": LTR_MODEL_KIND,
            "catboost": self._config.to_dict(),
            "feature_columns": list(self._feature_columns),
            "feature_count": len(self._feature_columns),
            "score": "RawFormulaVal",
            "tie_break": ["ranker_score DESC", "item_id ASC"],
            "group_weight_diagnostics": self.group_weight_diagnostics,
        }
        if self._model is not None:
            result.update(
                tree_count=self.tree_count,
                best_iteration=self.best_iteration,
                best_score=self.best_score,
            )
        return result

    def save(self, artifact_dir: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        destination = Path(artifact_dir)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite model artifact: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = (
            destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
        )
        temporary.mkdir()
        try:
            model_path = temporary / "model.cbm"
            self._model.save_model(model_path.as_posix(), format="cbm")
            metadata = {
                "artifact_version": LTR_ARTIFACT_VERSION,
                "kind": LTR_MODEL_KIND,
                "catboost_version": catboost.__version__,
                **self.get_config(),
                "model_sha256": sha256_file(model_path),
            }
            write_json_atomic(temporary / "model_config.json", metadata)
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
            raise

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> CatBoostRankerModel:
        root = Path(artifact_dir)
        config_path = root / "model_config.json"
        model_path = root / "model.cbm"
        if not config_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(root)
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read CatBoost LTR artifact: {error}") from error
        if (
            not isinstance(metadata, dict)
            or metadata.get("artifact_version") != LTR_ARTIFACT_VERSION
            or metadata.get("kind") != LTR_MODEL_KIND
            or metadata.get("model") != LTR_MODEL_KIND
            or metadata.get("catboost_version") != catboost.__version__
        ):
            raise ContractValidationError("invalid CatBoost LTR model artifact")
        if sha256_file(model_path) != metadata.get("model_sha256"):
            raise ContractValidationError("CatBoost LTR model checksum differs")
        model = cls(
            CatBoostLTRConfig.from_mapping(metadata["catboost"]),
            feature_columns=metadata["feature_columns"],
        )
        fitted = CatBoostRanker()
        fitted.load_model(model_path.as_posix(), format="cbm")
        if tuple(fitted.feature_names_) != model.feature_columns:
            raise ContractValidationError("restored CatBoost LTR feature order differs")
        model._model = fitted
        model._best_iteration_override = int(metadata["best_iteration"])
        raw_best_score = metadata.get("best_score", {})
        if not isinstance(raw_best_score, dict):
            raise ContractValidationError("restored CatBoost LTR score is invalid")
        model._best_score_override = {
            str(dataset): {
                str(metric): float(value) for metric, value in scores.items()
            }
            for dataset, scores in raw_best_score.items()
            if isinstance(scores, dict)
        }
        raw_group = metadata.get("group_weight_diagnostics")
        model._group_weight_diagnostics = (
            dict(raw_group) if isinstance(raw_group, dict) else None
        )
        if model.tree_count != int(metadata["tree_count"]):
            raise ContractValidationError("restored CatBoost LTR tree count differs")
        return model


def prepare_weighted_rows(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold: str,
    seed: int,
    negative_keep_probability: float,
) -> pl.DataFrame:
    """Select training rows and add inverse-probability sample weights."""

    features = validate_feature_columns(feature_columns)
    probability = _probability(
        negative_keep_probability,
        name="negative_keep_probability",
        allow_one=True,
    )
    required = {
        "user_id",
        "item_id",
        "label",
        "is_training_sample",
        "sampling_probability",
        *features,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ContractValidationError(f"ranker shard lacks columns: {sorted(missing)}")
    selected = frame.filter(pl.col("is_training_sample"))
    if probability < 1.0 and selected.height:
        uniform = deterministic_sampling_uniform(selected, seed=seed, fold=fold)
        selected = (
            selected.with_columns(
                pl.Series("__extra_uniform", uniform, dtype=pl.Float64)
            )
            .filter((pl.col("label") == 1) | (pl.col("__extra_uniform") < probability))
            .drop("__extra_uniform")
        )
    result = selected.select(
        pl.col("label").cast(pl.UInt8),
        (
            1.0
            / (
                pl.col("sampling_probability").cast(pl.Float64)
                * pl.when(pl.col("label") == 1).then(1.0).otherwise(probability)
            )
        )
        .cast(pl.Float32)
        .alias("sample_weight"),
        *(pl.col(name).cast(pl.Float32) for name in features),
    )
    if result.height == 0:
        raise ContractValidationError("weighted ranker rows must not be empty")
    if not result.get_column("sample_weight").is_finite().all():
        raise ContractValidationError("sample weights must be finite")
    return result


def write_catboost_dsv_part(
    frame: pl.DataFrame,
    destination: str | Path,
    *,
    feature_columns: Sequence[str],
    fold: str,
    seed: int,
    negative_keep_probability: float,
) -> dict[str, Any]:
    """Atomically write one numerical CatBoost DSV part from a task-07 shard."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError(output)
    rows = prepare_weighted_rows(
        frame,
        feature_columns=feature_columns,
        fold=fold,
        seed=seed,
        negative_keep_probability=negative_keep_probability,
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
    positives = int(rows.get_column("label").sum())
    return {
        "rows": rows.height,
        "positive_rows": positives,
        "negative_rows": rows.height - positives,
        "sample_weight_sum": float(rows.get_column("sample_weight").sum()),
        "size_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }


def write_column_description(
    destination: str | Path, *, feature_columns: Sequence[str]
) -> None:
    """Write the immutable Label/Weight/Num CatBoost column description."""

    features = validate_feature_columns(feature_columns)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    lines = ["0\tLabel", "1\tWeight"]
    lines.extend(f"{index + 2}\tNum\t{name}" for index, name in enumerate(features))
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ranker_scores_to_candidates(
    scores: pl.DataFrame,
    *,
    k: int = 20,
    source_name: str = POINTWISE_MODEL_KIND,
) -> pl.DataFrame:
    """Convert scores to deterministic per-user top-k candidate rows."""

    checked_k = _positive_int(k, name="k")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("source_name must be non-empty")
    validate_ranker_output(scores)
    result = (
        scores.sort(
            ("user_id", "ranker_score", "item_id"),
            descending=(False, True, False),
        )
        .with_columns(
            pl.col("item_id").cum_count().over("user_id").cast(pl.UInt32).alias("rank")
        )
        .filter(pl.col("rank") <= checked_k)
        .rename({"ranker_score": "score"})
        .with_columns(pl.lit(source_name).alias("source"))
        .select(CANDIDATE_SCHEMA.names())
        .cast(CANDIDATE_SCHEMA)
        .sort(("user_id", "source", "rank"))
    )
    return result


__all__ = [
    "LTR_ARTIFACT_VERSION",
    "LTR_MODEL_KIND",
    "POINTWISE_MODEL_KIND",
    "CatBoostFitBatch",
    "CatBoostLTRConfig",
    "CatBoostLTRDataLoader",
    "CatBoostPointwiseConfig",
    "CatBoostPointwiseModel",
    "CatBoostPredictBatch",
    "CatBoostRankerDataLoader",
    "CatBoostRankerModel",
    "apply_inverse_positive_group_weights",
    "prepare_weighted_rows",
    "ranker_scores_to_candidates",
    "validate_feature_columns",
    "validate_pool_groups",
    "write_catboost_dsv_part",
    "write_column_description",
]

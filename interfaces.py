"""Stable model and data-loader contracts for the recommender pipeline.

The orchestration layer owns paths, temporal splits, joins, and construction of
model-specific loaders.  A model only consumes batches exposed by its loader.
Concrete loaders implement the protected hooks below; the public methods keep
the lifecycle rules identical for every model family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from numbers import Integral
from typing import Any, Final, Generic, Self, TypeVar

import polars as pl

CANDIDATE_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "score": pl.Float64,
        "rank": pl.UInt32,
        "source": pl.String,
    }
)
FEATURE_TABLE_REQUIRED_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
    }
)
RANKER_OUTPUT_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_id": pl.Int32,
        "ranker_score": pl.Float64,
    }
)
FINAL_RECOMMENDATION_SCHEMA: Final = pl.Schema(
    {
        "user_id": pl.UInt64,
        "item_ids": pl.List(pl.Int32),
    }
)


class DataLoaderStateError(RuntimeError):
    """Raised when loader methods are called outside their lifecycle order."""


class DataLoaderState(str, Enum):
    """State of one independent fit or prediction side of a loader."""

    EMPTY = "empty"
    LOADED = "loaded"
    PREPARED = "prepared"


FitBatchT = TypeVar("FitBatchT")
PredictBatchT = TypeVar("PredictBatchT")
LoaderT = TypeVar("LoaderT", bound="DataLoader[Any, Any]")


def _checked_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be a positive integer or None")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return int(batch_size)


class DataLoader(ABC, Generic[FitBatchT, PredictBatchT]):
    """Prepare deterministic, model-specific fit and prediction batches.

    ``load_*`` hooks should register inputs and use lazy reads where a source
    supports them.  ``prepare_*`` hooks own filtering, temporal selection,
    feature construction, ID mappings, ordering, and conversion to the final
    batch representation.  Iteration must not perform those transformations.

    Reloading one side invalidates only that side's prepared state.  Every
    ``iter_*_batches`` call delegates to a fresh iterator.  Any randomness in a
    concrete implementation must be derived from the explicit ``seed``.
    """

    def __init__(self, *, seed: int = 42) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self._seed = int(seed)
        self._fit_state = DataLoaderState.EMPTY
        self._predict_state = DataLoaderState.EMPTY

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def fit_state(self) -> DataLoaderState:
        return self._fit_state

    @property
    def predict_state(self) -> DataLoaderState:
        return self._predict_state

    def load_fit_data(self, **kwargs: Any) -> Self:
        self._load_fit_data(**kwargs)
        self._fit_state = DataLoaderState.LOADED
        return self

    def prepare_fit_data(self, **kwargs: Any) -> Self:
        self._require_state("fit", self._fit_state, DataLoaderState.LOADED)
        self._prepare_fit_data(**kwargs)
        self._fit_state = DataLoaderState.PREPARED
        return self

    def iter_fit_batches(
        self, *, batch_size: int | None = None
    ) -> Iterator[FitBatchT]:
        self._require_state("fit", self._fit_state, DataLoaderState.PREPARED)
        checked_size = _checked_batch_size(batch_size)
        return iter(self._iter_fit_batches(batch_size=checked_size))

    def load_predict_data(self, **kwargs: Any) -> Self:
        self._load_predict_data(**kwargs)
        self._predict_state = DataLoaderState.LOADED
        return self

    def prepare_predict_data(self, **kwargs: Any) -> Self:
        self._require_state(
            "predict", self._predict_state, DataLoaderState.LOADED
        )
        self._prepare_predict_data(**kwargs)
        self._predict_state = DataLoaderState.PREPARED
        return self

    def iter_predict_batches(
        self, *, batch_size: int | None = None
    ) -> Iterator[PredictBatchT]:
        self._require_state(
            "predict", self._predict_state, DataLoaderState.PREPARED
        )
        checked_size = _checked_batch_size(batch_size)
        return iter(self._iter_predict_batches(batch_size=checked_size))

    @staticmethod
    def _require_state(
        phase: str, actual: DataLoaderState, expected: DataLoaderState
    ) -> None:
        if actual is not expected:
            raise DataLoaderStateError(
                f"{phase} data must be {expected.value} before this operation; "
                f"current state is {actual.value}"
            )

    @abstractmethod
    def _load_fit_data(self, **kwargs: Any) -> None:
        """Register fit inputs without preparing model batches."""

    @abstractmethod
    def _prepare_fit_data(self, **kwargs: Any) -> None:
        """Build all fit-time features and the final batch representation."""

    @abstractmethod
    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[FitBatchT]:
        """Return a new deterministic iterator over prepared fit batches."""

    @abstractmethod
    def _load_predict_data(self, **kwargs: Any) -> None:
        """Register prediction inputs without preparing model batches."""

    @abstractmethod
    def _prepare_predict_data(self, **kwargs: Any) -> None:
        """Build prediction features and the final batch representation."""

    @abstractmethod
    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[PredictBatchT]:
        """Return a new deterministic iterator over prediction batches."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable loader configuration."""


CandidateFitBatchT = TypeVar("CandidateFitBatchT")
CandidatePredictBatchT = TypeVar("CandidatePredictBatchT")
RankerFitBatchT = TypeVar("RankerFitBatchT")
RankerPredictBatchT = TypeVar("RankerPredictBatchT")


class CandidateDataLoader(
    DataLoader[CandidateFitBatchT, CandidatePredictBatchT],
    Generic[CandidateFitBatchT, CandidatePredictBatchT],
):
    """Loader boundary for candidate generators."""


class RankerDataLoader(
    DataLoader[RankerFitBatchT, RankerPredictBatchT],
    Generic[RankerFitBatchT, RankerPredictBatchT],
):
    """Loader boundary for rankers.

    Its batches must already contain ordered features and, for fitting, labels.
    A ranker must not join, reorder, or build features from raw sources.
    """


class Model(ABC, Generic[LoaderT]):
    """A model that updates or applies parameters from loader batches only."""

    @abstractmethod
    def fit(self, loader: LoaderT, **kwargs: Any) -> Self:
        """Fit model parameters from prepared loader batches."""

    @abstractmethod
    def predict(self, loader: LoaderT, **kwargs: Any) -> pl.DataFrame:
        """Apply model parameters and assemble a typed Polars result."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable model configuration."""


class CandidateModel(Model[CandidateDataLoader[Any, Any]], ABC):
    """Base class for a single, stably named candidate source."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Stable non-empty name written to every output candidate row."""

    def fit(self, loader: CandidateDataLoader[Any, Any], **kwargs: Any) -> Self:
        self._require_candidate_loader(loader)
        self._fit(loader, **kwargs)
        return self

    def predict(
        self,
        loader: CandidateDataLoader[Any, Any],
        *,
        k: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        self._require_candidate_loader(loader)
        checked_k = _checked_batch_size(k)
        if checked_k is None:  # pragma: no cover - k is statically non-optional
            raise TypeError("k must be a positive integer")
        source_before = self._checked_source_name()
        result = self._predict(loader, k=checked_k, **kwargs)
        if not isinstance(result, pl.DataFrame):
            raise TypeError("candidate predict must return polars.DataFrame")
        if self._checked_source_name() != source_before:
            raise RuntimeError("source_name changed during prediction")
        return result

    @staticmethod
    def _require_candidate_loader(loader: object) -> None:
        if not isinstance(loader, CandidateDataLoader):
            raise TypeError("CandidateModel requires a CandidateDataLoader")

    def _checked_source_name(self) -> str:
        source = self.source_name
        if not isinstance(source, str) or not source:
            raise ValueError("source_name must be a non-empty string")
        return source

    @abstractmethod
    def _fit(
        self, loader: CandidateDataLoader[Any, Any], **kwargs: Any
    ) -> None:
        """Update parameters using only ``loader.iter_fit_batches``."""

    @abstractmethod
    def _predict(
        self,
        loader: CandidateDataLoader[Any, Any],
        *,
        k: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Score prepared prediction batches and assemble candidate rows."""


class RankerModel(Model[RankerDataLoader[Any, Any]], ABC):
    """Base class for models scoring already prepared candidate features."""

    def fit(self, loader: RankerDataLoader[Any, Any], **kwargs: Any) -> Self:
        self._require_ranker_loader(loader)
        self._fit(loader, **kwargs)
        return self

    def predict(
        self, loader: RankerDataLoader[Any, Any], **kwargs: Any
    ) -> pl.DataFrame:
        self._require_ranker_loader(loader)
        result = self._predict(loader, **kwargs)
        if not isinstance(result, pl.DataFrame):
            raise TypeError("ranker predict must return polars.DataFrame")
        return result

    @staticmethod
    def _require_ranker_loader(loader: object) -> None:
        if not isinstance(loader, RankerDataLoader):
            raise TypeError("RankerModel requires a RankerDataLoader")

    @abstractmethod
    def _fit(self, loader: RankerDataLoader[Any, Any], **kwargs: Any) -> None:
        """Update parameters using ordered feature/label batches."""

    @abstractmethod
    def _predict(
        self, loader: RankerDataLoader[Any, Any], **kwargs: Any
    ) -> pl.DataFrame:
        """Score already ordered feature batches."""


__all__ = [
    "CANDIDATE_SCHEMA",
    "FEATURE_TABLE_REQUIRED_SCHEMA",
    "FINAL_RECOMMENDATION_SCHEMA",
    "RANKER_OUTPUT_SCHEMA",
    "CandidateDataLoader",
    "CandidateModel",
    "DataLoader",
    "DataLoaderState",
    "DataLoaderStateError",
    "Model",
    "RankerDataLoader",
    "RankerModel",
]

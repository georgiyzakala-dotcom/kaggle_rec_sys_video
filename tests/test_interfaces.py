from __future__ import annotations

import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import polars as pl

from interfaces import (
    CANDIDATE_SCHEMA,
    FINAL_RECOMMENDATION_SCHEMA,
    RANKER_OUTPUT_SCHEMA,
    CandidateDataLoader,
    CandidateModel,
    DataLoaderState,
    DataLoaderStateError,
    RankerDataLoader,
    RankerModel,
)
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_deterministic_iteration,
    validate_feature_table,
    validate_final_recommendations,
    validate_json_config,
    validate_loader,
    validate_model_config,
    validate_ranker_output,
)


@dataclass(frozen=True)
class CandidatePredictBatch:
    user_ids: tuple[int, ...]


class DummyCandidateDataLoader(
    CandidateDataLoader[pl.DataFrame, CandidatePredictBatch]
):
    def __init__(self, *, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._fit_lazy: pl.LazyFrame | None = None
        self._predict_lazy: pl.LazyFrame | None = None
        self._fit_frame: pl.DataFrame | None = None
        self._predict_frame: pl.DataFrame | None = None

    def _load_fit_data(self, **kwargs: Any) -> None:
        frame = kwargs["frame"]
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("test fit source must be a Polars frame")
        self._fit_lazy = frame.lazy()
        self._fit_frame = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        assert self._fit_lazy is not None
        self._fit_frame = self._fit_lazy.collect().sort(
            ("item_id", "strength")
        )

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[pl.DataFrame]:
        assert self._fit_frame is not None
        size = batch_size or max(self._fit_frame.height, 1)
        for offset in range(0, self._fit_frame.height, size):
            yield self._fit_frame.slice(offset, size)

    def _load_predict_data(self, **kwargs: Any) -> None:
        frame = kwargs["frame"]
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("test predict source must be a Polars frame")
        self._predict_lazy = frame.lazy()
        self._predict_frame = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        assert self._predict_lazy is not None
        self._predict_frame = self._predict_lazy.collect().sort("user_id")

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[CandidatePredictBatch]:
        assert self._predict_frame is not None
        size = batch_size or max(self._predict_frame.height, 1)
        for offset in range(0, self._predict_frame.height, size):
            frame = self._predict_frame.slice(offset, size)
            yield CandidatePredictBatch(
                user_ids=tuple(frame.get_column("user_id").to_list())
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "dummy_candidate",
            "seed": self.seed,
            "fit_order": ["item_id", "strength"],
            "predict_order": ["user_id"],
        }


class DummyCandidateModel(CandidateModel):
    def __init__(self) -> None:
        self._scores: list[tuple[int, float]] = []

    @property
    def source_name(self) -> str:
        return "dummy_candidate"

    def _fit(
        self, loader: CandidateDataLoader[Any, Any], **kwargs: Any
    ) -> None:
        batches = list(loader.iter_fit_batches(batch_size=kwargs.get("batch_size")))
        fit_frame = pl.concat(batches)
        self._scores = [
            (row["item_id"], row["score"])
            for row in (
                fit_frame.group_by("item_id")
                .agg(pl.col("strength").sum().alias("score"))
                .sort(("score", "item_id"), descending=(True, False))
                .iter_rows(named=True)
            )
        ]

    def _predict(
        self,
        loader: CandidateDataLoader[Any, Any],
        *,
        k: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        rows: list[tuple[int, int, float, int, str]] = []
        for batch in loader.iter_predict_batches(
            batch_size=kwargs.get("batch_size")
        ):
            for user_id in batch.user_ids:
                rows.extend(
                    (
                        user_id,
                        item_id,
                        score,
                        rank,
                        self.source_name,
                    )
                    for rank, (item_id, score) in enumerate(
                        self._scores[:k], start=1
                    )
                )
        return pl.DataFrame(rows, schema=CANDIDATE_SCHEMA, orient="row").sort(
            ("user_id", "source", "rank")
        )

    def get_config(self) -> dict[str, Any]:
        return {"model": "dummy_candidate", "score": "sum_strength"}


@dataclass(frozen=True)
class RankerFitBatch:
    user_ids: tuple[int, ...]
    item_ids: tuple[int, ...]
    features: tuple[float, ...]
    labels: tuple[int, ...]


class DummyRankerDataLoader(RankerDataLoader[RankerFitBatch, pl.DataFrame]):
    def __init__(self, *, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._fit_lazy: pl.LazyFrame | None = None
        self._predict_lazy: pl.LazyFrame | None = None
        self._fit_frame: pl.DataFrame | None = None
        self._predict_frame: pl.DataFrame | None = None

    def _load_fit_data(self, **kwargs: Any) -> None:
        self._fit_lazy = kwargs["frame"].lazy()
        self._fit_frame = None

    def _prepare_fit_data(self, **kwargs: Any) -> None:
        assert self._fit_lazy is not None
        self._fit_frame = self._fit_lazy.collect().sort(("user_id", "item_id"))

    def _iter_fit_batches(
        self, *, batch_size: int | None
    ) -> Iterator[RankerFitBatch]:
        assert self._fit_frame is not None
        size = batch_size or max(self._fit_frame.height, 1)
        for offset in range(0, self._fit_frame.height, size):
            frame = self._fit_frame.slice(offset, size)
            yield RankerFitBatch(
                user_ids=tuple(frame.get_column("user_id").to_list()),
                item_ids=tuple(frame.get_column("item_id").to_list()),
                features=tuple(frame.get_column("feature").to_list()),
                labels=tuple(frame.get_column("label").to_list()),
            )

    def _load_predict_data(self, **kwargs: Any) -> None:
        self._predict_lazy = kwargs["frame"].lazy()
        self._predict_frame = None

    def _prepare_predict_data(self, **kwargs: Any) -> None:
        assert self._predict_lazy is not None
        self._predict_frame = self._predict_lazy.collect().sort(
            ("user_id", "item_id")
        )

    def _iter_predict_batches(
        self, *, batch_size: int | None
    ) -> Iterator[pl.DataFrame]:
        assert self._predict_frame is not None
        size = batch_size or max(self._predict_frame.height, 1)
        for offset in range(0, self._predict_frame.height, size):
            yield self._predict_frame.slice(offset, size)

    def get_config(self) -> dict[str, Any]:
        return {
            "loader": "dummy_ranker",
            "seed": self.seed,
            "feature_columns": ["feature"],
            "label_column": "label",
        }


class DummyRankerModel(RankerModel):
    def __init__(self) -> None:
        self._label_mean = 0.0

    def _fit(self, loader: RankerDataLoader[Any, Any], **kwargs: Any) -> None:
        labels = [
            label
            for batch in loader.iter_fit_batches(
                batch_size=kwargs.get("batch_size")
            )
            for label in batch.labels
        ]
        self._label_mean = sum(labels) / len(labels)

    def _predict(
        self, loader: RankerDataLoader[Any, Any], **kwargs: Any
    ) -> pl.DataFrame:
        outputs = [
            batch.select(
                "user_id",
                "item_id",
                (pl.col("feature") + self._label_mean)
                .cast(pl.Float64)
                .alias("ranker_score"),
            )
            for batch in loader.iter_predict_batches(
                batch_size=kwargs.get("batch_size")
            )
        ]
        return pl.concat(outputs).cast(RANKER_OUTPUT_SCHEMA)

    def get_config(self) -> dict[str, Any]:
        return {"model": "dummy_ranker", "features": ["feature"]}


class InterfaceLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.high_user_id = 2**53 + 123
        self.candidate_fit = pl.DataFrame(
            {
                "item_id": pl.Series([10, 11, 12, 10, 13], dtype=pl.Int32),
                "strength": pl.Series(
                    [1.0, 3.0, 3.0, 4.0, 1.0], dtype=pl.Float64
                ),
            }
        )
        self.predict_users = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [self.high_user_id, 5, 3, 4, 2], dtype=pl.UInt64
                )
            }
        )

    def _prepared_candidate_loader(self) -> DummyCandidateDataLoader:
        return (
            DummyCandidateDataLoader(seed=42)
            .load_fit_data(frame=self.candidate_fit)
            .prepare_fit_data()
            .load_predict_data(frame=self.predict_users)
            .prepare_predict_data()
        )

    def test_lifecycle_errors_and_state_transitions(self) -> None:
        loader = DummyCandidateDataLoader(seed=42)
        self.assertEqual(loader.fit_state, DataLoaderState.EMPTY)
        self.assertEqual(loader.predict_state, DataLoaderState.EMPTY)
        with self.assertRaises(DataLoaderStateError):
            loader.prepare_fit_data()
        with self.assertRaises(DataLoaderStateError):
            loader.iter_fit_batches()
        with self.assertRaises(DataLoaderStateError):
            loader.prepare_predict_data()
        with self.assertRaises(DataLoaderStateError):
            loader.iter_predict_batches()

        self.assertIs(loader.load_fit_data(frame=self.candidate_fit), loader)
        self.assertEqual(loader.fit_state, DataLoaderState.LOADED)
        self.assertIs(loader.prepare_fit_data(), loader)
        self.assertEqual(loader.fit_state, DataLoaderState.PREPARED)

        loader.load_fit_data(frame=self.candidate_fit)
        self.assertEqual(loader.fit_state, DataLoaderState.LOADED)
        with self.assertRaises(DataLoaderStateError):
            loader.iter_fit_batches()

    def test_batch_sizes_and_repeatable_iteration(self) -> None:
        loader = self._prepared_candidate_loader()
        fit_batches = list(loader.iter_fit_batches(batch_size=2))
        predict_batches = list(loader.iter_predict_batches(batch_size=2))
        self.assertEqual([batch.height for batch in fit_batches], [2, 2, 1])
        self.assertEqual(
            [len(batch.user_ids) for batch in predict_batches], [2, 2, 1]
        )
        self.assertIsInstance(fit_batches[0], pl.DataFrame)
        self.assertIsInstance(predict_batches[0], CandidatePredictBatch)
        validate_deterministic_iteration(loader, phase="fit", batch_size=2)
        validate_deterministic_iteration(loader, phase="predict", batch_size=2)

    def test_batch_size_must_be_positive_integer_or_none(self) -> None:
        loader = self._prepared_candidate_loader()
        for invalid in (0, -1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                loader.iter_fit_batches(batch_size=invalid)
        for invalid in (True, 1.5, "2"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                loader.iter_predict_batches(batch_size=invalid)  # type: ignore[arg-type]
        self.assertEqual(len(list(loader.iter_fit_batches(batch_size=None))), 1)

    def test_candidate_model_full_cycle_and_uint64_preservation(self) -> None:
        loader = self._prepared_candidate_loader()
        model = DummyCandidateModel()
        self.assertIs(model.fit(loader, batch_size=2), model)
        output = model.predict(loader, k=3, batch_size=2)
        validate_candidate_output(output, k=3, source_name=model.source_name)
        self.assertEqual(output.schema, CANDIDATE_SCHEMA)
        self.assertEqual(output.get_column("user_id").dtype, pl.UInt64)
        self.assertIn(self.high_user_id, output.get_column("user_id").to_list())
        self.assertEqual(
            output.filter(pl.col("user_id") == self.high_user_id)
            .get_column("item_id")
            .to_list(),
            [10, 11, 12],
        )

    def test_models_reject_raw_frames_instead_of_loaders(self) -> None:
        candidate_model = DummyCandidateModel()
        with self.assertRaises(TypeError):
            candidate_model.fit(self.candidate_fit)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            candidate_model.predict(self.predict_users, k=2)  # type: ignore[arg-type]

        ranker_model = DummyRankerModel()
        raw = pl.DataFrame()
        with self.assertRaises(TypeError):
            ranker_model.fit(raw)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ranker_model.predict(raw)  # type: ignore[arg-type]

    def test_configs_are_json_serializable_and_separate(self) -> None:
        loader = self._prepared_candidate_loader()
        model = DummyCandidateModel()
        validate_loader(loader)
        validate_model_config(model)
        self.assertNotEqual(loader.get_config(), model.get_config())
        validate_json_config(
            {
                "data_loader_config": loader.get_config(),
                "model_config": model.get_config(),
            }
        )
        with self.assertRaises(ContractValidationError):
            validate_json_config({"bad": {1, 2}})
        with self.assertRaises(ContractValidationError):
            validate_json_config({"bad": float("nan")})


class RankerInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        high_user_id = 2**53 + 321
        self.fit_frame = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [high_user_id, 1, 1, 2, 2], dtype=pl.UInt64
                ),
                "item_id": pl.Series([10, 10, 11, 11, 12], dtype=pl.Int32),
                "feature": pl.Series([0.8, 0.2, 0.3, 0.4, 0.5], dtype=pl.Float64),
                "label": pl.Series([1, 0, 1, 0, 1], dtype=pl.UInt8),
            }
        )
        self.predict_frame = self.fit_frame.select(
            "user_id", "item_id", "feature"
        )

    def test_ranker_loader_supports_distinct_batch_types_and_model_cycle(self) -> None:
        loader = (
            DummyRankerDataLoader(seed=42)
            .load_fit_data(frame=self.fit_frame)
            .prepare_fit_data()
            .load_predict_data(frame=self.predict_frame)
            .prepare_predict_data()
        )
        fit_batches = list(loader.iter_fit_batches(batch_size=2))
        predict_batches = list(loader.iter_predict_batches(batch_size=2))
        self.assertEqual([len(batch.labels) for batch in fit_batches], [2, 2, 1])
        self.assertEqual([batch.height for batch in predict_batches], [2, 2, 1])
        self.assertIsInstance(fit_batches[0], RankerFitBatch)
        self.assertIsInstance(predict_batches[0], pl.DataFrame)
        validate_deterministic_iteration(loader, phase="fit", batch_size=2)
        validate_deterministic_iteration(loader, phase="predict", batch_size=2)

        model = DummyRankerModel()
        self.assertIs(model.fit(loader, batch_size=2), model)
        output = model.predict(loader, batch_size=2)
        validate_ranker_output(output)
        self.assertEqual(output.schema, RANKER_OUTPUT_SCHEMA)
        self.assertIn(
            2**53 + 321, output.get_column("user_id").to_list()
        )
        validate_model_config(model)
        validate_loader(loader)


class BoundaryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = pl.DataFrame(
            [
                (1, 10, 2.0, 1, "source_a"),
                (1, 11, 2.0, 2, "source_a"),
                (1, 12, 1.0, 3, "source_a"),
                (2, 12, 4.0, 1, "source_a"),
            ],
            schema=CANDIDATE_SCHEMA,
            orient="row",
        )

    def test_valid_candidate_and_tie_break(self) -> None:
        validate_candidate_output(self.valid, k=3, source_name="source_a")

    def test_candidate_rejects_null_wrong_dtype_and_duplicate(self) -> None:
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                self.valid.with_columns(
                    pl.when(pl.col("item_id") == 11)
                    .then(None)
                    .otherwise(pl.col("user_id"))
                    .cast(pl.UInt64)
                    .alias("user_id")
                ),
                k=3,
                source_name="source_a",
            )
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                self.valid.with_columns(pl.col("user_id").cast(pl.UInt32)),
                k=3,
                source_name="source_a",
            )
        duplicate = pl.concat([self.valid, self.valid.slice(0, 1)]).sort(
            ("user_id", "source", "rank")
        )
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                duplicate, k=4, source_name="source_a"
            )

    def test_candidate_rejects_cap_rank_gap_and_source_change(self) -> None:
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(self.valid, k=2, source_name="source_a")
        rank_gap = self.valid.with_columns(
            pl.when((pl.col("user_id") == 1) & (pl.col("rank") == 3))
            .then(pl.lit(4, dtype=pl.UInt32))
            .otherwise(pl.col("rank"))
            .alias("rank")
        )
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(rank_gap, k=4, source_name="source_a")
        changed_source = self.valid.with_columns(
            pl.when(pl.col("user_id") == 2)
            .then(pl.lit("source_b"))
            .otherwise(pl.col("source"))
            .alias("source")
        ).sort(("user_id", "source", "rank"))
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                changed_source, k=3, source_name="source_a"
            )

    def test_candidate_rejects_bad_tie_break_and_physical_order(self) -> None:
        bad_tie_break = self.valid.with_columns(
            pl.when((pl.col("user_id") == 1) & (pl.col("item_id") == 10))
            .then(pl.lit(11, dtype=pl.Int32))
            .when((pl.col("user_id") == 1) & (pl.col("item_id") == 11))
            .then(pl.lit(10, dtype=pl.Int32))
            .otherwise(pl.col("item_id"))
            .alias("item_id")
        )
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                bad_tie_break, k=3, source_name="source_a"
            )
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                self.valid.reverse(), k=3, source_name="source_a"
            )

    def test_candidate_rejects_non_finite_score(self) -> None:
        non_finite = self.valid.with_columns(
            pl.when(pl.col("item_id") == 10)
            .then(float("inf"))
            .otherwise(pl.col("score"))
            .alias("score")
        )
        with self.assertRaises(ContractValidationError):
            validate_candidate_output(
                non_finite, k=3, source_name="source_a"
            )

    def test_feature_ranker_and_final_schemas(self) -> None:
        features = pl.DataFrame(
            {
                "user_id": pl.Series([1, 2], dtype=pl.UInt64),
                "item_id": pl.Series([10, 11], dtype=pl.Int32),
                "feature": pl.Series([0.5, 0.7], dtype=pl.Float32),
            }
        )
        validate_feature_table(features)
        with self.assertRaises(ContractValidationError):
            validate_feature_table(pl.concat([features, features.slice(0, 1)]))

        ranker_output = features.select(
            "user_id",
            "item_id",
            pl.col("feature").cast(pl.Float64).alias("ranker_score"),
        )
        validate_ranker_output(ranker_output)
        with self.assertRaises(ContractValidationError):
            validate_ranker_output(
                ranker_output.with_columns(
                    pl.col("ranker_score").cast(pl.Float32)
                )
            )

        final = pl.DataFrame(
            {
                "user_id": pl.Series([1, 2], dtype=pl.UInt64),
                "item_ids": pl.Series(
                    [[10, 11], [12, 13]], dtype=pl.List(pl.Int32)
                ),
            },
            schema=FINAL_RECOMMENDATION_SCHEMA,
        )
        validate_final_recommendations(final, expected_k=2)
        with self.assertRaises(ContractValidationError):
            validate_final_recommendations(
                final.with_columns(
                    pl.when(pl.col("user_id") == 1)
                    .then(pl.lit([10, 10], dtype=pl.List(pl.Int32)))
                    .otherwise(pl.col("item_ids"))
                    .alias("item_ids")
                ),
                expected_k=2,
            )


if __name__ == "__main__":
    unittest.main()

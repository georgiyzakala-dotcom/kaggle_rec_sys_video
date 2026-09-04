from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from interfaces import CANDIDATE_SCHEMA, FINAL_RECOMMENDATION_SCHEMA
from popularity import (
    ITEM_POPULARITY_STATS_SCHEMA,
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
    candidates_to_recommendations,
    fill_with_global_popularity,
)
from scripts.run_global_popularity import run_experiment
from scripts.run_recency_popularity import run_experiment as run_recency_experiment
from validation import (
    ContractValidationError,
    validate_candidate_output,
    validate_deterministic_iteration,
    validate_recommendations_against_history,
)


def daily_frame(rows: list[tuple[int, int, int, int, int]]) -> pl.DataFrame:
    """Create daily rows from user, item, day offset, views, positive."""

    values = [
        (
            user_id,
            item_id,
            date(2024, 1, 1 + day_offset),
            datetime(2024, 1, 1 + day_offset, 12, tzinfo=UTC).replace(
                tzinfo=None
            ),
            views,
            70 if positive else 10,
            0,
            0,
            positive,
        )
        for user_id, item_id, day_offset, views, positive in rows
    ]
    return pl.DataFrame(values, schema=DAILY_INTERACTION_SCHEMA, orient="row")


def candidate_frame(
    rows: list[tuple[int, int, float, int, str]],
) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=CANDIDATE_SCHEMA, orient="row")


def timestamped_daily_frame(
    rows: list[tuple[int, int, datetime, int, int]],
) -> pl.DataFrame:
    values = [
        (
            user_id,
            item_id,
            dt.date(),
            dt,
            views,
            70 if positive else 10,
            0,
            0,
            positive,
        )
        for user_id, item_id, dt, views, positive in rows
    ]
    return pl.DataFrame(values, schema=DAILY_INTERACTION_SCHEMA, orient="row")


class PopularityStatisticsTests(unittest.TestCase):
    def test_all_score_definitions_use_daily_history_contract(self) -> None:
        high_user_id = 2**53 + 17
        history = daily_frame(
            [
                (high_user_id, 10, 0, 3, 1),
                (high_user_id, 10, 1, 2, 1),
                (2, 10, 0, 5, 0),
                (2, 11, 0, 1, 1),
                (3, 11, 0, 4, 0),
            ]
        )
        loader = (
            PopularityDataLoader(seed=42)
            .load_fit_data(history=history)
            .prepare_fit_data()
        )
        self.assertEqual(loader.item_stats.schema, ITEM_POPULARITY_STATS_SCHEMA)
        self.assertEqual(
            loader.item_stats.rows(named=True),
            [
                {
                    "item_id": 10,
                    "raw_interaction_count": 10,
                    "distinct_interacting_users": 2,
                    "relevant_interaction_count": 2,
                    "distinct_relevant_users": 1,
                },
                {
                    "item_id": 11,
                    "raw_interaction_count": 5,
                    "distinct_interacting_users": 2,
                    "relevant_interaction_count": 1,
                    "distinct_relevant_users": 1,
                },
            ],
        )
        validate_deterministic_iteration(loader, phase="fit", batch_size=1)

    def test_score_tie_break_is_item_id_ascending(self) -> None:
        history = daily_frame(
            [
                (1, 11, 0, 1, 1),
                (2, 11, 0, 1, 0),
                (1, 10, 0, 1, 1),
                (2, 10, 0, 1, 0),
            ]
        )
        loader = (
            PopularityDataLoader()
            .load_fit_data(history=history)
            .prepare_fit_data()
        )
        model = GlobalPopularityModel(PopularityScore.DISTINCT_INTERACTING_USERS)
        model.fit(loader, batch_size=1)
        self.assertEqual(model.item_ranking.get_column("item_id").to_list(), [10, 11])
        self.assertEqual(model.item_ranking.get_column("global_rank").to_list(), [1, 2])


class RecencyPopularityConfigTests(unittest.TestCase):
    def test_config_round_trip_for_all_score_families(self) -> None:
        values = [
            {
                "config_id": "window_full",
                "score_kind": "window",
                "signal": "raw_views",
                "window_hours": None,
            },
            {
                "config_id": "decay",
                "score_kind": "decay",
                "signal": "positive_daily_rows",
                "half_life_hours": 6,
            },
            {
                "config_id": "trend",
                "score_kind": "trending",
                "signal": "raw_views",
                "short_window_hours": 6,
                "long_window_hours": 24,
                "smoothing": 10,
            },
            {
                "config_id": "blend",
                "score_kind": "window_blend",
                "signal": "positive_daily_rows",
                "window_weights": [
                    {"window_hours": 6, "weight": 0.75},
                    {"window_hours": None, "weight": 0.25},
                ],
            },
        ]
        for value in values:
            with self.subTest(config_id=value["config_id"]):
                config = RecencyPopularityConfig.from_dict(value)
                self.assertEqual(
                    RecencyPopularityConfig.from_dict(config.to_dict()), config
                )

    def test_config_rejects_invalid_or_mixed_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than"):
            RecencyPopularityConfig.from_dict(
                {
                    "config_id": "bad",
                    "score_kind": "trending",
                    "signal": "raw_views",
                    "short_window_hours": 24,
                    "long_window_hours": 6,
                    "smoothing": 10,
                }
            )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            RecencyPopularityConfig.from_dict(
                {
                    "config_id": "bad",
                    "score_kind": "decay",
                    "signal": "raw_views",
                    "half_life_hours": 6,
                    "window_hours": 24,
                }
            )


class RecencyPopularityModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_time = datetime(
            2024, 1, 3, 0, 0, tzinfo=UTC
        ).replace(tzinfo=None)
        self.high_user_id = 2**53 + 321
        self.history = timestamped_daily_frame(
            [
                (
                    100,
                    1,
                    self.reference_time - timedelta(hours=6),
                    2,
                    0,
                ),
                (
                    101,
                    2,
                    self.reference_time
                    - timedelta(hours=6, microseconds=1),
                    10,
                    1,
                ),
                (
                    102,
                    3,
                    self.reference_time - timedelta(hours=3),
                    1,
                    1,
                ),
                (
                    103,
                    4,
                    self.reference_time - timedelta(hours=24),
                    4,
                    0,
                ),
                (
                    self.high_user_id,
                    5,
                    self.reference_time - timedelta(hours=25),
                    8,
                    1,
                ),
            ]
        )
        self.targets = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [self.high_user_id, 1], dtype=pl.UInt64
                )
            }
        ).cast(TARGET_USER_SCHEMA)

    def prepared_loader(self) -> RecencyPopularityDataLoader:
        return (
            RecencyPopularityDataLoader(
                reference_time=self.reference_time,
                windows_hours=[6, 24, 72],
                half_lives_hours=[6, 24],
                seed=42,
            )
            .load_fit_data(history=self.history)
            .prepare_fit_data()
            .load_predict_data(
                history=self.history, target_users=self.targets
            )
            .prepare_predict_data()
        )

    def test_window_boundaries_and_raw_positive_signals(self) -> None:
        loader = self.prepared_loader()
        raw_model = RecencyPopularityModel(
            {
                "config_id": "raw_6h",
                "score_kind": "window",
                "signal": "raw_views",
                "window_hours": 6,
            }
        ).fit(loader)
        self.assertEqual(
            raw_model.item_ranking.select("item_id", "score").rows(),
            [(1, 2.0), (3, 1.0)],
        )
        positive_model = RecencyPopularityModel(
            {
                "config_id": "positive_6h",
                "score_kind": "window",
                "signal": "positive_daily_rows",
                "window_hours": 6,
            }
        ).fit(loader)
        self.assertEqual(
            positive_model.item_ranking.select("item_id", "score").rows(),
            [(3, 1.0)],
        )

    def test_decay_trending_and_artifact_restore_are_deterministic(self) -> None:
        loader = self.prepared_loader()
        decay_config = RecencyPopularityConfig.from_dict(
            {
                "config_id": "decay_6h",
                "score_kind": "decay",
                "signal": "raw_views",
                "half_life_hours": 6,
            }
        )
        decay_model = RecencyPopularityModel(decay_config).fit(loader)
        score_item_1 = (
            decay_model.item_ranking.filter(pl.col("item_id") == 1)
            .get_column("score")
            .item()
        )
        self.assertTrue(math.isclose(score_item_1, 1.0, rel_tol=1e-12))

        trend_model = RecencyPopularityModel(
            {
                "config_id": "trend",
                "score_kind": "trending",
                "signal": "raw_views",
                "short_window_hours": 6,
                "long_window_hours": 72,
                "smoothing": 100,
            }
        ).fit(loader)
        self.assertTrue(trend_model.item_ranking["score"].is_finite().all())
        first = trend_model.predict(loader, k=3, batch_size=1)
        restored = RecencyPopularityModel.from_fitted_ranking(
            trend_model.config.to_dict(), trend_model.item_ranking
        )
        second = restored.predict(loader, k=3, batch_size=2)
        self.assertTrue(first.equals(second))
        validate_candidate_output(
            first, k=3, source_name="recency_popularity"
        )
        self.assertEqual(first.get_column("user_id").dtype, pl.UInt64)

    def test_future_timestamp_is_rejected(self) -> None:
        invalid = timestamped_daily_frame(
            [(1, 1, self.reference_time, 1, 1)]
        )
        loader = RecencyPopularityDataLoader(
            reference_time=self.reference_time,
            windows_hours=[6],
            half_lives_hours=[6],
        ).load_fit_data(history=invalid)
        with self.assertRaisesRegex(ContractValidationError, "at or after"):
            loader.prepare_fit_data()

    def test_candidate_source_is_partial_and_fallback_stays_separate(self) -> None:
        loader = self.prepared_loader()
        model = RecencyPopularityModel(
            {
                "config_id": "positive_6h",
                "score_kind": "window",
                "signal": "positive_daily_rows",
                "window_hours": 6,
            }
        ).fit(loader)
        candidates = model.predict(loader, k=20)
        self.assertEqual(candidates.height, 2)
        self.assertEqual(candidates.get_column("source").unique().to_list(), [
            "recency_popularity"
        ])


class PopularityPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.high_user_id = 2**53 + 123
        rows: list[tuple[int, int, int, int, int]] = []
        for item_id in range(1, 11):
            rows.append((100 + item_id, item_id, 0, 12 - item_id, item_id % 2))
        rows.extend(
            (self.high_user_id, item_id, 1, 1, 0) for item_id in range(1, 8)
        )
        rows.append((2, 10, 1, 1, 0))
        self.history = daily_frame(rows)
        self.targets = pl.DataFrame(
            {
                "user_id": pl.Series(
                    [self.high_user_id, 2], dtype=pl.UInt64
                )
            }
        ).cast(TARGET_USER_SCHEMA)

    def prepared_loader(self) -> PopularityDataLoader:
        return (
            PopularityDataLoader(seed=42)
            .load_fit_data(history=self.history)
            .prepare_fit_data()
            .load_predict_data(
                history=self.history, target_users=self.targets
            )
            .prepare_predict_data()
        )

    def test_predict_filters_seen_fills_deep_users_and_preserves_uint64(self) -> None:
        loader = self.prepared_loader()
        model = GlobalPopularityModel(PopularityScore.RAW_INTERACTION_COUNT)
        model.fit(loader)
        candidates = model.predict(loader, k=3, batch_size=1)
        validate_candidate_output(
            candidates, k=3, source_name=model.source_name
        )
        high_items = candidates.filter(
            pl.col("user_id") == self.high_user_id
        ).get_column("item_id")
        self.assertEqual(high_items.to_list(), [8, 9, 10])
        self.assertEqual(candidates.get_column("user_id").dtype, pl.UInt64)
        self.assertTrue(
            candidates.equals(model.predict(loader, k=3, batch_size=2))
        )
        validate_deterministic_iteration(loader, phase="predict", batch_size=1)

    def test_global_ranking_restore_does_not_refit(self) -> None:
        loader = self.prepared_loader()
        fitted = GlobalPopularityModel(
            PopularityScore.RAW_INTERACTION_COUNT
        ).fit(loader)
        expected = fitted.predict(loader, k=3, batch_size=1)
        restored = GlobalPopularityModel.from_fitted_ranking(
            fitted.score_type, fitted.item_ranking
        )
        actual = restored.predict(loader, k=3, batch_size=2)
        self.assertTrue(expected.equals(actual))

    def test_predict_rejects_insufficient_unseen_catalog(self) -> None:
        loader = self.prepared_loader()
        model = GlobalPopularityModel(PopularityScore.RAW_INTERACTION_COUNT)
        model.fit(loader)
        with self.assertRaisesRegex(ValueError, "known unseen"):
            model.predict(loader, k=4, batch_size=1)

    def test_candidates_convert_to_exact_recommendations(self) -> None:
        loader = self.prepared_loader()
        model = GlobalPopularityModel(PopularityScore.RAW_INTERACTION_COUNT)
        model.fit(loader)
        candidates = model.predict(loader, k=3, batch_size=2)
        recommendations = candidates_to_recommendations(
            candidates, self.targets, k=3
        )
        self.assertEqual(recommendations.schema, FINAL_RECOMMENDATION_SCHEMA)
        validate_recommendations_against_history(
            recommendations,
            target_users=self.targets,
            history_daily=self.history,
            expected_k=3,
        )


class PopularityFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = pl.DataFrame(
            {"user_id": pl.Series([1, 2], dtype=pl.UInt64)}
        )
        self.history = daily_frame(
            [
                (1, 1, 0, 1, 0),
                (2, 2, 0, 1, 0),
                (10, 3, 0, 1, 0),
                (10, 4, 0, 1, 0),
                (10, 5, 0, 1, 0),
                (10, 6, 0, 1, 0),
            ]
        )

    def test_fallback_filters_seen_unknown_and_duplicates(self) -> None:
        primary = candidate_frame(
            [
                (1, 1, 10.0, 1, "primary"),
                (1, 3, 9.0, 2, "primary"),
                (1, 3, 8.0, 3, "primary"),
                (2, 999, 10.0, 1, "primary"),
                (2, 4, 9.0, 2, "primary"),
            ]
        )
        fallback = candidate_frame(
            [
                (1, 3, 6.0, 1, "global_popularity"),
                (1, 4, 5.0, 2, "global_popularity"),
                (1, 5, 4.0, 3, "global_popularity"),
                (1, 6, 3.0, 4, "global_popularity"),
                (2, 3, 6.0, 1, "global_popularity"),
                (2, 4, 5.0, 2, "global_popularity"),
                (2, 5, 4.0, 3, "global_popularity"),
                (2, 6, 3.0, 4, "global_popularity"),
            ]
        )
        recommendations = fill_with_global_popularity(
            primary,
            fallback,
            self.targets,
            self.history,
            k=3,
        ).sort("user_id")
        self.assertEqual(
            recommendations.get_column("item_ids").to_list(),
            [[3, 4, 5], [4, 3, 5]],
        )
        validate_recommendations_against_history(
            recommendations,
            target_users=self.targets,
            history_daily=self.history,
            expected_k=3,
        )

    def test_semantic_validator_rejects_seen_and_unknown(self) -> None:
        invalid_seen = pl.DataFrame(
            [(1, [1, 3]), (2, [3, 4])],
            schema=FINAL_RECOMMENDATION_SCHEMA,
            orient="row",
        )
        with self.assertRaisesRegex(ContractValidationError, "seen_pairs=1"):
            validate_recommendations_against_history(
                invalid_seen,
                target_users=self.targets,
                history_daily=self.history,
                expected_k=2,
            )
        invalid_unknown = pl.DataFrame(
            [(1, [3, 999]), (2, [3, 4])],
            schema=FINAL_RECOMMENDATION_SCHEMA,
            orient="row",
        )
        with self.assertRaisesRegex(ContractValidationError, "unknown_items=1"):
            validate_recommendations_against_history(
                invalid_unknown,
                target_users=self.targets,
                history_daily=self.history,
                expected_k=2,
            )


class PopularityRunnerTests(unittest.TestCase):
    def _write_fold(
        self,
        root: Path,
        *,
        name: str,
        cutoff: str,
        end: str | None,
    ) -> Path:
        fold = root / name
        fold.mkdir()
        history = daily_frame(
            [(100, item_id, 0, 1, item_id % 2) for item_id in range(1, 26)]
        )
        targets = pl.DataFrame(
            {"user_id": pl.Series([1, 2], dtype=pl.UInt64)}
        ).cast(TARGET_USER_SCHEMA)
        ground_truth = pl.DataFrame(
            [(1, 1), (2, 2)],
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
            orient="row",
        )
        history.write_parquet(fold / "history_daily.parquet")
        targets.write_parquet(fold / "target_users.parquet")
        ground_truth.write_parquet(fold / "target_ground_truth.parquet")
        (fold / "config.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "mode": "full",
                    "split": {
                        "cutoff": cutoff,
                        "validation_end_exclusive": end,
                    },
                }
            ),
            encoding="utf-8",
        )
        (fold / "metrics.json").write_text(
            json.dumps(
                {
                    "mode": "full",
                    "deterministic_diagnostics": {"output_sha256": {}},
                }
            ),
            encoding="utf-8",
        )
        return fold

    def test_runner_selects_before_canonical_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection = [
                self._write_fold(
                    root,
                    name="fold1",
                    cutoff="2024-01-02T00:00:00",
                    end="2024-01-03T00:00:00",
                ),
                self._write_fold(
                    root,
                    name="fold2",
                    cutoff="2024-01-03T00:00:00",
                    end="2024-01-04T00:00:00",
                ),
                self._write_fold(
                    root,
                    name="fold3",
                    cutoff="2024-01-04T00:00:00",
                    end="2024-01-05T00:00:00",
                ),
            ]
            canonical = self._write_fold(
                root,
                name="canonical",
                cutoff="2024-01-05T00:00:00",
                end=None,
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "candidate_k": 20,
                        "final_k": 20,
                        "folds": {
                            "selection": [path.as_posix() for path in selection],
                            "canonical": canonical.as_posix(),
                        },
                        "predict_batch_size": 1,
                        "score_types": [
                            score.value for score in PopularityScore
                        ],
                        "seed": 42,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            metrics = run_experiment(
                config_path=config_path,
                output_dir=output,
                run_id="test_run",
            )
            self.assertEqual(
                metrics["selected_score_type"], "raw_interaction_count"
            )
            self.assertEqual(len(metrics["selection_folds"]), 3)
            self.assertTrue(metrics["deterministic_recommendations_match"])
            self.assertEqual(metrics["precision_at_20_all_targets"], 0.05)
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "item_ranking.parquet").is_file())
            self.assertTrue((output / "recommendations.parquet").is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(
                    config_path=config_path,
                    output_dir=output,
                    run_id="test_run",
                )


class RecencyPopularityRunnerTests(unittest.TestCase):
    def _write_fold(
        self,
        root: Path,
        *,
        name: str,
        cutoff: datetime,
        end: datetime | None,
    ) -> Path:
        fold = root / name
        fold.mkdir()
        history = timestamped_daily_frame(
            [
                (
                    100 + item_id,
                    item_id,
                    cutoff - timedelta(hours=1),
                    1,
                    item_id % 2,
                )
                for item_id in range(1, 26)
            ]
        )
        targets = pl.DataFrame(
            {"user_id": pl.Series([1, 2], dtype=pl.UInt64)}
        ).cast(TARGET_USER_SCHEMA)
        ground_truth = pl.DataFrame(
            [(1, 1), (2, 2)],
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
            orient="row",
        )
        history.write_parquet(fold / "history_daily.parquet")
        targets.write_parquet(fold / "target_users.parquet")
        ground_truth.write_parquet(fold / "target_ground_truth.parquet")
        (fold / "config.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "mode": "full",
                    "split": {
                        "cutoff": cutoff.isoformat(),
                        "validation_end_exclusive": (
                            end.isoformat() if end is not None else None
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        (fold / "metrics.json").write_text(
            json.dumps(
                {
                    "mode": "full",
                    "deterministic_diagnostics": {
                        "output_sha256": {"history_daily.parquet": "synthetic"}
                    },
                }
            ),
            encoding="utf-8",
        )
        return fold

    def _write_baseline(self, root: Path, canonical: Path) -> Path:
        baseline = root / "baseline"
        baseline.mkdir()
        history_path = canonical / "history_daily.parquet"
        targets = pl.read_parquet(canonical / "target_users.parquet")
        loader = (
            PopularityDataLoader(seed=42)
            .load_fit_data(history=history_path)
            .prepare_fit_data()
            .load_predict_data(history=history_path, target_users=targets)
            .prepare_predict_data()
        )
        model = GlobalPopularityModel(
            PopularityScore.RELEVANT_INTERACTION_COUNT
        ).fit(loader)
        recommendations = candidates_to_recommendations(
            model.predict(loader, k=20), targets, k=20
        )
        recommendations.write_parquet(baseline / "recommendations.parquet")
        (baseline / "config.json").write_text(
            json.dumps({"run_id": "synthetic_task02"}), encoding="utf-8"
        )
        (baseline / "metrics.json").write_text(
            json.dumps({"precision_at_20_all_targets": 0.05}),
            encoding="utf-8",
        )
        return baseline

    def test_runner_publishes_reusable_model_and_freezes_before_canonical(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cutoffs = [
                datetime(2024, 1, day, tzinfo=UTC).replace(tzinfo=None)
                for day in (2, 3, 4)
            ]
            selection = [
                self._write_fold(
                    root,
                    name=f"fold{index}",
                    cutoff=cutoff,
                    end=cutoff + timedelta(days=1),
                )
                for index, cutoff in enumerate(cutoffs, start=1)
            ]
            canonical = self._write_fold(
                root,
                name="canonical",
                cutoff=datetime(2024, 1, 5, tzinfo=UTC).replace(tzinfo=None),
                end=None,
            )
            baseline = self._write_baseline(root, canonical)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "baseline_artifact": baseline.as_posix(),
                        "candidate_k": 20,
                        "fallback_score_type": "relevant_interaction_count",
                        "final_k": 20,
                        "folds": {
                            "selection": [
                                path.as_posix() for path in selection
                            ],
                            "canonical": canonical.as_posix(),
                        },
                        "half_lives_hours": [6],
                        "model_configs": [
                            {
                                "config_id": "first",
                                "score_kind": "window",
                                "signal": "raw_views",
                                "window_hours": None,
                            },
                            {
                                "config_id": "second",
                                "score_kind": "window",
                                "signal": "raw_views",
                                "window_hours": None,
                            },
                        ],
                        "predict_batch_size": 1,
                        "seed": 42,
                        "windows_hours": [6],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            metrics = run_recency_experiment(
                config_path=config_path,
                output_dir=output,
                run_id="recency_test_run",
            )
            self.assertEqual(metrics["selected_config_id"], "first")
            self.assertEqual(len(metrics["selection_folds"]), 3)
            self.assertTrue(metrics["deterministic_recommendations_match"])
            self.assertTrue(
                metrics["baseline_comparison"][
                    "recommendations_match_stored_artifact"
                ]
            )
            portable = json.loads(
                (output / "model_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                RecencyPopularityConfig.from_dict(
                    portable["recency_config"]
                ).config_id,
                "first",
            )
            self.assertTrue((output / "item_ranking.parquet").is_file())
            self.assertTrue((output / "recommendations.parquet").is_file())
            with self.assertRaises(FileExistsError):
                run_recency_experiment(
                    config_path=config_path,
                    output_dir=output,
                    run_id="recency_test_run",
                )


if __name__ == "__main__":
    unittest.main()

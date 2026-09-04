from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from implicit_model import (
    COLLAPSED_ALS_HISTORY_SCHEMA,
    ImplicitALSConfig,
    ImplicitALSDataLoader,
    ImplicitALSModel,
    _build_confidence_csr,
    _collapse_with_confidence,
)
from scripts import run_implicit_als as implicit_runner
from scripts.run_implicit_als import BestModelCheckpoint, BestModelSnapshot
from validation import ContractValidationError, validate_candidate_output

REFERENCE = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)


def daily_frame(
    rows: list[tuple[int, int, datetime, int, int, int, int, int]],
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            (
                user_id,
                item_id,
                dt.date(),
                dt,
                views,
                watch_time,
                is_like,
                is_favorite,
                is_positive,
            )
            for (
                user_id,
                item_id,
                dt,
                views,
                watch_time,
                is_like,
                is_favorite,
                is_positive,
            ) in rows
        ],
        schema=DAILY_INTERACTION_SCHEMA,
        orient="row",
    )


def targets(*user_ids: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"user_id": pl.Series(user_ids, dtype=pl.UInt64)}
    ).cast(TARGET_USER_SCHEMA)


def small_history(*, item_count: int = 12) -> pl.DataFrame:
    rows: list[tuple[int, int, datetime, int, int, int, int, int]] = []
    for user_id in range(1, 9):
        for offset in range(4):
            item_id = ((user_id * 2 + offset) % item_count) + 1
            rows.append(
                (
                    user_id,
                    item_id,
                    REFERENCE - timedelta(hours=user_id + offset + 1),
                    offset + 1,
                    61 if offset == 0 else 60,
                    int(offset == 1),
                    int(offset == 2),
                    int(offset < 3),
                )
            )
    return daily_frame(rows)


def fitted(
    history: pl.DataFrame,
    config: ImplicitALSConfig,
    target_users: pl.DataFrame,
) -> tuple[ImplicitALSDataLoader, ImplicitALSModel]:
    loader = (
        ImplicitALSDataLoader(
            config=config,
            reference_time=REFERENCE,
            seed=config.seed,
        )
        .load_fit_data(history=history)
        .prepare_fit_data()
        .load_predict_data(history=history, target_users=target_users)
        .prepare_predict_data()
    )
    return loader, ImplicitALSModel(config).fit(loader, show_progress=False)


class ImplicitALSConfigTests(unittest.TestCase):
    def test_round_trip_and_validation(self) -> None:
        config = ImplicitALSConfig(
            config_id="event",
            factors=8,
            regularization=0.1,
            iterations=3,
            num_threads=1,
            interaction_weight=1,
            view_weight=1,
            long_watch_weight=1,
            like_weight=2,
            favorite_weight=2,
            half_life_hours=24,
        )
        encoded = json.loads(json.dumps(config.to_dict(), allow_nan=False))
        self.assertEqual(ImplicitALSConfig.from_dict(encoded), config)
        with self.assertRaises(ValueError):
            ImplicitALSConfig(
                config_id="zero",
                interaction_weight=0,
                view_weight=0,
                long_watch_weight=0,
                like_weight=0,
                favorite_weight=0,
            )
        with self.assertRaises(ValueError):
            ImplicitALSConfig(config_id="nan", view_weight=float("nan"))
        with self.assertRaises(ValueError):
            ImplicitALSConfig.from_dict(
                {"config_id": "bad", "unexpected": True}
            )


class ImplicitALSLoaderTests(unittest.TestCase):
    def test_uint64_mapping_is_sorted_stable_and_never_float(self) -> None:
        large = 2**53 + 99
        history = daily_frame(
            [
                (large, 30, REFERENCE - timedelta(hours=1), 1, 1, 0, 0, 0),
                (2, 20, REFERENCE - timedelta(hours=2), 1, 1, 0, 0, 0),
                (large, 10, REFERENCE - timedelta(hours=3), 1, 1, 0, 0, 0),
            ]
        )
        config = ImplicitALSConfig(
            config_id="mapping", factors=2, iterations=1, num_threads=1
        )
        first = (
            ImplicitALSDataLoader(config=config, reference_time=REFERENCE)
            .load_fit_data(history=history)
            .prepare_fit_data()
        )
        second = (
            ImplicitALSDataLoader(config=config, reference_time=REFERENCE)
            .load_fit_data(history=history.reverse())
            .prepare_fit_data()
        )
        self.assertTrue(first.user_mapping.equals(second.user_mapping))
        self.assertTrue(first.item_mapping.equals(second.item_mapping))
        self.assertEqual(
            first.user_mapping.get_column("user_id").to_list(), [2, large]
        )
        self.assertEqual(first.user_mapping.schema["user_id"], pl.UInt64)
        self.assertEqual(first.user_mapping.schema["user_index"], pl.UInt32)
        self.assertEqual(
            first.item_mapping.get_column("item_id").to_list(), [10, 20, 30]
        )

    def test_aggregation_confidence_components_and_decay_boundary(self) -> None:
        history = daily_frame(
            [
                (1, 10, REFERENCE - timedelta(hours=48), 2, 60, 0, 1, 1),
                (1, 10, REFERENCE - timedelta(hours=24), 3, 59, 1, 0, 1),
                (1, 20, REFERENCE - timedelta(hours=24), 1, 61, 0, 0, 1),
            ]
        )
        config = ImplicitALSConfig(
            config_id="confidence",
            factors=2,
            iterations=1,
            num_threads=1,
            interaction_weight=1,
            view_weight=1,
            long_watch_weight=3,
            like_weight=2,
            favorite_weight=4,
            half_life_hours=24,
        )
        collapsed = _collapse_with_confidence(
            history.lazy(), reference_time=REFERENCE, config=config
        )
        self.assertEqual(collapsed.schema, COLLAPSED_ALS_HISTORY_SCHEMA)
        item10 = collapsed.filter(pl.col("item_id") == 10).row(0, named=True)
        self.assertEqual(item10["views"], 5)
        self.assertEqual(item10["watch_time"], 60)
        self.assertEqual(item10["is_like"], 1)
        self.assertEqual(item10["is_favorite"], 1)
        expected10 = 1 + (1 + np.log1p(4) + 2 + 4) * 0.5
        self.assertAlmostEqual(item10["confidence"], expected10, places=6)
        item20 = collapsed.filter(pl.col("item_id") == 20).row(0, named=True)
        self.assertAlmostEqual(item20["confidence"], 1 + (1 + 3) * 0.5)

    def test_csr_float32_and_rejects_duplicates_nonpositive_and_leakage(self) -> None:
        config = ImplicitALSConfig(
            config_id="csr", factors=2, iterations=1, num_threads=1
        )
        history = small_history()
        loader = (
            ImplicitALSDataLoader(config=config, reference_time=REFERENCE)
            .load_fit_data(history=history)
            .prepare_fit_data()
        )
        matrix = loader.fit_matrix
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(matrix.has_sorted_indices)
        self.assertGreater(matrix.nnz, 0)

        collapsed = _collapse_with_confidence(
            history.lazy(), reference_time=REFERENCE, config=config
        )
        duplicate = pl.concat((collapsed, collapsed.head(1)))
        with self.assertRaises(ContractValidationError):
            _build_confidence_csr(
                duplicate, loader.user_mapping, loader.item_mapping
            )
        nonpositive = collapsed.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(pl.col("confidence"))
            .alias("confidence")
        )
        with self.assertRaises(ContractValidationError):
            _build_confidence_csr(
                nonpositive, loader.user_mapping, loader.item_mapping
            )

        future = history.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(REFERENCE, dtype=pl.Datetime("us")))
            .otherwise(pl.col("dt"))
            .alias("dt")
        )
        with self.assertRaises(ContractValidationError):
            (
                ImplicitALSDataLoader(config=config, reference_time=REFERENCE)
                .load_fit_data(history=future)
                .prepare_fit_data()
            )

        duplicate_daily = pl.concat((history, history.head(1)))
        with self.assertRaises(ContractValidationError):
            (
                ImplicitALSDataLoader(config=config, reference_time=REFERENCE)
                .load_fit_data(history=duplicate_daily)
                .prepare_fit_data()
            )


class ImplicitALSModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = small_history(item_count=50)
        self.config = ImplicitALSConfig(
            config_id="model",
            factors=4,
            regularization=0.05,
            iterations=2,
            num_threads=1,
            seed=42,
        )

    def test_scores_match_manual_factor_dot_product_and_seen_are_filtered(self) -> None:
        loader, model = fitted(self.history, self.config, targets(1, 2))
        candidates = model.predict(loader, k=5, batch_size=2)
        validate_candidate_output(candidates, k=5, source_name="implicit_als")
        seen = set(
            self.history.filter(pl.col("user_id") == 1)
            .get_column("item_id")
            .to_list()
        )
        user_one = candidates.filter(pl.col("user_id") == 1)
        self.assertTrue(seen.isdisjoint(user_one.get_column("item_id").to_list()))
        first = user_one.row(0, named=True)
        user_index = model.user_mapping.filter(
            pl.col("user_id") == 1
        ).get_column("user_index").item()
        item_index = model.item_mapping.filter(
            pl.col("item_id") == first["item_id"]
        ).get_column("item_index").item()
        manual = float(
            model.backend.user_factors[user_index]
            @ model.backend.item_factors[item_index]
        )
        self.assertAlmostEqual(first["score"], manual, places=6)

    def test_fit_callback_reports_every_iteration(self) -> None:
        loader = (
            ImplicitALSDataLoader(
                config=self.config,
                reference_time=REFERENCE,
                seed=self.config.seed,
            )
            .load_fit_data(history=self.history)
            .prepare_fit_data()
        )
        iterations: list[int] = []
        ImplicitALSModel(self.config).fit(
            loader,
            show_progress=False,
            callback=lambda iteration, _elapsed, _loss: iterations.append(
                iteration
            ),
        )
        self.assertEqual(iterations, [0, 1])

    def test_unknown_users_are_empty_and_ties_use_item_id(self) -> None:
        loader, model = fitted(self.history, self.config, targets(1, 2**53 + 7))
        model.backend.user_factors[:] = 0
        model.backend.item_factors[:] = 0
        candidates = model.predict(loader, k=5, batch_size=1)
        self.assertEqual(
            candidates.filter(pl.col("user_id") == 2**53 + 7).height, 0
        )
        seen = set(
            self.history.filter(pl.col("user_id") == 1)
            .get_column("item_id")
            .to_list()
        )
        expected = [
            item
            for item in model.item_mapping.get_column("item_id").to_list()
            if item not in seen
        ][:5]
        self.assertEqual(candidates.get_column("item_id").to_list(), expected)
        self.assertEqual(candidates.get_column("score").to_list(), [0.0] * 5)

    def test_save_load_does_not_fit_and_restores_identical_topk(self) -> None:
        loader, model = fitted(self.history, self.config, targets(1, 2, 99))
        expected = model.predict(loader, k=5, batch_size=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "artifact"
            model.save(
                artifact,
                metadata={
                    "fit_reference_time": REFERENCE.isoformat(),
                    "fit_history_sha256": "synthetic",
                },
            )
            with patch.object(
                ImplicitALSModel,
                "fit",
                side_effect=AssertionError("restore must not fit"),
            ):
                restored = ImplicitALSModel.from_artifact(artifact)
            restore_loader = (
                ImplicitALSDataLoader(
                    config=restored.config,
                    reference_time=REFERENCE,
                    user_mapping=restored.user_mapping,
                    item_mapping=restored.item_mapping,
                    seed=42,
                )
                .load_predict_data(history=self.history, target_users=targets(1, 2, 99))
                .prepare_predict_data()
            )
            actual = restored.predict(restore_loader, k=5, batch_size=2)
            self.assertTrue(expected.equals(actual))
            self.assertTrue(
                np.array_equal(
                    model.backend.user_factors,
                    restored.backend.user_factors,
                )
            )
            self.assertTrue(
                np.array_equal(
                    model.backend.item_factors,
                    restored.backend.item_factors,
                )
            )
            with self.assertRaises(FileExistsError):
                model.save(artifact)

    def test_best_model_callback_replaces_only_on_metric_improvement(self) -> None:
        loader, first_model = fitted(
            self.history, self.config, targets(1, 2)
        )
        better_config = ImplicitALSConfig(
            **{
                **self.config.to_dict(),
                "config_id": "better",
                "regularization": 0.1,
            }
        )
        second_model = ImplicitALSModel(better_config).fit(
            loader, show_progress=False
        )
        base_summary = {
            "mean_union_oracle_p20_all_targets_gain": 0.01,
            "mean_candidate_oracle_p20_all_targets": 0.02,
            "mean_precision_at_20_all_targets": 0.003,
            "mean_runtime_seconds": 10.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = BestModelCheckpoint(
                Path(temp_dir) / "best",
                run_id="callback_test",
                source_config_sha256="synthetic",
            )

            def snapshot(
                model: ImplicitALSModel,
                summary: dict[str, float],
            ) -> BestModelSnapshot:
                return BestModelSnapshot(
                    stage="stage",
                    config=model.config,
                    fold_label="rolling_3",
                    cutoff=REFERENCE,
                    model=model,
                    fold_metrics={"precision_at_20_all_targets": 0.0},
                    selection_summary=summary,
                )

            checkpoint(snapshot(first_model, base_summary))
            pointer_path = checkpoint.directory / "best_model.json"
            first_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            checkpoint(
                snapshot(
                    first_model,
                    {
                        **base_summary,
                        "mean_union_oracle_p20_all_targets_gain": 0.009,
                    },
                )
            )
            self.assertEqual(
                json.loads(pointer_path.read_text(encoding="utf-8")),
                first_pointer,
            )
            checkpoint(
                snapshot(
                    second_model,
                    {
                        **base_summary,
                        "mean_union_oracle_p20_all_targets_gain": 0.011,
                    },
                )
            )
            final_pointer = json.loads(
                pointer_path.read_text(encoding="utf-8")
            )
            self.assertEqual(final_pointer["config_id"], "better")
            self.assertNotEqual(
                final_pointer["model_path"], first_pointer["model_path"]
            )
            self.assertEqual(
                len(list((checkpoint.directory / "versions").iterdir())), 1
            )


class ImplicitALSRunnerTests(unittest.TestCase):
    def _write_fold(
        self,
        root: Path,
        *,
        name: str,
        cutoff: datetime,
        selection: bool,
    ) -> Path:
        fold = root / name
        fold.mkdir()
        rows = [
            (
                1000 + item_id,
                item_id,
                cutoff - timedelta(hours=2),
                1,
                70,
                0,
                0,
                1,
            )
            for item_id in range(1, 31)
        ]
        rows.extend(
            [
                (101, 1, cutoff - timedelta(hours=3), 1, 70, 0, 0, 1),
                (101, 2, cutoff - timedelta(hours=1), 1, 70, 0, 0, 1),
                (102, 3, cutoff - timedelta(hours=3), 1, 70, 0, 0, 1),
                (102, 4, cutoff - timedelta(hours=1), 1, 70, 0, 0, 1),
                (1, 1, cutoff - timedelta(hours=1), 1, 10, 0, 0, 0),
                (2, 3, cutoff - timedelta(hours=1), 1, 10, 0, 0, 0),
            ]
        )
        daily_frame(rows).write_parquet(fold / "history_daily.parquet")
        targets(1, 2).write_parquet(fold / "target_users.parquet")
        pl.DataFrame(
            [(1, 2), (2, 4)],
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
            orient="row",
        ).write_parquet(fold / "target_ground_truth.parquet")
        end = cutoff + timedelta(days=1) if selection else None
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
                        "output_sha256": {
                            "history_daily.parquet": "synthetic"
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return fold

    def _write_fixed_sources(self, root: Path) -> tuple[Path, Path, Path]:
        task02 = root / "task02"
        task03 = root / "task03"
        task04 = root / "task04"
        task02.mkdir()
        task03.mkdir()
        task04.mkdir()
        (task02 / "config.json").write_text(
            json.dumps(
                {
                    "model_config": {
                        "score_type": "relevant_interaction_count"
                    }
                }
            ),
            encoding="utf-8",
        )
        (task03 / "model_config.json").write_text(
            json.dumps(
                {
                    "recency_config": {
                        "config_id": "fixed_recency",
                        "score_kind": "window",
                        "signal": "positive_daily_rows",
                        "window_hours": None,
                    }
                }
            ),
            encoding="utf-8",
        )
        (task04 / "model_config.json").write_text(
            json.dumps(
                {
                    "item2item_config": {
                        "config_id": "fixed_item2item",
                        "history_cap": 2,
                        "neighbor_k": 2,
                        "seed_k": 1,
                        "seed_recency_half_life_hours": None,
                    }
                }
            ),
            encoding="utf-8",
        )
        return task02, task03, task04

    def test_runner_isolates_canonical_and_publishes_atomically(self) -> None:
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
                    selection=True,
                )
                for index, cutoff in enumerate(cutoffs, start=1)
            ]
            canonical = self._write_fold(
                root,
                name="canonical",
                cutoff=datetime(2024, 1, 5, tzinfo=UTC).replace(tzinfo=None),
                selection=False,
            )
            task02, task03, task04 = self._write_fixed_sources(root)
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ablation": {
                            "confidence_profiles": [
                                {
                                    "name": "binary",
                                    "interaction_weight": 1,
                                    "view_weight": 0,
                                    "long_watch_weight": 0,
                                    "like_weight": 0,
                                    "favorite_weight": 0,
                                }
                            ],
                            "half_life_hours": [None],
                            "factors": [2],
                            "regularization": [0.05],
                            "iterations": [1],
                        },
                        "base_config": {
                            "config_id": "base",
                            "factors": 2,
                            "regularization": 0.05,
                            "iterations": 1,
                            "num_threads": 1,
                            "seed": 42,
                        },
                        "candidate_k": 20,
                        "final_k": 20,
                        "predict_batch_size": 1,
                        "fallback_score_type": "relevant_interaction_count",
                        "seed": 42,
                        "folds": {
                            "selection": [path.as_posix() for path in selection],
                            "canonical": canonical.as_posix(),
                        },
                        "task02_artifact": task02.as_posix(),
                        "task03_artifact": task03.as_posix(),
                        "task04_artifact": task04.as_posix(),
                    }
                ),
                encoding="utf-8",
            )
            calls: list[bool] = []
            original_validate_fold = implicit_runner._validate_fold

            def recording_validate_fold(
                path: Path, *, selection: bool
            ) -> tuple[dict[str, Path], dict[str, object]]:
                calls.append(selection)
                return original_validate_fold(path, selection=selection)

            output = root / "run"
            log_file = root / "task05.log"
            best_model_dir = root / "best-model"
            with patch.object(
                implicit_runner,
                "_validate_fold",
                side_effect=recording_validate_fold,
            ):
                metrics = implicit_runner.run_experiment(
                    config_path=config_path,
                    output_dir=output,
                    run_id="implicit_test",
                    log_file=log_file,
                    show_progress=False,
                    best_model_dir=best_model_dir,
                )
            self.assertEqual(calls, [True, True, True, False])
            self.assertEqual(len(metrics["selection_stages"]), 5)
            self.assertEqual(
                [stage["new_fit_count"] for stage in metrics["selection_stages"]],
                [3, 0, 0, 0, 0],
            )
            self.assertEqual(metrics["canonical_evaluated_config_count"], 1)
            self.assertTrue(metrics["deterministic_recommendations_match"])
            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("event=run_start", log_text)
            self.assertIn('stage="stage1_confidence_profile"', log_text)
            self.assertIn('config="stage1_confidence_binary"', log_text)
            self.assertIn('fold="rolling_1"', log_text)
            self.assertIn("event=fit_iteration", log_text)
            self.assertIn("event=best_model_save_finish", log_text)
            self.assertIn('status="completed"', log_text)
            self.assertLess(log_file.stat().st_size, 100_000)
            best_pointer = json.loads(
                (best_model_dir / "best_model.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                best_pointer["kind"], "task05_rolling_best_model"
            )
            self.assertEqual(
                best_pointer["config_id"], "stage1_confidence_binary"
            )
            self.assertEqual(
                metrics["rolling_best_model_checkpoint"]["config_id"],
                "stage1_confidence_binary",
            )
            checkpoint_model = ImplicitALSModel.from_artifact(
                best_model_dir / best_pointer["model_path"]
            )
            self.assertEqual(
                checkpoint_model.config.config_id,
                "stage1_confidence_binary",
            )
            self.assertEqual(
                len(list((best_model_dir / "versions").iterdir())), 1
            )
            self.assertFalse((output / "selection_union_hit_cache").exists())
            self.assertFalse((output / "canonical_union_hits.parquet").exists())
            for name in (
                "config.json",
                "metrics.json",
                "model_config.json",
                "als_model.npz",
                "user_mapping.parquet",
                "item_mapping.parquet",
                "recommendations.parquet",
            ):
                self.assertTrue((output / name).is_file())
            with self.assertRaises(FileExistsError):
                implicit_runner.run_experiment(
                    config_path=config_path,
                    output_dir=output,
                    run_id="implicit_test",
                )


if __name__ == "__main__":
    unittest.main()

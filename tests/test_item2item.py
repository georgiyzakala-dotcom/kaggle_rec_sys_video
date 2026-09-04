from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from data_utils import DAILY_INTERACTION_SCHEMA, TARGET_USER_SCHEMA
from item2item import (
    NEIGHBOR_TABLE_SCHEMA,
    Item2ItemConfig,
    Item2ItemDataLoader,
    Item2ItemModel,
)
from popularity import (
    GlobalPopularityModel,
    PopularityDataLoader,
    PopularityScore,
    RecencyPopularityConfig,
    RecencyPopularityDataLoader,
    RecencyPopularityModel,
    candidates_to_recommendations,
)
from scripts.run_item2item import run_experiment
from validation import ContractValidationError, validate_candidate_output

REFERENCE = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)


def daily_frame(
    rows: list[tuple[int, int, datetime, int, int, int, int, int]],
) -> pl.DataFrame:
    values = [
        (user_id, item_id, dt.date(), dt, views, watch, like, favorite, positive)
        for user_id, item_id, dt, views, watch, like, favorite, positive in rows
    ]
    return pl.DataFrame(
        values,
        schema=DAILY_INTERACTION_SCHEMA,
        orient="row",
    )


def targets(*user_ids: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"user_id": pl.Series(user_ids, dtype=pl.UInt64)}
    ).cast(TARGET_USER_SCHEMA)


def fitted_model(
    history: pl.DataFrame,
    config: Item2ItemConfig,
    *,
    target_users: pl.DataFrame | None = None,
) -> tuple[Item2ItemDataLoader, Item2ItemModel]:
    loader = (
        Item2ItemDataLoader(
            reference_time=REFERENCE,
            max_history_items=10,
            max_seed_items=10,
        )
        .load_fit_data(history=history)
        .prepare_fit_data()
    )
    model = Item2ItemModel(config).fit(loader)
    if target_users is not None:
        loader.load_predict_data(
            history=history, target_users=target_users
        ).prepare_predict_data()
    return loader, model


class Item2ItemConfigTests(unittest.TestCase):
    def test_config_round_trip_and_validation(self) -> None:
        config = Item2ItemConfig(
            config_id="directed_positive",
            profile="positive",
            direction="directed",
            pair_weight="time_distance",
            normalization="cosine",
            min_pair_users=2,
            neighbor_k=50,
            seed_k=3,
            seed_recency_half_life_hours=None,
            seed_strength="event_strength",
        )
        encoded = json.loads(json.dumps(config.to_dict(), allow_nan=False))
        self.assertEqual(Item2ItemConfig.from_dict(encoded), config)
        with self.assertRaises(ValueError):
            Item2ItemConfig(config_id="bad", seed_k=11, history_cap=10)
        with self.assertRaises(ValueError):
            Item2ItemConfig(config_id="bad", normalization="dense")
        with self.assertRaises(ValueError):
            Item2ItemConfig.from_dict(
                {"config_id": "bad", "unexpected": True}
            )


class Item2ItemFitTests(unittest.TestCase):
    def test_collapse_sort_and_history_cap(self) -> None:
        history = daily_frame(
            [
                (1, 30, REFERENCE - timedelta(hours=4), 1, 10, 0, 0, 0),
                (1, 20, REFERENCE - timedelta(hours=1), 2, 70, 0, 0, 1),
                (1, 10, REFERENCE - timedelta(hours=1), 3, 20, 1, 0, 1),
                (1, 30, REFERENCE - timedelta(hours=2), 4, 80, 0, 1, 1),
                (2, 40, REFERENCE - timedelta(hours=1), 1, 10, 0, 0, 0),
            ]
        )
        loader = (
            Item2ItemDataLoader(
                reference_time=REFERENCE,
                max_history_items=2,
                max_seed_items=2,
            )
            .load_fit_data(history=history)
            .prepare_fit_data()
        )
        user_one = loader.fit_profiles.filter(pl.col("user_id") == 1)
        self.assertEqual(user_one.get_column("item_id").to_list(), [10, 20])
        self.assertEqual(user_one.get_column("history_rank").to_list(), [1, 2])
        collapsed_30 = (
            Item2ItemDataLoader(
                reference_time=REFERENCE,
                max_history_items=3,
                max_seed_items=3,
            )
            .load_fit_data(history=history)
            .prepare_fit_data()
            .fit_profiles
            .filter((pl.col("user_id") == 1) & (pl.col("item_id") == 30))
            .row(0, named=True)
        )
        self.assertEqual(collapsed_30["views"], 5)
        self.assertEqual(collapsed_30["watch_time"], 80)
        self.assertEqual(collapsed_30["is_favorite"], 1)
        self.assertEqual(collapsed_30["last_dt"], REFERENCE - timedelta(hours=2))

    def test_positive_profile_pair_counts_min_support_and_normalization(self) -> None:
        history = daily_frame(
            [
                (1, 1, REFERENCE - timedelta(hours=1), 1, 70, 0, 0, 1),
                (1, 2, REFERENCE - timedelta(hours=2), 1, 70, 0, 0, 1),
                (1, 3, REFERENCE - timedelta(hours=3), 1, 10, 0, 0, 0),
                (2, 1, REFERENCE - timedelta(hours=1), 1, 70, 0, 0, 1),
                (2, 2, REFERENCE - timedelta(hours=2), 1, 70, 0, 0, 1),
                (3, 1, REFERENCE - timedelta(hours=1), 1, 70, 0, 0, 1),
                (3, 3, REFERENCE - timedelta(hours=2), 1, 70, 0, 0, 1),
            ]
        )
        _, raw_model = fitted_model(
            history,
            Item2ItemConfig(
                config_id="raw",
                profile="positive",
                min_pair_users=2,
                seed_recency_half_life_hours=None,
            ),
        )
        raw = raw_model.neighbor_table
        self.assertEqual(
            set(raw.iter_rows()),
            {(1, 2, 2.0, 2, 1), (2, 1, 2.0, 2, 1)},
        )

        for normalization, expected in (
            ("cosine", 2 / (3 * 2) ** 0.5),
            ("jaccard", 2 / 3),
        ):
            _, model = fitted_model(
                history,
                Item2ItemConfig(
                    config_id=normalization,
                    profile="positive",
                    normalization=normalization,
                    min_pair_users=2,
                    seed_recency_half_life_hours=None,
                ),
            )
            value = model.neighbor_table.filter(
                (pl.col("item_id") == 1)
                & (pl.col("neighbor_item_id") == 2)
            ).get_column("score").item()
            self.assertAlmostEqual(value, expected)

    def test_directed_is_strictly_older_to_newer(self) -> None:
        history = daily_frame(
            [
                (1, 1, REFERENCE - timedelta(hours=3), 1, 1, 0, 0, 0),
                (1, 2, REFERENCE - timedelta(hours=1), 1, 1, 0, 0, 0),
                (1, 3, REFERENCE - timedelta(hours=1), 1, 1, 0, 0, 0),
            ]
        )
        _, model = fitted_model(
            history,
            Item2ItemConfig(
                config_id="directed",
                direction="directed",
                seed_recency_half_life_hours=None,
            ),
        )
        links = set(
            model.neighbor_table.select(
                "item_id", "neighbor_item_id"
            ).iter_rows()
        )
        self.assertEqual(links, {(1, 2), (1, 3)})

    def test_time_distance_and_top_neighbor_pruning(self) -> None:
        history = daily_frame(
            [
                (1, 1, REFERENCE - timedelta(hours=1), 1, 1, 0, 0, 0),
                (1, 2, REFERENCE - timedelta(hours=25), 1, 1, 0, 0, 0),
                (2, 1, REFERENCE - timedelta(hours=1), 1, 1, 0, 0, 0),
                (2, 3, REFERENCE - timedelta(hours=1), 1, 1, 0, 0, 0),
            ]
        )
        _, model = fitted_model(
            history,
            Item2ItemConfig(
                config_id="decayed",
                pair_weight="time_distance",
                neighbor_k=1,
                seed_recency_half_life_hours=None,
            ),
        )
        item_one = model.neighbor_table.filter(pl.col("item_id") == 1)
        self.assertEqual(item_one.height, 1)
        self.assertEqual(item_one.get_column("neighbor_item_id").item(), 3)
        self.assertAlmostEqual(item_one.get_column("score").item(), 1.0)


class Item2ItemPredictionTests(unittest.TestCase):
    def _history_and_neighbors(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        history = daily_frame(
            [
                (1, 1, REFERENCE - timedelta(hours=4), 1, 10, 0, 0, 0),
                (1, 2, REFERENCE - timedelta(hours=1), 3, 70, 1, 0, 1),
                (2, 9, REFERENCE - timedelta(hours=1), 1, 10, 0, 0, 0),
            ]
        )
        neighbors = pl.DataFrame(
            [
                (1, 10, 2.0, 1, 1),
                (1, 11, 1.0, 1, 2),
                (2, 10, 3.0, 1, 1),
                (2, 12, 1.0, 1, 2),
            ],
            schema=NEIGHBOR_TABLE_SCHEMA,
            orient="row",
        )
        return history, neighbors

    def test_seed_aggregation_uses_max_seen_filter_and_tie_break(self) -> None:
        history, neighbors = self._history_and_neighbors()
        config = Item2ItemConfig(
            config_id="predict",
            neighbor_k=2,
            seed_k=2,
            seed_recency_half_life_hours=None,
        )
        loader = (
            Item2ItemDataLoader(
                reference_time=REFERENCE,
                max_history_items=10,
                max_seed_items=10,
            )
            .load_predict_data(history=history, target_users=targets(1, 2))
            .prepare_predict_data()
        )
        model = Item2ItemModel.from_fitted_neighbors(config, neighbors)
        candidates = model.predict(loader, k=10, batch_size=1)
        validate_candidate_output(candidates, k=10, source_name="item2item")
        user_one = candidates.filter(pl.col("user_id") == 1)
        self.assertEqual(user_one.get_column("item_id").to_list(), [10, 11, 12])
        self.assertEqual(user_one.get_column("score").to_list(), [3.0, 1.0, 1.0])
        self.assertEqual(user_one.get_column("rank").to_list(), [1, 2, 3])
        self.assertEqual(
            candidates.filter(pl.col("user_id") == 2).height,
            0,
        )

    def test_seed_recency_strength_and_artifact_restore(self) -> None:
        history, neighbors = self._history_and_neighbors()
        loader = (
            Item2ItemDataLoader(
                reference_time=REFERENCE,
                max_history_items=10,
                max_seed_items=10,
            )
            .load_predict_data(history=history, target_users=targets(1))
            .prepare_predict_data()
        )
        recency_config = Item2ItemConfig(
            config_id="recency",
            neighbor_k=2,
            seed_k=2,
            seed_recency_half_life_hours=1,
        )
        recency_model = Item2ItemModel.from_fitted_neighbors(
            recency_config, neighbors
        )
        recency = recency_model.predict(loader, k=10)
        score_10 = recency.filter(pl.col("item_id") == 10).get_column("score").item()
        self.assertAlmostEqual(score_10, 1.5)

        strength_config = Item2ItemConfig(
            config_id="strength",
            neighbor_k=2,
            seed_k=2,
            seed_recency_half_life_hours=None,
            seed_strength="event_strength",
        )
        restored = Item2ItemModel.from_fitted_neighbors(
            Item2ItemConfig.from_dict(strength_config.to_dict()),
            neighbors.clone(),
        )
        first = restored.predict(loader, k=10)
        second = restored.predict(loader, k=10)
        self.assertTrue(first.equals(second))
        self.assertGreater(
            first.filter(pl.col("item_id") == 10).get_column("score").item(),
            3.0,
        )

    def test_restore_rejects_invalid_neighbors(self) -> None:
        _, neighbors = self._history_and_neighbors()
        bad = neighbors.with_columns(
            pl.when(pl.col("item_id") == 1)
            .then(pl.lit(1, dtype=pl.Int32))
            .otherwise(pl.col("neighbor_item_id"))
            .alias("neighbor_item_id")
        )
        with self.assertRaises(ContractValidationError):
            Item2ItemModel.from_fitted_neighbors(
                Item2ItemConfig(config_id="bad", neighbor_k=2), bad
            )


class Item2ItemRunnerTests(unittest.TestCase):
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
        history = daily_frame(rows)
        target_users = targets(1, 2)
        ground_truth = pl.DataFrame(
            [(1, 2), (2, 4)],
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
            orient="row",
        )
        history.write_parquet(fold / "history_daily.parquet")
        target_users.write_parquet(fold / "target_users.parquet")
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
                        "output_sha256": {
                            "history_daily.parquet": "synthetic"
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return fold

    def _write_baselines(
        self, root: Path, canonical: Path, cutoff: datetime
    ) -> tuple[Path, Path]:
        history_path = canonical / "history_daily.parquet"
        target_users = pl.read_parquet(canonical / "target_users.parquet")
        popularity_loader = (
            PopularityDataLoader()
            .load_fit_data(history=history_path)
            .prepare_fit_data()
            .load_predict_data(history=history_path, target_users=target_users)
            .prepare_predict_data()
        )
        global_model = GlobalPopularityModel(
            PopularityScore.RELEVANT_INTERACTION_COUNT
        ).fit(popularity_loader)
        global_recommendations = candidates_to_recommendations(
            global_model.predict(popularity_loader, k=20),
            target_users,
            k=20,
        )
        task02 = root / "task02"
        task02.mkdir()
        global_recommendations.write_parquet(
            task02 / "recommendations.parquet"
        )
        (task02 / "config.json").write_text(
            json.dumps({"run_id": "synthetic_task02"}), encoding="utf-8"
        )
        (task02 / "metrics.json").write_text("{}", encoding="utf-8")

        recency_config = RecencyPopularityConfig(
            config_id="synthetic_recency",
            score_kind="window",
            signal="raw_views",
            window_hours=None,
        )
        recency_loader = (
            RecencyPopularityDataLoader(
                reference_time=cutoff,
                windows_hours=[],
                half_lives_hours=[],
            )
            .load_fit_data(history=history_path)
            .prepare_fit_data()
            .load_predict_data(history=history_path, target_users=target_users)
            .prepare_predict_data()
        )
        recency_model = RecencyPopularityModel(recency_config).fit(
            recency_loader
        )
        recency_recommendations = candidates_to_recommendations(
            recency_model.predict(recency_loader, k=20),
            target_users,
            k=20,
        )
        task03 = root / "task03"
        task03.mkdir()
        recency_recommendations.write_parquet(
            task03 / "recommendations.parquet"
        )
        recency_model.item_ranking.write_parquet(
            task03 / "item_ranking.parquet"
        )
        (task03 / "config.json").write_text(
            json.dumps({"run_id": "synthetic_task03"}), encoding="utf-8"
        )
        (task03 / "metrics.json").write_text("{}", encoding="utf-8")
        (task03 / "model_config.json").write_text(
            json.dumps({"recency_config": recency_config.to_dict()}),
            encoding="utf-8",
        )
        return task02, task03

    def test_runner_uses_rolling_selection_then_atomic_canonical_publish(
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
            canonical_cutoff = datetime(
                2024, 1, 5, tzinfo=UTC
            ).replace(tzinfo=None)
            canonical = self._write_fold(
                root,
                name="canonical",
                cutoff=canonical_cutoff,
                end=None,
            )
            task02, task03 = self._write_baselines(
                root, canonical, canonical_cutoff
            )
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ablation": {
                            "directions": ["undirected"],
                            "min_pair_users": [1],
                            "neighbor_k": [2],
                            "normalizations": ["raw"],
                            "pair_weights": ["uniform"],
                            "profiles": ["all"],
                            "seed_k": [1],
                            "seed_recency_half_life_hours": [None],
                            "seed_strengths": ["uniform"],
                        },
                        "base_config": {
                            "config_id": "base",
                            "history_cap": 2,
                            "neighbor_k": 2,
                            "seed_k": 1,
                            "seed_recency_half_life_hours": None,
                        },
                        "candidate_k": 20,
                        "fallback_score_type": "relevant_interaction_count",
                        "final_k": 20,
                        "folds": {
                            "canonical": canonical.as_posix(),
                            "selection": [
                                path.as_posix() for path in selection
                            ],
                        },
                        "predict_batch_size": 1,
                        "seed": 42,
                        "task02_artifact": task02.as_posix(),
                        "task03_artifact": task03.as_posix(),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            metrics = run_experiment(
                config_path=config_path,
                output_dir=output,
                run_id="item2item_test",
            )
            self.assertEqual(len(metrics["selection_stages"]), 7)
            self.assertEqual(metrics["canonical_evaluated_config_count"], 1)
            self.assertTrue(metrics["deterministic_recommendations_match"])
            self.assertEqual(metrics["precision_at_20_all_targets"], 0.05)
            for name in (
                "config.json",
                "metrics.json",
                "model_config.json",
                "neighbor_table.parquet",
                "recommendations.parquet",
            ):
                self.assertTrue((output / name).is_file())
            self.assertFalse((output / "selection_neighbor_cache").exists())
            with self.assertRaises(FileExistsError):
                run_experiment(
                    config_path=config_path,
                    output_dir=output,
                    run_id="item2item_test",
                )


if __name__ == "__main__":
    unittest.main()

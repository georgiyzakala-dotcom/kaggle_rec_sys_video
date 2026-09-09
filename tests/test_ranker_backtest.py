from __future__ import annotations

import io
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
from catboost import Pool

from experiment_utils import (
    EventProgressReporter,
    config_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ranker_backtest import (
    FOLD_ORDER,
    allocate_training_probabilities,
    load_backtest_config,
    metrics_from_hits,
    quantize_training_parts,
    sample_training_rows,
    select_evaluation_users,
    select_tree_checkpoint,
    validate_temporal_training_scope,
)
from rankers import (
    CatBoostPointwiseConfig,
    CatBoostPointwiseModel,
    CatBoostRankerDataLoader,
    validate_feature_columns,
)
from scripts.run_ranker_backtest import BacktestRun, run_backtest, verify_artifact
from validation import ContractValidationError

FEATURES = ("feature_a", "feature_b")


def example_rows() -> pl.DataFrame:
    users = [2**63 + 17, 2**63 + 19, 2**63 + 23]
    rows = [
        (
            user,
            item,
            int(item < 3 and user != users[-1]),
            float(25 - item),
            float(item % 7),
        )
        for user in users
        for item in range(25)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.UInt64,
            "item_id": pl.Int32,
            "label": pl.UInt8,
            "feature_a": pl.Float32,
            "feature_b": pl.Float32,
        },
        orient="row",
    ).with_columns(
        pl.lit(True).alias("is_training_sample"),
        pl.when(pl.col("label") == 1)
        .then(1.0)
        .otherwise(0.25)
        .cast(pl.Float32)
        .alias("sampling_probability"),
    )


def fit_example(root: Path, frame: pl.DataFrame) -> CatBoostPointwiseModel:
    model = CatBoostPointwiseModel(
        CatBoostPointwiseConfig(
            task_type="CPU", iterations=12, depth=3, thread_count=2, metric_period=4
        ),
        feature_columns=FEATURES,
    )
    pool = Pool(
        frame.select(FEATURES).to_numpy(),
        label=frame["label"].to_numpy(),
        feature_names=list(FEATURES),
    )
    loader = (
        CatBoostRankerDataLoader(feature_columns=FEATURES)
        .load_fit_data(train_pool=pool)
        .prepare_fit_data()
    )
    model.fit(
        loader,
        fixed_tree_budget=True,
        train_dir=root / "train",
        log_cout=io.StringIO(),
        log_cerr=io.StringIO(),
    )
    return model


def synthetic_inputs(root: Path) -> Path:
    """Tiny immutable folds with a real portable baseline and known unseen items."""
    frame = example_rows()
    dataset = root / "inputs/task07"
    dataset.mkdir(parents=True)
    history = root / "inputs/history.parquet"
    pl.DataFrame(
        {
            "user_id": pl.Series([1] * 25, dtype=pl.UInt64),
            "item_id": pl.Series(range(25), dtype=pl.Int32),
        }
    ).write_parquet(history)
    schema = {"feature_columns": list(FEATURES), "feature_count": 2}
    write_json_atomic(dataset / "feature_schema.json", schema)
    manifests = {}
    for index, fold in enumerate(FOLD_ORDER):
        folder = dataset / "folds" / fold
        parts = folder / "ranker_data"
        parts.mkdir(parents=True)
        part = parts / "part-00000.parquet"
        frame.write_parquet(part)
        users = folder / "target_users.parquet"
        truth = folder / "target_ground_truth.parquet"
        frame.select("user_id").unique().sort("user_id").write_parquet(users)
        frame.filter(pl.col("label") == 1).select("user_id", "item_id").write_parquet(
            truth
        )
        manifest = {
            "fold": fold,
            "part_count": 1,
            "parts": {part.name: sha256_file(part)},
            "target_users_sha256": sha256_file(users),
            "target_ground_truth_sha256": sha256_file(truth),
            "history_path": str(history),
            "cutoff": (date(2024, 11, 29) + timedelta(days=index)).isoformat(),
        }
        path = folder / "fold_manifest.json"
        write_json_atomic(path, manifest)
        manifests[fold] = {
            "manifest": path.relative_to(dataset).as_posix(),
            "manifest_sha256": sha256_file(path),
        }
    write_json_atomic(
        dataset / "dataset_manifest.json",
        {
            "kind": "task07_ranker_datasets",
            "mode": "full",
            "schema_sha256": config_sha256(schema),
            "folds": manifests,
        },
    )
    baseline = root / "inputs/task08/model"
    fit_example(root, frame).save(baseline)
    borders = root / "inputs/borders.tsv"
    borders.write_text("0\t10.5\n0\t20.5\n")
    config = read_json("configs/task12_ranker_backtest_smoke_v1.json")
    config.update(
        run_id="synthetic_backtest",
        paths={
            "task07_dataset": str(dataset),
            "task08_model": str(baseline),
            "task08_borders": str(borders),
        },
    )
    config["smoke"]["user_count"] = config["selection"]["user_count"] = 3
    config["training"]["target_rows"] = 100
    path = root / "config.json"
    write_json_atomic(path, config)
    return path


class RankerBacktestTests(unittest.TestCase):
    def test_configs_and_temporal_labels_cannot_overlap(self):
        for name in ("v1", "smoke_v1"):
            load_backtest_config(f"configs/task12_ranker_backtest_{name}.json")
        cutoffs = {"a": "2024-12-01T08:00:00", "b": "2024-12-02T08:00:00"}
        validate_temporal_training_scope(["a"], "b", cutoffs)
        for train, evaluation in [(["b"], "a"), (["a"], "a"), (["a", "a"], "b")]:
            with self.assertRaises(ValueError):
                validate_temporal_training_scope(train, evaluation, cutoffs)
        cutoffs["b"] = "2024-12-02T07:59:59"
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_temporal_training_scope(["a"], "b", cutoffs)

    def test_sampling_keeps_positives_and_multiplies_probabilities(self):
        source = example_rows()
        first = sample_training_rows(
            source,
            features=FEATURES,
            fold="rolling_1",
            seed=42,
            negative_probability=0.5,
        )
        second = sample_training_rows(
            source.reverse(),
            features=FEATURES,
            fold="rolling_1",
            seed=42,
            negative_probability=0.5,
        )
        self.assertTrue(first.equals(second))
        self.assertEqual(first.schema["user_id"], pl.UInt64)
        self.assertEqual(first.filter(pl.col("label") == 1).height, 6)
        self.assertEqual(
            first.filter(pl.col("label") == 1)["sample_weight"].unique().to_list(),
            [1.0],
        )
        self.assertEqual(
            first.filter(pl.col("label") == 0)["sample_weight"].unique().to_list(),
            [8.0],
        )
        for bad in (
            source.with_columns(pl.lit(False).alias("is_training_sample")),
            source.with_columns(pl.lit(0.0).alias("sampling_probability")),
        ):
            with self.assertRaises(ContractValidationError):
                sample_training_rows(
                    bad,
                    features=FEATURES,
                    fold="rolling_1",
                    seed=42,
                    negative_probability=0.5,
                )
        for name in (
            "fold_id",
            "sampling_stratum",
            "full_sampling_probability",
            "label",
            "sample_weight",
        ):
            with self.assertRaises(ValueError):
                validate_feature_columns(("feature_a", name))

    def test_balanced_expected_fold_budget_and_positive_floor(self):
        counts = {
            "a": {"rows": 100, "positives": 10},
            "b": {"rows": 1000, "positives": 20},
        }
        plan = allocate_training_probabilities(counts, target_rows=101)
        for fold, expected in [("a", 51), ("b", 50)]:
            c = counts[fold]
            self.assertAlmostEqual(
                c["positives"]
                + (c["rows"] - c["positives"])
                * plan[fold]["secondary_negative_probability"],
                expected,
            )
        with self.assertRaises(ValueError):
            allocate_training_probabilities(counts, target_rows=20)

    def test_complete_user_sampling_is_independent_of_order_and_labels(self):
        users = example_rows().select("user_id").unique()
        self.assertTrue(
            select_evaluation_users(users, count=2, seed=42).equals(
                select_evaluation_users(users.reverse(), count=2, seed=42)
            )
        )
        self.assertTrue(
            select_evaluation_users(users, count=3, seed=42).equals(
                users.sort("user_id")
            )
        )
        with self.assertRaises(ContractValidationError):
            select_evaluation_users(
                users.cast({"user_id": pl.Float64}), count=2, seed=42
            )

    def test_p20_selection_uses_hits_not_logloss_and_both_denominators(self):
        curve = [
            {
                "tree_count": n,
                "full_candidate_logloss": loss,
                **metrics_from_hits(hits=hits, targets=3, labeled=2),
            }
            for n, loss, hits in [(4, 0.01, 3), (8, 0.1, 5), (12, 0.2, 5)]
        ]
        best = select_tree_checkpoint(curve)
        self.assertEqual(best["tree_count"], 8)
        self.assertEqual(best["precision_at_20_all_targets"], 5 / 60)
        self.assertEqual(best["precision_at_20_labeled_users"], 5 / 40)
        self.assertEqual(
            metrics_from_hits(hits=0, targets=3, labeled=0)[
                "precision_at_20_labeled_users"
            ],
            0,
        )
        with self.assertRaises(ValueError):
            select_tree_checkpoint([curve[0], {**curve[1], "target_users": 4}])

    def test_frozen_borderless_feature_stays_ignored_and_fresh_learns_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = example_rows().with_columns(
                pl.lit(1.0).cast(pl.Float32).alias("sample_weight")
            )
            part = root / "part.parquet"
            source.write_parquet(part)
            borders = root / "frozen.tsv"
            borders.write_text("0\t10.5\n0\t20.5\n")
            common = {
                "parts": [part],
                "features": FEATURES,
                "frozen_borders": borders,
                "border_count": 4,
                "feature_border_type": "GreedyLogSum",
                "seed": 42,
                "thread_count": 2,
                "maximum_dsv_bytes": 100000,
            }
            frozen = quantize_training_parts(
                **common, policy="frozen_task08", destination=root / "frozen"
            )
            fresh = quantize_training_parts(
                **common, policy="fit_training", destination=root / "fresh"
            )
            repeated = quantize_training_parts(
                **common, policy="fit_training", destination=root / "repeat"
            )
            self.assertEqual(frozen["ignored_borderless_features"], ["feature_b"])
            self.assertEqual(frozen["features"][0]["borders"], [10.5, 20.5])
            self.assertGreater(fresh["features"][1]["border_count"], 0)
            self.assertEqual(
                fresh["actual_borders_sha256"], repeated["actual_borders_sha256"]
            )
            self.assertTrue(all(row["border_count"] <= 4 for row in fresh["features"]))
            self.assertFalse(list(root.glob("*/transport.tsv")))
            with self.assertRaisesRegex(ValueError, "size limit"):
                quantize_training_parts(
                    **{**common, "maximum_dsv_bytes": 1},
                    policy="fit_training",
                    destination=root / "failed",
                )
            self.assertFalse((root / "failed/transport.tsv").exists())

    def test_staged_prefix_scores_match_saved_prefix_without_mutating_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = example_rows()
            model = fit_example(root, frame)
            loader = (
                CatBoostRankerDataLoader(feature_columns=FEATURES)
                .load_predict_data(frame=frame)
                .prepare_predict_data()
            )
            scores = model.predict_checkpoints(
                loader, tree_counts=[3, 8, 12], batch_size=17
            )
            for count, predicted in scores.items():
                prefix = model.with_tree_count(count)
                prefix.save(root / f"model_{count}")
                restored = CatBoostPointwiseModel.from_artifact(root / f"model_{count}")
                direct = restored.predict(loader, batch_size=13)
                self.assertTrue(
                    direct.select("user_id", "item_id").equals(
                        predicted.select("user_id", "item_id")
                    )
                )
                np.testing.assert_allclose(
                    direct["ranker_score"].to_numpy(),
                    predicted["ranker_score"].to_numpy(),
                    rtol=0,
                    atol=1e-15,
                )
                self.assertTrue(model.predict(loader, tree_count=count).equals(direct))
            self.assertEqual(model.tree_count, 12)
            with self.assertRaises(ValueError):
                model.predict_checkpoints(loader, tree_counts=[13])

    def test_atomic_operation_adopts_publication_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "run_id": "operation_test",
                "resources": {"minimum_free_disk_gib": 0.01},
            }
            reporter = EventProgressReporter(
                task_name="test",
                total_phases=1,
                log_file=root / "test.log",
                show_progress=False,
            )
            try:
                run = BacktestRun(
                    config,
                    output=root / "output",
                    work=root / "work",
                    reporter=reporter,
                )
                destination = run.work / "operation"

                def build(path):
                    (path / "result.txt").write_text("complete")
                    return {"done": True}

                with (
                    patch.object(
                        run.checkpoint, "complete", side_effect=KeyboardInterrupt
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    run.operation("stage", "config", "fold", destination, build)
                self.assertTrue(destination.is_dir())
                self.assertIsNone(
                    run.checkpoint.get(stage="stage", config="config", fold="fold")
                )
                with patch(
                    "shutil.disk_usage",
                    side_effect=AssertionError("must reuse completed work"),
                ):
                    self.assertEqual(
                        run.operation("stage", "config", "fold", destination, build),
                        {"done": True},
                    )
                (destination / "result.txt").write_text("corrupt")
                with self.assertRaisesRegex(ContractValidationError, "checksum"):
                    run.operation("stage", "config", "fold", destination, build)
            finally:
                reporter.close()

    def test_end_to_end_pause_resume_portability_and_no_canonical_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = synthetic_inputs(root)
            output, work = root / "output", root / "work"
            args = {
                "output": output,
                "work": work,
                "log_file": root / "run.log",
                "show_progress": False,
            }
            original_load = BacktestRun.load_folds

            def forbid_canonical(run, names):
                self.assertNotIn("canonical", names)
                return original_load(run, names)

            with patch.object(BacktestRun, "load_folds", forbid_canonical):
                paused = run_backtest(config_path, stop_after_selection=True, **args)
            self.assertEqual(paused["status"], "paused_after_selection")
            self.assertFalse(output.exists())
            selected = (work / "winner.json").read_bytes()
            model_hash = sha256_file(
                work
                / "selected_models"
                / paused["winner"]["policy"]
                / "model/model.cbm"
            )
            measured = run_backtest(config_path, **args)
            self.assertEqual((output / "winner.json").read_bytes(), selected)
            self.assertEqual(
                sha256_file(output / "selection_model/model.cbm"), model_hash
            )
            self.assertFalse(measured["canonical_used_for_selection"])
            self.assertEqual(measured["target_users"], 3)
            self.assertEqual(measured["labeled_users"], 2)
            self.assertTrue(verify_artifact(output)["portable_repeat_equal"])
            self.assertFalse((work / "pools").exists())
            self.assertIn("operation_resume_skip", (root / "run.log").read_text())
            self.assertNotIn("\x1b", (root / "run.log").read_text())
            with self.assertRaises(FileExistsError):
                run_backtest(config_path, **args)


if __name__ == "__main__":
    unittest.main()

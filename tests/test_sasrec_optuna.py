from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import polars as pl
import torch
from test_sasrec import synthetic_fold

from experiment_utils import read_json, sha256_file, write_json_atomic
from interfaces import CANDIDATE_SCHEMA
from sasrec_data import SequenceStore
from sasrec_selection import SOURCES, SelectionEvaluator, id_sample, overlap
from scripts.run_sasrec_benchmark import BenchmarkPaused
from scripts.run_sasrec_optuna import (
    SearchRunner,
    load_config,
    resolve_config,
    resume_config,
    suggest,
    tell_result,
    trial_handle,
    verify_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def candidates(source, mapping):
    return pl.DataFrame(
        [
            (u, i, -float(rank), rank, source)
            for u, items in mapping.items()
            for rank, i in enumerate(items, 1)
        ],
        schema=CANDIDATE_SCHEMA,
        orient="row",
    )


def fixture(root):
    fold, cutoff = synthetic_fold(root)
    data = root / "fixed_sources"
    data.mkdir()
    write_json_atomic(
        data / "dataset_manifest.json",
        {
            "fold": "rolling_1",
            "cutoff": cutoff.isoformat(),
            "history_path": str(fold / "history_daily.parquet"),
        },
    )
    history = pl.read_parquet(fold / "history_daily.parquet")
    catalog = sorted(history["item_id"].unique())
    users = history["user_id"].unique().sort().to_list()
    seen = {u: set(history.filter(pl.col("user_id") == u)["item_id"]) for u in users}
    for source in SOURCES:
        directory = data / "sources" / source
        directory.mkdir(parents=True)
        frame = candidates(
            source, {u: [i for i in catalog if i not in seen[u]] for u in users}
        )
        frame.write_parquet(directory / "candidates.parquet")
        write_json_atomic(
            directory / "metadata.json",
            {
                "source": source,
                "fold": "rolling_1",
                "cutoff": cutoff.isoformat(),
                "candidate_k": 200,
                "candidates_sha256": sha256_file(directory / "candidates.parquet"),
            },
        )
    config = load_config(ROOT / "configs/task15_sasrec_optuna_cpu_smoke_v1.json")
    config["data"].update(fold_artifact=str(fold), source_fold=str(data))
    config["search"].update(startup_trials=1)
    config["training"].update(max_users=8)
    return config


class SelectionMetricTests(unittest.TestCase):
    def setUp(self):
        self.users = pl.DataFrame(
            {"user_id": [2**63 + 17, 2**63 + 18, 2**64 - 7]},
            schema={"user_id": pl.UInt64},
        )
        ids = self.users["user_id"].to_list()
        self.store = SequenceStore(
            self.users["user_id"].to_numpy(),
            np.arange(1, 1001, dtype=np.int32),
            np.array([1, 2], dtype=np.uint32),
            np.array([0, 2, 2, 2]),
            np.array([1], dtype=np.uint32),
            np.array([0, 1, 1, 1]),
            {"history_sha256": "synthetic"},
        )
        self.truth = pl.DataFrame(
            {"user_id": [ids[0]] * 3 + [ids[1]], "item_id": [600, 2, 100, 100]},
            schema={"user_id": pl.UInt64, "item_id": pl.Int32},
        )
        self.sources = {
            "global_popularity": candidates(
                "global_popularity", {u: range(100, 300) for u in ids}
            ),
            "recency_popularity": candidates(
                "recency_popularity", {u: range(150, 350) for u in ids}
            ),
            "item2item": candidates("item2item", {u: range(350, 550) for u in ids}),
            "implicit_als": candidates("implicit_als", {ids[0]: range(2, 202)}),
        }
        self.native = candidates("sasrec", {ids[0]: range(600, 800)})

    def test_equal_budget_objective_and_empty_history_denominators(self):
        evaluator = SelectionEvaluator(self.users, self.truth, self.sources, self.store)
        result = evaluator.evaluate(self.native)
        self.assertEqual(result["objective"], 0.25)
        self.assertEqual(result["selected_policy"], "blend_150_50")
        self.assertEqual(result["policies"]["replace_als"]["delta_recall"], 0.0)
        self.assertEqual(result["native_sasrec"]["candidate_recall"], 0.25)
        self.assertEqual(result["native_sasrec"]["coverage"], 1 / 3)
        self.assertAlmostEqual(
            result["standalone_with_global_fallback"]["precision_at_20_all_targets"],
            2 / 60,
        )
        self.assertAlmostEqual(
            result["standalone_with_global_fallback"]["precision_at_20_labeled_users"],
            2 / 40,
        )
        self.assertFalse(result["add_source_1000"]["eligible_for_selection"])
        self.assertEqual(result["add_source_1000"]["new_positive_hits"], 1)
        pair = result["overlap_sasrec_vs_sources"]["implicit_als"]
        self.assertEqual(
            (
                pair["shared_positive_hits"],
                pair["left_only_positive_hits"],
                pair["right_only_positive_hits"],
            ),
            (0, 1, 2),
        )
        self.assertEqual(pair["per_user"]["share_left_in_right"]["undefined_users"], 2)

    def test_seen_unknown_and_extra_users_are_rejected(self):
        evaluator = SelectionEvaluator(self.users, self.truth, self.sources, self.store)
        user = self.users["user_id"][0]
        for frame in (
            candidates("sasrec", {user: [1]}),
            candidates("sasrec", {user: [1001]}),
            candidates("sasrec", {5: [600]}),
        ):
            with self.assertRaises(ValueError):
                evaluator.evaluate(frame)

    def test_id_sample_and_overlap_dedup(self):
        a = id_sample(self.users, 2, 42)
        b = id_sample(self.users.reverse(), 2, 42)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.schema["user_id"], pl.UInt64)
        result = overlap(
            pl.concat([self.native, self.native]), self.native, self.truth, self.users
        )
        self.assertEqual(result["intersection_pairs"], 200)
        self.assertEqual(result["shared_positive_hits"], 1)
        self.assertEqual(result["micro_jaccard"], 1.0)


class OptunaRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def test_tell_complete_pruned_failed_and_result_identity(self):
        study = optuna.create_study(direction="maximize")
        for name in ("COMPLETE", "PRUNED", "FAIL"):
            with self.subTest(state=name):
                trial = study.ask()
                trial.report(0.25, step=4)
                result = {"trial_number": trial.number, "state": name, "value": 0.25}
                frozen = tell_result(study, trial, result)
                self.assertEqual(frozen.state.name, name)
                self.assertEqual(frozen.value, None if name == "FAIL" else 0.25)
        trial = study.ask()
        with self.assertRaises(ValueError):
            tell_result(
                study, trial, {"trial_number": 999, "state": "COMPLETE", "value": 1.0}
            )

    def test_pruned_result_recovery_does_not_train_the_completed_trial(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            config["run_id"] = "optuna_test_pruned_recovery"
            try:
                os.chdir(directory)
                c = resolve_config(config)
                with patch(
                    "scripts.run_sasrec_optuna.optuna.pruners.MedianPruner",
                    return_value=optuna.pruners.ThresholdPruner(lower=1.0),
                ):
                    with (
                        patch(
                            "scripts.run_sasrec_optuna.tell_result",
                            side_effect=RuntimeError("simulated crash before tell"),
                        ),
                        self.assertRaisesRegex(RuntimeError, "simulated crash"),
                    ):
                        SearchRunner(c, show_progress=False).run()
                    work = Path("artifacts/.optuna_test_pruned_recovery.work")
                    record = read_json(work / "trials/trial_0000/result.json")
                    self.assertEqual(record["state"], "PRUNED")
                    model = (
                        work
                        / "trials/trial_0000"
                        / record["best"]["path"]
                        / "model/weights.pt"
                    )
                    digest = sha256_file(model)
                    with patch(
                        "scripts.run_sasrec_optuna.SASRecCandidateModel.fit",
                        side_effect=AssertionError("completed trial must not train"),
                    ) as fit:
                        with self.assertRaises(BenchmarkPaused):
                            SearchRunner(
                                c, show_progress=False, stop_after_trial=1
                            ).run()
                        fit.assert_not_called()
                    self.assertEqual(sha256_file(model), digest)
                    study = optuna.load_study(
                        study_name=c["run_id"],
                        storage="sqlite:///" + str((work / "study.sqlite3").resolve()),
                    )
                    self.assertEqual(study.trials[0].state.name, "PRUNED")
                    self.assertEqual(study.trials[0].value, record["value"])
                    output = SearchRunner(c, show_progress=False).run()
                    self.assertTrue(verify_artifact(output)["verified"])
                    self.assertEqual(
                        [
                            t["state"]
                            for t in read_json(output / "trials.json")["trials"]
                        ],
                        ["PRUNED", "PRUNED"],
                    )
            finally:
                os.chdir(previous)

    def test_reviewed_runner_patch_preserves_legacy_checkpoints(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            config["run_id"] = "optuna_test_legacy_resume"
            current = resolve_config(config)
            legacy = copy.deepcopy(current)
            key = "scripts/run_sasrec_optuna.py"
            legacy["implementation_sha256"][key] = "reviewed-old-runner"
            compatibility = root / "compatibility.json"
            write_json_atomic(
                compatibility,
                {
                    "transitions": [
                        {
                            "from_sha256": "reviewed-old-runner",
                            "to_sha256": current["implementation_sha256"][key],
                            "reason": "synthetic API-only fix",
                        }
                    ]
                },
            )
            try:
                os.chdir(root)
                with self.assertRaises(BenchmarkPaused):
                    SearchRunner(
                        legacy, show_progress=False, stop_after_epoch=(0, 1)
                    ).run()
                work = Path("artifacts/.optuna_test_legacy_resume.work")
                config_hash = sha256_file(work / "config.json")
                for name in ("seed", "implementation", "unreviewed_runner"):
                    changed = copy.deepcopy(current)
                    if name == "seed":
                        changed["seed"] = 43
                    elif name == "implementation":
                        changed["implementation_sha256"]["sasrec_model.py"] = (
                            "modified-training"
                        )
                    else:
                        changed["implementation_sha256"][key] = "unreviewed-code"
                    with self.assertRaises(ValueError):
                        resume_config(changed, work, compatibility_path=compatibility)
                with patch(
                    "scripts.run_sasrec_optuna.resume_config",
                    side_effect=lambda c, w: resume_config(
                        c, w, compatibility_path=compatibility
                    ),
                ):
                    output = SearchRunner(current, show_progress=False).run()
                self.assertEqual(sha256_file(work / "config.json"), config_hash)
                self.assertEqual(read_json(output / "config.json"), legacy)
                self.assertEqual(
                    read_json(output / "runtime_patch.json")["to_sha256"],
                    current["implementation_sha256"][key],
                )
                self.assertTrue(verify_artifact(output)["verified"])
            finally:
                os.chdir(previous)

    def test_config_rejects_full_cpu_and_unbounded_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "configs/task15_sasrec_optuna_v1.json")
            config["device"] = "cpu"
            p = Path(directory) / "config.json"
            p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(p)
            config = load_config(
                ROOT / "configs/task15_sasrec_optuna_cpu_smoke_v1.json"
            )
            config["search"]["trials"] = 24
            p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(p)

    def test_sampler_partial_suggestion_resume_and_median_pruning(self):
        config = load_config(ROOT / "configs/task15_sasrec_optuna_v1.json")
        uninterrupted = optuna.create_study(direction="maximize")
        expected = suggest(uninterrupted, uninterrupted.ask(), config)
        resumed = optuna.create_study(direction="maximize")
        trial = resumed.ask()
        resumed.sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=6)
        trial.suggest_categorical(
            "embedding_dim", config["space"]["embedding_dim"]["choices"]
        )
        actual = suggest(resumed, trial_handle(resumed, resumed.trials[0]), config)
        self.assertEqual(expected, actual)
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=1, n_warmup_steps=2, n_min_trials=1
            ),
        )
        first = study.ask()
        first.report(0.2, 2)
        study.tell(first, 0.2)
        second = study.ask()
        second.report(0.01, 1)
        self.assertFalse(second.should_prune())
        second.report(0.01, 2)
        self.assertTrue(second.should_prune())

    def test_end_to_end_epoch_and_trial_resume_match_independent_search(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            config["run_id"] = "optuna_test_resume"
            config["search"]["patience_evaluations"] = 1
            try:
                os.chdir(root)
                c = resolve_config(config)
                with self.assertRaises(BenchmarkPaused):
                    SearchRunner(c, show_progress=False, stop_after_epoch=(0, 1)).run()
                checkpoint = (
                    root
                    / "artifacts/.optuna_test_resume.work/trials/trial_0000/checkpoint.json"
                )
                self.assertEqual(read_json(checkpoint)["epoch"], 1)
                changed = copy.deepcopy(c)
                changed["seed"] = 43
                with self.assertRaises(ValueError):
                    SearchRunner(changed, show_progress=False)
                with self.assertRaises(BenchmarkPaused):
                    SearchRunner(c, show_progress=False, stop_after_trial=1).run()
                output = SearchRunner(c, show_progress=False).run()
                self.assertTrue(verify_artifact(output)["verified"])
                with self.assertRaises(FileExistsError):
                    SearchRunner(c, show_progress=False)
                config["run_id"] = "optuna_test_repeat"
                repeat = SearchRunner(resolve_config(config), show_progress=False).run()
                left = read_json(output / "trials.json")["trials"]
                right = read_json(repeat / "trials.json")["trials"]
                self.assertTrue(all(t["stop_reason"] == "early_stopping" for t in left))
                self.assertEqual(
                    [(t["params"], t["state"], t.get("value")) for t in left],
                    [(t["params"], t["state"], t.get("value")) for t in right],
                )
                for n in range(2):
                    left_path = (
                        Path(f"artifacts/.optuna_test_resume.work/trials/trial_{n:04d}")
                        / left[n]["best"]["path"]
                        / "model/weights.pt"
                    )
                    right_path = (
                        Path(f"artifacts/.optuna_test_repeat.work/trials/trial_{n:04d}")
                        / right[n]["best"]["path"]
                        / "model/weights.pt"
                    )
                    a, b = (
                        torch.load(left_path, weights_only=True),
                        torch.load(right_path, weights_only=True),
                    )
                    self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
                self.assertTrue(
                    pl.read_parquet(output / "selection_candidates.parquet").equals(
                        pl.read_parquet(repeat / "selection_candidates.parquet")
                    )
                )
                self.assertNotIn(
                    "\x1b", Path("logs/optuna_test_resume.log").read_text()
                )
                self.assertFalse(Path("experiments/results.csv").exists())
            finally:
                os.chdir(previous)

    def test_fold_mismatch_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            path = (
                Path(config["data"]["source_fold"])
                / "sources/implicit_als/metadata.json"
            )
            value = read_json(path)
            value["fold"] = "canonical"
            write_json_atomic(path, value)
            with self.assertRaises(ValueError):
                resolve_config(config)


if __name__ == "__main__":
    unittest.main()

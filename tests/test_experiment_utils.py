from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment_utils import (
    AtomicBestConfig,
    CheckpointStore,
    EventProgressReporter,
    config_sha256,
    publish_directory_atomic,
)


class ExperimentUtilityTests(unittest.TestCase):
    def test_event_log_is_structured_and_has_no_ansi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.log"
            reporter = EventProgressReporter(
                task_name="test",
                total_phases=1,
                log_file=log_path,
                show_progress=False,
            )
            started = reporter.phase_start("phase", stage="stage_a")
            operation = reporter.operation_start(
                stage="stage_a",
                config="config_a",
                fold="fold_a",
                operation="fit",
            )
            reporter.operation_finish(
                stage="stage_a",
                config="config_a",
                fold="fold_a",
                operation="fit",
                started=operation,
                metric=0.25,
            )
            reporter.phase_finish("phase", started)
            reporter.close()
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("event=operation_finish", text)
            self.assertIn('stage="stage_a"', text)
            self.assertIn('config="config_a"', text)
            self.assertIn('fold="fold_a"', text)
            self.assertNotIn("\x1b[", text)
            for line in text.splitlines():
                self.assertIn("stage=", line)
                self.assertIn("config=", line)
                self.assertIn("fold=", line)
                self.assertIn("operation=", line)

    def test_event_log_rotates_at_the_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.log"
            reporter = EventProgressReporter(
                task_name="rotation_test",
                total_phases=1,
                log_file=log_path,
                show_progress=False,
                log_max_bytes=256,
                log_backup_count=2,
            )
            for index in range(20):
                reporter.event(
                    "bounded_event",
                    stage="selection",
                    config=f"config_{index}",
                    fold="rolling",
                    operation="fit",
                )
            reporter.close()
            self.assertTrue(log_path.is_file())
            self.assertTrue((log_path.parent / "events.log.1").is_file())
            self.assertLessEqual(
                len(list(log_path.parent.glob("events.log*"))),
                3,
            )

    def test_checkpoint_is_idempotent_and_rejects_incompatible_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            digest = config_sha256({"a": 1})
            store = CheckpointStore(root, run_id="run", config_digest=digest)
            store.complete(
                stage="stage",
                config="config",
                fold="fold",
                metadata={"sha256": "abc"},
            )
            resumed = CheckpointStore(root, run_id="run", config_digest=digest)
            self.assertEqual(
                resumed.get(stage="stage", config="config", fold="fold"),
                {"sha256": "abc"},
            )
            resumed.complete(
                stage="stage",
                config="config",
                fold="fold",
                metadata={"sha256": "abc"},
            )
            with self.assertRaisesRegex(ValueError, "already differs"):
                resumed.complete(
                    stage="stage",
                    config="config",
                    fold="fold",
                    metadata={"sha256": "different"},
                )
            with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
                CheckpointStore(root, run_id="run", config_digest="other")

    def test_best_config_callback_updates_only_on_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            best = AtomicBestConfig(directory, run_id="run", config_digest="digest")
            self.assertTrue(best.update(score=(0.1, 0.2), payload={"config": "a"}))
            self.assertFalse(best.update(score=(0.1, 0.1), payload={"config": "b"}))
            self.assertTrue(best.update(score=(0.2, 0.0), payload={"config": "c"}))
            self.assertEqual(best.read()["payload"], {"config": "c"})

    def test_atomic_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / "value.txt").write_text("complete", encoding="utf-8")
            output = root / "output"
            publish_directory_atomic(staging, output)
            self.assertEqual(
                (output / "value.txt").read_text(encoding="utf-8"), "complete"
            )
            second = root / "second"
            second.mkdir()
            with self.assertRaises(FileExistsError):
                publish_directory_atomic(second, output)


if __name__ == "__main__":
    unittest.main()

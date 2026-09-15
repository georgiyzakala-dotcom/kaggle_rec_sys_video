"""Deletion is confined to verified, idle work state; published hard links survive."""

import fcntl
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiment_utils import config_sha256, read_json, sha256_file, write_json_atomic
from scripts.cleanup_expanded_ranker import KIND, cleanup, inventory


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.run_id = "cleanup_test"
        self.output = self.repo / "artifacts" / self.run_id
        self.work = self.repo / "artifacts" / f".{self.run_id}.work"
        self.trash = self.repo / "artifacts" / f".{self.run_id}.cleanup-trash"
        self.audit = self.repo / "artifacts" / f"{self.run_id}_cleanup_v1"
        self.output.mkdir(parents=True)
        self.work.mkdir()
        config = {"kind": KIND, "run_id": self.run_id, "seed": 42}
        for root in (self.output, self.work):
            write_json_atomic(root / "config.json", config)
        write_json_atomic(
            self.work / "checkpoint.json",
            {"run_id": self.run_id, "config_sha256": config_sha256(config)},
        )
        (self.work / "full_sequences").mkdir()
        self.private = self.work / "full_sequences" / "items.bin"
        self.private.write_bytes(b"sequence cache" * 1000)
        (self.output / "model.bin").write_bytes(b"kept model" * 2000)
        (self.output / "submission.csv").write_text("user_id,item_ids\n")
        (self.work / "publication").mkdir()
        os.link(self.output / "model.bin", self.work / "publication/model.bin")
        os.link(self.output / "model.bin", self.work / "publication/model2.bin")
        self.protected = {p.name: sha256_file(p) for p in self.output.iterdir()}
        write_json_atomic(self.output / "manifest.json", {"files": self.protected})
        (self.repo / "data").mkdir()
        (self.repo / "data/raw.parquet").write_bytes(b"raw immutable sentinel")

    def verify(self, output):
        for name, digest in self.protected.items():
            self.assertEqual(sha256_file(output / name), digest)
        return {"verified": True}

    def call(self, **kw):
        verifier = kw.pop("verifier", self.verify)
        with redirect_stdout(io.StringIO()):
            return cleanup(
                self.repo, self.run_id, verifier=verifier, show_progress=False, **kw
            )

    def test_preview_does_not_delete_or_create_audit(self):
        before = inventory(self.work)
        result = self.call()
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(inventory(self.work), before)
        self.assertFalse(self.audit.exists())

    def test_only_private_blocks_are_counted_and_shared_files_survive(self):
        plan = inventory(self.work)
        self.assertEqual(
            plan["externally_linked_bytes"], (self.output / "model.bin").stat().st_size
        )
        expected = sum(
            p.stat().st_blocks * 512
            for p in [
                self.work / "config.json",
                self.work / "checkpoint.json",
                self.private,
            ]
        )
        self.assertEqual(plan["estimated_reclaimable_bytes"], expected)
        result = self.call(apply=True)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(self.work.exists())
        self.assertFalse(self.trash.exists())
        self.verify(self.output)
        self.assertEqual(
            (self.repo / "data/raw.parquet").read_bytes(), b"raw immutable sentinel"
        )
        self.assertEqual(read_json(self.audit / "metrics.json")["status"], "completed")
        self.assertEqual(self.call(apply=True)["status"], "already_clean")

    def test_interrupted_unlink_resumes_from_receipt(self):
        original_unlink = Path.unlink

        def interrupted(path, *args, **kw):
            if path.name == "items.bin":
                raise KeyboardInterrupt(
                    "simulated interruption after ownership files were removed"
                )
            return original_unlink(path, *args, **kw)

        with (
            patch.object(Path, "unlink", interrupted),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.call(apply=True)
        self.assertTrue(self.trash.exists())
        self.assertFalse((self.trash / "config.json").exists())
        self.assertEqual(
            read_json(self.audit / "metrics.json")["status"], "interrupted_or_failed"
        )
        self.assertEqual(self.call(apply=True)["status"], "completed")
        self.verify(self.output)

    def test_active_runner_lock_blocks_cleanup(self):
        logs = self.repo / "logs"
        logs.mkdir()
        with (logs / "expanded_ranker.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "active"):
                self.call(apply=True)
        self.assertTrue(self.private.exists())

    def test_failed_publication_verification_preserves_everything(self):
        def failed(path):
            raise ValueError("broken model checksum")

        with self.assertRaisesRegex(ValueError, "checksum"):
            self.call(apply=True, verifier=failed)
        self.assertTrue(self.private.exists())
        self.assertFalse(self.audit.exists())

    def test_invalid_run_id_and_incomplete_output_are_refused(self):
        with self.assertRaisesRegex(ValueError, "run ID"):
            cleanup(self.repo, "../data", apply=True, verifier=self.verify)
        (self.output / "config.json").unlink()
        with self.assertRaises(ValueError):
            self.call(apply=True)
        self.assertTrue(self.private.exists())

    def test_mismatched_work_config_is_refused(self):
        c = read_json(self.work / "config.json")
        c["seed"] = 123
        write_json_atomic(self.work / "config.json", c)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.call(apply=True)
        self.assertTrue(self.private.exists())

    def test_unexpected_root_file_is_preserved(self):
        (self.work / "notes.txt").write_text("manual notes")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.call(apply=True)
        self.assertTrue(self.private.exists())

    def test_symlink_to_raw_data_is_refused(self):
        (self.work / "full_sequences" / "escape").symlink_to(
            self.repo / "data", target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.call(apply=True)
        self.assertTrue((self.repo / "data/raw.parquet").exists())
        self.assertTrue(self.private.exists())

    def test_work_directory_symlink_is_refused(self):
        moved = self.work.with_name("protected_work")
        self.work.rename(moved)
        self.work.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.call(apply=True)
        self.assertTrue(moved.exists())

    def test_trash_without_receipt_is_refused(self):
        self.work.rename(self.trash)
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.call(apply=True)
        self.assertTrue(self.trash.exists())


if __name__ == "__main__":
    unittest.main()

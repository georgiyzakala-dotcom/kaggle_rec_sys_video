#!/usr/bin/env python3
"""Materialize the shared canonical temporal-fold data artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_utils import prepare_temporal_fold


def _datetime_argument(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an ISO-8601 datetime, got {value!r}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split raw events by timestamp, aggregate each side independently, "
            "and materialize an immutable temporal-fold artifact."
        )
    )
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument(
        "--target-users-path", default="data/target_user_ids.parquet"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/task01_canonical_data_v1"
    )
    parser.add_argument(
        "--cutoff",
        type=_datetime_argument,
        help="Explicit fold cutoff; default is max(raw date) minus one day.",
    )
    parser.add_argument(
        "--validation-end-exclusive",
        type=_datetime_argument,
        help="Optional exclusive validation end for rolling folds.",
    )
    parser.add_argument(
        "--smoke-user-limit",
        type=int,
        help=(
            "Run a non-canonical limited smoke on the first N sorted target "
            "users. Use a non-canonical output path."
        ),
    )
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_temporal_fold(
        train_path=args.train_path,
        target_users_path=args.target_users_path,
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        validation_end_exclusive=args.validation_end_exclusive,
        smoke_user_limit=args.smoke_user_limit,
        run_id=args.run_id,
    )
    summary = {
        "output_dir": result.output_dir.as_posix(),
        "mode": result.metrics["mode"],
        "runtime_seconds": result.metrics["runtime_seconds"],
        "peak_memory_mb": result.metrics["peak_memory_mb"],
        "split": result.metrics["deterministic_diagnostics"]["split"],
        "ground_truth_funnel": result.metrics["deterministic_diagnostics"][
            "ground_truth_funnel"
        ],
        "output_sha256": result.metrics["deterministic_diagnostics"][
            "output_sha256"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

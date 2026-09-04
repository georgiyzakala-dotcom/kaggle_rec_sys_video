"""Strict serialization and validation for leaderboard submissions."""

from __future__ import annotations

import csv
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import polars as pl

from data_utils import TARGET_USER_SCHEMA
from experiment_utils import sha256_file
from interfaces import FINAL_RECOMMENDATION_SCHEMA
from validation import (
    ContractValidationError,
    validate_final_recommendations,
    validate_recommendations_against_history,
)

SUBMISSION_COLUMNS = ("user_id", "item_ids")
_DECIMAL_UINT64 = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_UINT64_MAX = 2**64 - 1
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_user_id(value: str, *, row_number: int) -> int:
    if not _DECIMAL_UINT64.fullmatch(value):
        raise ContractValidationError(
            f"row {row_number} user_id is not a canonical decimal UInt64"
        )
    result = int(value, 10)
    if result > _UINT64_MAX:
        raise ContractValidationError(f"row {row_number} user_id exceeds UInt64")
    return result


def _parse_item_ids(value: str, *, row_number: int, expected_k: int) -> list[int]:
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ContractValidationError(
            f"row {row_number} item_ids is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, list):
        raise ContractValidationError(f"row {row_number} item_ids must be a list")
    if len(parsed) != expected_k:
        raise ContractValidationError(
            f"row {row_number} item_ids must contain exactly {expected_k} items"
        )
    if any(type(item) is not int for item in parsed):
        raise ContractValidationError(
            f"row {row_number} item_ids must contain JSON integers only"
        )
    if any(item < _INT32_MIN or item > _INT32_MAX for item in parsed):
        raise ContractValidationError(
            f"row {row_number} item_id is outside the Int32 range"
        )
    if len(set(parsed)) != len(parsed):
        raise ContractValidationError(
            f"row {row_number} item_ids contains duplicate items"
        )
    return parsed


def read_submission(path: str | Path, *, expected_k: int = 20) -> pl.DataFrame:
    """Read a submission with the standard CSV and strict JSON parsers."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    users: list[int] = []
    items: list[list[int]] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ContractValidationError("submission is empty") from error
        if header != list(SUBMISSION_COLUMNS):
            raise ContractValidationError(
                f"submission header must be {list(SUBMISSION_COLUMNS)}, got {header}"
            )
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ContractValidationError(
                    f"row {row_number} must contain exactly two CSV fields"
                )
            users.append(_parse_user_id(row[0], row_number=row_number))
            items.append(
                _parse_item_ids(row[1], row_number=row_number, expected_k=expected_k)
            )
    frame = pl.DataFrame(
        {
            "user_id": pl.Series(users, dtype=pl.UInt64),
            "item_ids": pl.Series(items, dtype=pl.List(pl.Int32)),
        },
        schema=FINAL_RECOMMENDATION_SCHEMA,
    )
    validate_final_recommendations(frame, expected_k=expected_k)
    return frame


def write_submission(
    recommendations: pl.DataFrame,
    path: str | Path,
    *,
    expected_k: int = 20,
) -> dict[str, Any]:
    """Atomically write and round-trip a deterministic submission CSV."""

    validate_final_recommendations(recommendations, expected_k=expected_k)
    destination = Path(path)
    if destination.name != "submission.csv":
        raise ContractValidationError("submission filename must be submission.csv")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = recommendations.sort("user_id")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(SUBMISSION_COLUMNS)
            for user_id, item_ids in ordered.iter_rows():
                writer.writerow(
                    (
                        str(int(user_id)),
                        json.dumps(item_ids, ensure_ascii=True, allow_nan=False),
                    )
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    restored = read_submission(destination, expected_k=expected_k)
    if not restored.equals(ordered):
        raise ContractValidationError(
            "submission CSV round-trip differs from internal recommendations"
        )
    return {
        "filename": destination.name,
        "columns": list(SUBMISSION_COLUMNS),
        "rows": restored.height,
        "expected_k": expected_k,
        "sha256": sha256_file(destination),
        "round_trip_equal": True,
        "serializer": "csv.writer+json.dumps",
    }


def validate_submission_against_artifacts(
    path: str | Path,
    *,
    recommendations: pl.DataFrame,
    target_users: pl.DataFrame,
    history_daily: pl.DataFrame | pl.LazyFrame,
    expected_k: int = 20,
) -> dict[str, Any]:
    """Validate CSV, exact internal parity, target universe, and history rules."""

    if target_users.schema != TARGET_USER_SCHEMA:
        raise ContractValidationError("target users have invalid schema")
    restored = read_submission(path, expected_k=expected_k)
    expected = recommendations.sort("user_id")
    if not restored.equals(expected):
        raise ContractValidationError(
            "submission content differs from internal recommendations"
        )
    validate_recommendations_against_history(
        restored,
        target_users=target_users,
        history_daily=history_daily,
        expected_k=expected_k,
    )
    return {
        "filename": Path(path).name,
        "columns": list(SUBMISSION_COLUMNS),
        "rows": restored.height,
        "target_users": target_users.height,
        "items_per_user": expected_k,
        "duplicate_users": 0,
        "duplicate_items": 0,
        "missing_users": 0,
        "extra_users": 0,
        "null_values": 0,
        "unknown_items": 0,
        "seen_pairs": 0,
        "round_trip_equal": True,
        "sha256": sha256_file(path),
    }


def submission_schema() -> dict[str, Any]:
    return {
        "filename": "submission.csv",
        "header": list(SUBMISSION_COLUMNS),
        "user_id": "canonical decimal UInt64",
        "item_ids": "strict JSON list[Int32]",
        "rows": "one per target user",
        "rank_order": "deterministic",
    }


__all__ = [
    "SUBMISSION_COLUMNS",
    "read_submission",
    "submission_schema",
    "validate_submission_against_artifacts",
    "write_submission",
]

#!/usr/bin/env bash
set -Eeuo pipefail

cleanup_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${cleanup_repo}"
export OMP_NUM_THREADS=4
export POLARS_MAX_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec .venv/bin/python scripts/cleanup_expanded_ranker.py "$@"

#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

run_id="${TASK05_RUN_ID:-task05_implicit_als_v1}"
config_path="${TASK05_CONFIG:-configs/task05_implicit_als_v1.json}"
output_dir="${TASK05_OUTPUT_DIR:-artifacts/${run_id}}"
log_file="${TASK05_LOG_FILE:-logs/${run_id}.log}"
best_model_dir="${TASK05_BEST_MODEL_DIR:-artifacts/.${run_id}.best-model}"
lock_file="logs/${run_id}.lock"

mkdir -p logs

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/python is missing or not executable." >&2
    exit 1
fi
if [[ ! -f "${config_path}" ]]; then
    echo "ERROR: config does not exist: ${config_path}" >&2
    exit 1
fi
if [[ -e "${output_dir}" ]]; then
    echo "ERROR: refusing to overwrite existing artifact: ${output_dir}" >&2
    exit 1
fi

exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: another ${run_id} overnight run is already active." >&2
    exit 1
fi

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Task05 full run is starting."
echo "Artifact: ${output_dir}"
echo "Compact log: ${log_file}"
echo "Rolling best model: ${best_model_dir}/best_model.json"
echo "Monitor from another terminal: tail -f ${log_file}"
echo "Stop gracefully in this terminal: Ctrl-C"

exec ./.venv/bin/python scripts/run_implicit_als.py \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    --run-id "${run_id}" \
    --log-file "${log_file}" \
    --best-model-dir "${best_model_dir}"

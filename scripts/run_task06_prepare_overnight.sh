#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

run_id="${TASK06_DATASET_RUN_ID:-task06_candidate_datasets_v1}"
config_path="${TASK06_CONFIG:-configs/task06_candidate_ensemble_v1.json}"
output_dir="${TASK06_DATASET_OUTPUT_DIR:-artifacts/${run_id}}"
checkpoint_dir="${TASK06_DATASET_CHECKPOINT_DIR:-artifacts/.${run_id}.checkpoint}"
log_file="${TASK06_DATASET_LOG_FILE:-logs/${run_id}.log}"
lock_file="logs/${run_id}.lock"
pid_file="logs/${run_id}.pid"

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
    echo "ERROR: another ${run_id} preparation is already active." >&2
    exit 1
fi

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export POLARS_MAX_THREADS="${TASK06_POLARS_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "$$" > "${pid_file}"
echo "Task06 offline candidate preparation is starting."
echo "Artifact: ${output_dir}"
echo "Resume checkpoint: ${checkpoint_dir}/checkpoint.json"
echo "Compact log: ${log_file}"
echo "Monitor: tail -f ${log_file}"
echo "Stop gracefully here: Ctrl-C"
echo "Stop from another terminal: kill -TERM $(cat "${pid_file}")"

exec ./.venv/bin/python scripts/prepare_candidate_datasets.py \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    --run-id "${run_id}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --log-file "${log_file}"

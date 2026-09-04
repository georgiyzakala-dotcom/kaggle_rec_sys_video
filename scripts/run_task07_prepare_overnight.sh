#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

run_id="${TASK07_RUN_ID:-task07_ranker_dataset_v1}"
config_path="${TASK07_CONFIG:-configs/task07_ranker_dataset_v1.json}"
output_dir="${TASK07_OUTPUT_DIR:-artifacts/${run_id}}"
checkpoint_dir="${TASK07_CHECKPOINT_DIR:-artifacts/.${run_id}.checkpoint}"
log_file="${TASK07_LOG_FILE:-logs/${run_id}.log}"
lock_file="logs/${run_id}.lock"
pid_file="logs/${run_id}.pid"
minimum_free_gb="${TASK07_MIN_FREE_GB:-150}"

mkdir -p logs artifacts

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
if [[ ! "${minimum_free_gb}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TASK07_MIN_FREE_GB must be a non-negative integer." >&2
    exit 1
fi

available_kb="$(df -Pk artifacts | awk 'NR == 2 {print $4}')"
required_kb="$((minimum_free_gb * 1024 * 1024))"
if (( available_kb < required_kb )); then
    echo "ERROR: less than ${minimum_free_gb} GiB is available for task07." >&2
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
export POLARS_MAX_THREADS="${TASK07_POLARS_THREADS:-8}"
export PYTHONUNBUFFERED=1

child_pid=""
cleanup() {
    local status=$?
    trap - EXIT
    rm -f -- "${pid_file}"
    exit "${status}"
}
terminate_child() {
    if [[ -n "${child_pid}" ]]; then
        kill -TERM "${child_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap terminate_child INT TERM

echo "Task07 ranker-dataset preparation is starting."
echo "Artifact: ${output_dir}"
echo "Resume checkpoint: ${checkpoint_dir}/checkpoint.json"
echo "Compact progress log: ${log_file}"
echo "Free disk requirement: ${minimum_free_gb} GiB"
echo "Monitor: tail -f ${log_file}"
echo "Stop gracefully here: Ctrl-C"

./.venv/bin/python scripts/prepare_ranker_datasets.py \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    --run-id "${run_id}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --log-file "${log_file}" &
child_pid=$!
echo "${child_pid}" > "${pid_file}"
echo "Stop from another terminal: kill -TERM ${child_pid}"
wait "${child_pid}"

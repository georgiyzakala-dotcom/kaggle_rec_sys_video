#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

run_id="${TASK08_RUN_ID:-task08_catboost_pointwise_v1}"
config_path="${TASK08_CONFIG:-configs/task08_catboost_pointwise_v1.json}"
output_dir="${TASK08_OUTPUT_DIR:-artifacts/${run_id}}"
checkpoint_dir="${TASK08_CHECKPOINT_DIR:-artifacts/.${run_id}.checkpoint}"
pool_cache_dir="${TASK08_POOL_CACHE_DIR:-artifacts/task08_catboost_pools_v1}"
best_model_dir="${TASK08_BEST_MODEL_DIR:-artifacts/.${run_id}.best-model}"
log_file="${TASK08_LOG_FILE:-logs/${run_id}.log}"
lock_file="logs/${run_id}.lock"
pid_file="logs/${run_id}.pid"
minimum_free_gb="${TASK08_MIN_FREE_GB:-250}"
cleanup_recoverable_on_success="${TASK08_CLEANUP_ON_SUCCESS:-1}"

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
    echo "ERROR: TASK08_MIN_FREE_GB must be a non-negative integer." >&2
    exit 1
fi
if [[ "${cleanup_recoverable_on_success}" != "0" && "${cleanup_recoverable_on_success}" != "1" ]]; then
    echo "ERROR: TASK08_CLEANUP_ON_SUCCESS must be 0 or 1." >&2
    exit 1
fi
if ! .venv/bin/python -c 'import catboost; assert catboost.__version__ == "1.2.10"'; then
    echo "ERROR: catboost==1.2.10 is not installed in .venv." >&2
    exit 1
fi
if ! nvidia-smi >/dev/null; then
    echo "ERROR: NVIDIA GPU is not available to CatBoost." >&2
    exit 1
fi

available_kb="$(df -Pk artifacts | awk 'NR == 2 {print $4}')"
required_kb="$((minimum_free_gb * 1024 * 1024))"
if (( available_kb < required_kb )); then
    echo "ERROR: less than ${minimum_free_gb} GiB is available for task08." >&2
    exit 1
fi

exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: another ${run_id} CatBoost run is already active." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${TASK08_CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export POLARS_MAX_THREADS="${TASK08_POLARS_THREADS:-8}"
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

echo "Task08 pointwise CatBoost training is starting."
echo "Artifact: ${output_dir}"
echo "Reusable quantized pools: ${pool_cache_dir}"
echo "Resume checkpoint: ${checkpoint_dir}/checkpoint.json"
echo "CatBoost snapshot: ${checkpoint_dir}/catboost_training/snapshot.cbsnapshot"
echo "Portable best model: ${best_model_dir}/model"
echo "Compact progress log: ${log_file}"
echo "Free disk requirement: ${minimum_free_gb} GiB"
echo "Cleanup recoverable state on success: ${cleanup_recoverable_on_success}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Monitor log: tail -f ${log_file}"
echo "Monitor GPU: watch -n 2 nvidia-smi"
echo "Stop gracefully here: Ctrl-C"

runner_args=(
    .venv/bin/python scripts/run_catboost_ranker.py
    --config "${config_path}"
    --output-dir "${output_dir}"
    --run-id "${run_id}"
    --checkpoint-dir "${checkpoint_dir}"
    --pool-cache-dir "${pool_cache_dir}"
    --best-model-dir "${best_model_dir}"
    --log-file "${log_file}"
)
if [[ "${cleanup_recoverable_on_success}" == "1" ]]; then
    runner_args+=(--cleanup-recoverable-on-success)
fi

"${runner_args[@]}" &
child_pid=$!
echo "${child_pid}" > "${pid_file}"
echo "Stop from another terminal: kill -TERM ${child_pid}"
wait "${child_pid}"

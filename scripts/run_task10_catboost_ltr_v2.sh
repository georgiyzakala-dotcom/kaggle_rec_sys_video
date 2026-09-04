#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

run_id="task10_catboost_ltr_v2"
config_path="configs/task10_catboost_ltr_v2.json"
output_dir="artifacts/${run_id}"
checkpoint_dir="artifacts/.${run_id}.checkpoint"
best_model_dir="artifacts/.${run_id}.best-model"
log_file="logs/${run_id}.log"
lock_file="logs/${run_id}.lock"
pid_file="logs/${run_id}.pid"
minimum_free_gib="${TASK10_MIN_FREE_GIB:-30}"
minimum_available_ram_gib="${TASK10_MIN_AVAILABLE_RAM_GIB:-40}"
minimum_gpu_total_mib="${TASK10_MIN_GPU_TOTAL_MIB:-15000}"
minimum_gpu_free_mib="${TASK10_MIN_GPU_FREE_MIB:-14000}"
windows_host_drive="G"
minimum_windows_host_free_gib="70"
cleanup_recoverable_on_success="${TASK10_CLEANUP_ON_SUCCESS:-1}"
gpu_id="${TASK10_CUDA_DEVICE:-0}"

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
for value_name in \
    minimum_free_gib \
    minimum_available_ram_gib \
    minimum_gpu_total_mib \
    minimum_gpu_free_mib \
    minimum_windows_host_free_gib \
    gpu_id; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${value_name} must be a non-negative integer." >&2
        exit 1
    fi
done
if [[ "${cleanup_recoverable_on_success}" != "0" && "${cleanup_recoverable_on_success}" != "1" ]]; then
    echo "ERROR: TASK10_CLEANUP_ON_SUCCESS must be 0 or 1." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=8
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

if ! .venv/bin/python -c 'import catboost; assert catboost.__version__ == "1.2.10"'; then
    echo "ERROR: catboost==1.2.10 is required in .venv." >&2
    exit 1
fi
if ! nvidia-smi --id="${gpu_id}" >/dev/null; then
    echo "ERROR: NVIDIA GPU ${gpu_id} is unavailable." >&2
    exit 1
fi
if ! .venv/bin/python -c 'from catboost.utils import get_gpu_device_count; assert get_gpu_device_count() >= 1'; then
    echo "ERROR: CatBoost cannot initialize a CUDA device." >&2
    exit 1
fi

gpu_total_mib="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR == 1 {gsub(/ /, ""); print $1}')"
gpu_free_mib="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR == 1 {gsub(/ /, ""); print $1}')"
if ((gpu_total_mib < minimum_gpu_total_mib)); then
    echo "ERROR: GPU has ${gpu_total_mib} MiB; at least ${minimum_gpu_total_mib} MiB is required." >&2
    exit 1
fi
if ((gpu_free_mib < minimum_gpu_free_mib)); then
    echo "ERROR: GPU has only ${gpu_free_mib} MiB free; at least ${minimum_gpu_free_mib} MiB is required." >&2
    exit 1
fi

available_kib="$(df -Pk artifacts | awk 'NR == 2 {print $4}')"
required_kib="$((minimum_free_gib * 1024 * 1024))"
if ((available_kib < required_kib)); then
    echo "ERROR: less than ${minimum_free_gib} GiB is available inside WSL." >&2
    exit 1
fi
available_ram_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
required_ram_kib="$((minimum_available_ram_gib * 1024 * 1024))"
if ((available_ram_kib < required_ram_kib)); then
    echo "ERROR: less than ${minimum_available_ram_gib} GiB RAM is available." >&2
    exit 1
fi

powershell_exe="$(command -v powershell.exe || true)"
if [[ -z "${powershell_exe}" && -x /mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe ]]; then
    powershell_exe="/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"
fi
if [[ -z "${powershell_exe}" ]]; then
    echo "ERROR: powershell.exe is required to check Windows host storage." >&2
    exit 1
fi
host_free_bytes="$(${powershell_exe} -NoProfile -NonInteractive -Command "[Console]::Write((Get-PSDrive -Name '${windows_host_drive}').Free)")"
if [[ ! "${host_free_bytes}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: cannot determine free space on Windows drive ${windows_host_drive}:" >&2
    exit 1
fi
required_host_bytes="$((minimum_windows_host_free_gib * 1024 * 1024 * 1024))"
if ((host_free_bytes < required_host_bytes)); then
    echo "ERROR: less than ${minimum_windows_host_free_gib} GiB is free on Windows drive ${windows_host_drive}:" >&2
    exit 1
fi

exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: another ${run_id} run is already active." >&2
    exit 1
fi
if [[ -f "${pid_file}" ]]; then
    stale_pid="$(<"${pid_file}")"
    if [[ "${stale_pid}" =~ ^[0-9]+$ ]] && kill -0 "${stale_pid}" 2>/dev/null; then
        echo "ERROR: PID ${stale_pid} from ${pid_file} is still active." >&2
        exit 1
    fi
    rm -f -- "${pid_file}"
fi

child_pid=""
cleanup_pid() {
    local status=$?
    trap - EXIT
    rm -f -- "${pid_file}"
    exit "${status}"
}
terminate_child() {
    if [[ -n "${child_pid}" ]]; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}
trap cleanup_pid EXIT
trap terminate_child INT TERM

echo "Task 10 CatBoost LTR recovery selection v2 is starting."
echo "Artifact: ${output_dir}"
echo "Resume checkpoint: ${checkpoint_dir}/checkpoint.json"
echo "Imported completed profile: task10_catboost_ltr_v1/s10_qsm_beta_1"
echo "Reusable grouped pools: artifacts/task10_catboost_grouped_pools_*_v1"
echo "Compact rotating log: ${log_file}"
echo "Resource thresholds: RAM=${minimum_available_ram_gib} GiB, WSL disk=${minimum_free_gib} GiB, Windows ${windows_host_drive}: ${minimum_windows_host_free_gib} GiB, GPU free=${minimum_gpu_free_mib} MiB"
echo "Monitor log: tail -F ${log_file}"
echo "Monitor GPU: watch -n 2 nvidia-smi"
echo "Monitor RAM/disk: watch -n 30 'free -h; df -h artifacts; du -sh artifacts/.${run_id}.checkpoint artifacts/task10_catboost_grouped_pools_*_v1 2>/dev/null'"
echo "Monitor Windows host disk in PowerShell: Get-PSDrive ${windows_host_drive}"
echo "Stop gracefully here: Ctrl-C"

runner_args=(
    .venv/bin/python scripts/run_catboost_ltr.py
    --config "${config_path}"
    --output-dir "${output_dir}"
    --run-id "${run_id}"
    --checkpoint-dir "${checkpoint_dir}"
    --best-model-dir "${best_model_dir}"
    --log-file "${log_file}"
)
if [[ "${cleanup_recoverable_on_success}" == "1" ]]; then
    runner_args+=(--cleanup-recoverable-on-success)
fi

"${runner_args[@]}" &
child_pid=$!
pid_tmp="${pid_file}.tmp-${child_pid}"
printf '%s\n' "${child_pid}" >"${pid_tmp}"
mv -f -- "${pid_tmp}" "${pid_file}"
echo "Stop from another terminal: kill -TERM ${child_pid}"
wait "${child_pid}"

exit 0

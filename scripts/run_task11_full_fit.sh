#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-configs/task11_full_fit_v1.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -x .venv/bin/python ]]; then
    printf 'ERROR: missing executable .venv/bin/python\n' >&2
    exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    printf 'ERROR: config does not exist: %s\n' "${CONFIG_PATH}" >&2
    exit 2
fi

readarray -t CONFIG_VALUES < <(
    .venv/bin/python -c '
import json, sys
source = json.load(open(sys.argv[1], encoding="utf-8"))
resources = source["resources"]
print(source["run_id"])
print(source["mode"])
print(max(
    float(resources["minimum_windows_g_free_gib"]),
    float(resources["stop_windows_g_free_gib"])
    + float(resources["projected_peak_growth_gib"])
    + float(resources["disk_safety_margin_gib"]),
))
print(float(resources["minimum_linux_free_gib"]))
print(float(resources["minimum_available_ram_gib"]))
print(int(resources["minimum_free_vram_mib"]))
' "${CONFIG_PATH}"
)

RUN_ID="${CONFIG_VALUES[0]}"
MODE="${CONFIG_VALUES[1]}"
WINDOWS_G_GATE="${CONFIG_VALUES[2]}"
LINUX_GATE="${CONFIG_VALUES[3]}"
RAM_GATE="${CONFIG_VALUES[4]}"
VRAM_GATE="${CONFIG_VALUES[5]}"
OUTPUT_DIR="artifacts/${RUN_ID}"
WORK_DIR="artifacts/.${RUN_ID}.work"
LOG_FILE="logs/${RUN_ID}.log"
LOCK_FILE="artifacts/.${RUN_ID}.lock"
PID_FILE="artifacts/.${RUN_ID}.pid"

mkdir -p artifacts logs
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    printf 'ERROR: another %s launcher owns %s\n' "${RUN_ID}" "${LOCK_FILE}" >&2
    exit 3
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    printf 'ERROR: completed output already exists: %s\n' "${OUTPUT_DIR}" >&2
    exit 4
fi
if [[ -e "${PID_FILE}" ]]; then
    EXISTING_PID="$(<"${PID_FILE}")"
    if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
        printf 'ERROR: active %s process from %s: PID %s\n' "${RUN_ID}" "${PID_FILE}" "${EXISTING_PID}" >&2
        exit 3
    fi
    rm -f -- "${PID_FILE}"
fi
if pgrep -af '[r]un_full_fit_inference.py' >/dev/null; then
    printf 'ERROR: another Task 11 Python process is active:\n' >&2
    pgrep -af '[r]un_full_fit_inference.py' >&2
    exit 3
fi

.venv/bin/python -c '
import catboost, sys
assert sys.version_info[:2] == (3, 12), sys.version
assert catboost.__version__ == "1.2.10", catboost.__version__
print(f"preflight python={sys.version.split()[0]} catboost={catboost.__version__}")
'

GPU_ROW="$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits --id=0)"
VRAM_FREE="$(printf '%s\n' "${GPU_ROW}" | awk -F, '{gsub(/ /, "", $4); print $4}')"
LINUX_FREE="$(df -B1 --output=avail . | tail -n 1 | awk '{printf "%.6f", $1 / 1073741824}')"
RAM_AVAILABLE="$(awk '/^MemAvailable:/ {printf "%.6f", $2 * 1024 / 1073741824}' /proc/meminfo)"
POWERSHELL='/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe'
if [[ ! -x "${POWERSHELL}" ]]; then
    printf 'ERROR: PowerShell is unavailable; cannot validate Windows G: free space\n' >&2
    exit 5
fi
WINDOWS_G_FREE="$(${POWERSHELL} -NoProfile -NonInteractive -Command "[double](Get-PSDrive -Name 'G').Free / 1GB" | tr ',' '.' | tail -n 1)"

.venv/bin/python -c '
import sys
checks = {
    "Windows G: free GiB": (float(sys.argv[1]), float(sys.argv[2])),
    "Linux free GiB": (float(sys.argv[3]), float(sys.argv[4])),
    "available RAM GiB": (float(sys.argv[5]), float(sys.argv[6])),
    "free VRAM MiB": (float(sys.argv[7]), float(sys.argv[8])),
}
for name, (actual, required) in checks.items():
    print(f"preflight {name}={actual:.3f} required={required:.3f}")
    if actual < required:
        raise SystemExit(f"ERROR: {name} gate failed")
' "${WINDOWS_G_FREE}" "${WINDOWS_G_GATE}" "${LINUX_FREE}" "${LINUX_GATE}" "${RAM_AVAILABLE}" "${RAM_GATE}" "${VRAM_FREE}" "${VRAM_GATE}"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=8
export POLARS_MAX_THREADS=8
export PYTHONUNBUFFERED=1

CHILD_PID=""
cleanup_pid() {
    if [[ -n "${CHILD_PID}" && -f "${PID_FILE}" && "$(<"${PID_FILE}")" == "${CHILD_PID}" ]]; then
        rm -f -- "${PID_FILE}"
    fi
}
forward_term() {
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        printf 'Forwarding SIGTERM to PID %s\n' "${CHILD_PID}" >&2
        kill -TERM "${CHILD_PID}"
        wait "${CHILD_PID}" || true
    fi
    exit 143
}
trap cleanup_pid EXIT
trap forward_term TERM INT HUP

printf 'Launching run_id=%s mode=%s output=%s work=%s log=%s\n' \
    "${RUN_ID}" "${MODE}" "${OUTPUT_DIR}" "${WORK_DIR}" "${LOG_FILE}"
.venv/bin/python scripts/run_full_fit_inference.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --work-dir "${WORK_DIR}" \
    --log-file "${LOG_FILE}" \
    "$@" &
CHILD_PID=$!
printf '%s\n' "${CHILD_PID}" >"${PID_FILE}"
wait "${CHILD_PID}"

#!/usr/bin/env bash
set -Eeuo pipefail

task14_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${task14_repo}"
if [[ "${1:-}" == "--help" ]]; then
    exec .venv/bin/python scripts/run_profile_full_fit.py --help
fi
task14_config="${1:-configs/task14_full_fit_v1.json}"
if (( $# > 0 )); then shift; fi
if [[ ! -x .venv/bin/python || ! -f "${task14_config}" ]]; then
    echo 'ERROR: .venv/bin/python or the requested config is missing.' >&2
    exit 2
fi
readarray -t task14_settings < <(.venv/bin/python -c '
import json, re, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"])
assert isinstance(c["catboost"]["thread_count"], int) and c["catboost"]["thread_count"] > 0
print(c["run_id"])
print(c["catboost"]["thread_count"])
' "${task14_config}")
if (( ${#task14_settings[@]} != 2 )); then
    echo 'ERROR: invalid launcher config.' >&2
    exit 2
fi
task14_run="${task14_settings[0]}"
task14_threads="${task14_settings[1]}"
task14_pid_file="logs/${task14_run}.pid"
mkdir -p artifacts logs
exec 9>"logs/${task14_run}.launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher already owns ${task14_run}." >&2
    exit 3
fi
if [[ -e "artifacts/${task14_run}" ]]; then
    echo "ERROR: refusing to overwrite artifacts/${task14_run}." >&2
    exit 4
fi

export OMP_NUM_THREADS="${task14_threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${task14_threads}"
export POLARS_MAX_THREADS="${task14_threads}"
export PYTHONUNBUFFERED=1

task14_child=""
task14_cleanup() {
    if [[ -n "${task14_child}" && -f "${task14_pid_file}" && "$(<"${task14_pid_file}")" == "${task14_child}" ]]; then
        rm -f -- "${task14_pid_file}"
    fi
}
task14_stop() {
    if [[ -n "${task14_child}" ]] && kill -0 "${task14_child}" 2>/dev/null; then
        kill -TERM "${task14_child}"
        wait "${task14_child}" || true
    fi
    exit 143
}
trap task14_cleanup EXIT
trap task14_stop INT TERM HUP

printf 'Run: %s\nOutput: artifacts/%s\nResume: artifacts/.%s.work/checkpoint.json\nLog: logs/%s.log\n' \
    "${task14_run}" "${task14_run}" "${task14_run}" "${task14_run}"
printf 'Stop: Ctrl-C or kill -TERM $(cat %s)\nRerun the same command to resume.\n' "${task14_pid_file}"
.venv/bin/python scripts/run_profile_full_fit.py --config "${task14_config}" "$@" &
task14_child=$!
printf '%s\n' "${task14_child}" > "${task14_pid_file}"
wait "${task14_child}"

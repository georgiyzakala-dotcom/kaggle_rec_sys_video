#!/usr/bin/env bash
set -Eeuo pipefail

task13_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${task13_repo}"
if [[ "${1:-}" == "--help" ]]; then
    exec .venv/bin/python scripts/run_history_profiles.py --help
fi
task13_config="${1:-configs/task13_history_profiles_v1.json}"
if (( $# > 0 )); then shift; fi
if [[ ! -x .venv/bin/python || ! -f "${task13_config}" ]]; then
    echo 'ERROR: .venv/bin/python or the requested config is missing.' >&2
    exit 2
fi
readarray -t task13_settings < <(.venv/bin/python -c '
import json, re, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"])
assert isinstance(c["catboost"]["thread_count"], int) and c["catboost"]["thread_count"] > 0
print(c["run_id"])
print(c["catboost"]["thread_count"])
' "${task13_config}")
if (( ${#task13_settings[@]} != 2 )); then
    echo 'ERROR: invalid launcher config.' >&2
    exit 2
fi
task13_run="${task13_settings[0]}"
task13_threads="${task13_settings[1]}"
task13_pid_file="logs/${task13_run}.pid"
mkdir -p artifacts logs
exec 9>"logs/${task13_run}.launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher already owns ${task13_run}." >&2
    exit 3
fi
if [[ -e "artifacts/${task13_run}" ]]; then
    echo "ERROR: refusing to overwrite artifacts/${task13_run}." >&2
    exit 4
fi

export OMP_NUM_THREADS="${task13_threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${task13_threads}"
export POLARS_MAX_THREADS="${task13_threads}"
export PYTHONUNBUFFERED=1

task13_child=""
task13_cleanup() {
    if [[ -n "${task13_child}" && -f "${task13_pid_file}" && "$(<"${task13_pid_file}")" == "${task13_child}" ]]; then
        rm -f -- "${task13_pid_file}"
    fi
}
task13_stop() {
    if [[ -n "${task13_child}" ]] && kill -0 "${task13_child}" 2>/dev/null; then
        kill -TERM "${task13_child}"
        wait "${task13_child}" || true
    fi
    exit 143
}
trap task13_cleanup EXIT
trap task13_stop INT TERM HUP

printf 'Run: %s\nOutput: artifacts/%s\nResume: artifacts/.%s.work/checkpoint.json\nLog: logs/%s.log\n' \
    "${task13_run}" "${task13_run}" "${task13_run}" "${task13_run}"
printf 'Stop: Ctrl-C or kill -TERM $(cat %s)\nRerun the same command to resume.\n' "${task13_pid_file}"
.venv/bin/python scripts/run_history_profiles.py --config "${task13_config}" "$@" &
task13_child=$!
printf '%s\n' "${task13_child}" > "${task13_pid_file}"
wait "${task13_child}"

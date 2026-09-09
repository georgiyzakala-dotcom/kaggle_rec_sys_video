#!/usr/bin/env bash
set -Eeuo pipefail

task12_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${task12_repo}"
if [[ "${1:-}" == "--help" ]]; then
    exec .venv/bin/python scripts/run_ranker_backtest.py --help
fi
task12_config="${1:-configs/task12_ranker_backtest_v1.json}"
if (( $# > 0 )); then shift; fi
if [[ ! -x .venv/bin/python || ! -f "${task12_config}" ]]; then
    echo 'ERROR: .venv/bin/python or the requested config is missing.' >&2
    exit 2
fi
readarray -t task12_settings < <(.venv/bin/python -c '
import json, re, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"])
assert isinstance(c["catboost"]["thread_count"], int) and c["catboost"]["thread_count"] > 0
print(c["run_id"])
print(c["catboost"]["thread_count"])
' "${task12_config}")
if (( ${#task12_settings[@]} != 2 )); then
    echo 'ERROR: invalid launcher config.' >&2
    exit 2
fi
task12_run="${task12_settings[0]}"
task12_threads="${task12_settings[1]}"
task12_pid_file="logs/${task12_run}.pid"
mkdir -p artifacts logs
exec 9>"logs/${task12_run}.launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher already owns ${task12_run}." >&2
    exit 3
fi
if [[ -e "artifacts/${task12_run}" ]]; then
    echo "ERROR: refusing to overwrite artifacts/${task12_run}." >&2
    exit 4
fi

export OMP_NUM_THREADS="${task12_threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${task12_threads}"
export POLARS_MAX_THREADS="${task12_threads}"
export PYTHONUNBUFFERED=1

task12_child=""
task12_cleanup() {
    if [[ -n "${task12_child}" && -f "${task12_pid_file}" && "$(<"${task12_pid_file}")" == "${task12_child}" ]]; then
        rm -f -- "${task12_pid_file}"
    fi
}
task12_stop() {
    if [[ -n "${task12_child}" ]] && kill -0 "${task12_child}" 2>/dev/null; then
        kill -TERM "${task12_child}"
        wait "${task12_child}" || true
    fi
    exit 143
}
trap task12_cleanup EXIT
trap task12_stop INT TERM HUP

printf 'Run: %s\nOutput: artifacts/%s\nResume: artifacts/.%s.work/checkpoint.json\nLog: logs/%s.log\n' \
    "${task12_run}" "${task12_run}" "${task12_run}" "${task12_run}"
printf 'Stop: Ctrl-C or kill -TERM $(cat %s)\nRerun the same command to resume.\n' "${task12_pid_file}"
.venv/bin/python scripts/run_ranker_backtest.py --config "${task12_config}" "$@" &
task12_child=$!
printf '%s\n' "${task12_child}" > "${task12_pid_file}"
wait "${task12_child}"

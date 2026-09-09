#!/usr/bin/env bash
set -Eeuo pipefail

task15_ranker_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${task15_ranker_repo}"
if [[ "${1:-}" == "--help" ]]; then
    exec .venv/bin/python scripts/run_expanded_ranker.py --help
fi
task15_ranker_config="${1:-configs/task15_sasrec_ranker_v1.json}"
if (( $# > 0 )); then shift; fi
if [[ ! -x .venv/bin/python || ! -f "${task15_ranker_config}" ]]; then
    echo 'ERROR: .venv/bin/python or the requested config is missing.' >&2
    exit 2
fi
readarray -t task15_ranker_settings < <(.venv/bin/python -c '
import json, re, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"])
assert isinstance(c["catboost"]["thread_count"], int) and c["catboost"]["thread_count"] > 0
print(c["run_id"])
print(c["catboost"]["thread_count"])
' "${task15_ranker_config}")
if (( ${#task15_ranker_settings[@]} != 2 )); then
    echo 'ERROR: invalid launcher config.' >&2
    exit 2
fi
task15_ranker_run="${task15_ranker_settings[0]}"
task15_ranker_threads="${task15_ranker_settings[1]}"
task15_ranker_pid_file="logs/${task15_ranker_run}.pid"
mkdir -p artifacts logs
exec 9>"logs/${task15_ranker_run}.launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher already owns ${task15_ranker_run}." >&2
    exit 3
fi
if [[ -e "artifacts/${task15_ranker_run}" ]]; then
    echo "ERROR: refusing to overwrite artifacts/${task15_ranker_run}." >&2
    exit 4
fi

export OMP_NUM_THREADS="${task15_ranker_threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${task15_ranker_threads}"
export POLARS_MAX_THREADS="${task15_ranker_threads}"
export PYTHONUNBUFFERED=1

task15_ranker_child=""
task15_ranker_cleanup() {
    if [[ -n "${task15_ranker_child}" && -f "${task15_ranker_pid_file}" && "$(<"${task15_ranker_pid_file}")" == "${task15_ranker_child}" ]]; then
        rm -f -- "${task15_ranker_pid_file}"
    fi
}
task15_ranker_stop() {
    if [[ -n "${task15_ranker_child}" ]] && kill -0 "${task15_ranker_child}" 2>/dev/null; then
        kill -TERM "${task15_ranker_child}"
        wait "${task15_ranker_child}" || true
    fi
    exit 143
}
trap task15_ranker_cleanup EXIT
trap task15_ranker_stop INT TERM HUP

printf 'Run: %s\nOutput: artifacts/%s\nResume: artifacts/.%s.work/checkpoint.json\nLog: logs/%s.log\n' \
    "${task15_ranker_run}" "${task15_ranker_run}" "${task15_ranker_run}" "${task15_ranker_run}"
printf 'Stop: Ctrl-C or kill -TERM $(cat %s)\nRerun the same command to resume.\n' "${task15_ranker_pid_file}"
.venv/bin/python scripts/run_expanded_ranker.py --config "${task15_ranker_config}" "$@" &
task15_ranker_child=$!
printf '%s\n' "${task15_ranker_child}" > "${task15_ranker_pid_file}"
wait "${task15_ranker_child}"

#!/usr/bin/env bash
set -Eeuo pipefail

sasrec_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${sasrec_repo}"
if [[ "${1:-}" == "--help" ]]; then
    exec .venv/bin/python scripts/run_sasrec_top600.py --help
fi
sasrec_config="${1:-configs/task15_sasrec_top600_v1.json}"
if (( $# > 0 )); then shift; fi
if [[ ! -x .venv/bin/python || ! -f "${sasrec_config}" ]]; then
    echo 'ERROR: existing .venv/bin/python and config are required.' >&2
    exit 2
fi
readarray -t sasrec_settings < <(.venv/bin/python -c '
import json, re, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", c["run_id"])
assert type(c["cpu_threads"]) is int and 1 <= c["cpu_threads"] <= 8
print(c["run_id"])
print(c["cpu_threads"])
' "${sasrec_config}")
if (( ${#sasrec_settings[@]} != 2 )); then
    echo 'ERROR: invalid launcher config.' >&2
    exit 2
fi
sasrec_run="${sasrec_settings[0]}"
sasrec_threads="${sasrec_settings[1]}"
sasrec_pid_file="logs/${sasrec_run}.pid"
mkdir -p logs artifacts
exec 9>"logs/${sasrec_run}.launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher owns ${sasrec_run}." >&2
    exit 3
fi
if [[ -e "artifacts/${sasrec_run}" ]]; then
    echo "ERROR: completed artifact exists: artifacts/${sasrec_run}." >&2
    exit 4
fi
export OMP_NUM_THREADS="${sasrec_threads}"
export POLARS_MAX_THREADS="${sasrec_threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${sasrec_threads}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

sasrec_child=""
sasrec_cleanup() {
    if [[ -n "${sasrec_child}" && -f "${sasrec_pid_file}" && "$(<"${sasrec_pid_file}")" == "${sasrec_child}" ]]; then
        rm -f -- "${sasrec_pid_file}"
    fi
}
sasrec_stop() {
    if [[ -n "${sasrec_child}" ]] && kill -0 "${sasrec_child}" 2>/dev/null; then
        kill -TERM "${sasrec_child}"
        wait "${sasrec_child}" || true
    fi
    exit 143
}
trap sasrec_cleanup EXIT
trap sasrec_stop INT TERM HUP
printf 'SASRec/ALS top600 evaluation: %s\nOutput: artifacts/%s/metrics.json\nLog: logs/%s.log\n' "${sasrec_run}" "${sasrec_run}" "${sasrec_run}"
printf 'Stop: Ctrl-C or kill -TERM $(cat %s)\nRepeat this command to resume from the last completed shard/fold.\n' "${sasrec_pid_file}"
.venv/bin/python scripts/run_sasrec_top600.py --config "${sasrec_config}" "$@" &
sasrec_child=$!
printf '%s\n' "${sasrec_child}" > "${sasrec_pid_file}"
wait "${sasrec_child}"

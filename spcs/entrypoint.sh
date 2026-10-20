#!/bin/bash
# SPCS job entrypoint.  Usage: entrypoint.sh prepare [prepare.py args] | train <run_id> | tests
# /stage is the mounted internal stage: /stage/cache (prepared data, snapshots), /stage/runs/<run_id>/
set -uo pipefail
export SHAPE_AE_CACHE=/stage/cache
mode="${1:-}"; shift || true

case "$mode" in
  prepare)
    python prepare.py --source snowflake "$@" 2>&1 | tee /stage/cache_prepare.log
    exit "${PIPESTATUS[0]}"
    ;;
  train)
    run_id="$1"
    run_dir="/stage/runs/${run_id}"
    if [ ! -f "${run_dir}/train.py" ]; then echo "missing ${run_dir}/train.py"; exit 2; fi
    cp "${run_dir}/train.py" /app/train.py
    export SHAPE_AE_OUTPUT_DIR="${run_dir}/outputs"
    if [ -f "${run_dir}/events.csv" ]; then export SHAPE_AE_EVENTS_CSV="${run_dir}/events.csv"; fi
    nvidia-smi -L 2>/dev/null || echo "no GPU visible"
    timeout "${JOB_TIMEOUT_SECONDS:-900}" python train.py 2>&1 | tee "${run_dir}/run.log"
    rc="${PIPESTATUS[0]}"
    echo "${rc}" > "${run_dir}/exit_code"
    exit "${rc}"
    ;;
  tests)
    rc=0
    for t in tests/test_*.py; do python "$t" || rc=1; done
    exit "$rc"
    ;;
  *)
    echo "usage: entrypoint.sh prepare|train <run_id>|tests"; exit 2
    ;;
esac

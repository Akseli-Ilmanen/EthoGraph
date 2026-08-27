#!/usr/bin/env bash
# Runs the training stages listed at the bottom, one after another, for an
# unattended overnight run. They all train models on the GPU, so they run
# strictly sequentially, never in parallel.
#
# Each stage is independently resumable (bench_cells.tsv / study.db are
# append-only), so a crash partway through loses nothing already trained --
# rerun this script and finished work is skipped.
#
# A failure in one stage does NOT stop the others: they don't depend on each
# other's output. Check the per-stage log and the final summary in the
# morning.
#
#   ./scripts/run_overnight.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../data/bench_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="$LOG_DIR/overnight_${STAMP}.log"

# Everything from here on is both printed and appended to $RUN_LOG, so the
# preflight banner and the final summary survive even if the terminal that
# launched this is gone by morning.
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== run_overnight $STAMP ==="
echo "python resolves to: $(command -v python)"
python --version 2>&1 || true
echo "logs: $LOG_DIR"

# `python` can resolve to the Windows Store's App Execution Alias stub (which
# prints "Python was not found..." and exits 9009) when this script runs in a
# shell that never activated the project's venv/conda env -- e.g. launched
# from a different terminal or profile than the one you've been working in.
# That failure is easy to miss (each stage's own log is a single short line),
# so fail loudly here before burning the whole night on three dead stages.
if ! python -c "import ethograph" > /dev/null 2>&1; then
    echo "FATAL: 'python' cannot import ethograph -- wrong (or no) environment activated." >&2
    echo "Activate the project's venv/conda env in the shell that runs this script, then retry." >&2
    exit 1
fi

echo

declare -a NAMES=()
declare -a STATUSES=()

run_stage() {
    local name="$1"
    local script="$2"
    local log="$LOG_DIR/${name}_${STAMP}.log"
    local started
    started="$(date +%s)"

    echo "=== $(date '+%F %T') starting $name -> $log ==="
    python "$SCRIPT_DIR/$script" > "$log" 2>&1
    local status=$?

    local elapsed=$(( $(date +%s) - started ))
    if [ "$status" -eq 0 ]; then
        echo "=== $(date '+%F %T') $name finished OK (${elapsed}s) ==="
    else
        echo "=== $(date '+%F %T') $name FAILED, exit $status (${elapsed}s) -- see $log ===" >&2
    fi

    NAMES+=("$name")
    STATUSES+=("$status")
}

run_stage bench bench.py

echo
echo "=== summary ($(date '+%F %T')) ==="
overall=0
for i in "${!NAMES[@]}"; do
    if [ "${STATUSES[$i]}" -eq 0 ]; then
        echo "  OK     ${NAMES[$i]}"
    else
        echo "  FAILED ${NAMES[$i]} (exit ${STATUSES[$i]})"
        overall=1
    fi
done

exit "$overall"

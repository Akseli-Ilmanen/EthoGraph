#!/usr/bin/env bash
# Train the two trunks on data/spot/project.yaml, one after the other, and compare them.
#
#   bash scripts/spot_arch_sweep.sh              # msagsm, then gsm
#   bash scripts/spot_arch_sweep.sh gsm          # one of them
#   bash scripts/spot_arch_sweep.sh msagsm gsm   # any order
#
# Everything else — features:, clip:, train: — comes from project.yaml. Each run
# gets its own name (the auto name carries only the clip + `_features`, so two
# trunks would otherwise land in one folder) and its own log under
# data/spot/runs/. The dataset is materialised once; a run whose last
# checkpoint exists is skipped, so the script can be re-run after an
# interruption. Evaluate scores every trained run and writes
# data/spot/runs/compare.tsv — with features fed in, each run is scored a
# second time with them zeroed (test_metrics_nofeatures.yaml).

set -euo pipefail
cd "$(dirname "$0")/.."

# Git Bash's $HOME can point somewhere the Windows profile isn't, and neither
# $USERPROFILE nor $HOMEDRIVE is guaranteed to reach bash, so try each in turn.
if [ -z "${PYTHON:-}" ]; then
    profile="${USERPROFILE:-${HOMEDRIVE:-}${HOMEPATH:-}}"
    for root in "${profile//\\//}" "${HOME:-}" /c/Users/"${USERNAME:-$(id -un)}"; do
        [ -n "$root" ] || continue
        for conda in anaconda3 miniconda3 miniforge3; do
            candidate="$root/$conda/envs/ethograph/python.exe"
            [ -x "$candidate" ] && PYTHON="$candidate" && break 2
        done
    done
fi
[ -x "${PYTHON:-}" ] || { echo "ethograph python not found; set PYTHON=/path/to/python.exe" >&2; exit 1; }
echo "python: $PYTHON"
export PYTHONUTF8=1
mkdir -p data/spot/runs

archs=("$@")
[ ${#archs[@]} -eq 0 ] && archs=(msagsm gsm)

"$PYTHON" scripts/spot.py materialise

for arch in "${archs[@]}"; do
    name="ctx2s_res10ms_${arch}_features"
    log="data/spot/runs/${name}.log"
    echo "== ${name} -> ${log}"
    "$PYTHON" scripts/spot.py baseline \
        --set "model.architecture=rny008_${arch}" "train.run_name=${name}" 2>&1 | tee "$log"
done

"$PYTHON" scripts/spot.py evaluate

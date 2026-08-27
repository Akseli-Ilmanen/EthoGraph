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

PYTHON="${PYTHON:-$HOME/anaconda3/envs/ethograph/python.exe}"
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

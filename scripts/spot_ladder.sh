#!/usr/bin/env bash
# The stride ladder: does E2E-Spot fail at 200 fps because its temporal
# aperture is 8x too small in real time? See scripts/spot_point_events.md,
# "200 fps is not 25 fps". Five runs, ~45 min each.
#
#   bash scripts/spot_ladder.sh                 # all five, in order
#   bash scripts/spot_ladder.sh A0 A3           # just these
#
# Close anything else using the GPU first — a 200-frame batch needs ~6 GB
# free, and a run that pages is 20x slower. Check with:
#   nvidia-smi --query-gpu=memory.used --format=csv,noheader
set -uo pipefail
cd "$(dirname "$0")/.."

# The interpreter, not $HOME: WSL's bash and Git bash disagree about what
# $HOME is, and only one of them can see the Windows conda env.
if [ -z "${PY:-}" ]; then
  for candidate in \
      "${USERPROFILE:-/nonexistent}/anaconda3/envs/ethograph/python.exe" \
      "$HOME/anaconda3/envs/ethograph/python.exe" \
      "$HOME/anaconda3/envs/ethograph/bin/python"; do
    [ -x "$candidate" ] && { PY=$candidate; break; }
  done
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
  echo "no ethograph python found; set PY=/path/to/python" >&2
  exit 1
fi
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

# name  stride  clip_len  acc_grad  dilate_len   (dilate holds the positive
# window at +-10 ms real time, so dilation is not a confound across strides)
declare -A LADDER=(
  [A0]="1 200 4 2"   # 1.0 s context,  5 ms quantisation — matched baseline
  [A3]="4 200 4 0"   # 4.0 s context, 20 ms — the paper's regime
  [A2]="2 200 4 1"   # 2.0 s context, 10 ms
  [A1]="2 100 4 1"   # 1.0 s context, 10 ms — prices resolution alone
  [A4]="8 200 4 0"   # 8.0 s context, 40 ms — diagnostic, past the budget
)
ORDER=(A0 A3 A2 A1 A4)
[ $# -gt 0 ] && ORDER=("$@")
failed=()

for name in "${ORDER[@]}"; do
  read -r stride clip acc dilate <<<"${LADDER[$name]}"
  echo "=== $(date +%H:%M) starting $name (stride $stride, clip $clip, dilate $dilate) ==="
  "$PY" scripts/spot_point_events.py train --save-dir "runs/$name" \
    --stride "$stride" --clip-len "$clip" --acc-grad "$acc" \
    --epochs 8 --retries 2 -- \
    --epoch_num_frames 250000 --warm_up_epochs 1 \
    --criterion map --start_val_epoch 1 --dilate_len "$dilate"
  status=$?
  if [ $status -ne 0 ]; then
    failed+=("$name")
    echo "=== $(date +%H:%M) FAILED $name (exit $status) ==="
  else
    echo "=== $(date +%H:%M) finished $name ==="
  fi
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "=== LADDER INCOMPLETE: ${failed[*]} failed ==="
  exit 1
fi
echo "=== LADDER COMPLETE ==="

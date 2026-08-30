#!/usr/bin/env bash
# Paired baseline-vs-masked runner for the MEAP-style masking experiment.
#
# One run:
#   ./experiments/run.sh --arm masked --p 0.15 --seed 3
#
# A paired block (same INIT_SEED for both arms, so the init cancels out):
#   for s in 1 2 3 4 5; do
#     ./experiments/run.sh --arm baseline --seed $s
#     ./experiments/run.sh --arm masked --p 0.15 --seed $s
#   done
#
# Annealed arm:  --arm masked --p 0.15 --p-end 0.0 --seed $s
set -euo pipefail

ARM=baseline; P=0.0; P_END=""; SEED=1; STEPS=""; GPUS=${GPUS:-8}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2;;
    --p) P="$2"; shift 2;;
    --p-end) P_END="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;       # override num_scheduled_iterations for step-removal tests
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ "$ARM" == "baseline" ]] && P=0.0 && P_END=""

export MASK_P_START="$P"
export MASK_P_END="${P_END:-$P}"
export MASK_SEED="$SEED"
export INIT_SEED="$SEED"        # pairs the arms: same init, different objective

TAG="${ARM}_p${MASK_P_START}-${MASK_P_END}_seed${SEED}${STEPS:+_steps${STEPS}}"
OUT="experiments/logs/${TAG}"
mkdir -p "$OUT"

# The step count is a module constant, so a step-removal test edits it into a scratch copy
# rather than mutating the tracked file.
SCRIPT=train_gpt.py
if [[ -n "$STEPS" ]]; then
  SCRIPT="${OUT}/train_gpt.py"
  sed "s/^    num_scheduled_iterations: int = .*/    num_scheduled_iterations: int = ${STEPS}/" train_gpt.py > "$SCRIPT"
  grep -n "num_scheduled_iterations: int" "$SCRIPT" | head -1
fi

{
  echo "commit:  $(git rev-parse HEAD)"
  echo "dirty:   $(git status --porcelain -- train_gpt.py | wc -l | tr -d ' ') tracked change(s) to train_gpt.py"
  echo "arm:     $ARM"
  echo "p:       $MASK_P_START -> $MASK_P_END"
  echo "seed:    $SEED (init and mask)"
  echo "gpus:    $GPUS"
} | tee "$OUT/config.txt"

torchrun --standalone --nproc_per_node="$GPUS" "$SCRIPT" 2>&1 | tee "$OUT/train.log"

echo "final: $(grep -E 'val_loss:[0-9.]+' "$OUT/train.log" | tail -1)"

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
#
# Step removal:  --arm masked --p 0.15 --seed $s --steps 1255
#   --steps sets num_scheduled_iterations (default 1270). The run is that plus the 15
#   extension steps, so 1255 means 15 steps removed. Compare against the FULL baseline.
#
# Copy mixture:  --arm baseline --seed $s --copy-mix [--eval-ws 6:26,8:20]
#   Eval-only. Training is the untouched baseline, so the run also counts as a baseline
#   replicate; every validation additionally prints the copy-mixture loss on the same weights.
#   --eval-ws rescoring of other (short:long) eval windows happens after the final validation.
#
# Noise floor:   --arm baseline --seed $s --rep 2
#   Same seed, same arm, run again. The spread between replicates is the run-to-run noise
#   that pairing CANNOT cancel (kernel nondeterminism), so it sets the smallest effect the
#   paired design can detect. Measure it before spending the budget on screening.
set -euo pipefail

ARM=baseline; P=0.0; P_END=""; SEED=1; STEPS=""; REP=1; GPUS=${GPUS:-8}; COPY_MIX=0; EVAL_WS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2;;
    --p) P="$2"; shift 2;;
    --p-end) P_END="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;       # override num_scheduled_iterations for step-removal tests
    --rep) REP="$2"; shift 2;;           # replicate index: an identical rerun that does not overwrite rep 1
    --copy-mix) COPY_MIX=1; shift;;      # score the document-local copy mixture at every validation
    --eval-ws) EVAL_WS="$2"; shift 2;;   # extra eval windows, "short:long,..." in blocks; not part of the tag
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ "$ARM" == "baseline" ]] && P=0.0 && P_END=""

export MASK_P_START="$P"
export MASK_P_END="${P_END:-$P}"
export MASK_SEED="$SEED"
export INIT_SEED="$SEED"        # pairs the arms: same init, different objective

export COPY_MIX
export EVAL_WS_SWEEP="$EVAL_WS"

TAG="${ARM}_p${MASK_P_START}-${MASK_P_END}_seed${SEED}${STEPS:+_steps${STEPS}}"
[[ "$COPY_MIX" == "1" ]] && TAG="${TAG}_copymix"
[[ "$REP" != "1" ]] && TAG="${TAG}_rep${REP}"
OUT="${LOG_ROOT:-experiments/logs}/${TAG}"   # LOG_ROOT: remote launchers point this at persistent storage
mkdir -p "$OUT"
# DUMP_ROOT: set by launchers with persistent storage. The final per-token losses (about 50 MB)
# let any variant of the mixture be rescored later on CPU, without another training run.
[[ "$COPY_MIX" == "1" && -n "${DUMP_ROOT:-}" ]] && export COPY_MIX_DUMP="$DUMP_ROOT/$TAG"

# The step count is a module constant, so a step-removal test edits it into a scratch copy
# rather than mutating the tracked file.
SCRIPT=train_gpt.py
if [[ -n "$STEPS" ]]; then
  SCRIPT="${OUT}/train_gpt.py"
  sed "s/^    num_scheduled_iterations: int = .*/    num_scheduled_iterations: int = ${STEPS}/" train_gpt.py > "$SCRIPT"
  grep -n "num_scheduled_iterations: int" "$SCRIPT" | head -1
  # train_gpt.py opens and imports its kernel files relative to its own directory, so
  # they have to sit next to the scratch copy or it dies before the first import.
  cp triton_kernels.py dc_triton_kernels.py "$OUT/"
fi

{
  # GIT_COMMIT / GIT_DIRTY: set by launchers that ship the files without the .git directory
  echo "commit:  ${GIT_COMMIT:-$(git rev-parse HEAD)}"
  echo "dirty:   ${GIT_DIRTY:-$(git status --porcelain -- train_gpt.py | wc -l | tr -d ' ')} tracked change(s) to train_gpt.py"
  echo "arm:     $ARM"
  echo "p:       $MASK_P_START -> $MASK_P_END"
  echo "seed:    $SEED (init and mask)"
  echo "steps:   ${STEPS:-default}"
  echo "rep:     $REP"
  echo "gpus:    $GPUS"
  echo "copy_mix: $COPY_MIX"
  echo "eval_ws: ${EVAL_WS:-none}"
  # Which hardware the run landed on. Compile-cache hits have varied between containers, and
  # the GPU model, driver and CPU are the first suspects; this makes that checkable for free.
  echo "gpu_name: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "cpu:     $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)"
} | tee "$OUT/config.txt"

torchrun --standalone --nproc_per_node="$GPUS" "$SCRIPT" 2>&1 | tee "$OUT/train.log"

echo "final: $(grep -E 'val_loss:[0-9.]+' "$OUT/train.log" | tail -1)"

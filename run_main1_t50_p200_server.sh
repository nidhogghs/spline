#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/main1_t50_p200_n15_server.json}"
SEED_START="${SEED_START:-0}"
N_SEEDS="${N_SEEDS:-5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/main1_t50_p200_n15_server}"
LOG_ROOT="${LOG_ROOT:-logs/main1_t50_p200_n15_server}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"

RUN_LOG="$LOG_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
echo "[start] $(date)" | tee "$RUN_LOG"
echo "[python] $PYTHON_BIN" | tee -a "$RUN_LOG"
echo "[config] $CONFIG_PATH" | tee -a "$RUN_LOG"
echo "[seeds] start=$SEED_START count=$N_SEEDS" | tee -a "$RUN_LOG"
echo "[checkpoints] $CHECKPOINT_ROOT" | tee -a "$RUN_LOG"
echo "[logs] $LOG_ROOT" | tee -a "$RUN_LOG"
echo "[threads] OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS" | tee -a "$RUN_LOG"

for ((offset=0; offset<N_SEEDS; offset++)); do
  seed=$((SEED_START + offset))
  ckpt_dir="$CHECKPOINT_ROOT/seed${seed}"
  hist_json="$ckpt_dir/history.json"
  seed_log="$LOG_ROOT/seed${seed}.log"

  mkdir -p "$ckpt_dir"

  echo "[seed-start] seed=$seed $(date)" | tee -a "$RUN_LOG"
  "$PYTHON_BIN" -u main_1.py \
    --config "$CONFIG_PATH" \
    --checkpoint-dir "$ckpt_dir" \
    --history-json "$hist_json" \
    --seed-data "$seed" \
    2>&1 | tee "$seed_log"
  echo "[seed-done] seed=$seed $(date)" | tee -a "$RUN_LOG"
done

echo "[done] $(date)" | tee -a "$RUN_LOG"

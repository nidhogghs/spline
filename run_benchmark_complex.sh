#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
LOG_FILE="logs/benchmark_complex_t100_s30_three_$(date +%Y%m%d_%H%M%S).log"
CONFIG_PATH="${CONFIG_PATH:-configs/benchmark_three_simple.json}"
T_FINAL="${T_FINAL:-0}"
COEF_INTERVAL="${COEF_INTERVAL:-90,100}"
CLEAN_SEED_DIR="${CLEAN_SEED_DIR:--1}"

# Python selection:
# 1) use $PYTHON_BIN if provided
# 2) fallback to current shell python (works for venv)
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=/path/to/venv/bin/python bash run_benchmark_complex.sh"
  exit 1
fi

echo "[python] $("$PYTHON_BIN" -V 2>&1)"
echo "[config] $CONFIG_PATH"
echo "[override] T_FINAL=$T_FINAL CLEAN_SEED_DIR=$CLEAN_SEED_DIR COEF_INTERVAL=$COEF_INTERVAL"

echo "[start] $(date) log=$LOG_FILE"
"$PYTHON_BIN" benchmark_compare_three.py \
  --config "$CONFIG_PATH" \
  --t-final "$T_FINAL" \
  --clean-seed-dir "$CLEAN_SEED_DIR" \
  2>&1 | tee "$LOG_FILE"
"$PYTHON_BIN" benchmark_visualize_three.py \
  --config "$CONFIG_PATH" \
  --coef-interval "$COEF_INTERVAL" \
  2>&1 | tee -a "$LOG_FILE"
echo "[done] $(date)"

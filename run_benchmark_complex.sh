#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
LOG_FILE="logs/benchmark_complex_t100_s30_three_$(date +%Y%m%d_%H%M%S).log"

echo "[start] $(date) log=$LOG_FILE"
conda run -n work python benchmark_compare_three.py --config configs/benchmark_complex_t100_s30.json | tee "$LOG_FILE"
conda run -n work python benchmark_visualize_three.py --config configs/benchmark_complex_t100_s30.json | tee -a "$LOG_FILE"
echo "[done] $(date)"

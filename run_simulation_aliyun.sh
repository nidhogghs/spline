#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_CONFIG="${SUITE_CONFIG:-configs/simulation_suite_paper_v3_lowdim.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/paper_simulation_suite}"
SCENARIOS="${SCENARIOS:-}"

# 资源控制：
# CPU_MODE=auto: 根据 RESERVE_CPUS + CPU_UTILIZATION 自动计算 worker
# CPU_MODE=manual: 使用 MAX_WORKERS
CPU_MODE="${CPU_MODE:-auto}"
MAX_WORKERS="${MAX_WORKERS:-0}"
RESERVE_CPUS="${RESERVE_CPUS:-0}"
CPU_UTILIZATION="${CPU_UTILIZATION:-1.00}"
MAX_WORKERS_CAP="${MAX_WORKERS_CAP:-0}"
BLAS_THREADS="${BLAS_THREADS:-1}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/suite_$(date +%Y%m%d_%H%M%S).log"

echo "[start] $(date)" | tee "$LOG_FILE"
echo "[python] $PYTHON_BIN" | tee -a "$LOG_FILE"
echo "[suite] $SUITE_CONFIG" | tee -a "$LOG_FILE"
echo "[output] $OUTPUT_ROOT" | tee -a "$LOG_FILE"
echo "[resource] CPU_MODE=$CPU_MODE MAX_WORKERS=$MAX_WORKERS RESERVE_CPUS=$RESERVE_CPUS CPU_UTILIZATION=$CPU_UTILIZATION MAX_WORKERS_CAP=$MAX_WORKERS_CAP BLAS_THREADS=$BLAS_THREADS" | tee -a "$LOG_FILE"

"$PYTHON_BIN" simulation_suite_runner.py \
  --suite-config "$SUITE_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --cpu-mode "$CPU_MODE" \
  --max-workers "$MAX_WORKERS" \
  --reserve-cpus "$RESERVE_CPUS" \
  --cpu-utilization "$CPU_UTILIZATION" \
  --max-workers-cap "$MAX_WORKERS_CAP" \
  --blas-threads "$BLAS_THREADS" \
  --scenarios "$SCENARIOS" \
  --continue-on-error "$CONTINUE_ON_ERROR" \
  --dry-run "$DRY_RUN" \
  2>&1 | tee -a "$LOG_FILE"

echo "[done] $(date)" | tee -a "$LOG_FILE"

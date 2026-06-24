#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${COSCI_PYTHON:-/home/lyj/miniconda3/envs/coscientist/bin/python}"

ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(ts)] root=$ROOT"
echo "[$(ts)] python=$PY"
"$PY" -V
echo "[$(ts)] starting DeepSeek AML smoke bench"

"$PY" -u -m co_scientist.cli bench \
  --preset deepseek-aml-uplift \
  -c deepseek-session=openai_compatible:deepseek-v4-pro@session \
  -c deepseek-pipeline=openai_compatible:deepseek-v4-pro@pipeline \
  --candidate-universe data/bench/aml_smoke_candidates.txt \
  --n 1 \
  --matches 0 \
  --budget-per-candidate 0.75 \
  --judge-budget 0.05

echo "[$(ts)] bench command finished"

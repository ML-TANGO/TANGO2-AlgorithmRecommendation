#!/usr/bin/env bash
# Web demo. Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/demo.sh [port]   → http://127.0.0.1:<port> (default 8004)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"
CHECKPOINT="${AR_CHECKPOINT:-checkpoints/gemma4-lora-final}"
EXTRA=(); [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]] && EXTRA+=(--local-files-only)
$PY -m algorithm_recommend demo --checkpoint "$CHECKPOINT" --host "${HOST:-127.0.0.1}" --port "${1:-${DEMO_PORT:-8004}}" "${EXTRA[@]}"

#!/usr/bin/env bash
# Recommend from the trained checkpoint. Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh "<task description>" "<requirements>"
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh            # asks for both fields
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"
CHECKPOINT="${AR_CHECKPOINT:-checkpoints/gemma4-lora-final}"
EXTRA=(); [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]] && EXTRA+=(--local-files-only)
if [[ $# -ge 2 ]]; then
  $PY -m algorithm_recommend infer --checkpoint "$CHECKPOINT" --task-description "$1" --requirements "$2" "${EXTRA[@]}" "${@:3}"
else
  $PY -m algorithm_recommend infer --checkpoint "$CHECKPOINT" --interactive "${EXTRA[@]}" "$@"
fi

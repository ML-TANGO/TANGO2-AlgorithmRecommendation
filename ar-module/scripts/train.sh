#!/usr/bin/env bash
# LoRA SFT of the recommender. Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh [output_dir] [finetuning]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"
OUTPUT_DIR="${1:-artifacts/outputs/gemma4-lora}"
FINETUNING="${2:-lora}"
MODEL="${AR_BASE_MODEL:-google/gemma-4-E4B-it}"
EXTRA=(); [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]] && EXTRA+=(--local-files-only)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  $PY -m algorithm_recommend train --model "$MODEL" --dry-run --output-dir "artifacts/reports/tokenization" "${EXTRA[@]}"
  exit 0
fi
$PY -m algorithm_recommend train \
  --model "$MODEL" \
  --finetuning "$FINETUNING" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-3}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-4096}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --seed "${SEED:-42}" \
  "${EXTRA[@]}"

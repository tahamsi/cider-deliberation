#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
DEV_PER_DATASET="${DEV_PER_DATASET:-20}"
TEST_PER_DATASET="${TEST_PER_DATASET:-100}"
MISTRAL_SUBSET_PER_DATASET="${MISTRAL_SUBSET_PER_DATASET:-20}"
QWEN_SUBSET_PER_DATASET="${QWEN_SUBSET_PER_DATASET:-4}"
DATASETS="${DATASETS:-mmlu_pro gsm8k math500 truthfulqa_mc strategyqa medqa medmcqa aime2024}"

run() {
  printf '+ %s\n' "$*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ ! -x "$PYTHON_BIN" && "$DRY_RUN" != "1" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  echo "Run scripts/setup_a100_ubuntu.sh first, or set PYTHON_BIN." >&2
  exit 1
fi

run "$PYTHON_BIN" datasets/download_preprocess.py \
  --datasets $DATASETS \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED"

run "$PYTHON_BIN" experiments/prepare_v4_splits.py \
  --input "$OUTPUT_DIR/all.jsonl" \
  --out_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --dev_per_dataset "$DEV_PER_DATASET" \
  --test_per_dataset "$TEST_PER_DATASET"

run "$PYTHON_BIN" experiments/prepare_v6_subsets.py \
  --input "$OUTPUT_DIR/real_llm_v4_test_balanced.jsonl" \
  --out_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --mistral_per_dataset "$MISTRAL_SUBSET_PER_DATASET" \
  --qwen_per_dataset "$QWEN_SUBSET_PER_DATASET"

cat <<EOF

Dataset preparation complete.

Full balanced test split:
  $OUTPUT_DIR/real_llm_v4_test_balanced.jsonl

Smaller V6 subsets:
  $OUTPUT_DIR/real_llm_v6_mistral_160.jsonl
  $OUTPUT_DIR/real_llm_v6_qwen3_32.jsonl

Manifest files:
  $OUTPUT_DIR/real_llm_v4_split_manifest.json
  $OUTPUT_DIR/real_llm_v6_subset_manifest.json
EOF

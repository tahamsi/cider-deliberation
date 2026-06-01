#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
A100_MODELS="${A100_MODELS:-mistral:latest qwen3:32b qwen2.5:32b llama3.1:70b}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_OLLAMA_INSTALL="${SKIP_OLLAMA_INSTALL:-0}"
SKIP_MODEL_PULL="${SKIP_MODEL_PULL:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

run() {
  printf '+ %s\n' "$*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_shell() {
  printf '+ %s\n' "$*"
  if [[ "$DRY_RUN" != "1" ]]; then
    bash -lc "$*"
  fi
}

if [[ "$SKIP_GPU_CHECK" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not available. Install NVIDIA driver/CUDA before running the full campaign, or set SKIP_GPU_CHECK=1 for dry provisioning." >&2
    exit 1
  fi
  run nvidia-smi
fi

if [[ "$SKIP_APT" != "1" ]]; then
  run sudo apt-get update
  run sudo apt-get install -y \
    curl git ca-certificates build-essential pkg-config \
    python3 python3-venv python3-pip python3-dev
fi

if [[ "$SKIP_OLLAMA_INSTALL" != "1" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    run_shell "curl -fsSL https://ollama.com/install.sh | sh"
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  run_shell "sudo systemctl enable ollama || true"
  run_shell "sudo systemctl start ollama || true"
else
  echo "systemctl not available; start Ollama manually with: ollama serve"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if ! curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is not responding at http://${OLLAMA_HOST}. Start it with: ollama serve" >&2
    exit 1
  fi
fi

run "$PYTHON_BIN" -m venv "$VENV_DIR"
run "$VENV_DIR/bin/python" -m pip install -U pip wheel setuptools
run "$VENV_DIR/bin/python" -m pip install -r requirements.txt
run "$VENV_DIR/bin/python" -m pip install -e ".[test,stats]"
run "$VENV_DIR/bin/python" -m pip install reportlab

if [[ "$SKIP_MODEL_PULL" != "1" ]]; then
  for model in $A100_MODELS; do
    run ollama pull "$model"
  done
fi

cat <<EOF

A100 environment setup complete.

Root:        $ROOT_DIR
Venv:        $ROOT_DIR/$VENV_DIR
Ollama host: http://$OLLAMA_HOST
Models:      $A100_MODELS

Next:
  source "$VENV_DIR/bin/activate"
  scripts/download_preprocess_all.sh
  python scripts/run_a100_full_campaign.py --out_dir outputs/a100_full_v7
EOF

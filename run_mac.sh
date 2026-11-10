#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================================"
echo " Starting Qwen3-TTS for macOS (Apple Silicon / CPU)"
echo "========================================================"

# Enable MPS kernel fallback to CPU for unsupported Metal operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Prefer uv if available, fallback to python3 venv
if command -v uv >/dev/null 2>&1; then
    echo "[+] uv detected."
    if [ ! -d ".venv" ]; then
        echo "[+] Initializing Python 3.12 virtual environment..."
        uv venv .venv --python 3.12
    fi
    source .venv/bin/activate
    echo "[+] Checking/installing editable package dependencies..."
    uv pip install -e .
else
    echo "[!] uv not found in PATH, using python3..."
    if [ ! -d ".venv" ]; then
        echo "[+] Initializing Python venv..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    echo "[+] Checking/installing package dependencies..."
    pip install -e .
fi

echo "[+] Launching Unified Web UI..."
python app.py "$@"

#!/usr/bin/env bash
# GPU evaluation venv for the ExecuTorch 1.3.1 quantizer: torch 2.12.0 CUDA 13
# wheel + the same executorch / torchao / tosa-tools / vela versions kit 26.06
# pins. Separate from the kit venv (which stays on the kit's +cpu pins).
# A constraints file keeps pip from drifting any torch-family version while
# resolving executorch's dependencies.
# Usage: python_env/build_venv_gpu.sh <venv dir>     e.g. ~/venv_et131_gpu
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$1"
CU=https://download.pytorch.org/whl/cu130
if [ -e "$VENV" ]; then echo "refusing: $VENV exists"; exit 1; fi
echo "== host $(hostname) start $(date '+%F %T')"
python3.12 -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install -q -U pip
CONS="$VENV/constraints.txt"
printf 'torch==2.12.0+cu130\ntorchao==0.17.0+cu130\ntorchaudio==2.11.0+cu130\n' > "$CONS"
export PIP_CONSTRAINT="$CONS"
"$PIP" install --index-url "$CU" torch==2.12.0+cu130 torchao==0.17.0+cu130
"$PIP" install --extra-index-url "$CU" executorch==1.3.1 tosa-tools==2026.2.1 ethos-u-vela==5.1.0
"$PIP" install --no-deps --index-url "$CU" torchaudio==2.11.0+cu130 || echo "torchaudio cu130 install failed (non-fatal)"
"$PIP" install --extra-index-url "$CU" -r "$HERE/requirements-gpu.txt"
echo "== check"
"$VENV/bin/python3" "$HERE/check_env.py" --cuda "$HERE/requirements-gpu.txt"
echo "== done $(date +%T)"

#!/usr/bin/env bash
# GPU evaluation venv for the ExecuTorch 1.3.1 quantizer: torch 2.12.0 CUDA 13
# wheel + the same executorch / torchao / tosa-tools / vela versions kit 26.06
# pins. Separate from the kit venv (which stays on the kit's +cpu pins) and
# from the 25.12 receipts' environment. A constraints file keeps pip from
# drifting any torch-family version while resolving executorch's dependencies.
set -euo pipefail
VENV="$HOME/venv_et131_gpu"
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
echo "== versions"
"$VENV/bin/python3" - <<'PY'
import importlib.metadata as md, torch
for p in ("torch","executorch","torchao","tosa-tools","ethos-u-vela","torchaudio"):
    try: print(p, md.version(p))
    except Exception: print(p, "MISSING")
print("cuda", torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import executorch.backends.arm.ethosu, executorch.backends.arm.quantizer; print("executorch arm backend import ok")
PY
echo "== done $(date +%T)"

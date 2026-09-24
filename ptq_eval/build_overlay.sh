#!/usr/bin/env bash
# Install the model / metric packages that must shadow the venv's versions (see requirements-overlay.txt).
# Usage: PTQ_PY=~/venv_et131_gpu/bin/python3 ./build_overlay.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
"$PY" -m pip install --no-deps --target "$HERE/ovl" -r "$HERE/requirements-overlay.txt"
echo "overlay ready: export PYTHONPATH=$HERE/ovl:$HERE"

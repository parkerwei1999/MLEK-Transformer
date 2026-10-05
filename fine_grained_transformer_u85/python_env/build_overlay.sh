#!/usr/bin/env bash
# Install the model / metric packages that must shadow the venv's versions (see requirements-overlay.txt)
# into fine_grained_transformer_u85/ovl, which goes FIRST on PYTHONPATH.
# Usage: python_env/build_overlay.sh <venv python>     e.g. ~/venv_et131_gpu/bin/python3
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$1"
"$PY" -m pip install --no-deps --target "$HERE/../ovl" -r "$HERE/requirements-overlay.txt"
PYTHONPATH="$HERE/../ovl" "$PY" "$HERE/check_env.py" "$HERE/requirements-overlay.txt"
echo "overlay ready: export PYTHONPATH=$(cd "$HERE/.." && pwd)/ovl:$(cd "$HERE/.." && pwd)"

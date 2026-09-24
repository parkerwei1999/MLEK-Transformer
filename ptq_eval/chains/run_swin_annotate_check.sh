#!/usr/bin/env bash
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S="$PTQ_EVAL"
OVL="$PTQ_EVAL/ovl"
OUT="$PTQ_EVAL/annotate_check"
export PYTHONPATH="$OVL"
cd "$S"
echo "== ExecuTorch 1.0.0 (25.12 kit venv) $(date +%T)"
PATH="$HOME/ml-embedded-evaluation-kit/resources_downloaded/env/bin:$PATH" "$HOME/ml-embedded-evaluation-kit/resources_downloaded/env/bin/python3" swin_annotate_check.py --out "$OUT/et100_a8w8.json" 2>&1 | grep -E "^==|^  |Traceback|Error"
echo "== ExecuTorch 1.3.1 (GPU venv, cpu device) $(date +%T)"
PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH" "${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}" swin_annotate_check.py --out "$OUT/et131_a8w8.json" 2>&1 | grep -E "^==|^  |Traceback|Error"
echo "== done $(date +%T)"

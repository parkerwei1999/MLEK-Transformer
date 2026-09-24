#!/usr/bin/env bash
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
echo "== whisper a16w8 stock (reference) $(date +%T)"
"$PY" lower_probe.py --model whisper-tiny-encoder --quant-config a16w8 --out-dir "$W/lower_a16_stock" 2>&1 | grep -E "^===|^  \[|Traceback"
echo "== whisper a16w8 + LN int32 internals $(date +%T)"
"$PY" lower_probe.py --model whisper-tiny-encoder --quant-config a16w8 --prec-rules "$LN32" --out-dir "$W/lower_a16_ln32" 2>&1 | grep -E "^===|^  \[|Traceback"
echo "== deit-t a8w8 + LN int32 internals $(date +%T)"
"$PY" lower_probe.py --model deit_tiny --quant-config a8w8 --prec-rules 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8' --out-dir "$W/lower_deit_ln32" 2>&1 | grep -E "^===|^  \[|Traceback"
echo "== done $(date +%T)"

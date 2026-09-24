#!/usr/bin/env bash
# Whisper: LayerNorm internals int32, rest a8w8; waits for the W1-W4 chain to finish first.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_rules.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
echo "== whisper a8w8_ln32 $(date +%T)"
"$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_a8w8_ln32" --stages int8 --n-cal 200 --n-eval 200 \
    --quant-config a8w8 --prec-rules 'layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8' 2>&1 \
    | grep -E "rewritten|200/200|Traceback|Error"
echo "== done $(date +%T)"

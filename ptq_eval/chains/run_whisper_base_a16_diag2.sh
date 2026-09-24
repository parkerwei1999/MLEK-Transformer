#!/usr/bin/env bash
# Whisper-base: int16 scale floor 2^-16 instead of 2^-12 (with LN int32 internals). Waits for the first diag chain.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_base_a16_diag.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8e16'
echo "== whisper-base a16e16_ln32 $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$W/et131_wb_a16e16_ln32" --stages int8 --n-cal 200 --n-eval 200 \
    --quant-config a16w8e16 --prec-rules "$LN32" 2>&1 | grep -E "200/200|Traceback|Error"
echo "== done $(date +%T)"

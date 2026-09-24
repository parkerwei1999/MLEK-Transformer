#!/usr/bin/env bash
# Whisper-small / base: a16w8 + LN int32 internals with MinMax observers (histogram clipping was the base's 1.5 pt). Waits for diag2.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_base_a16_diag2.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== $1 $2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id "openai/$1" --out-dir "$W/et131_${1}_$2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 "${@:3}" 2>&1 | grep -E "200/200|Traceback|Error"; }
run whisper-small a16_ln32_minmax  --act-observer minmax --prec-rules "$LN32"
run whisper-tiny  a16_ln32_minmax  --act-observer minmax --prec-rules "$LN32"
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Whisper-tiny: which int16 piece of the LayerNorm breaks it? (a) int16 LN with MinMax observers, (b) LN fully int32, (c) only rsqrt int32.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() {  # tag, rules
  echo "== whisper $1 [$2] $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_$1" --stages int8 --n-cal 200 --n-eval 200 \
      --quant-config a8w8 --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"
}
run ln16_minmax 'layer_norm$=a16w8:minmax'
run ln32_all    'layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,rsqrt,reshape,view;layer_norm$=a16w8'
run ln16_rsqrt32 'layer_norm$=a32w8@rsqrt;layer_norm$=a16w8'
echo "== done $(date +%T)"

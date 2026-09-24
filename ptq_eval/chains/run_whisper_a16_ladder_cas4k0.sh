#!/usr/bin/env bash
# a16w8 regime (no int8 softmax chaos): LN int32 internals vs LN int16 MinMax, via rules. Runs on cas4k0.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() { echo "== whisper $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w3_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 "${@:2}" 2>&1 \
      | grep -E "200/200|Traceback|Error"; }
run a16_rule_ln32_int  --prec-rules 'layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run a16_rule_ln16_minmax --prec-rules 'layer_norm$=a16w8:minmax'
run a16_stock_rulepath --prec-rules 'layer_norm$=a16w8'
echo "== done $(date +%T)"

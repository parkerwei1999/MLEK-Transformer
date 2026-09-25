#!/usr/bin/env bash
# Overnight (casada, RTX 4090): whisper-medium — stock a16w8 (MinMax), deploy v4 (int16 residual) with Newton 0 / 1, dual-range rsqrt, dual + Newton1.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,clamp;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-medium $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-medium --out-dir "$OUT/pcrw/wmedium_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax "${@:2}" 2>&1 | grep -E "dual-range|NewtonLayerNorm|200/200|Traceback|Error|out of memory"; }
run stock_a16w8_mm
run v4_newton0 --prec-rules "$INT8;$LN32" --ln-newton-steps 0
run v4_newton1 --prec-rules "$INT8;$LN32" --ln-newton-steps 1
run v4_dual_newton0 --prec-rules "$INT8;$LN32" --ln-dual-q 0.9 --ln-newton-steps 0
run v4_dual_newton1 --prec-rules "$INT8;$LN32" --ln-dual-q 0.9 --ln-newton-steps 1
echo "== done $(date +%T)"

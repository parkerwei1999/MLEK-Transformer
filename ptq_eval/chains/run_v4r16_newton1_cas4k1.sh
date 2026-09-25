#!/usr/bin/env bash
# Whisper tiny/base/small at 200u with the full-set recipe (int8 demotions + int16 per-tensor
# residual + LN-i32 + Newton-1): gives Newton-1 vs dual k=8 at the same residual format,
# and the 200u counterpart of the 2620u numbers. Gated behind the dual+PCS chain.
set -uo pipefail
until grep -q "^== done" "$OUT/run_dual_pcs_cas4k1.log" 2>/dev/null; do sleep 300; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-$1 v4r16_newton1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$1 --out-dir "$OUT/pcrw/w$1_v4r16_newton1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;$LN32" --ln-newton-steps 1 2>&1 | grep -E "NewtonLayerNorm|200/200|Traceback|Error"; }
run tiny; run base; run small
echo "== done $(date +%T)"

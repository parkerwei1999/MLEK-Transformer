#!/usr/bin/env bash
# Dual-range rsqrt v2: c = 4 x median per-token variance, fine path with eps 2^-16 observers. Whisper small/tiny/base, deploy v4 structure.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"; export LN_DUAL_K=8
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-$1 $2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$1 --out-dir "$OUT/pcrw/w$1_$2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;$LNR" "${@:3}" 2>&1 | grep -E "dual-range|NewtonLayerNorm|200/200|Traceback|Error"; }
run small dualk8_newton0 --ln-dual-k 8 --ln-newton-steps 0
run small dualk8_newton1 --ln-dual-k 8 --ln-newton-steps 1
run tiny  dualk8_newton0 --ln-dual-k 8 --ln-newton-steps 0
run base  dualk8_newton0 --ln-dual-k 8 --ln-newton-steps 0
echo "== done $(date +%T)"

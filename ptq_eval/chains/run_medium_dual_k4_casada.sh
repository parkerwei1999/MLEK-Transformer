#!/usr/bin/env bash
# whisper-medium dual-range v2 (c = 4 x median, e16 fine path), Newton 0 / 1. Gated on the first medium chain.
set -uo pipefail
until grep -q "^== done" "$OUT/run_overnight_medium_casada.log" 2>/dev/null; do sleep 300; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"; export LN_DUAL_K=8
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
for n in 0 1; do echo "== whisper-medium v4_dualk8_newton$n $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-medium --out-dir "$OUT/pcrw/wmedium_v4_dualk8_newton$n" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;$LNR" --ln-dual-k 8 --ln-newton-steps $n 2>&1 | grep -E "dual-range|NewtonLayerNorm|200/200|Traceback|Error|memory"; done
echo "== done $(date +%T)"

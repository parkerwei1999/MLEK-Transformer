#!/usr/bin/env bash
# whisper-large-v3, 100 dev-clean cal / 100 test-clean eval, after the tokenizer fix
# (num_languages=100 for the 51866-token vocab). fp32 reference first, then the cell that
# decides the size-scaling story (dual k=8), then Newton-1 and the stock a16w8 baseline.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"; export LN_DUAL_K=8
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-large-v3 $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-large-v3 --out-dir "$OUT/pcrw/wlarge2_$1" --n-cal 100 --n-eval 100 --quant-config a16w8 --act-observer minmax "${@:2}" 2>&1 | grep -E "dual-range|NewtonLayerNorm|100/100|Traceback|Error|out of memory"; }
run fp32_static --stages fp32-static
run v4_dualk8_newton0 --stages int8 --prec-rules "$INT8;$LNR" --ln-dual-k 8 --ln-newton-steps 0
run v4_newton1 --stages int8 --prec-rules "$INT8;$LN32" --ln-newton-steps 1
run stock_a16w8_mm --stages int8
echo "== done $(date +%T)"

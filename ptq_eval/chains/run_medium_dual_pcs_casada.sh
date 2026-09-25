#!/usr/bin/env bash
# whisper-medium: dual k=8 LayerNorm + PCS residual (like-for-like with the deploy-v4 cells);
# waits for the large-v3 chain so the two large fake-quant models do not share the GPU.
set -uo pipefail
until grep -q "^== done" "$OUT/run_large2_casada.log" 2>/dev/null; do sleep 600; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"; export LN_DUAL_K=8
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
PCS='layers\.\d+$=a16inpc8symout@add'
echo "== whisper-medium dualk8pcs_newton0 $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-medium --out-dir "$OUT/pcrw/wmedium_dualk8pcs_newton0" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;$PCS;$LNR" --ln-dual-k 8 --ln-newton-steps 0 2>&1 | grep -E "dual-range|NewtonLayerNorm|200/200|Traceback|Error|out of memory"
echo "== done $(date +%T)"

#!/usr/bin/env bash
# whisper-small dual q=0.9 Newton0 scored 6.99% (worse than plain LN-i32 4.21): dump both rsqrt grids + fp32 var stats per LN.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,clamp;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
echo "== whisper-small dual q0.9 diag $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$OUT/pcrw/wsmall_dual_diag" --stages int8 --n-cal 200 --n-eval 20 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;$LN32" --ln-dual-q 0.9 --ln-newton-steps 0 --dump-ln-scales 2>&1 | grep -E "dual-range|NewtonLayerNorm|20/20|Traceback|Error|ln_scales" | cut -c1-160
echo "== done $(date +%T)"

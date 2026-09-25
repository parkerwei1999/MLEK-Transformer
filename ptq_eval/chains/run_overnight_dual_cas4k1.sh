#!/usr/bin/env bash
# Overnight (cas4k1): dual-range rsqrt LayerNorm on Whisper. Base structure = deploy v4 with int16 residual
# (int8 conv/GELU/proj/logit/fc1, fc2 int16, residual int16, LN-i32). Reference: small Newton1 2.29 / Newton0 4.21, tiny 6.59/6.01, base 4.13/3.98.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,clamp;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-$1 $2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$1 --out-dir "$OUT/pcrw/w$1_$2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;$LN32" "${@:3}" 2>&1 | grep -E "dual-range|NewtonLayerNorm|200/200|Traceback|Error"; }
run small dual_newton0 --ln-dual-q 0.9 --ln-newton-steps 0
run small dual_newton1 --ln-dual-q 0.9 --ln-newton-steps 1
run tiny  dual_newton0 --ln-dual-q 0.9 --ln-newton-steps 0
run base  dual_newton0 --ln-dual-q 0.9 --ln-newton-steps 0
run small dual_q95_newton0 --ln-dual-q 0.95 --ln-newton-steps 0
echo "== done $(date +%T)"

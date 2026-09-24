#!/usr/bin/env bash
# Whisper mixed v2: a16w8 + LN-i32 + MinMax base with conv+front GELU int8, projections int8, logits int8, fc1 int8, fc2 output PTF int8 (input int16 = GELU out), residual PTF.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
MIX2="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8;fc2\$=a16inptf8out;layers\\.\\d+\$=a16inptf8out@add;$LN32"
run() { echo "== $1 $2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id "openai/$1" --out-dir "$W/et131_${1}_$2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$3" 2>&1 | grep -E "200/200|Traceback|Error"; }
run whisper-tiny  mixed_v2 "$MIX2"
run whisper-base  mixed_v2 "$MIX2"
run whisper-small mixed_v2 "$MIX2"
echo "== done $(date +%T)"

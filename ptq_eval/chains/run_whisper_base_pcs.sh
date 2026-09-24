#!/usr/bin/env bash
# whisper-base: fc2 output int8 per-channel MinMax (unconstrained, "PCS") = 4.32% while PTF (power-of-two scales) = 71.6%.
# Deployable candidate with unconstrained per-channel everywhere per-channel is used: residual add output pc8 instead of PTF.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
PRE="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-base $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$W/et131_wb_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run pcs_fc2_res "$PRE;fc2\$=a16inpc8out;layers\\.\\d+\$=a16inpc8out@add;$LN32"
run pcs_res_only "$PRE;layers\\.\\d+\$=a16inpc8out@add;$LN32"
echo "== done $(date +%T)"

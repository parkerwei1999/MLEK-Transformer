#!/usr/bin/env bash
# whisper-base fc2 OUTPUT per-channel int8 variants on top of mixed_final (4.47%): PTF alpha<=12 collapsed (71.6%).
#  fc2ptf_a16 : PTF with alpha clamp 16 (no clipping possible)  -> if still collapsed, clipping is not the cause
#  fc2pc      : unconstrained per-channel MinMax int8 (a16inpc8out) -> if collapsed, any per-channel int8 grid fails here
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
PRE="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8"
run() { echo "== whisper-base $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$W/et131_wb_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
PTF_MAX_ALPHA=16 run fc2ptf_a16 "$PRE;fc2\$=a16inptf8out;layers\\.\\d+\$=a16inptf8out@add;$LN32"
run fc2pc "$PRE;fc2\$=a16inpc8out;layers\\.\\d+\$=a16inptf8out@add;$LN32"
echo "== done $(date +%T)"

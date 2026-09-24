#!/usr/bin/env bash
# whisper-base deployable candidate (mixed_final + fc1 int8 = 4.81%) with the fc2 OUTPUT as int16 PER-CHANNEL MinMax.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
PRE="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
echo "== whisper-base fc1i8_fc2pc16 $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$W/et131_wb_fc1i8_fc2pc16" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$PRE;fc2\$=a16inpc16out;layers\\.\\d+\$=a16inptf8out@add;$LN32" 2>&1 | grep -E "200/200|Traceback|Error"
echo "== done $(date +%T)"

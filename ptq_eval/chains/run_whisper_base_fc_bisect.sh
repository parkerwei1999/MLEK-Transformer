#!/usr/bin/env bash
# Bisect the whisper-base mixed_v2 collapse (68.5% WER, 28 loops): mixed_final (4.47%) + fc1 int8 only / + fc2 PTF-out only.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
MIX="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;layers\\.\\d+\$=a16inptf8out@add;$LN32"
MIX_FC1="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8;layers\\.\\d+\$=a16inptf8out@add;$LN32"
MIX_FC2PTF="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc2\$=a16inptf8out;layers\\.\\d+\$=a16inptf8out@add;$LN32"
run() { echo "== $1 $2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id "openai/$1" --out-dir "$W/et131_${1}_$2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$3" 2>&1 | grep -E "200/200|Traceback|Error"; }
run whisper-base mixed_fc1i8   "$MIX_FC1"
run whisper-base mixed_fc2ptf  "$MIX_FC2PTF"
echo "== done $(date +%T)"

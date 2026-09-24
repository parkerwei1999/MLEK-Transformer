#!/usr/bin/env bash
# whisper-base deploy v4 (int8 conv/GELU/proj/logit/fc1, fc2 int16, SYMMETRIC PCS residual, LN-i32 + Newton) with the
# per-channel lowering rewrite applied in fake-quant: does the lowered form keep 4.06%?
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-base $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$PTQ_EVAL/chains/pcrw/wb_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;layers\\.\\d+\$=a16inpc8symout@add;$LN32" --ln-newton-steps 1 $2 2>&1 | grep -E "pc_rewrite|200/200|Traceback|Error"; }
run deploy_v4_newton1 ""
run deploy_v4_newton1_pcrw "--pc-rewrite"
echo "== done $(date +%T)"

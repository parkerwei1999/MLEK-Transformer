#!/usr/bin/env bash
# Whisper-tiny on the a16w8 + LN-W32 + MinMax base: (a) conv1/conv2 AND their front-end GELUs at int8; (b) only k_proj/v_proj at int8 (what a KV cache would hold).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w11_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a16_convgelu8 "conv[12]\$=a8w8;encoder\$=a8w8@gelu;$LN32"
run a16_kv8       "(k_proj|v_proj)\$=a8w8;$LN32"
run a16_kv8_out8  "(k_proj|v_proj|out_proj)\$=a8w8;$LN32"
echo "== done $(date +%T)"

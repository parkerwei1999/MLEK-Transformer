#!/usr/bin/env bash
# DeiT-T / DeiT-S with the bridged recipe: residual PTF int8 (int16 adder inputs) + LN-W32, everything else per-tensor int8.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=90
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== $1 rules[$2] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_deitptf_$i" --model-name "$1" --quant-configs a8w8 --act-observers histogram \
      --keep-fp32-sets none --prec-rules "$2" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run deit_tiny_patch16_224  "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
run deit_small_patch16_224 "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
run deit_small_patch16_224 "blocks\\.\\d+\$=a16w8@add;$VLN32"
echo "== done $(date +%T)"

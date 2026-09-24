#!/usr/bin/env bash
# DeiT-B demotion ladder from a16w8 + LN-i32 (81.1 / fp32 82.2): one family to per-tensor int8 at a time.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=110
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== deit_base rules[$1] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_deitb_dem_$i" --model-name deit_base_patch16_224 --quant-configs a16w8 --act-observers minmax \
      --keep-fp32-sets none --prec-rules "$1" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run "$VLN32"
run "attn\\.(qkv|proj)\$=a8w8;$VLN32"
run "mlp\\.(fc1|fc2)\$=a8w8;$VLN32"
run "attn\$=a8w8;$VLN32"
run "mlp\\.act\$=a8w8;$VLN32"
run "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
run "blocks\\.\\d+\$=a8w8;$VLN32"
echo "== done $(date +%T)"

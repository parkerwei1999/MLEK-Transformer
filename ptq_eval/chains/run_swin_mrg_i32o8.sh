#!/usr/bin/env bash
# Swin-T bridged with the minimal merge token mrg:ptf8i32o8 (slices/cat read PTF via int16 expansion, merge LN i32, reduction output int8),
# and the no-merge-rule control (slices/cat per-tensor int8 = requant leakage).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=150
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== swin_tiny rules[$1] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_mrg_$i" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
      --keep-fp32-sets none --mask-aware --prec-rules "$1" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run "blocks\\.\\d+\$=a16inptf8out@add;layers\\.2\\.downsample\\.norm\$=a32w8@sub,mul,sum.dim_IntList,add;layers\\.2\\.downsample\\.reduction\$=a16in8out;layers\\.2\\.downsample=a16w8;$VLN32"
run "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
echo "== done $(date +%T)"

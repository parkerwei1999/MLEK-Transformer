#!/usr/bin/env bash
# Complete the MinMax table: int16 per-tensor residual (instead of PTF) on the a8w8 + LN-i32 base, DeiT-B/S and Swin-T.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_minmax_recipe.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=140
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== $1 rules[$2] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_mmres16_$i" --model-name "$1" --quant-configs a8w8 --act-observers minmax \
      --keep-fp32-sets none --mask-aware --prec-rules "$2" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run deit_base_patch16_224  "blocks\\.\\d+\$=a16w8@add;$VLN32"
run deit_small_patch16_224 "blocks\\.\\d+\$=a16w8@add;$VLN32"
run swin_tiny_patch4_window7_224 "blocks\\.\\d+\$=a16w8@add;layers\\.2\\.downsample=a16w8;$VLN32"
run deit_base_patch16_224  "blocks\\.\\d+\$=a16in8out@add;$VLN32"
echo "== done $(date +%T)"

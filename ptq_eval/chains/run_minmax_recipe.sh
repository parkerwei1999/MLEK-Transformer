#!/usr/bin/env bash
# Bridged a8w8 recipe with MinMax observers everywhere (the histogram clipping hurt DeiT-B even at int16): DeiT-S/B, Swin-T; plus DeiT-S a16w8 MinMax reference.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=130
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== $1 qc=$2 obs=$3 rules[$4] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_mmrecipe_$i" --model-name "$1" --quant-configs "$2" --act-observers "$3" \
      --keep-fp32-sets none --mask-aware --prec-rules "$4" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run deit_base_patch16_224  a8w8 minmax "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
run deit_small_patch16_224 a8w8 minmax "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
run deit_small_patch16_224 a16w8 minmax "$VLN32"
run swin_tiny_patch4_window7_224 a8w8 minmax "blocks\\.\\d+\$=a16inptf8out@add;layers\\.2\\.downsample=a16w8;$VLN32"
run swin_tiny_patch4_window7_224 a16w8 minmax "$VLN32"
run deit_tiny_patch16_224  a8w8 minmax "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
echo "== done $(date +%T)"

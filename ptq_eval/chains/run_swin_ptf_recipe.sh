#!/usr/bin/env bash
# The FQ-ViT recipe inside the ExecuTorch flow: a8w8 everywhere except residual stored PTF int8, the stage-3->4 PatchMerging reading it
# losslessly (int16), LN int32 internals. Swin-T seed 1, Swin-S, Swin-B; plus the variant with ALL PatchMergings int16.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=80
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
RECIPE="blocks\\.\\d+\$=a16inptf8out@add;layers\\.2\\.downsample=a16w8;$VLN32"
run() { i=$((i+1)); echo "== $1 seed=$3 rules[$2] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_recipe_$i" --model-name "$1" --quant-configs a8w8 --act-observers histogram \
      --keep-fp32-sets none --mask-aware --prec-rules "$2" --seed "$3" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run swin_tiny_patch4_window7_224  "$RECIPE" 1
run swin_small_patch4_window7_224 "$RECIPE" 0
run swin_base_patch4_window7_224  "$RECIPE" 0
run swin_tiny_patch4_window7_224  "blocks\\.\\d+\$=a16inptf8out@add;downsample=a16w8;$VLN32" 0
echo "== done $(date +%T)"

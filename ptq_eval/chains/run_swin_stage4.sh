#!/usr/bin/env bash
# Swin-T: the fp32 divergence appears at the stage 3->4 PatchMerging; how much of stage 4 / the merge must be int16?
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=0
for rules in \
  'layers\.2\.downsample=a16w8;norm\d?$=fp32' \
  'layers\.2\.downsample=a16w8;layers\.3\.=a16w8;norm\d?$=fp32' \
  'layers\.3\..*norm\d?$=fp32;layers\.2\.downsample\.norm$=fp32;layers\.2\.downsample=a16w8;layers\.3\.=a16w8;norm\d?$=fp32' \
  'layers\.[23]\.=a16w8;norm\d?$=fp32' ; do
  i=$((i+1)); echo "== swin rules[$rules] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_stage4_$i" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --mask-aware --prec-rules "$rules" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
echo "== done $(date +%T)"

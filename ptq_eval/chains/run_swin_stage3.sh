#!/usr/bin/env bash
# Swin-T: stage 3 (layers.2) is where massive activations grow; which part of it must be int16? plus a seed-1 repeat of the 77.3 cell.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=0
run() { i=$((i+1)); echo "== swin rules[$1] seed=$2 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_stage3_$i" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --mask-aware --prec-rules "$1" --seed "$2" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run 'layers\.2\.=a16w8;norm\d?$=fp32' 0
run 'layers\.2\..*norm\d?$=fp32;layers\.2\.=a16w8;norm\d?$=fp32' 0
run 'layers\.2\..*(attn|mlp)=a16w8;norm\d?$=fp32' 0
run 'layers\.2\..*mlp=a16w8;norm\d?$=fp32' 0
run 'layers\.[23]\.=a16w8;norm\d?$=fp32' 1
run 'norm\d?$=fp32' 1
echo "== done $(date +%T)"

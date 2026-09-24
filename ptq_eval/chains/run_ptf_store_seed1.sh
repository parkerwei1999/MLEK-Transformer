#!/usr/bin/env bash
# Seed-1 repeat of the residual-storage cells (int16 / int8 per-tensor / PTF int8 per-channel) for DeiT-B and Swin-T stage 3.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=20
LN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== $1 rules[$2] seed=1 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_ptf_$i" --model-name "$1" --quant-configs a8w8 --act-observers histogram \
      --keep-fp32-sets none --mask-aware --prec-rules "$2" --seed 1 --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
for cfg in a16w8 a16in8out a16inptf8out; do run deit_base_patch16_224 "blocks\\.\\d+\$=$cfg@add;$LN32"; done
for cfg in a16w8 a16in8out a16inptf8out; do run swin_tiny_patch4_window7_224 "layers\\.2\\.blocks\\.\\d+\$=$cfg@add;layers\\.2\\.=a16w8;norm\\d?\$=fp32"; done
echo "== done $(date +%T)"

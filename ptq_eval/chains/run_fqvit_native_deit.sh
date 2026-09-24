#!/usr/bin/env bash
# FQ-ViT's own PTQ on DeiT-T/S/B (its first-1000 subset): fp32 vs --ptf minmax (no LIS), for the "gap vs FQ-ViT" question.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
export PYTHONPATH="$PTQ_EVAL/ovl"
cd "$HOME/tvm_andersen/3rdparty/FQ-ViT"
for m in deit_tiny deit_small deit_base; do for mode in "" "--quant --ptf --quant-method minmax"; do
  echo "== $m [$mode] $(date +%T)"
  "$PY" test_quant.py $m /home/shared/ImageNet $mode --num-images 1000 --calib-batchsize 100 --calib-iter 10 --val-batchsize 100 2>&1 \
      | grep -v -i "warn\|register_constant" | grep -E "^ \* Prec@1|Traceback|Error" | tail -1
done; done
echo "== done $(date +%T)"

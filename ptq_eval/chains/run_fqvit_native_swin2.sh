#!/usr/bin/env bash
# FQ-ViT's own PTQ on Swin-T without LIS (fake 8-bit softmax), minmax and percentile; plus PTF off for reference.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
export PYTHONPATH="$PTQ_EVAL/ovl"
cd "$HOME/tvm_andersen/3rdparty/FQ-ViT"
for mode in "--quant --ptf --quant-method minmax" "--quant --ptf --quant-method percentile" "--quant --quant-method minmax" "--quant --lis --quant-method minmax"; do
  echo "== swin_tiny [$mode] $(date +%T)"
  "$PY" test_quant.py swin_tiny /home/shared/ImageNet $mode --num-images 1000 --calib-batchsize 100 --calib-iter 10 --val-batchsize 100 2>&1 \
      | grep -v -i "warn\|register_constant" | grep -E "^ \* Prec@1|Traceback|Error" | tail -2
done
echo "== done $(date +%T)"

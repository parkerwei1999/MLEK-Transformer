#!/usr/bin/env bash
# FQ-ViT's own PTQ (test_quant.py) on Swin-T: fp32 and --quant --ptf --lis --quant-method minmax, 1000 val images, in the torch-2.12 venv.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
export PYTHONPATH="$PTQ_EVAL/ovl"
cd "$HOME/tvm_andersen/3rdparty/FQ-ViT"
for mode in "" "--quant --ptf --lis --quant-method minmax" "--quant --ptf --lis --quant-method percentile"; do
  echo "== swin_tiny [$mode] $(date +%T)"
  "$PY" test_quant.py swin_tiny /home/shared/ImageNet $mode --num-images 1000 --calib-batchsize 100 --calib-iter 10 --val-batchsize 100 2>&1 \
      | grep -v -i "warn\|register_constant" | grep -E "Prec@1|prec1|Acc@1|top1|Traceback|Error|acc" | tail -3
done
echo "== done $(date +%T)"

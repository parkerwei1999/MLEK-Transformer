#!/usr/bin/env bash
# FQ-ViT native (fork test_quant.py, official 888 = PTF + LIS, MinMax) on the FULL 50k val, same venv, as the credible baseline.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$PTQ_EVAL/ovl"
cd "${FQVIT_DIR:-$HOME/tvm_andersen/3rdparty/FQ-ViT}"  # FQ-ViT checkout at the commit pinned in fqvit_models/VERSION.md (test_quant.py is not vendored)
for m in deit_tiny deit_small deit_base swin_tiny swin_small swin_base; do
  for mode in "" "--quant --ptf --lis --quant-method minmax"; do
    echo "== $m [$mode] $(date +%T)"
    "$PY" test_quant.py $m /home/shared/ImageNet $mode --num-images 50000 --calib-batchsize 100 --calib-iter 10 --val-batchsize 100 2>&1 | grep -E "Acc@1|Prec@1|\* |Traceback|Error" | tail -3
  done
done
echo "== done $(date +%T)"

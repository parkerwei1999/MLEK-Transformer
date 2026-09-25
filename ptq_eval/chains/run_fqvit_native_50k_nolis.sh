#!/usr/bin/env bash
# The fork's canonical FQ-ViT reproduction (README: 80.472 Swin-T = upstream) is PTF + MinMax WITHOUT LIS. Rerun all six on full 50k.
set -uo pipefail
until grep -q "^== done" "$OUT/run_fqvit_native_50k.log" 2>/dev/null; do sleep 120; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$PTQ_EVAL/ovl"
cd "${FQVIT_DIR:-$HOME/tvm_andersen/3rdparty/FQ-ViT}"  # FQ-ViT checkout at the commit pinned in fqvit_models/VERSION.md (test_quant.py is not vendored)
for m in swin_tiny deit_tiny deit_small deit_base swin_small swin_base; do
  echo "== $m [--quant --ptf --quant-method minmax] $(date +%T)"
  "$PY" test_quant.py $m /home/shared/ImageNet --quant --ptf --quant-method minmax --num-images 50000 --calib-batchsize 100 --calib-iter 10 --val-batchsize 100 2>&1 | grep -E "\* Prec@1|Traceback|Error" | tail -2
done
echo "== done $(date +%T)"

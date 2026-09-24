#!/usr/bin/env bash
# Run-to-run spread of identical configurations (1000-cal/1000-eval, ET 1.3.1).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
for r in 1 2; do
  echo "== swin LN-off maskaware repeat $r $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_noise_swin_$r" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets layernorm --mask-aware \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
for r in 1 2; do
  echo "== deit stock + LN-int16 repeat $r $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_noise_deit_stock_$r" --model-name deit_tiny_patch16_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_noise_deit_ln16_$r" --model-name deit_tiny_patch16_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --prec-rules 'norm\d?$=a16w8' \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
echo "== done $(date +%T)"

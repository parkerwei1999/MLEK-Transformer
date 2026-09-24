#!/usr/bin/env bash
# DeiT-Small / DeiT-Base on ExecuTorch 1.3.1, 1000-cal/1000-eval:
# stock a8w8 + a16w8, LN-off, LN-only int16. Waits for the Swin attention chain
# so the cas4k0 GPU is not shared.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_swin_attn_a16.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
for m in deit_small_patch16_224 deit_base_patch16_224; do
  short=${m%%_patch*}
  echo "== $m stock a8w8+a16w8 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_${short}_stock" --model-name "$m" \
      --quant-configs a8w8 a16w8 --act-observers histogram --keep-fp32-sets none \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
  echo "== $m LN-off $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_${short}_lnoff" --model-name "$m" \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets layernorm \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
  echo "== $m LN int16 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_${short}_ln16" --model-name "$m" \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --prec-rules 'norm\d?$=a16w8' \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
echo "== done $(date +%T)"

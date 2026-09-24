#!/usr/bin/env bash
# Swin-S / Swin-B: fp32, stock a8w8, LN-off, a16w8, then the stage 3+4 int16 rule (1000 cal / 1000 eval).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
for m in swin_small_patch4_window7_224 swin_base_patch4_window7_224; do
  echo "== $m native $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_sb_native_$m" --model-name "$m" \
      --quant-configs a8w8 a16w8 --act-observers histogram --keep-fp32-sets none layernorm --mask-aware \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
  for rules in 'layers\.[23]\.=a16w8;norm\d?$=fp32' 'layers\.2\.=a16w8;norm\d?$=fp32'; do
    echo "== $m rules[$rules] $(date +%T)"
    "$PY" vision_ptq_eval.py --out-dir "$V/et131_sb_rule_${m}_$(echo "$rules" | md5sum | cut -c1-6)" --model-name "$m" \
        --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --mask-aware --prec-rules "$rules" \
        --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
  done
done
echo "== done $(date +%T)"

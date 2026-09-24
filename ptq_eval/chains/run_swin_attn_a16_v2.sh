#!/usr/bin/env bash
# Re-run the Swin attention-core int16 tests with the dtype-correct mask rule;
# waits for the noise check on this GPU.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_noise_check.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=0
for rules in 'attn(\.log_int_softmax|\.attn_drop)?$=a16w8' 'reduction$=a16w8;attn(\.log_int_softmax|\.attn_drop)?$=a16w8' 'blocks\.\d+$=a16w8;norm\d?$=fp32'; do
  i=$((i+1)); echo "== swin LN-off rules[$rules] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_attn_v2_$i" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets layernorm --mask-aware --prec-rules "$rules" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
echo "== done $(date +%T)"

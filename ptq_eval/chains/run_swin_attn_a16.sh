#!/usr/bin/env bash
# Swin-Tiny, a8w8 + LN-off + mask-aware base: attention core, patch embed,
# GELU at int16, and reduction+attention combined.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
i=0
for rules in 'attn(\.log_int_softmax|\.attn_drop)?$=a16w8' 'patch_embed(\.proj|\.norm)?$=a16w8' 'mlp\.act$=a16w8' 'reduction$=a16w8;attn(\.log_int_softmax|\.attn_drop)?$=a16w8'; do
  i=$((i+1)); echo "== swin LN-off rules[$rules] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_attn_$i" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets layernorm --mask-aware --prec-rules "$rules" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "/1000 \(|top1|Traceback|Error"
done
echo "== done $(date +%T)"

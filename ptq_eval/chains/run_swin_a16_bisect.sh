#!/usr/bin/env bash
# Swin-Tiny a8w8 + LN-off + mask-aware; one module family at a time to a16w8.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
for rx in 'mlp\.fc2$' 'mlp\.fc1$' 'attn\.qkv$' 'attn\.proj$' 'reduction$' 'head$' '(fc1|fc2|qkv|proj|reduction|head)$'; do
  tag=$(echo "$rx" | tr -cd 'a-z0-9|' | tr '|' '_')
  echo "== swin a8w8 LN-off maskaware a16=[$rx] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_a16_${tag}_cal1000_n1000" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets layernorm --mask-aware --a16-modules "$rx" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "a16w8 modules|/1000 \(|top1|Traceback|Error"
done
echo "== done $(date +%T)"

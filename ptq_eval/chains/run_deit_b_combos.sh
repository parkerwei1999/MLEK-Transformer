#!/usr/bin/env bash
# DeiT-Base: LN int16 plus one more family at int16; waits for the DeiT-B chain.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_deit_sb.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=0
for extra in 'mlp\.fc2$=a16w8' 'mlp\.fc1$=a16w8' 'blocks\.\d+$=a16w8' 'attn(\.log_int_softmax|\.attn_drop)?$=a16w8' '(fc1|fc2|qkv|proj)$=a16w8'; do
  i=$((i+1)); rules="norm\\d?\$=a16w8;$extra"
  echo "== deit_base LN16 + [$extra] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_deit_base_combo_$i" --model-name deit_base_patch16_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --prec-rules "$rules" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
echo "== done $(date +%T)"

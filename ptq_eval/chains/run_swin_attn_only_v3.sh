#!/usr/bin/env bash
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_swin_attn_a16_v2.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
echo "== swin LN-off attention-only int16 (fixed rule) $(date +%T)"
"$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_attn_v3" --model-name swin_tiny_patch4_window7_224 \
    --quant-configs a8w8 --act-observers histogram --keep-fp32-sets layernorm --mask-aware \
    --prec-rules 'attn(\.log_int_softmax|\.attn_drop)?$=a16w8' --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
echo "== done $(date +%T)"

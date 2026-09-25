#!/usr/bin/env bash
# 50k for the LOWERABLE per-channel variants (symmetric PCS residual, LN-i32, transparent memory ops): DeiT-B :mm, Swin-T/S/B. Gated on the dual chain.
set -uo pipefail
until grep -q "^== done" "$OUT/run_overnight_dual_cas4k1.log" 2>/dev/null; do sleep 300; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { echo "== $1 pcsym8i32o8 obs=$2 50k $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$PTQ_EVAL/imagenet50k/$1__pcsym8i32o8-$2" --model-name "$1" --quant-configs a8w8 --act-observers "$2" \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inpc8symout@add;$VLN32" --n-cal 1000 --n-eval 50000 2>&1 | grep -E "top1|Traceback|Error"; }
run deit_base_patch16_224 minmax
run swin_tiny_patch4_window7_224 histogram
run swin_small_patch4_window7_224 histogram
run swin_base_patch4_window7_224 histogram
echo "== done $(date +%T)"

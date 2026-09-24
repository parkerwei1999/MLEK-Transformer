#!/usr/bin/env bash
# Fill the holes in the fixed ladders: DeiT-S res8i32o8 (hist, mm), Swin-S/B res8i32o8 (LN-i32 alone, expected chaotic).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_swin_mrg_i32o8.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=160
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { i=$((i+1)); echo "== $1 obs=$2 rules[$3] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_fill_$i" --model-name "$1" --quant-configs a8w8 --act-observers "$2" \
      --keep-fp32-sets none --mask-aware --prec-rules "$3" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run deit_small_patch16_224 histogram "$VLN32"
run deit_small_patch16_224 minmax    "$VLN32"
run swin_small_patch4_window7_224 histogram "$VLN32"
run swin_base_patch4_window7_224  histogram "$VLN32"
echo "== done $(date +%T)"

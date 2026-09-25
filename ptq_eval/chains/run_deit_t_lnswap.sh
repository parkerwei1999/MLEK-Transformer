#!/usr/bin/env bash
# DeiT-T res8i32o8 accuracy with the LayerNorm module swap (0 / 1 Newton steps); reference res8i32o8 72.7.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
for n in 0 1; do echo "== deit_tiny res8i32o8 ln-swap newton=$n $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/deit_t_lnswap$n" --model-name deit_tiny_patch16_224 --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --mask-aware \
    --prec-rules "$VLN32" --ln-newton-steps $n --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|NewtonLayerNorm|Traceback|Error"; done
echo "== done $(date +%T)"

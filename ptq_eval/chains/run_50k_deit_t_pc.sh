#!/usr/bin/env bash
# DeiT-T full-50k for the two per-channel residual formats, same settings as the
# res8i32o8 50k ladder cell (histogram observer, mask-aware, LN-i32), to answer
# whether per-channel residual storage buys anything on a model without a sink.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { echo "== deit_tiny_patch16_224 $1 obs=histogram 50k $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$PTQ_EVAL/imagenet50k/deit_tiny_patch16_224__$1-histogram" --model-name deit_tiny_patch16_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=$2@add;$VLN32" --n-cal 1000 --n-eval 50000 2>&1 | grep -E "top1|Traceback|Error"; }
run ptf8i32o8 a16inptf8out
run pcsym8i32o8 a16inpc8symout
echo "== done $(date +%T)"

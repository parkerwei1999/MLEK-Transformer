#!/usr/bin/env bash
# Swin-T: int16 on stage 1 / 2 / 4 alone (with LN-i32), to complete the s3a16 picture (s3a16 = 79.9, all-int8+LN-i32 = 56.0).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { echo "== swin_tiny s${1}a16 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/swin_t_s${1}a16" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "layers\\.$2\\.=a16w8;$VLN32" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run 1 0
run 2 1
run 4 3
echo "== done $(date +%T)"

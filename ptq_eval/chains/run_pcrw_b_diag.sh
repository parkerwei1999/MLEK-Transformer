#!/usr/bin/env bash
# DeiT-B (MinMax) rewrite drop 81.90 -> 80.70: finer int32 grid (bits 16) and seed 1 with / without the rewrite.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { echo "== deit_base pcsym8i32o8:mm seed=$1 $2 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/deit_b_s$1_$3" --model-name deit_base_patch16_224 --quant-configs a8w8 --act-observers minmax --seed "$1" \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inpc8symout@add;$VLN32" --n-cal 1000 --n-eval 1000 $2 2>&1 | grep -E "top1|Traceback|Error"; }
run 0 "--pc-rewrite --pc-rewrite-bits 16" rw16
run 1 "" plain
run 1 "--pc-rewrite" rw8
echo "== done $(date +%T)"

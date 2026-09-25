#!/usr/bin/env bash
# Is the -0.5 of the per-channel rewrite on DeiT-T (72.40 -> 71.90) noise? Seed 1 on DeiT-T, and DeiT-B (MinMax) with / without the rewrite.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { for rw in "" "--pc-rewrite"; do
  echo "== $1 pcsym8i32o8 obs=$2 seed=$3 ${rw:-plain} $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/$1_$2_s$3${rw}" --model-name "$1" --quant-configs a8w8 --act-observers "$2" --seed "$3" \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inpc8symout@add;$VLN32" --n-cal 1000 --n-eval 1000 $rw 2>&1 | grep -E "top1|Traceback|Error"
done; }
run deit_tiny_patch16_224 histogram 1
run deit_base_patch16_224 minmax 0
echo "== done $(date +%T)"

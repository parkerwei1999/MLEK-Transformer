#!/usr/bin/env bash
# DeiT-T pcsym8i32o8: fake-quant accuracy with vs without the per-channel lowering rewrite (kit package).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
for rw in "" "--pc-rewrite"; do
  echo "== deit_tiny pcsym8i32o8 ${rw:-plain} $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/deit_t${rw}" --model-name deit_tiny_patch16_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inpc8symout@add;$VLN32" --n-cal 1000 --n-eval 1000 $rw 2>&1 | grep -E "top1|pc_rewrite|Traceback|Error"
done
echo "== done $(date +%T)"

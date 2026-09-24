#!/usr/bin/env bash
# 50k DeiT ladder, remaining rungs (observer rule: int8-only rungs histogram, rungs with int16 carriers MinMax). Waits for batch 1 (deit).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_50k_deit.log" 2>/dev/null; do sleep 120; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"; O="$V/imagenet50k"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; mkdir -p "$O"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { local m="$1" tag="$2" qc="$3" obs="$4" rules="$5"; echo "== $m $tag $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$O/${m}__$tag" --model-name "$m" --quant-configs "$qc" --act-observers "$obs" \
      --keep-fp32-sets none --mask-aware ${rules:+--prec-rules "$rules"} --n-cal 1000 --n-eval 50000 --batch-size 100 2>&1 | grep -E "50000/50000|top1|Traceback|Error" | tail -1; }
for m in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224; do
  run $m res8i16o8     a8w8  histogram "norm\\d?\$=a16w8"
  [ "$m" != deit_tiny_patch16_224 ] && run $m res8i32o8 a8w8 histogram "$VLN32"
  run $m res16i32o8-mm a8w8  minmax    "blocks\\.\\d+\$=a16w8@add;$VLN32"
  [ "$m" = deit_tiny_patch16_224 ] && run $m ptf8i32o8-mm a8w8 minmax "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
  run $m orig-a16w8-mm a16w8 minmax    ""
done
echo "== done $(date +%T)"

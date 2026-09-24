#!/usr/bin/env bash
# 50k ImageNet val, batch 1: per model orig-a8w8, the most practical mixed config, orig-a16w8. Usage: run_50k_batch1.sh <deit|swin>
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAM="$1"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"; O="$V/imagenet50k"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; mkdir -p "$O"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { local m="$1" tag="$2" qc="$3" obs="$4" rules="$5"; echo "== $m $tag $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$O/${m}__$tag" --model-name "$m" --quant-configs "$qc" --act-observers "$obs" \
      --keep-fp32-sets none --mask-aware ${rules:+--prec-rules "$rules"} --n-cal 1000 --n-eval 50000 --batch-size 100 2>&1 | grep -E "50000/50000|top1|Traceback|Error" | tail -1; }
if [ "$FAM" = deit ]; then
  run deit_tiny_patch16_224  orig-a8w8        a8w8  histogram ""
  run deit_tiny_patch16_224  res8i32o8        a8w8  histogram "$VLN32"
  run deit_tiny_patch16_224  orig-a16w8       a16w8 histogram ""
  run deit_small_patch16_224 orig-a8w8        a8w8  histogram ""
  run deit_small_patch16_224 ptf8i32o8-mm     a8w8  minmax    "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
  run deit_small_patch16_224 orig-a16w8       a16w8 histogram ""
  run deit_base_patch16_224  orig-a8w8        a8w8  histogram ""
  run deit_base_patch16_224  ptf8i32o8-mm     a8w8  minmax    "blocks\\.\\d+\$=a16inptf8out@add;$VLN32"
  run deit_base_patch16_224  orig-a16w8       a16w8 histogram ""
else
  for m in swin_tiny_patch4_window7_224 swin_small_patch4_window7_224 swin_base_patch4_window7_224; do
    run $m orig-a8w8   a8w8  histogram ""
    run $m ptf8i32o8-mrg a8w8 histogram "blocks\\.\\d+\$=a16inptf8out@add;layers\\.2\\.downsample=a16w8;$VLN32"
    run $m s3a16       a8w8  histogram "layers\\.2\\.=a16w8;$VLN32"
    run $m orig-a16w8  a16w8 histogram ""
  done
fi
echo "== done $(date +%T)"

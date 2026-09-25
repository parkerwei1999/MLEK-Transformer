#!/usr/bin/env bash
# s3a16 refinement: int16 only from stage-3 block k onward (blocks before k stay int8), with LN-i32. s3a16 = 79.9.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { echo "== swin_tiny s3a16 from block $1 ($2) $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/swin_t_s3from$1" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "layers\\.2\\.blocks\\.$2\\.=a16w8;layers\\.2\\.downsample=a16w8;$VLN32" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run 2 "[2-5]"
run 3 "[3-5]"
run 4 "[4-5]"
echo "== done $(date +%T)"

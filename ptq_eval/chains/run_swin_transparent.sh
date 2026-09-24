#!/usr/bin/env bash
# Swin-T ptf8i32o8 with NO merge rule under the transparent memory-op pass (per-channel producers pass through untouched).
# First a 100-image smoke (crash check + counters), then the 1000-image cell comparable to run_swin_global_mem (int16-carrier version).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { echo "== swin_tiny ptf8i32o8 transparent n_eval=$1 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_transparent_$1" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inptf8out@add;$VLN32" --n-cal 1000 --n-eval "$1" 2>&1 | grep -E "top1|Traceback|Error|memory-op pass"; }
run 100
run 1000
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Swin-T bridged WITHOUT any merge rule, after the global memory-op widening: expect ~80 (was 72 with requant leakage).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
echo "== swin_tiny ptf8i32o8 no-merge-rule, global memory-op widening $(date +%T)"
"$PY" vision_ptq_eval.py --out-dir "$V/et131_globalmem_1" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inptf8out@add;$VLN32" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error|widened"
echo "== done $(date +%T)"

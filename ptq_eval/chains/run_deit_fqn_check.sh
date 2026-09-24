#!/usr/bin/env bash
# DeiT-T sanity after the FQN fallback: LN16 and LN32 should reproduce 72.1 / 72.7.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=0
for rules in 'norm\d?$=a16w8' 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8' 'norm\d?$=fp32'; do
  i=$((i+1)); echo "== deit_tiny rules[$rules] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_deit_fqncheck_$i" --model-name deit_tiny_patch16_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --prec-rules "$rules" \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
done
echo "== done $(date +%T)"

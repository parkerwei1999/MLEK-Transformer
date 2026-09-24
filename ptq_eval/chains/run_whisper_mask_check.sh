#!/usr/bin/env bash
# Whisper: is the rule-path failure the MaskAwareQuantizer rewrite? keep_fp32 + mask-aware, then the LN ladder without the rewrite.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
I32='a32w8@sub,mul,sum.dim_IntList,add,rsqrt,reshape,view'
run() { echo "== whisper $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a8w8 "${@:2}" 2>&1 \
      | grep -E "200/200|Traceback|Error"; }
run lnoff_maskaware   --keep-fp32 layernorm --mask-aware
run rule_fp32_nomask  --prec-rules 'layer_norm$=fp32'
run rule_ln16_nomask  --prec-rules 'layer_norm$=a16w8'
run rule_ln32_nomask  --prec-rules "layer_norm\$=$I32;layer_norm\$=a16w8"
echo "== done $(date +%T)"

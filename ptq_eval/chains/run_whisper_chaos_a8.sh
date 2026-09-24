#!/usr/bin/env bash
# Is the a8w8 + LN-fp32 5.82% a chaotic coin flip? Perturb the calibration set (199 / 198 utts) on the keep path and the rule path.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_ln_ladder2.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() { echo "== whisper $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w3_$1" --stages int8 --n-eval 200 --quant-config a8w8 "${@:2}" 2>&1 \
      | grep -E "200/200|Traceback|Error"; }
run keep_cal199  --n-cal 199 --keep-fp32 layernorm
run keep_cal198  --n-cal 198 --keep-fp32 layernorm
run rule_cal199  --n-cal 199 --prec-rules 'layer_norm$=fp32'
run a16_rule_fp32 --n-cal 200 --quant-config a16w8 --prec-rules 'layer_norm$=fp32'
run a16_rule_ln32_all --n-cal 200 --quant-config a16w8 --prec-rules 'layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,rsqrt,reshape,view;layer_norm$=a16w8'
echo "== done $(date +%T)"

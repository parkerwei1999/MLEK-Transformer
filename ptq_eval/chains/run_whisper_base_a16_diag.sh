#!/usr/bin/env bash
# Whisper-base a16w8 + LN int32 sits 1.5 pt above fp32 with errors spread uniformly: observer / calibration-size / mask checks.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-base $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$W/et131_wb_$1" --stages int8 --n-eval 200 --quant-config a16w8 "${@:2}" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a16_ln32_minmax   --n-cal 200 --act-observer minmax --prec-rules "$LN32"
run a16_ln32_cal400   --n-cal 400 --prec-rules "$LN32"
run a16_ln32_maskaware --n-cal 200 --mask-aware --prec-rules "$LN32"
echo "== done $(date +%T)"

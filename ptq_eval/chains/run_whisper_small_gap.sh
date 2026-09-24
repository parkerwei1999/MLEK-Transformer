#!/usr/bin/env bash
# Whisper-small residual 1.9 pt at a16w8 + LN32 + MinMax: more calibration (400 utts) and the lower int16 floor.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() { echo "== whisper-small $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_$1" --stages int8 --n-eval 200 --act-observer minmax "${@:2}" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a16_ln32_minmax_cal400 --n-cal 400 --quant-config a16w8 --prec-rules 'layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run a16e16_ln32_minmax     --n-cal 200 --quant-config a16w8e16 --prec-rules 'layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8e16'
echo "== done $(date +%T)"

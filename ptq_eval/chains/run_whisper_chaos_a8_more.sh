#!/usr/bin/env bash
# More calibration sets for the a8w8 + LN fp32 keep path (chaos distribution) and the mask rewrite in the a16w8 regime.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() { echo "== whisper $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w4_$1" --stages int8 --n-eval 200 "${@:2}" 2>&1 \
      | grep -E "200/200|Traceback|Error"; }
run keep_cal150 --quant-config a8w8 --n-cal 150 --keep-fp32 layernorm
run keep_cal201 --quant-config a8w8 --n-cal 201 --keep-fp32 layernorm
run keep_cal250 --quant-config a8w8 --n-cal 250 --keep-fp32 layernorm
run a16_keep_maskaware --quant-config a16w8 --n-cal 200 --keep-fp32 layernorm --mask-aware
echo "== done $(date +%T)"

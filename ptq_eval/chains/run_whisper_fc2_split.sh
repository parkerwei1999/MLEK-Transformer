#!/usr/bin/env bash
# Whisper-tiny: fc1 int8 is free (5.63%), fc2 int8 kills (18.9%). Split fc2: input int8 / output int16 vs input int16 / output int8.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w13_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a16_fc2_in8_out16 "fc2\$=a8in16out;$LN32"
run a16_fc2_in16_out8 "fc2\$=a16in8out;$LN32"
run a16_fc2_in16_outptf "fc2\$=a16inptf8out;$LN32"
echo "== done $(date +%T)"

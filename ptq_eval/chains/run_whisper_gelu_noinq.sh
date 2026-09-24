#!/usr/bin/env bash
# Whisper-tiny: MLP path int8 costs 10 pt; does GELU-without-input-quant (fc1 out int16 + GELU int16, fc2 in/out int8) recover it?
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_mixed_final.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w12_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a16_mlp8_gf     "fc1\$=a8in16out;activation_fn\$=a16w8;fc2\$=a8w8;$LN32"
run a16_fc2_8_only  "fc2\$=a8w8;$LN32"
run a16_fc1_8_only  "fc1\$=a8w8;$LN32"
echo "== done $(date +%T)"

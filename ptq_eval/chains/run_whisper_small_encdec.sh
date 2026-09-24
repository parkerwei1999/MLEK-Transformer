#!/usr/bin/env bash
# Whisper-small's remaining 1.9 pt at a16w8 + LN-i32 + MinMax: is it in the encoder or the decoder? (rule path fp32 on one half; `.*=fp32` control = 5.54% on tiny)
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-small $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run enc_fp32_dec_a16 "encoder.*=fp32;$LN32"
run dec_fp32_enc_a16 "decoder.*=fp32;proj_out\$=fp32;$LN32"
echo "== done $(date +%T)"

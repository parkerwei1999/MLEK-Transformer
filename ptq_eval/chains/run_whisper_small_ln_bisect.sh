#!/usr/bin/env bash
# Whisper-small: LN fp32 recovers 2.29% (fp32 2.27) while LN-i32 gives 4.21%. Bisect the LN internals:
# rsqrt (int16 TABLE) in fp32, the mean-path reshape (int16 by the a16w8 fallback) in fp32.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-small $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_lnb_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" "${@:3}" 2>&1 | grep -E "200/200|Traceback|Error"; }
run ln_rsqrt_fp32    "layer_norm\$=fp32@rsqrt;$LN32"
run ln_reshape_fp32  "layer_norm\$=fp32@reshape;$LN32"
echo "== done $(date +%T)"

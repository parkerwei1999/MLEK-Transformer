#!/usr/bin/env bash
# Whisper-small LN-i32 baseline (4.21%) with --dump-ln-scales: per-LN rsqrt-input grids vs fp32 token variance.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
echo "== whisper-small ln_i32_dump $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_lnb_dump" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$LN32" --dump-ln-scales 2>&1 | grep -E "200/200|Traceback|Error|ln_scales"
echo "== done $(date +%T)"

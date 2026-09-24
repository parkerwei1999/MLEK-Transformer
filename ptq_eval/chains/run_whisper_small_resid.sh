#!/usr/bin/env bash
# Whisper-small inversion (4.21% vs base 3.70%): is the residual's int16 MinMax grid too coarse for ordinary tokens? residual int32 + LN scan.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
echo "== whisper-small scan (minmax) $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_scan" --stages int8 --n-cal 200 --n-eval 2 --quant-config a16w8 --act-observer minmax --prec-rules "$LN32" --dump-ln-scales 2>&1 | grep -E "LN scales|Traceback|Error"
echo "== whisper-small a16_ln32_minmax_res32 $(date +%T)"
"$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_res32" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "layers\\.\\d+\$=a32w8@add;$LN32" 2>&1 | grep -E "200/200|Traceback|Error"
echo "== done $(date +%T)"

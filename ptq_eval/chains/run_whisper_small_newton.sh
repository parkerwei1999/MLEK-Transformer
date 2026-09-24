#!/usr/bin/env bash
# Gated on the LN bisect: only if rsqrt-in-fp32 recovers whisper-small (WER < 3.0) is the rsqrt input grid the
# cause, and then NewtonLayerNorm (int16 table seed + int32 Newton steps, lowerable) is the candidate fix.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"; L="$PTQ_EVAL/chains/run_whisper_small_ln_bisect.log"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
until [ "$(grep -c 'WER=' "$L" 2>/dev/null)" -ge 1 ]; do sleep 60; done
wer=$(grep -o 'WER=[0-9.]*' "$L" | head -1 | cut -d= -f2)
echo "== gate: ln_rsqrt_fp32 WER=$wer $(date +%T)"
if ! awk -v w="$wer" 'BEGIN{exit !(w < 3.0)}'; then echo "== gate failed: rsqrt-in-fp32 did not recover; Newton cells skipped"; echo "== done $(date +%T)"; exit 0; fi
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-small $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$LN32" --ln-newton-steps "$2" 2>&1 | grep -E "NewtonLayerNorm|200/200|Traceback|Error"; }
run ln_newton2 2
run ln_newton1 1
echo "== done $(date +%T)"

#!/usr/bin/env bash
# --dump-ln-scales for whisper tiny / base (LN-i32, MinMax): rsqrt-input grid vs per-token variance, to test the calibration-only "Newton needed?" criterion.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
for m in tiny base; do echo "== whisper-$m ln_dump $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$m --out-dir "$OUT/pcrw/w${m}_lndump" --stages int8 --n-cal 200 --n-eval 20 --quant-config a16w8 --act-observer minmax --prec-rules "$LN32" --dump-ln-scales 2>&1 | grep -E "ln_scales|Traceback|Error"; done
echo "== done $(date +%T)"

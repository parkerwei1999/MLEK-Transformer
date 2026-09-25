#!/usr/bin/env bash
# fp32 references (fp32-static stage) for the sizes whose chains have none:
# whisper-medium 200 utt and whisper-large-v3 100 utt, same cal/eval protocol as the int8 cells.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
run() { echo "== whisper-$1 fp32_static n=$2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$1 --out-dir "$OUT/pcrw/w$1_fp32_static" --stages fp32-static --n-cal $2 --n-eval $2 --quant-config a16w8 --act-observer minmax 2>&1 | grep -E "WER=|Traceback|Error|out of memory"; }
run medium 200
run large-v3 100
echo "== done $(date +%T)"

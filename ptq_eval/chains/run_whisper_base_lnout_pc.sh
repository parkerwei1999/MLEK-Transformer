#!/usr/bin/env bash
# whisper-base deploy structure (int16 residual, LN-i32 + Newton1, int8 conv/GELU/proj/logit/fc1): LN OUTPUT format
#  o8    : int8 per-tensor at q/k/v/fc1 inputs (deploy v4, 3.98)
#  optf8 : int8 PER-CHANNEL (PTF) at q/k/v/fc1 inputs (folds into the next Linear's weights on device)
#  o16   : int16 per-tensor at q/k/v/fc1 inputs (projections a16in8out)
set -uo pipefail
until grep -q "^== done" "$OUT/run_fullset_whisper_cas4k0.log" 2>/dev/null; do sleep 600; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,clamp;layer_norm$=a16w8'
BASE="conv[12]\$=a8w8;encoder\$=a8w8@gelu;out_proj\$=a8w8;proj_out\$=a8w8"
run() { echo "== whisper-base $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$OUT/pcrw/wb_lnout_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" --ln-newton-steps 1 2>&1 | grep -E "200/200|Traceback|Error"; }
run optf8 "$BASE;(q_proj|k_proj|v_proj|fc1)\$=aptf8in8out;$LN32"
run o16   "$BASE;(q_proj|k_proj|v_proj|fc1)\$=a16in8out;$LN32"
run res16pc16 "$BASE;(q_proj|k_proj|v_proj|fc1)\$=a8w8;layers\\.\\d+\$=a16inpc16out@add;$LN32"
echo "== done $(date +%T)"

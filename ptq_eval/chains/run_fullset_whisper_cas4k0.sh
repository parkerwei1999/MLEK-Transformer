#!/usr/bin/env bash
# Full LibriSpeech test-clean (2620 utterances): whisper-small fp32-static and deploy v4 (int8 conv/GELU/proj/logit/fc1, fc2 int16,
# residual int16, LN-i32 + Newton1). Gated on the 50k Swin chain finishing on this host.
set -uo pipefail
until grep -q "^== done" "$OUT/run_50k_swin.log" 2>/dev/null; do sleep 300; done
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,clamp;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-$1 $2 full test-clean $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$1 --out-dir "$OUT/fullset/w$1_$2" --n-cal 200 --n-eval 2620 --quant-config a16w8 --act-observer minmax "${@:3}" 2>&1 | grep -E "2620/2620|Traceback|Error"; }
run small v4_newton1 --stages int8 --prec-rules "$INT8;$LN32" --ln-newton-steps 1
run small fp32_static --stages fp32-static
echo "== done $(date +%T)"

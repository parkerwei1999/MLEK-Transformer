#!/usr/bin/env bash
# whisper tiny/base/small on the full test-clean with the dual k=8 recipe (fp32 and Newton-1 exist).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"; export LN_DUAL_K=8
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-$1 $2 full test-clean $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$1 --out-dir "$OUT/fullset/w$1_$2" --n-cal 200 --n-eval 2620 --quant-config a16w8 --act-observer minmax "${@:3}" 2>&1 | grep -E "dual-range|2620/2620|Traceback|Error|out of memory"; }
run tiny v4r16_dualk8 --stages int8 --prec-rules "$INT8;$LNR" --ln-dual-k 8 --ln-newton-steps 0
run base v4r16_dualk8 --stages int8 --prec-rules "$INT8;$LNR" --ln-dual-k 8 --ln-newton-steps 0
run small v4r16_dualk8 --stages int8 --prec-rules "$INT8;$LNR" --ln-dual-k 8 --ln-newton-steps 0
echo "== done $(date +%T)"

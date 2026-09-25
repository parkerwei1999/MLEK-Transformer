#!/usr/bin/env bash
# (a) Is Newton neutral where the sink is small? Whisper tiny/base deploy v4 structure with NO LN swap (Newton 0) vs Newton 1 (6.01 / 3.98).
# (b) Rule-only fix for the LN reshapes: DeiT-T res8i32o8 with reshape/view in the int32 op list (res8i32o8 = 72.7).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
for m in tiny base; do echo "== whisper-$m deploy_v4_newton0 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-$m --out-dir "$OUT/pcrw/w${m}_v4_newton0" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;layers\\.\\d+\$=a16inpc8symout@add;$LN32" 2>&1 | grep -E "200/200|Traceback|Error"; done
echo "== deit_tiny res8i32o8 reshape-in-int32 $(date +%T)"
"$PY" vision_ptq_eval.py --out-dir "$OUT/pcrw/deit_t_ln32_reshape32" --model-name deit_tiny_patch16_224 --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none --mask-aware \
  --prec-rules 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add,reshape,view;norm\d?$=a16w8' --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
echo "== done $(date +%T)"

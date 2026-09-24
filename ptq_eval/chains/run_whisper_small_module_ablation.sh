#!/usr/bin/env bash
# Whisper-small module ablation (apo-style "pull one module to fp32"): from a16w8 + LN-i32 + MinMax (4.21%), set one family fp32 via rules.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_small_encdec.log" 2>/dev/null; do sleep 60; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-small $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_abl_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run attn_fp32     "(self_attn|encoder_attn)\$=fp32;$LN32"
run mlp_fp32      "(fc1|fc2|activation_fn)\$=fp32;$LN32"
run resid_fp32    "layers\\.\\d+\$=fp32;$LN32"
run ln_fp32       "layer_norm\$=fp32"
run proj_fp32     "(q_proj|k_proj|v_proj|out_proj)\$=fp32;$LN32"
echo "== done $(date +%T)"

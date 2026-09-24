#!/usr/bin/env bash
# Whisper-tiny demotion ladder: start from the good a16w8 + LN32 (MinMax) and demote ONE family to per-tensor int8 at a time.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_a8base_projout.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w9_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run demote_linear   "(q_proj|k_proj|v_proj|out_proj|fc1|fc2)\$=a8w8;$LN32"
run demote_gelu     "activation_fn\$=a8w8;$LN32"
run demote_conv     "conv[12]\$=a8w8;$LN32"
run demote_layerops "layers\\.\\d+\$=a8w8;$LN32"
run demote_projout  "proj_out\$=a8w8;$LN32"
run demote_attn     "(self_attn|encoder_attn)\$=a8w8;$LN32"
echo "== done $(date +%T)"

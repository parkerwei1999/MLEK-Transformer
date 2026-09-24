#!/usr/bin/env bash
# Whisper deployable candidate v3: conv / front GELU / projections / logits / fc1 int8; fc2 int16; residual add output
# int8 per-channel with UNCONSTRAINED scales (PCS, a16inpc8out); LN-i32 with NewtonLayerNorm (2 steps).
# Base: PTF residual 4.47/4.81 -> PCS residual 3.76; small: LN-i32 4.21 -> Newton2 2.20.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
DEPLOY="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8;layers\\.\\d+\$=a16inpc8out@add;$LN32"
run() { echo "== $1 $2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id "openai/$1" --out-dir "$W/et131_${1}_$2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$DEPLOY" --ln-newton-steps 2 2>&1 | grep -E "200/200|Traceback|Error"; }
run whisper-tiny  deploy_pcs_newton2
run whisper-base  deploy_pcs_newton2
run whisper-small deploy_pcs_newton2
echo "== done $(date +%T)"

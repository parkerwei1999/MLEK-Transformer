#!/usr/bin/env bash
# whisper-small: LN-i32 + Newton2 alone = 2.20%, full deploy v3 = 2.95% (1 loop). Which demotion costs the 0.75?
#  newton_pcsres : Newton2 + PCS residual only (projections / conv / fc1 stay int16)
#  newton_int8proj: Newton2 + int8 conv / front GELU / proj / logit / fc1 (residual stays int16 per-tensor)
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
run() { echo "== whisper-small $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-small --out-dir "$W/et131_ws_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" --ln-newton-steps 2 2>&1 | grep -E "200/200|Traceback|Error"; }
run newton_pcsres   "layers\\.\\d+\$=a16inpc8out@add;$LN32"
run newton_int8proj "$INT8;$LN32"
echo "== done $(date +%T)"

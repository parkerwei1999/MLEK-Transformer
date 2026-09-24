#!/usr/bin/env bash
# Whisper-tiny, apo-int-aligned variants on top of mixed v2: (a) GELU int8-in / int16-out; (b) softmax s8i32o16 with SxV 16x8;
# (c) both. Waits for mixed v2.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
BASE="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8;fc2\$=a16inptf8out;layers\\.\\d+\$=a16inptf8out@add"
GELU='activation_fn$=a8in16out'
ATT='(self_attn|encoder_attn)$=a8in16out@exp;(self_attn|encoder_attn)$=a8w8@amax,sub;(self_attn|encoder_attn)$=a32w8@sum.dim_IntList,reciprocal'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w14_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run apo_gelu8i16o   "$GELU;$BASE;$LN32"
run apo_sm8i32o16   "$ATT;$BASE;$LN32"
run apo_both        "$GELU;$ATT;$BASE;$LN32"
echo "== done $(date +%T)"

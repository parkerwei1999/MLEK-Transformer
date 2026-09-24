#!/usr/bin/env bash
# Whisper-tiny: projections vs MLP. (a) a8-base + fc2 int16 only; (b) a8-base + fc1+fc2 int16 (MLP int16, projections int8);
# (c) a16w8-MinMax-LN32 with q/k/v/out_proj int8; (d) same with fc1/fc2 int8.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
BASE='(self_attn|encoder_attn)$=a16w8;layers\.\d+$=a16w8@add'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w10_$1" --stages int8 --n-cal 200 --n-eval 200 "${@:2}" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a8base_fc2_16      --quant-config a8w8  --prec-rules "fc2\$=a16w8;$BASE;$LN32"
run a8base_mlp16       --quant-config a8w8  --prec-rules "(fc1|fc2)\$=a16w8;$BASE;$LN32"
run a16_proj8          --quant-config a16w8 --act-observer minmax --prec-rules "(q_proj|k_proj|v_proj|out_proj)\$=a8w8;$LN32"
run a16_mlp8           --quant-config a16w8 --act-observer minmax --prec-rules "(fc1|fc2)\$=a8w8;$LN32"
echo "== done $(date +%T)"

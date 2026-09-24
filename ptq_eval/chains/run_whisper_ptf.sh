#!/usr/bin/env bash
# PTF on Whisper-tiny (200/200). (1-2) a16w8 + LN int32 internals with the residual stored int8 per-tensor / PTF.
# (3-6) apo-int-like mixed scheme on an a8w8 base: attention core int16, LN int32 internals, residual stored int16 / int8 / PTF (+199-cal chaos check).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
ATT='(self_attn|encoder_attn)$=a16w8'
run() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w5_$1" --stages int8 --n-eval 200 "${@:2}" 2>&1 | grep -E "200/200|Traceback|Error"; }
run a16_ln32_res8     --n-cal 200 --quant-config a16w8 --prec-rules "layers\\.\\d+\$=a16in8out@add;$LN32"
run a16_ln32_resptf   --n-cal 200 --quant-config a16w8 --prec-rules "layers\\.\\d+\$=a16inptf8out@add;$LN32"
run a8_att16_ln32_res16  --n-cal 200 --quant-config a8w8 --prec-rules "$ATT;layers\\.\\d+\$=a16w8@add;$LN32"
run a8_att16_ln32_res8   --n-cal 200 --quant-config a8w8 --prec-rules "$ATT;layers\\.\\d+\$=a16in8out@add;$LN32"
run a8_att16_ln32_resptf --n-cal 200 --quant-config a8w8 --prec-rules "$ATT;layers\\.\\d+\$=a16inptf8out@add;$LN32"
run a8_att16_ln32_resptf_cal199 --n-cal 199 --quant-config a8w8 --prec-rules "$ATT;layers\\.\\d+\$=a16inptf8out@add;$LN32"
echo "== done $(date +%T)"

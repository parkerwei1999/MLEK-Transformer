#!/usr/bin/env bash
# Zero-point-free per-channel int8 (a16inpc8symout): if accuracy holds, the lowering rewrite of a per-channel
# activation reduces to one int32 MUL by a per-channel constant on each side (no per-channel offset add/sub).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"; V="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
INT8="conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;proj_out\$=a8w8;fc1\$=a8w8"
if [ "$1" = whisper ]; then
  cd "$W"; echo "== whisper-base deploy_pcsym_newton2 $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id openai/whisper-base --out-dir "$W/et131_wb_deploy_pcsym_newton2" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax --prec-rules "$INT8;layers\\.\\d+\$=a16inpc8symout@add;$LN32" --ln-newton-steps 2 2>&1 | grep -E "200/200|Traceback|Error"
else
  cd "$V"; echo "== swin_tiny pcsym8i32o8 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_pcsym_swin_t" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inpc8symout@add;$VLN32" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
fi
echo "== done $(date +%T)"

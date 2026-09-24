#!/usr/bin/env bash
# Power-of-two per-channel (PTF) vs unconstrained per-channel residual storage: Swin-T recipe and Whisper-tiny/small a16 base.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
WLN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
cd "$V"
echo "== swin_tiny recipe with unconstrained per-channel residual $(date +%T)"
"$PY" vision_ptq_eval.py --out-dir "$V/et131_pc_swin" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --prec-rules "blocks\\.\\d+\$=a16inpc8out@add;layers\\.2\\.downsample=a16w8;$VLN32" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"
cd "$W"
for m in whisper-tiny whisper-small; do
  echo "== $m a16_ln32_minmax_respc $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id "openai/$m" --out-dir "$W/et131_${m}_respc" --stages int8 --n-cal 200 --n-eval 200 --quant-config a16w8 --act-observer minmax \
      --prec-rules "layers\\.\\d+\$=a16inpc8out@add;$WLN32" 2>&1 | grep -E "200/200|Traceback|Error"
done
echo "== done $(date +%T)"

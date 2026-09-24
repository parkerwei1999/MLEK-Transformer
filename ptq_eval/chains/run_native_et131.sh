#!/usr/bin/env bash
# Stock ExecuTorch 1.3.1 quantizer (histogram observer, LayerNorm quantized,
# no exclusions) on quick subsets: Whisper-Tiny 200-cal/200-eval, DeiT-Tiny and
# Swin-Tiny 1000-cal/1000-eval, a8w8 and a16w8. GPU venv, sequential so the
# single GPU is never shared.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
OVL="$PTQ_EVAL/ovl"
W="$PTQ_EVAL"
V="$PTQ_EVAL"
export PYTHONPATH="$OVL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
echo "== start $(date '+%F %T') on $(hostname)"
cd "$W"
for q in a8w8 a16w8; do
  echo "== whisper $q 200/200 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_${q}_cal200_n200" --stages int8 --quant-config "$q" \
      --n-cal 200 --n-eval 200 2>&1 | grep -E "prepared graph|export\+prepare|quantize\[|^quantized|200/200|Traceback|Error"
done
cd "$V"
for m in deit_tiny_patch16_224 swin_tiny_patch4_window7_224; do
  echo "== vision $m 1000/1000 $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_${m%%_*}_cal1000_n1000" --model-name "$m" \
      --quant-configs a8w8 a16w8 --act-observers histogram --keep-fp32-sets none \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "export\+prepare|calibrate\+convert|/1000 \(|top1|Traceback|Error"
done
echo "== done $(date +%T)"

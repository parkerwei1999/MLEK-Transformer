#!/usr/bin/env bash
# Swin-Tiny 1000/1000 on ExecuTorch 1.3.1: stock vs mask-aware annotation,
# LayerNorm quantized and kept FP32, a8w8 histogram.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
for flag in "" "--mask-aware"; do
  tag=$([ -n "$flag" ] && echo maskaware || echo stock)
  echo "== swin a8w8 $tag $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_${tag}_cal1000_n1000" --model-name swin_tiny_patch4_window7_224 \
      --quant-configs a8w8 --act-observers histogram --keep-fp32-sets none layernorm $flag \
      --n-cal 1000 --n-eval 1000 2>&1 | grep -E "/1000 \(|top1|Traceback|Error|rewritten"
done
echo "== done $(date +%T)"

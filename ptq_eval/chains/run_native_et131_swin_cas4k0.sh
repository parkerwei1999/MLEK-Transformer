#!/usr/bin/env bash
# Swin-Tiny 1000-cal/1000-eval on the stock ExecuTorch 1.3.1 quantizer, split
# off from run_native_et131.sh so it runs on cas4k0's idle GPU in parallel.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"
echo "== vision swin_tiny_patch4_window7_224 1000/1000 $(date +%T) on $(hostname)"
"$PY" vision_ptq_eval.py --out-dir "$V/et131_swin_cal1000_n1000" --model-name swin_tiny_patch4_window7_224 \
    --quant-configs a8w8 a16w8 --act-observers histogram --keep-fp32-sets none \
    --n-cal 1000 --n-eval 1000 2>&1 | grep -E "export\+prepare|calibrate\+convert|/1000 \(|top1|Traceback|Error"
echo "== done $(date +%T)"

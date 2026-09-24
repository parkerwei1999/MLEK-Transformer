#!/usr/bin/env bash
# Swin-T stage 3 with FQ-ViT's quantization points: residual PTF, fc2/proj outputs int8 per-tensor, fc1 output unquantized (int16) before GELU,
# LN int32 internals, everything else int8; histogram vs MinMax observers. Waits for the stage-3 ladder.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_stage3_ptf_ladder.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$V"; i=50
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
REPL='layers\.2\..*fc1$=a8in16out;layers\.2\.blocks\.\d+$=a16inptf8out@add'
run() { i=$((i+1)); echo "== swin_tiny obs=$1 rules[$2] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_repl_$i" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers "$1" \
      --keep-fp32-sets none --mask-aware --prec-rules "$2" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
run histogram "$REPL;$VLN32"
run minmax    "$REPL;$VLN32"
run minmax    "layers\\.2\\..*(fc2|proj)\$=a16inptf8out;$REPL;$VLN32"
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Calibration-only advisor on STOCK a8w8: does it flag exactly the places the test-set bisects found?
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
F='grep -v -i "warn\|deprecat\|Skipping\|meshgrid\|register_constant\|act_scale\|REAL ImageNet\|prepared graph\|memory-op"'
run() { echo "== $1 $2 rules=[$3] $(date +%T)"; "$PY" levels_report.py --model "$1" --quant-config "$2" --prec-rules "$3" --n-cal 100 --top 30 2>&1 | eval $F | grep -E "^===|^!!|^  |flagged|Traceback|Error" | cut -c1-170; }
run deit_tiny_patch16_224 a8w8 ""
run deit_base_patch16_224 a8w8 ""
run swin_tiny_patch4_window7_224 a8w8 ""
run whisper-base a8w8 ""
run deit_base_patch16_224 a8w8 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run swin_tiny_patch4_window7_224 a8w8 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
echo "== census whisper-large-v3 $(date +%T)"; "$PY" ln_var_census.py --model-id openai/whisper-large-v3 --n 8 2>&1 | grep -E "^===|^ *[0-9]|levels_med|LNs with|Traceback|Error|out of memory" | cut -c1-150
echo "== done $(date +%T)"

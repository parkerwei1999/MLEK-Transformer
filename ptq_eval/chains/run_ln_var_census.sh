#!/usr/bin/env bash
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; cd "$K"
for m in tiny base small medium large-v3; do echo "== whisper-$m $(date +%T)"; "$PY" ln_var_profile.py --model-id openai/whisper-$m --n 8 2>&1 | grep -E "^===|^ *[0-9]|levels_med|LNs with|Traceback|Error|out of memory" | cut -c1-150; done
echo "== done $(date +%T)"

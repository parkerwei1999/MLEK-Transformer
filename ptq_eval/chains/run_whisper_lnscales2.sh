#!/usr/bin/env bash
# Re-scan the int16 LayerNorm grids with the (x-mean)^2 element grid included; waits for the first scan chain.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_lnscales.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
echo "== whisper lnscales2 ln16 $(date +%T)"
"$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_lnscales2_ln16" --stages int8 --n-cal 200 --n-eval 2 \
    --quant-config a8w8 --prec-rules 'layer_norm$=a16w8' --dump-ln-scales 2>&1 | grep -E "LN scales|2/2|Traceback|Error"
echo "== done $(date +%T)"

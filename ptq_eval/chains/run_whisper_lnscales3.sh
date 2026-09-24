#!/usr/bin/env bash
# Scan again (rsqrt input grid + per-token variance) for LN16 and LN32 after the variants chain has finished.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_ln_variants.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
for tag_rules in "ln16|layer_norm\$=a16w8" "ln32|layer_norm\$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm\$=a16w8"; do
  tag="${tag_rules%%|*}"; rules="${tag_rules#*|}"
  echo "== whisper lnscales3 $tag $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_lnscales3_$tag" --stages int8 --n-cal 200 --n-eval 2 \
      --quant-config a8w8 --prec-rules "$rules" --dump-ln-scales 2>&1 | grep -E "LN scales|2/2|Traceback|Error"
done
echo "== done $(date +%T)"

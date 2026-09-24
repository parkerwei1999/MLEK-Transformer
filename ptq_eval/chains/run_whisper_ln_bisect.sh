#!/usr/bin/env bash
# Whisper: control (LN fp32 via the rule path) and encoder-only / decoder-only LN int32; waits for the variants chain.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_ln_variants.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
I32='a32w8@sub,mul,sum.dim_IntList,add,rsqrt,reshape,view'
run() { echo "== whisper $1 [$2] $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_$1" --stages int8 --n-cal 200 --n-eval 200 \
      --quant-config a8w8 --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
run ln_fp32_rule   'layer_norm$=fp32'
run ln32_enc_only  "encoder\..*layer_norm\$=$I32;encoder\..*layer_norm\$=a16w8;layer_norm\$=fp32"
run ln32_dec_only  "decoder\..*layer_norm\$=$I32;decoder\..*layer_norm\$=a16w8;layer_norm\$=fp32"
echo "== scan ln32_all $(date +%T)"
"$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_lnscales4_ln32all" --stages int8 --n-cal 200 --n-eval 2 \
    --quant-config a8w8 --prec-rules "layer_norm\$=$I32;layer_norm\$=a16w8" --dump-ln-scales 2>&1 | grep -E "LN scales|Traceback|Error"
echo "== done $(date +%T)"

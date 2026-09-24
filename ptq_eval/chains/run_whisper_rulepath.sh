#!/usr/bin/env bash
# Whisper: isolate the rule-path failure. (a) everything fp32 via rules, (b) LN fp32 rule with only the encoder quantized, (c) only the decoder.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_whisper_mask_check.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() { echo "== whisper $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a8w8 "${@:2}" 2>&1 \
      | grep -E "200/200|Traceback|Error"; }
run rule_allfp32          --prec-rules '.*=fp32'
run rule_lnfp32_enc_only  --prec-rules 'layer_norm$=fp32' --quant-parts encoder
run rule_lnfp32_dec_only  --prec-rules 'layer_norm$=fp32' --quant-parts decoder
run keepfp32_enc_only     --keep-fp32 layernorm --quant-parts encoder
run keepfp32_dec_only     --keep-fp32 layernorm --quant-parts decoder
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Whisper-base then Whisper-small: fp32, a8w8 + LN fp32 (200 / 199 cal), a16w8 + LN fp32, a16w8 + LN int32 internals, stock a16w8 (base only).
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { local model="$1" tag="$2"; shift 2; echo "== $model $tag $(date +%T)"
  "$PY" whisper_ptq_eval.py --model-id "openai/$model" --out-dir "$W/et131_${model}_$tag" --n-eval 200 "$@" 2>&1 \
      | grep -E "200/200|Traceback|Error"; }
for model in whisper-base whisper-small; do
  run "$model" fp32           --stages fp32-static --n-cal 8
  run "$model" a8_lnoff_200   --stages int8 --n-cal 200 --quant-config a8w8 --keep-fp32 layernorm
  run "$model" a8_lnoff_199   --stages int8 --n-cal 199 --quant-config a8w8 --keep-fp32 layernorm
  run "$model" a16_lnoff      --stages int8 --n-cal 200 --quant-config a16w8 --keep-fp32 layernorm
  run "$model" a16_ln32int    --stages int8 --n-cal 200 --quant-config a16w8 --prec-rules "$LN32"
  [ "$model" = whisper-base ] && run "$model" a16_stock --stages int8 --n-cal 200 --quant-config a16w8
done
echo "== done $(date +%T)"

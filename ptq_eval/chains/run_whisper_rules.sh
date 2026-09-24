#!/usr/bin/env bash
# Whisper-Tiny 200/200 on ExecuTorch 1.3.1: LN int16 and mask-aware variants.
# Waits for the Swin minmax chain on this GPU.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_swin_resid_minmax.log" 2>/dev/null; do sleep 20; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
run() {  # tag, args...
  local tag="$1"; shift
  echo "== whisper $tag $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w_$tag" --stages int8 --n-cal 200 --n-eval 200 "$@" 2>&1 \
      | grep -E "rewritten|200/200|Traceback|Error"
}
run a8w8_ln16            --quant-config a8w8  --prec-rules 'layer_norm$=a16w8'
run a8w8_ln16_maskaware  --quant-config a8w8  --prec-rules 'layer_norm$=a16w8' --mask-aware
run a16w8_maskaware      --quant-config a16w8 --mask-aware
run a16w8_lnoff          --quant-config a16w8 --keep-fp32 layernorm
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Whisper-tiny a8-base scheme: the remaining ~6 pt is spread; candidates not yet covered by any rule: the logits Linear (proj_out, int8 logits
# over a 51865 vocab feed greedy argmax) and the decoder embeddings. Waits for replica-3.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_fqvit_replica3.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
WLN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
BASE='(self_attn|encoder_attn)$=a16w8;layers\.\d+$=a16w8@add'
runw() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w8_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a8w8 --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
runw a8base_projout16          "proj_out\$=a16w8;$BASE;$WLN32"
runw a8base_projout16_fc2proj16 "proj_out\$=a16w8;(fc2|out_proj)\$=a16w8;$BASE;$WLN32"
runw a8base_embed16            "embed_(tokens|positions)\$=a16w8;$BASE;$WLN32"
runw a8base_all3               "proj_out\$=a16w8;(fc2|out_proj)\$=a16w8;embed_(tokens|positions)\$=a16w8;$BASE;$WLN32"
echo "== done $(date +%T)"

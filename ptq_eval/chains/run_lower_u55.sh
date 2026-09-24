#!/usr/bin/env bash
# Does the int32-interior LayerNorm (and the a16w8 Whisper config) lower for Ethos-U55-128?
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"; O="$PTQ_EVAL/chains/lower_cost"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
WLN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { local tag="$1" model="$2" qc="$3" rules="$4"; echo "== $tag $(date +%T)"
  "$PY" lower_probe.py --target ethos-u55-128 --model "$model" --quant-config "$qc" ${rules:+--prec-rules "$rules"} --out-dir "$O/$tag" > "$O/$tag.stdout" 2>&1
  grep -E "^===|^  \[" "$O/$tag.stdout" | cut -c1-260; }
run u55_deit_t_a8w8_stock deit_tiny_patch16_224 a8w8 ""
run u55_deit_t_a8w8_ln32  deit_tiny_patch16_224 a8w8 "$VLN32"
run u55_whisper_a16w8_ln32 whisper-tiny-encoder a16w8 "$WLN32"
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Cost of promoting the LN OUTPUT to int16 (projections a16in8out: int16 IFM, int8 OFM) vs o8, on Vela.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
run() { out=$OUT/lower_cost/$1; echo "== $1 $(date +%T)"; "$PY" lower_probe.py --model $2 --quant-config $3 --prec-rules "$4" "${@:5}" --out-dir $out > $out.stdout 2>&1
  grep -E "\[EthosU85\+Vela\] partitions|to_executorch|Traceback|Error" $out.stdout | cut -c1-120; grep -E "Batch Inference time" $out.stdout | head -1 | cut -c1-70; }
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run whisper_v4_o8  whisper-tiny-encoder a16w8 "conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)\$=a8w8;fc1\$=a8w8;$LN32" --ln-newton-steps 1
run whisper_v4_o16 whisper-tiny-encoder a16w8 "conv[12]\$=a8w8;encoder\$=a8w8@gelu;(q_proj|k_proj|v_proj|fc1)\$=a16in8out;out_proj\$=a8w8;$LN32" --ln-newton-steps 1
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run deit_t_res8i32o16 deit_tiny_patch16_224 a8w8 "(qkv|fc1)\$=a16in8out;$VLN32"
echo "== done $(date +%T)"

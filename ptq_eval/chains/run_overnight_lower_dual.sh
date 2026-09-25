#!/usr/bin/env bash
# Overnight (cas4k1 CPU): does the dual-range LN lower? whisper-tiny encoder deploy v4 + dual (+ Newton1); DeiT-T res8i32o8 + dual.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
run() { out=$OUT/lower_cost/$1; echo "== $1 $(date +%T)"; "$PY" lower_probe.py --model $2 --quant-config $3 --prec-rules "$4" "${@:5}" --out-dir $out > $out.stdout 2>&1
  grep -E "NewtonLayerNorm|\[TOSA\] partitions|\[EthosU85\+Vela\] partitions|to_executorch|Traceback|Error" $out.stdout | cut -c1-200; grep -E "Batch Inference time" $out.stdout | head -1 | cut -c1-70; }
run whisper_v4_dual_newton1 whisper-tiny-encoder a16w8 'conv[12]$=a8w8;encoder$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)$=a8w8;fc1$=a8w8;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add,clamp;layer_norm$=a16w8' --ln-dual-q 0.9 --ln-newton-steps 1
run deit_t_ln32_dual deit_tiny_patch16_224 a8w8 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add,clamp;norm\d?$=a16w8' --ln-dual-q 0.9 --ln-newton-steps 0
echo "== done $(date +%T)"

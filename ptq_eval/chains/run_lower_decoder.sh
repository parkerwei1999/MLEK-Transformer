#!/usr/bin/env bash
# First Whisper DECODER lowering: deploy v4 rules (int8 conv/GELU/proj/logit/fc1, fc2 int16, residual int16, LN-i32 + Newton1),
# with the token-embedding table fp32 (stock) vs int8 (our annotation). What stays on the CPU?
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
INT8='(q_proj|k_proj|v_proj|out_proj)$=a8w8;proj_out$=a8w8;fc1$=a8w8'
run() { out=$OUT/lower_cost/$1; echo "== $1 $(date +%T)"; "$PY" lower_probe.py --model whisper-tiny-decoder --quant-config a16w8 --n-cal 4 --prec-rules "$INT8;$LN32" --ln-newton-steps 1 "${@:2}" --out-dir $out > $out.stdout 2>&1
  grep -E "NewtonLayerNorm|\[TOSA\] partitions|\[EthosU85\+Vela\] partitions|to_executorch|Traceback|Error" $out.stdout | cut -c1-400; grep -E "Batch Inference time" $out.stdout | head -1 | cut -c1-70; grep -B2 -A8 Traceback $out.stdout | tail -10 | cut -c1-200; }
run whisper_dec_v4_embfp32
run whisper_dec_v4_emb8 --embedding-bits 8
echo "== done $(date +%T)"

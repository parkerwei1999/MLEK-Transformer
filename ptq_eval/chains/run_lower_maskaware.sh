#!/usr/bin/env bash
# The vision ACCURACY runs use --mask-aware; the lowering probes so far did not. Re-lower Swin-T pcsym + rewrite and DeiT-T res8i32o8 with mask-aware ON.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$PY}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { out=$OUT/lower_cost/$1; echo "== $1 $(date +%T)"; "$PY" lower_probe.py --model $2 --quant-config a8w8 --prec-rules "$3" --mask-aware "${@:4}" --out-dir $out > $out.stdout 2>&1
  grep -E "\[EthosU85\+Vela\] partitions|to_executorch|Traceback|Error" $out.stdout | cut -c1-160; grep -E "Batch Inference time" $out.stdout | awk '{s+=$4} END {printf "   Vela sum %.1f ms over %d partitions\n", s, NR}'; }
run deit_t_res8i32o8_maskaware deit_tiny_patch16_224 "$VLN32"
run swin_t_pcsym_pcrw_maskaware swin_tiny_patch4_window7_224 "blocks\\.\\d+\$=a16inpc8symout@add;$VLN32" --pc-rewrite
echo "== done $(date +%T)"

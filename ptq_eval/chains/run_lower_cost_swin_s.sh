#!/usr/bin/env bash
# Vela cost for Swin-S: stock a8w8 vs a16w8 vs stage-local widening (stage 3 int16 + LN-W32), to price the int16 stage that PTF would remove.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"; O="$PTQ_EVAL/chains/lower_cost"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
run() { local tag="$1" qc="$2" rules="$3"; echo "== $tag $(date +%T)"
  "$PY" lower_probe.py --model swin_small_patch4_window7_224 --quant-config "$qc" ${rules:+--prec-rules "$rules"} --out-dir "$O/$tag" > "$O/$tag.stdout" 2>&1
  grep -E "Batch Inference time|Total   DRAM bandwidth  |Total SRAM used" "$O/$tag.stdout" | head -3; }
run swin_s_a8w8_stock  a8w8  ""
run swin_s_a16w8_stock a16w8 ""
run swin_s_stage3_ln32 a8w8  "layers\\.2\\.=a16w8;$VLN32"
echo "== done $(date +%T)"

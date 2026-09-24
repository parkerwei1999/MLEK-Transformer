#!/usr/bin/env bash
# Lowering + Vela cost model (--verbose-cycle-estimate) for stock and deployable configs; full stdout kept per cell.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; W="$PTQ_EVAL"; O="$PTQ_EVAL/chains/lower_cost"
mkdir -p "$O"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$W"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
WLN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
run() { local tag="$1" model="$2" qc="$3" rules="$4"; echo "== $tag $(date +%T)"
  "$PY" lower_probe.py --model "$model" --quant-config "$qc" ${rules:+--prec-rules "$rules"} --out-dir "$O/$tag" > "$O/$tag.stdout" 2>&1
  grep -E "^===|^  \[" "$O/$tag.stdout" | cut -c1-200
  grep -E -A 14 "Network summary" "$O/$tag.stdout" | grep -E "Network summary|Accelerator|Total SRAM|Total DRAM|Total Off-chip|Total On-chip|Batch Inference|cycles|MAC" | head -12; }
run deit_t_a8w8_stock    deit_tiny_patch16_224 a8w8  ""
run deit_t_a16w8_stock   deit_tiny_patch16_224 a16w8 ""
run deit_t_a8w8_ln16     deit_tiny_patch16_224 a8w8  'norm\d?$=a16w8'
run deit_t_a8w8_ln32     deit_tiny_patch16_224 a8w8  "$VLN32"
run deit_b_a8w8_ln32_res16 deit_base_patch16_224 a8w8 "blocks\\.\\d+\$=a16w8@add;$VLN32"
run deit_b_a16w8_stock   deit_base_patch16_224 a16w8 ""
run swin_t_a8w8_stock    swin_tiny_patch4_window7_224 a8w8  ""
run swin_t_a16w8_stock   swin_tiny_patch4_window7_224 a16w8 ""
run swin_t_stage3_ln32   swin_tiny_patch4_window7_224 a8w8  "layers\\.2\\.=a16w8;$VLN32"
run whisper_a8w8_stock   whisper-tiny-encoder a8w8  ""
run whisper_a16w8_stock  whisper-tiny-encoder a16w8 ""
run whisper_a16w8_ln32   whisper-tiny-encoder a16w8 "$WLN32"
echo "== done $(date +%T)"

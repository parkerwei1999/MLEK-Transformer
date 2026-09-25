#!/usr/bin/env bash
# Re-run Vela on the already-lowered TOSA graphs under four Ethos-U85 configurations
# (accelerator x system config from the kit's default_vela.ini, Dedicated_Sram everywhere).
# No re-export, no re-quantization: the TOSA flatbuffers are the ones behind Table B.
set -uo pipefail
V="${VELA:-$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin/vela}"
CFG="$HOME/ml-embedded-evaluation-kit-26.06/scripts/vela/default_vela.ini"
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; LC="${PTQ_OUT:-$PTQ_EVAL/out}/lower_cost"; OUT="$LC/sweep"; mkdir -p "$OUT"
GRAPHS="deit_t_a8w8_stock deit_t_a8w8_ln32 deit_t_pcsym8i32o8_pcrw2 deit_t_a16w8_stock swin_t_a8w8_stock swin_t_stage3_ln32 swin_t_pcsym8i32o8_pcrw swin_t_a16w8_stock whisper_a8w8_stock whisper_a16w8_stock whisper_a16w8_ln32_newton1 whisper_tiny_enc_v4r16_newton1 whisper_v4_o8 whisper_v4_attn8"
CONFIGS="ethos-u85-256:Ethos_U85_SYS_DRAM_Low ethos-u85-512:Ethos_U85_SYS_DRAM_Mid_512 ethos-u85-1024:Ethos_U85_SYS_DRAM_Mid_1024 ethos-u85-2048:Ethos_U85_SYS_DRAM_High_2048"
for g in $GRAPHS; do for c in $CONFIGS; do acc="${c%%:*}"; sc="${c##*:}"; tag="${g}__${acc}"
  : > "$OUT/$tag.txt"
  for t in "$LC/$g"/tosa/*.tosa; do
    "$V" "$t" --accelerator-config "$acc" --system-config "$sc" --memory-mode Dedicated_Sram --config "$CFG" --verbose-performance --verbose-cycle-estimate --output-dir "/tmp/vela_sweep_$$/$tag" 2>&1 | grep -E "Batch Inference time|Total +DRAM bandwidth +per input|Total SRAM used|NPU cycles|^Warning|^Error" >> "$OUT/$tag.txt"
  done
  echo "== $tag $(grep -c 'Batch Inference' "$OUT/$tag.txt") partitions $(date +%T)"
done; done
echo "== done $(date +%T)"

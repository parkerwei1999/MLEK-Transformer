#!/usr/bin/env bash
# Vela accelerator / system-config sweep over EVERY lowered TOSA graph in lower_cost/
# (all models, all receipts), Dedicated_Sram, four Ethos-U85 configurations. Skips
# graph x config pairs that already have an output file.
set -uo pipefail
V="${VELA:-$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin/vela}"
CFG="$HOME/ml-embedded-evaluation-kit-26.06/scripts/vela/default_vela.ini"
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; LC="${PTQ_OUT:-$PTQ_EVAL/out}/lower_cost"; OUT="$LC/sweep"; mkdir -p "$OUT"
CONFIGS="ethos-u85-256:Ethos_U85_SYS_DRAM_Low ethos-u85-512:Ethos_U85_SYS_DRAM_Mid_512 ethos-u85-1024:Ethos_U85_SYS_DRAM_Mid_1024 ethos-u85-2048:Ethos_U85_SYS_DRAM_High_2048"
for d in "$LC"/*/tosa; do g=$(basename "$(dirname "$d")"); case "$g" in u55_*|*_pcrw|whisper_v4_dualk4_v3_newton1|deit_t_ln32_dual|whisper_v4_dual_newton1|deit_t_res8i32o16) [ "$g" = "deit_t_pcsym8i32o8_pcrw" ] && continue; case "$g" in *_pcrw) ;; *) continue;; esac;; esac
  for c in $CONFIGS; do acc="${c%%:*}"; sc="${c##*:}"; tag="${g}__${acc}"
    [ -s "$OUT/$tag.txt" ] && grep -q "Batch Inference" "$OUT/$tag.txt" && continue
    : > "$OUT/$tag.txt"
    for t in "$d"/*.tosa; do
      "$V" "$t" --accelerator-config "$acc" --system-config "$sc" --memory-mode Dedicated_Sram --config "$CFG" --verbose-performance --verbose-cycle-estimate --output-dir "/tmp/vela_sweep_$$/$tag" 2>&1 | grep -E "Batch Inference time|Total +DRAM bandwidth +per input|Total SRAM used|NPU cycles|^Error" >> "$OUT/$tag.txt"
    done
    echo "== $tag $(grep -c 'Batch Inference' "$OUT/$tag.txt") partitions $(date +%T)"
  done
done
rm -rf "/tmp/vela_sweep_$$"
echo "== done $(date +%T)"

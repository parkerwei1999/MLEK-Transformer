#!/usr/bin/env bash
# Whisper A16W8 default (stock) and repaired A16 (A16W8 + LN-i32 + Newton-1 for tiny/base/small,
# dual k=8 for medium/large-v3) graphs, lowered once and costed on all four Ethos-U85 configs.
# Lowering runs once (--tosa-only); Vela costs the dumped .tosa files: no second Ethos-U lowering, no .pte.
# Usage: run_lower_a16_pair.sh <size>:<enc|dec>:<default|repaired> ...
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
V="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin/vela"; CFG="$HOME/ml-embedded-evaluation-kit-26.06/scripts/vela/default_vela.ini"
LC="${PTQ_OUT:-$PTQ_EVAL/out}/lower_cost"; mkdir -p "$LC/sweep"; cd "$K"; export LN_DUAL_K=8
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
for cell in "$@"; do IFS=: read -r s p kind <<< "$cell"
  part=$([ "$p" = enc ] && echo encoder || echo decoder)
  if [ "$kind" = default ]; then name="whisper_${s}_${p}_a16w8_default"; R=""; F=""
  elif [ "$s" = medium ] || [ "$s" = large-v3 ]; then name="whisper_${s}_${p}_a16w8_repaired_dualk8"; R="$LNR"; F="--ln-dual-k 8 --ln-newton-steps 0"
  else name="whisper_${s}_${p}_a16w8_repaired_newton1"; R="$LN32"; F="--ln-newton-steps 1"; fi
  out=$LC/$name; echo "== $name $(hostname) $(date +%T)"
  "$PY" lower_probe.py --model whisper-$s-$part --quant-config a16w8 --prec-rules "$R" $F --tosa-only --out-dir $out > $out.stdout 2>&1
  grep -E "\[TOSA\] partitions=|LOWERING FAILED" $out.stdout | tail -2 | cut -c1-120
  for c in ethos-u85-256:Ethos_U85_SYS_DRAM_Low ethos-u85-512:Ethos_U85_SYS_DRAM_Mid_512 ethos-u85-1024:Ethos_U85_SYS_DRAM_Mid_1024 ethos-u85-2048:Ethos_U85_SYS_DRAM_High_2048; do
    acc=${c%%:*}; sc=${c##*:}; f=$LC/sweep/${name}__$acc.txt; : > $f
    for t in $out/tosa/*.tosa; do "$V" "$t" --accelerator-config $acc --system-config $sc --memory-mode Dedicated_Sram --config "$CFG" --verbose-performance --verbose-cycle-estimate --output-dir /tmp/vela_a16pair_$$ 2>&1 | grep -E "Batch Inference time|Total +DRAM bandwidth +per input|NPU cycles" >> $f; done
    echo "   $acc: $(grep -hoE 'Batch Inference time +[0-9.]+' $f | awk '{s+=$4} END{printf "%.1f ms", s}')"
  done
done
rm -rf /tmp/vela_a16pair_$$
echo "== done $(date +%T)"

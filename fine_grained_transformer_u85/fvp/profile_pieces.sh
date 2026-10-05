#!/usr/bin/env bash
# NPU latency of a model profiled in pieces (lower_probe.py --pieces all --tosa-only --out-dir <dir>):
# every piece is one profile_graph.sh cell (<tag>__<piece>), and the model latency is
#   sum over pieces of (piece NPU active cycles x repetitions)
# with no correction for overlap between pieces.
# Output: $U85_WORK/cells/<tag>__Z<macs>__pieces.tsv, one row per piece plus a TOTAL row:
#   piece  reps  n_partitions  npu_active_cycles  ms  (TOTAL: sum of cycles x reps, ms; n_partitions > 1 = CPU ops between)
# Usage: profile_pieces.sh <tag> <macs> <lower_probe out dir>
set -uo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/env.sh"
tag="$1"; c="$2"; D="$3"
[ -s "$D/pieces.tsv" ] || { echo "$D/pieces.tsv missing: run lower_probe.py --pieces all"; exit 1; }
R="$U85_WORK/cells/${tag}__Z${c}__pieces.tsv"; mkdir -p "$U85_WORK/cells"; : > "$R.part"
total=0
while IFS=$'\t' read -r piece reps _; do
  ls "$D/$piece"/tosa/*.tosa >/dev/null 2>&1 || { echo "$tag: no TOSA for piece $piece in $D/$piece/tosa"; exit 1; }
  row=$("$HERE/profile_graph.sh" "${tag}__$piece" "$c" "$D/$piece"/tosa/*.tosa | tail -1)
  a=$(cut -f4 <<< "$row"); [[ "$a" =~ ^[0-9]+$ ]] || { echo "$tag piece $piece: $row"; exit 1; }
  printf "%s\t%d\t%d\t%d\t%.4f\n" "$piece" "$reps" "$(cut -f3 <<< "$row")" "$a" "$(echo "$a / ${U85_CLOCK_HZ[$c]} * 1000" | bc -l)" >> "$R.part"
  total=$((total + a * reps))
done < "$D/pieces.tsv"
printf "TOTAL\t-\t-\t%d\t%.4f\n" "$total" "$(echo "$total / ${U85_CLOCK_HZ[$c]} * 1000" | bc -l)" >> "$R.part"
mv "$R.part" "$R"; cat "$R"

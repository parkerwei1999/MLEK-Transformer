#!/usr/bin/env bash
# NPU cycles of one graph on the Corstone-320 FVP: every TOSA partition is compiled by stock Vela and
# run on its own (model loaded at DYNAMIC_MODEL_BASE); the cell total is the sum over partitions.
# CPU work between partitions is not simulated. A run counts only if its log has "NPU TOTAL" and no
# failed invoke (the FVP exits 0 either way).
# Output: $U85_WORK/cells/<tag>__Z<macs>/result.tsv
#   tag  macs  n_partitions  npu_active_cycles  npu_total_cycles  ms (npu_active / clock)
# NPU TOTAL also counts cycles the NPU sits idle between the runner's PMU start and stop, so the
# latency is taken from NPU ACTIVE. A cell is cached under its tag; $O/inputs.sha256 records the
# partitions (content, order) and MAC count it was made from, and a mismatch is an error.
# Usage: profile_graph.sh <tag> <macs> <partition.tosa>...     (KEEP_TFLITE=1 keeps the Vela output)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
tag="$1"; c="$2"; shift 2
O="$U85_WORK/cells/${tag}__Z$c"; RUNNER="$U85_WORK/runners/Z$c.axf"
want=$( { echo "macs $c"; sha256sum "$@" | cut -d' ' -f1; } | sha256sum | cut -d' ' -f1)
if [ -d "$O" ]; then
  have=$(cat "$O/inputs.sha256" 2>/dev/null)
  [ "$have" = "$want" ] || { echo "$tag Z$c: $O was made from other inputs (or predates inputs.sha256); remove it or use another tag"; exit 1; }
  [ -s "$O/result.tsv" ] && { cat "$O/result.tsv"; exit 0; }
fi
[ -f "$RUNNER" ] || { echo "$tag Z$c: no runner, run build_runner.sh $c"; exit 1; }
M=$(ls "$FVP_HOME"/models/*/FVP_Corstone_SSE-320 | head -1)
mkdir -p "$O"; echo "$want" > "$O/inputs.sha256"; active=0; total=0; i=0
for tosa in "$@"; do
  V="$O/vela_$i"
  if ! ls "$V"/*_vela.tflite >/dev/null 2>&1; then
    "$VELA" "$tosa" --accelerator-config "ethos-u85-$c" --system-config "${U85_SYSTEM_CONFIG[$c]}" --memory-mode Dedicated_Sram \
      --config "$VELA_INI" --output-dir "$V" > "$V.log" 2>&1 || { echo "$tag Z$c part $i: VELA FAILED ($V.log)"; exit 1; }
  fi
  tfl=$(ls "$V"/*_vela.tflite)
  [ "$(stat -c %s "$tfl")" -gt 0 ] || { echo "$tag Z$c part $i: empty tflite"; exit 1; }
  [ "$(stat -c %s "$tfl")" -le $((DYNAMIC_MODEL_SIZE)) ] || { echo "$tag Z$c part $i: $(stat -c %s "$tfl") B > DYNAMIC_MODEL_SIZE, split the graph"; exit 1; }
  L="$O/fvp_$i.log"
  # runtime.sh rewrites the Python / library environment, so it is sourced only in the FVP's subshell
  ( export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"; set +u; source "$FVP_HOME/scripts/runtime.sh"; set -u
    stdbuf -oL -eL "$M" -a "$RUNNER" --data "$tfl@$DYNAMIC_MODEL_BASE" -C mps4_board.subsystem.ethosu.num_macs="$c" \
      -C mps4_board.visualisation.disable-visualisation=1 -C vis_hdlcd.disable_visualisation=1 \
      -C mps4_board.telnetterminal0.start_telnet=0 -C mps4_board.uart0.out_file=- -C mps4_board.uart0.shutdown_on_eot=1 --stat ) > "$L" 2>&1
  a=$(grep -oE "NPU ACTIVE: [0-9]+" "$L" | grep -oE "[0-9]+$"); t=$(grep -oE "NPU TOTAL: [0-9]+" "$L" | grep -oE "[0-9]+$")
  if [ -z "$t" ] || grep -q "failed to invoke\|Invoke failed" "$L"; then echo "$tag Z$c part $i: FVP run FAILED ($L)"; exit 1; fi
  active=$((active + a)); total=$((total + t)); i=$((i + 1))
  [ "${KEEP_TFLITE:-0}" = 1 ] || rm -f "$tfl"
done
printf "%s\t%s\t%d\t%d\t%d\t%.4f\n" "$tag" "$c" "$i" "$active" "$total" "$(echo "$active / ${U85_CLOCK_HZ[$c]} * 1000" | bc -l)" | tee "$O/result.tsv"

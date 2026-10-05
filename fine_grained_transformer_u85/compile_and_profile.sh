#!/usr/bin/env bash
# Compile one configuration to TOSA and measure its NPU latency on the Corstone-320 FVP.
# Usage: compile_and_profile.sh <tag> <macs> [--whole] [--lower-only] <lower_probe.py arguments...>
#   default  : in pieces (lower_probe.py --pieces all); latency = sum(piece x repetitions), fvp/profile_pieces.sh
#   --whole  : the whole graph as one cell, fvp/profile_graph.sh
#   --lower-only : stop after the lowering (a GPU host without Vela / the FVP); a later run of the same
#              tag and arguments on an FVP host reuses it (<macs> is then ignored)
# The lowering goes to $U85_WORK/lower/<tag>/ and is reused when the same tag is run again with the same
# arguments (a fresh lowering is not guaranteed to be byte-identical, and the FVP cells are keyed by their
# input bytes); the same tag with other arguments is an error.
# `python3` = the evaluation venv (the kit's resources_downloaded/env/bin goes AFTER it on PATH, see README).
# Needs a CUDA GPU (calibration) and the FVP set up by fvp/install.sh + fvp/build_runner.sh <macs>.
# Example: compile_and_profile.sh deit_t_default_a8w8 1024 --model deit_tiny_patch16_224 --quant-config a8w8
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/fvp/env.sh"
tag="$1"; c="$2"; shift 2
whole=0; lower_only=0
while :; do case "${1:-}" in --whole) whole=1; shift;; --lower-only) lower_only=1; shift;; *) break;; esac; done
for a in "$@"; do
  case "$a" in --out-dir|--pieces|--tosa-only) echo "$a is set by this script"; exit 1;; esac
done
O="$U85_WORK/lower/$tag"; mode=$([ $whole = 1 ] && echo whole || echo pieces)
want="$mode $*"
mkdir -p "$O"
# Several MAC counts of one tag may run at once: the tag lock makes one of them lower and the others
# reuse it; the per-host lock keeps lowerings (GPU calibration) from running side by side.
exec 8>"$O/.lock"; flock 8
if [ -f "$O/lower.done" ]; then
  [ "$(cat "$O/lower.args")" = "$want" ] || { echo "$tag: $O was lowered with other arguments ($(cat "$O/lower.args")); use another tag"; exit 1; }
else
  exec 9>"$U85_WORK/lower/.gpu_$(hostname).lock"; flock 9
  echo "$want" > "$O/lower.args"  # stale partitions are cleared per graph by lower_probe.py
  extra=(); [ $whole = 1 ] || extra=(--pieces all)
  python3 -u "$HERE/lower_probe.py" "$@" --tosa-only "${extra[@]}" --out-dir "$O" > "$O/lower.log" 2>&1 \
    || { echo "$tag: lowering failed ($O/lower.log)"; exit 1; }
  grep -q "LOWERING FAILED" "$O/lower.log" && { echo "$tag: lowering failed ($O/lower.log)"; exit 1; }
  touch "$O/lower.done"
  flock -u 9
fi
flock -u 8
[ $lower_only = 1 ] && { echo "$tag: lowered into $O"; exit 0; }
if [ $whole = 1 ]; then
  ls "$O"/tosa/*.tosa >/dev/null 2>&1 || { echo "$tag: no TOSA in $O/tosa"; exit 1; }
  "$HERE/fvp/profile_graph.sh" "$tag" "$c" "$O"/tosa/*.tosa
else
  "$HERE/fvp/profile_pieces.sh" "$tag" "$c" "$O"
fi

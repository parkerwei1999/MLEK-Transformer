#!/usr/bin/env bash
# Accuracy and latency runs, one command per line.
#   ./reproduce.sh list [regex]    print the commands (optionally only those matching regex)
#   ./reproduce.sh run  [regex]    run them in order
# e.g. ./reproduce.sh run 'latency .*deit_t'      ./reproduce.sh list 'accuracy .*Ours'
# Environment: README "Setup" (venv active, PYTHONPATH, kit bin appended to PATH). Accuracy needs a CUDA GPU
# and the datasets (--imagenet-dir / --librispeech-dir defaults); latency also needs the FVP setup
# (fvp/install.sh, fvp/build_runner.sh <macs>) and a host that runs the kit's Vela.
# Outputs: accuracy -> out/acc/<model>/<recipe>/summary.json; latency -> $U85_WORK/cells/<tag>__Z<macs>__pieces.tsv (TOTAL row).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DEIT="deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224"
SWIN="swin_tiny_patch4_window7_224 swin_small_patch4_window7_224 swin_base_patch4_window7_224"
WHISPER="tiny base small medium large-v3"
MACS="256 512 1024 2048"
VISION_RECIPES=("Default A8W8" "Default A16W8" "A8W8 + LN-32" "Ours A8W8")
SWIN_ONLY=("A8W8 + LN-32 + S2-A16" "Ours A8W8 + S2-A16")
WHISPER_RECIPES=("Default A8W8" "Default A16W8" "A16W8 + LN-32" "Ours A16W8" "Ours A16W8 + INT8-Linear")
# Swin-T recipes: layers.2.blocks.5's fc2 output in INT16 on top of Ours A8W8 or A8W8 + LN-32 (the latter on the
# 1000-image subset)
SWIN_T_RECIPES=("Ours A8W8 + S2-B5-FC2-A16" "Ours A8W8 + S2-B5-A16" "A8W8 + LN-32 + S2-B5-FC2-A16|--n-eval 1000")

slug() { echo "$1" | tr 'A-Z' 'a-z' | sed -E 's/ \+ /_/g; s/[ -]+/_/g'; }
short() { case "$1" in
  whisper-*) echo "$1" | sed -E 's/-encoder$/_enc/; s/-decoder$/_dec/; s/-/_/g' ;;
  *) echo "$1" | sed -E 's/_patch.*//; s/tiny/t/; s/small/s/; s/base/b/' ;; esac; }
q() { printf "'%s'" "$1"; }

cmds() {
  # accuracy: vision, ImageNet val 50k, 1000 train images for calibration
  for m in $DEIT $SWIN; do
    rs=("${VISION_RECIPES[@]}"); [[ $m == swin_* ]] && rs+=("${SWIN_ONLY[@]}")
    for r in "${rs[@]}"; do
      echo "accuracy $m | python3 vision_eval.py --model-name $m --recipe $(q "$r") --n-cal 1000 --n-eval 50000 --out-dir out/acc/$m/$(slug "$r")"
    done
  done
  for e in "${SWIN_T_RECIPES[@]}"; do r="${e%%|*}"; ne=$([[ $e == *"|"* ]] && echo "${e#*|}" || echo "--n-eval 50000")
    echo "accuracy swin_tiny_patch4_window7_224 | python3 vision_eval.py --model-name swin_tiny_patch4_window7_224 --recipe $(q "$r") --n-cal 1000 $ne --out-dir out/acc/swin_tiny_patch4_window7_224/$(slug "$r")"
  done
  # accuracy: Whisper, LibriSpeech test-clean subset (200 utterances, 100 for large-v3) and full set (2620)
  for s in $WHISPER; do
    for n in subset 2620; do
      ne=$([ $n = subset ] && echo "" || echo " --n-eval 2620 --resume")
      echo "accuracy whisper-$s fp32 $n | python3 whisper_eval.py --model-id openai/whisper-$s --stages fp32-static fp32-openai$ne --out-dir out/acc/whisper-$s/fp32_$n"
      for r in "${WHISPER_RECIPES[@]}"; do
        echo "accuracy whisper-$s $n | python3 whisper_eval.py --model-id openai/whisper-$s --recipe $(q "$r") --stages int8 --n-cal 200$ne --out-dir out/acc/whisper-$s/$(slug "$r")_$n"
      done
    done
  done
  # latency: FVP, every Ethos-U85 MAC count, in pieces
  for m in $DEIT $SWIN; do
    rs=("${VISION_RECIPES[@]}"); [[ $m == swin_* ]] && rs+=("${SWIN_ONLY[@]}")
    for r in "${rs[@]}"; do for c in $MACS; do
      echo "latency $(short $m) Z$c | ./compile_and_profile.sh $(short $m)_$(slug "$r") $c --model $m --recipe $(q "$r")"
    done; done
  done
  for r in "${SWIN_T_RECIPES[@]:0:2}"; do for c in $MACS; do
    echo "latency swin_t Z$c | ./compile_and_profile.sh swin_t_$(slug "$r") $c --model swin_tiny_patch4_window7_224 --recipe $(q "$r")"
  done; done
  for s in $WHISPER; do for p in encoder decoder; do
    g="whisper-$s-$p"
    for r in "${WHISPER_RECIPES[@]}"; do for c in $MACS; do
      echo "latency $(short $g) Z$c | ./compile_and_profile.sh $(short $g)_$(slug "$r") $c --model $g --recipe $(q "$r")"
    done; done
  done; done
}

mode="${1:-list}"; filter="${2:-.}"
case "$mode" in
  list) cmds | grep -E -- "$filter" | sed 's/^[^|]*| //' ;;
  run)  failed=0
        while IFS= read -r c; do echo "== $c"; eval "$c" || { echo "== FAILED: $c"; failed=$((failed + 1)); }
        done < <(cmds | grep -E -- "$filter" | sed 's/^[^|]*| //')
        echo "== $failed failed"; [ $failed -eq 0 ] ;;
  *) echo "usage: $0 list|run [regex]"; exit 1 ;;
esac

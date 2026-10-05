#!/usr/bin/env bash
# Download the Corstone-320 FVP (Ethos-U85) and the Arm GNU toolchain into $U85_WORK/tools.
# Both are Arm downloads under their own licence terms (the FVP installer asks to accept its EULA),
# so they are fetched here rather than committed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
FVP_URL="https://developer.arm.com/-/cdn-downloads/permalink/FVPs-Corstone-IoT/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64.tgz"
GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi.tar.xz"
T="$U85_WORK/tools"; mkdir -p "$T"; cd "$T"

if [ ! -x "$ARM_GCC_BIN/arm-none-eabi-gcc" ]; then
  curl -fL --retry 3 -o gcc.tar.xz "$GCC_URL" && tar -xf gcc.tar.xz && rm gcc.tar.xz
fi
if ! ls "$FVP_HOME"/models/*/FVP_Corstone_SSE-320 >/dev/null 2>&1; then
  curl -fL --retry 3 -o fvp.tgz "$FVP_URL" && tar -xzf fvp.tgz
  ./FVP_Corstone_SSE-320.sh --destination "$FVP_HOME"   # interactive: accept the EULA
  rm -f fvp.tgz
fi
"$ARM_GCC_BIN/arm-none-eabi-gcc" --version | head -1
ls "$FVP_HOME"/models/*/FVP_Corstone_SSE-320

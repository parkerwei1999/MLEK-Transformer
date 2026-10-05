#!/usr/bin/env bash
# Build ONE inference_runner per Ethos-U85 MAC count, with dynamic model loading: the Vela-compiled
# .tflite is passed to the FVP at run time (--data <model>@DYNAMIC_MODEL_BASE), so no per-model build.
# Output: $U85_WORK/runners/Z<macs>.axf (the build tree is removed, its logs are kept).
# Usage: build_runner.sh <256|512|1024|2048>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
c="$1"; R="$U85_WORK/runners"; BD="$R/build_Z$c"; mkdir -p "$R"
[ -f "$R/Z$c.axf" ] && { echo "Z$c runner exists: $R/Z$c.axf"; exit 0; }
export PATH="$ARM_GCC_BIN:$PATH"

# Linker script = the kit's MPS4 SSE-320 script with (1) DDR widened from 32 MiB to the whole 256 MiB
# secure window at 0x70000000 (tensor arena) and (2) the dynamic model / IFM / OFM regions resized to
# match DYNAMIC_MODEL_SIZE. Generated here so the kit's file stays the single source of the layout.
LD="$R/mps4-sse-320-u85.ld"
ifm=$(printf "0x%08X" $((DYNAMIC_MODEL_BASE + DYNAMIC_MODEL_SIZE))); ofm=$(printf "0x%08X" $((ifm + 0x01000000)))
sed -e "s/DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x02000000/DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x10000000/" \
    -e "s/DDR_dynamic_model (rx) : ORIGIN = $DYNAMIC_MODEL_BASE, LENGTH = 0x02000000/DDR_dynamic_model (rx) : ORIGIN = $DYNAMIC_MODEL_BASE, LENGTH = $DYNAMIC_MODEL_SIZE/" \
    -e "s/DDR_dynamic_ifm   (rx) : ORIGIN = 0x92000000/DDR_dynamic_ifm   (rx) : ORIGIN = $ifm/" \
    -e "s/DDR_dynamic_ofm   (rx) : ORIGIN = 0x93000000/DDR_dynamic_ofm   (rx) : ORIGIN = $ofm/" \
    "$KIT/scripts/cmake/platforms/mps4/sse-320/mps4-sse-320.gnu.ld" > "$LD"
[ "$(grep -c "LENGTH = 0x10000000\|LENGTH = $DYNAMIC_MODEL_SIZE\|ORIGIN = $ifm\|ORIGIN = $ofm" "$LD")" = 4 ] \
  || { echo "Z$c linker script: kit layout changed, check the sed patterns"; exit 1; }

cd "$KIT"
"$CMAKE" -B "$BD" -DTARGET_PLATFORM=mps4 -DTARGET_SUBSYSTEM=sse-320 \
  -DCMAKE_TOOLCHAIN_FILE="$KIT/scripts/cmake/toolchains/bare-metal-gcc.cmake" -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID="Z$c" \
  -DUSE_CASE_BUILD=inference_runner -Dinference_runner_DYNAMIC_MEM_LOAD_ENABLED=ON -DDYNAMIC_MODEL_SIZE="$DYNAMIC_MODEL_SIZE" \
  -Dinference_runner_ACTIVATION_BUF_SZ="$ACTIVATION_BUF_SZ" -DLINKER_SCRIPT_OVERRIDE_PATH="$LD" > "$R/Z$c.configure.log" 2>&1 \
  || { echo "Z$c configure FAILED ($R/Z$c.configure.log)"; exit 1; }
"$CMAKE" --build "$BD" -j4 > "$R/Z$c.build.log" 2>&1 || { echo "Z$c build FAILED ($R/Z$c.build.log)"; exit 1; }
cp "$BD/bin/mlek_inference_runner.axf" "$R/Z$c.axf" && rm -rf "$BD"
echo "Z$c runner: $R/Z$c.axf"

# Shared settings for the FVP latency flow; sourced by the other fvp/ scripts.
# Every path can be overridden from the environment.
FVP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
U85_ROOT="$(cd "$FVP_DIR/.." && pwd)"
KIT="${KIT:-$(cd "$U85_ROOT/.." && pwd)}"                     # ml-embedded-evaluation-kit checkout
U85_WORK="${U85_WORK:-$U85_ROOT/out/fvp}"                      # runners, Vela outputs, FVP logs
FVP_HOME="${FVP_HOME:-$U85_WORK/tools/FVP_Corstone_SSE-320}"   # from install.sh
ARM_GCC_BIN="${ARM_GCC_BIN:-$U85_WORK/tools/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi/bin}"
VELA="${VELA:-$KIT/resources_downloaded/env/bin/vela}"         # stock Vela: the command stream the NPU runs
CMAKE="${CMAKE:-$KIT/resources_downloaded/env/bin/cmake}"
VELA_INI="${VELA_INI:-$KIT/scripts/vela/default_vela.ini}"

# Vela system config per Ethos-U85 MAC count (same as the kit's defaults for each config).
declare -A U85_SYSTEM_CONFIG=([256]=Ethos_U85_SYS_DRAM_Low [512]=Ethos_U85_SYS_DRAM_Mid_512
                              [1024]=Ethos_U85_SYS_DRAM_Mid_1024 [2048]=Ethos_U85_SYS_DRAM_High_2048)
# NPU clock per config, for cycles -> ms (Vela ini core_clock of the system config above).
declare -A U85_CLOCK_HZ=([256]=500000000 [512]=1000000000 [1024]=1000000000 [2048]=1000000000)

# Secure DDR on MPS4 is one 256 MiB window per alias (0x70000000, 0x90000000); nothing may cross it.
# The model is loaded at run time into the 0x90000000 window, followed by the kit's IFM (16 MiB),
# OFM (16 MiB) and ML-framework scratch (32 MiB) regions, so the model gets at most 176 MiB.
# The tensor arena lives in the 0x70000000 window.
DYNAMIC_MODEL_BASE=0x90000000
DYNAMIC_MODEL_SIZE=0x0B000000
ACTIVATION_BUF_SZ=0x0C000000

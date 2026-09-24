#!/usr/bin/env bash
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PTQ_EVAL/ovl:$PTQ_EVAL"
cd "$PTQ_EVAL"
~/venv_et131_gpu/bin/python3 block_dag.py deit_base_patch16_224 blocks.5 'blocks\.\d+$=a16inptf8out@add;norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8' > "$PTQ_EVAL/chains/deit_b5_ptf_dag.txt" 2>&1
~/venv_et131_gpu/bin/python3 block_dag.py swin_tiny_patch4_window7_224 layers.2.blocks.5 'layers\.2\.blocks\.\d+$=a16inptf8out@add;layers\.2\.=a16w8;norm\d?$=fp32' > "$PTQ_EVAL/chains/swin_l2b5_ptf_dag.txt" 2>&1
echo "== done"

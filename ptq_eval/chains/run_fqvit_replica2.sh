#!/usr/bin/env bash
# Replica round 2. Swin-T stage 3: the block-level shape ops (view/roll/permute between LN->qkv and proj->add) were still int8 per-tensor
# in every "PTF + int8" cell; make them int16 (lossless carriers, as FQ-ViT never quantizes them). Whisper-tiny a8-base: same idea for the
# layer-level dropout before the residual add, plus GELU int16.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_fqvit_replica.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
WLN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
cd "$V"; i=60
runv() { i=$((i+1)); echo "== swin_tiny obs=$1 rules[$2] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_repl_$i" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers "$1" \
      --keep-fp32-sets none --mask-aware --prec-rules "$2" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
S3='layers\.2\.blocks\.\d+$=a16inptf8out@add;layers\.2\.blocks\.\d+$=a16w8'
runv histogram "$S3;$VLN32"
runv minmax    "$S3;$VLN32"
runv histogram "layers\\.2\\..*fc1\$=a8in16out;$S3;$VLN32"
runv histogram "layers\\.2\\..*(fc2|proj)\$=a16inptf8out;layers\\.2\\..*fc1\$=a8in16out;$S3;$VLN32"
cd "$W"
runw() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w7_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a8w8 --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
ATT='(self_attn|encoder_attn)$=a16w8'
runw a8base_layerops16        "$ATT;layers\\.\\d+\$=a16w8;$WLN32"
runw a8base_layerops16_gelu16 "$ATT;layers\\.\\d+\$=a16w8;activation_fn\$=a16w8;$WLN32"
runw a8base_layerops16_ptfres "layers\\.\\d+\$=a16inptf8out@add;$ATT;layers\\.\\d+\$=a16w8;$WLN32"
echo "== done $(date +%T)"

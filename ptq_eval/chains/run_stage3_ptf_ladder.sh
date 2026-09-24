#!/usr/bin/env bash
# (A) Swin-T: can stage 3 stay mostly int8 with PTF? residual + fc2/proj outputs stored PTF int8, LN int32 internals; then + attention core int16; then + qkv/fc1 int16.
# (B) Whisper-tiny: a8w8 base + attention int16 + LN32 + residual int16 (12.1%), then which int8 piece costs the rest: fc2/out_proj outputs PTF, all Linear int16, conv front-end int16.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
until grep -q "^== done" "$PTQ_EVAL/chains/run_swin_deploy.log" 2>/dev/null; do sleep 30; done
PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; V="$PTQ_EVAL"; W="$PTQ_EVAL"
export PYTHONPATH="$W/ovl:$W"
export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
WLN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
cd "$V"; i=40
runv() { i=$((i+1)); echo "== swin_tiny rules[$1] $(date +%T)"
  "$PY" vision_ptq_eval.py --out-dir "$V/et131_s3ptf_$i" --model-name swin_tiny_patch4_window7_224 --quant-configs a8w8 --act-observers histogram \
      --keep-fp32-sets none --mask-aware --prec-rules "$1" --n-cal 1000 --n-eval 1000 2>&1 | grep -E "top1|Traceback|Error"; }
S3PTF='layers\.2\..*(fc2|proj)$=a16inptf8out;layers\.2\.blocks\.\d+$=a16inptf8out@add'
runv "$S3PTF;$VLN32"
runv "$S3PTF;layers\\.2\\..*attn\$=a16w8;$VLN32"
runv "$S3PTF;layers\\.2\\..*attn\$=a16w8;layers\\.2\\..*(qkv|fc1)\$=a16w8;$VLN32"
cd "$W"
runw() { echo "== whisper-tiny $1 $(date +%T)"
  "$PY" whisper_ptq_eval.py --out-dir "$W/et131_w6_$1" --stages int8 --n-cal 200 --n-eval 200 --quant-config a8w8 --prec-rules "$2" 2>&1 | grep -E "200/200|Traceback|Error"; }
BASE='(self_attn|encoder_attn)$=a16w8;layers\.\d+$=a16w8@add'
runw a8base_fc2proj_ptf   "(fc2|out_proj)\$=a16inptf8out;$BASE;$WLN32"
runw a8base_alllinear16   "(q_proj|k_proj|v_proj|out_proj|fc1|fc2)\$=a16w8;$BASE;$WLN32"
runw a8base_conv16        "conv[12]\$=a16w8;$BASE;$WLN32"
runw a8base_fc2proj16     "(fc2|out_proj)\$=a16w8;$BASE;$WLN32"
echo "== done $(date +%T)"

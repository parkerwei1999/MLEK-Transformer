#!/usr/bin/env bash
# TOSA + Vela receipts for the sizes that had none, in each one's headline recipe:
# vision pcsym8i32o8 + pc_rewrite + mask-aware; Whisper encoders deploy v4 (PCS residual)
# + Newton-1 + pc_rewrite; medium encoder with the dual k=8 LayerNorm; base decoder.
set -uo pipefail
PTQ_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; K="$PTQ_EVAL"; PY="${PTQ_PY:-$HOME/venv_et131_gpu/bin/python3}"; OUT="${PTQ_OUT:-$PTQ_EVAL/out}"; mkdir -p "$OUT"
export PYTHONPATH="$K/ovl:$K"; export PATH="$HOME/ml-embedded-evaluation-kit-26.06/resources_downloaded/env/bin:$PATH"
cd "$K"; export LN_DUAL_K=8
VLN32='norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'
VPCS='blocks\.\d+$=a16inpc8symout@add'
LN32='layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
LNR='layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'
ENC8='conv[12]$=a8w8;encoder$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)$=a8w8;fc1$=a8w8'
DEC8='(q_proj|k_proj|v_proj|out_proj)$=a8w8;proj_out$=a8w8;fc1$=a8w8'
WPCS='layers\.\d+$=a16inpc8symout@add'
run() { out=$OUT/lower_cost/$1; echo "== $1 $(date +%T)"
  "$PY" lower_probe.py --model $2 --quant-config $3 --prec-rules "$4" "${@:5}" --out-dir $out > $out.stdout 2>&1
  grep -E "partitions=|LOWERING FAILED|Traceback|Batch Inference time" $out.stdout | tail -3 | cut -c1-160; }
run deit_s_pcsym8i32o8_pcrw   deit_small_patch16_224        a8w8  "$VPCS;$VLN32" --mask-aware --pc-rewrite
run swin_b_pcsym8i32o8_pcrw   swin_base_patch4_window7_224  a8w8  "$VPCS;$VLN32" --mask-aware --pc-rewrite
run whisper_base_enc_v4_pcrw  whisper-base-encoder          a16w8 "$ENC8;$WPCS;$LN32" --ln-newton-steps 1 --pc-rewrite
run whisper_small_enc_v4_pcrw whisper-small-encoder         a16w8 "$ENC8;$WPCS;$LN32" --ln-newton-steps 1 --pc-rewrite
run whisper_base_dec_v4_emb8  whisper-base-decoder          a16w8 "$DEC8;$LN32" --ln-newton-steps 1 --embedding-bits 8
run whisper_medium_enc_v4_dual_pcrw whisper-medium-encoder  a16w8 "$ENC8;$WPCS;$LNR" --ln-dual-k 8 --ln-newton-steps 0 --pc-rewrite
echo "== done $(date +%T)"

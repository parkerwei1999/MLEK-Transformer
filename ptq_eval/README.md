# ptq_eval: mixed-precision PTQ evaluation for the ExecuTorch Arm (Ethos-U85) flow

Fake-quant accuracy harness for DeiT / Swin (ImageNet) and Whisper (LibriSpeech) on the stock
ExecuTorch 1.3.1 Arm quantizer, plus TOSA / Vela lowering probes. Nothing in ExecuTorch, torchao or
Vela is patched: every extension is a quantizer subclass or a graph pass in this directory.

## Layout

| file | role |
|---|---|
| `whisper_ptq_eval.py` | Whisper harness (encoder + static-length greedy decoder), `QuantConfig` / `ActObserver` enums, `--prec-rules` parser, `--dump-ln-scales`, `--ln-newton-steps` |
| `vision_ptq_eval.py` | DeiT / Swin harness; imports the shared pieces from `whisper_ptq_eval.py` |
| `mask_aware_quantizer.py` | `MixedPrecisionQuantizer` (per-module / per-op precision rules), `MaskAwareQuantizer`, the memory-op pass (`_widen_memory_ops`) |
| `ptf_observer.py` | `PTFPerChannelObserver` (per-channel power-of-two int8, `PTF_MAX_ALPHA` env overrides the alpha clamp) and `KMedianObserver` (scale = k x median, saturating; the fine rsqrt grid of the dual-range LayerNorm) |
| `newton_layernorm.py` | `NewtonLayerNorm`: keepdim LayerNorm with an rsqrt table seed, optional int32 Newton steps (`--ln-newton-steps N`) and the optional dual-range rsqrt (`--ln-dual-k K`: two int16 tables blended by an int16 mask) |
| `pc_rewrite.py` | rewrites symmetric per-channel activation Q/DQ (`a16inpc8symout`) into per-tensor int8 Q/DQ plus two int32 MULs so the graph lowers to TOSA / Vela (`--pc-rewrite`) |
| `levels_report.py`, `ln_var_profile.py`, `ln_unit_check.py` | calibration-only diagnostics: levels left for the median token at every quantize node; per-LayerNorm variance range (sink tokens); standalone fake-quant error of one LayerNorm per rsqrt scheme |
| `whisper_executorch_wrapper.py` | HF Whisper split into exportable encoder / decoder modules |
| `lower_probe.py` | TOSA + Ethos-U85 Vela lowering check for one configuration (`--target`, `--vela-flags`) |
| `block_dag.py`, `dump_boundaries.py`, `fc2_profile.py` | diagnostics: per-block Q/DQ DAG, boundary dump, fp32 activation survey |
| `fqvit_models/` | vendored FQ-ViT model definitions + ImageNet loader (see its `VERSION.md`) |
| `data/` | LibriSpeech loader and the Whisper calibration / WER protocol |
| `chains/` | the experiment scripts behind the recorded results (`PTQ_EVAL` = this dir, `PTQ_PY` = python) |

## Setup

```bash
./build_venv_gpu.sh            # ~/venv_et131_gpu: torch 2.12 cu130, executorch 1.3.1, torchao 0.17, vela 5.1.0
./build_overlay.sh             # ptq_eval/ovl: transformers 4.57 / openai-whisper / jiwer (shadow the venv, see requirements-overlay.txt)
export PYTHONPATH="$PWD/ovl:$PWD"
export PATH="<kit>/resources_downloaded/env/bin:$PATH"   # flatc / vela binaries used by the lowering probes
```

Datasets: `--imagenet-dir` (train + val in torchvision ImageFolder layout), `--librispeech-dir`
(dev-clean, test-clean). Checkpoints download on first use (timm / torch.hub URLs, Hugging Face).

## Running

```bash
# DeiT-T, int8 everywhere except int32 LayerNorm internals with int16 in/out (res8i32o8)
python3 vision_ptq_eval.py --model-name deit_tiny_patch16_224 --quant-configs a8w8 --act-observers histogram \
    --keep-fp32-sets none --mask-aware --n-cal 1000 --n-eval 1000 --out-dir out/deit_t \
    --prec-rules 'norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8'

# Whisper-base, a16w8 base with int8 projections / conv / fc1, PTF residual, int32 LayerNorm
python3 whisper_ptq_eval.py --model-id openai/whisper-base --stages int8 --n-cal 200 --n-eval 200 \
    --quant-config a16w8 --act-observer minmax --out-dir out/whisper_base --prec-rules \
    'conv[12]$=a8w8;encoder$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)$=a8w8;proj_out$=a8w8;fc1$=a8w8;layers\.\d+$=a16inptf8out@add;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'

# Whisper-medium, the recipe that survives the large sizes: int8 conv / front GELU / projections / fc1,
# int16 residual, int32 LayerNorm internals, dual-range rsqrt (fine table = 8 x median variance)
LN_DUAL_K=8 python3 whisper_ptq_eval.py --model-id openai/whisper-medium --stages int8 --n-cal 200 --n-eval 200 \
    --quant-config a16w8 --act-observer minmax --ln-dual-k 8 --ln-newton-steps 0 --out-dir out/whisper_medium --prec-rules \
    'conv[12]$=a8w8;encoder$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)$=a8w8;proj_out$=a8w8;fc1$=a8w8;layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8'

# lowering check (TOSA + Vela) of a configuration; add --pc-rewrite for per-channel residuals,
# --ln-newton-steps / --ln-dual-k for the LayerNorm variants, --embedding-bits 8 for the decoder table
python3 lower_probe.py --model deit_tiny_patch16_224 --quant-config a8w8 --prec-rules '...' --target ethos-u85-256
python3 lower_probe.py --model whisper-tiny-encoder --quant-config a16w8 --prec-rules '...' --ln-newton-steps 1 --pc-rewrite
```

Every recorded number has a chain script under `chains/` that reproduces it (`chains/receipts.sh`
prints the source lines). Chain environment: `PTQ_EVAL` = this directory, `PTQ_PY` = the python to
use, `PTQ_OUT` = where logs and per-cell outputs go (default `out/`), `FQVIT_DIR` = an FQ-ViT checkout
at the commit in `fqvit_models/VERSION.md` for the two native-FQ-ViT baseline chains.

### Precision rules

`--prec-rules 'regex=config[:observer][@op,op,...];...'`, first match wins. The regex is matched against
the deepest module name of a node (nodes without a module stack inherit their first user's). `@ops`
restricts a rule to those op types (names as in `aten.<op>`, e.g. `sum.dim_IntList`, `mul`, `rsqrt`).
Configs: `a8w8`, `a16w8`, `a16w8e16` (int16 with a 2^-16 scale floor), `a32w8` (int32 activations, for
LayerNorm internals), `a8in16out`, `a16in8out`, `a16inptf8out` (PTF per-channel int8 output, fake-quant
only), `a16inpc8out` (per-channel int8 with zero points, fake-quant only), `a16inpc16out`,
`a16inpc8symout` (symmetric per-channel int8, the lowerable "PCS" form), `fp32`.
Observers: `histogram` (default for int8), `minmax`, `kmedian` (k x median, saturating; k from `LN_DUAL_K`).

### Memory-op pass

After annotation, shape / memory ops (view, permute, slice, cat, ...) whose producer is per-channel get
their annotation removed (the per-channel values pass through, the consumer re-quantizes); those whose
producer is int16 / int32 per-tensor get int16 carriers. `PTQ_MEMORY_OP_PASS=off` restores the stock
int8 boundaries (A/B control). The harness prints `memory-op pass: transparent=N int16-carrier=M`.

## Known limits

* Per-channel activations lower only in the symmetric form: `a16inpc8symout` + `--pc-rewrite` becomes
  per-tensor int8 Q/DQ plus two int32 MULs by constant vectors (one NPU partition on DeiT / Whisper).
  `a16inptf8out` and `a16inpc8out` carry per-channel zero points, which the rewrite does not handle
  and TOSA's `input_zp = 0` rule on the int16 / int32 path forbids, so they stay fake-quant experiments.
* The Whisper decoder is exported as a static 128-position graph without a KV cache and is run once
  per generated token (pad, run, read the logits of the current position); its Vela number is the
  cost of one such pass, i.e. the constant per-token latency of that scheme, and time-to-first-token
  is the encoder pass plus one decoder pass. A cached single-step decoder is not exported.
* Swin's `torch.roll` decomposes to ops the Arm partitioner leaves on the CPU; Vela cost numbers for
  Swin are sums over its partitions.

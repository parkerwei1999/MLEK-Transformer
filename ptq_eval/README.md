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
| `ptf_observer.py` | per-channel power-of-two-factor int8 activation observer (`PTF_MAX_ALPHA` env overrides the alpha clamp) |
| `newton_layernorm.py` | `NewtonLayerNorm`: rsqrt table seed + int32 Newton steps (`--ln-newton-steps N`) |
| `whisper_et_model.py` | HF Whisper split into exportable encoder / decoder modules |
| `lower_probe.py` | TOSA + Ethos-U85 Vela lowering check for one configuration (`--target`, `--vela-flags`) |
| `block_dag.py`, `dump_boundaries.py`, `fc2_census.py` | diagnostics: per-block Q/DQ DAG, boundary dump, fp32 activation survey |
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

# lowering check (TOSA + Vela) of a configuration
python3 lower_probe.py --model deit_tiny_patch16_224 --quant-config a8w8 --prec-rules '...' --target ethos-u85-256
```

### Precision rules

`--prec-rules 'regex=config[:observer][@op,op,...];...'`, first match wins. The regex is matched against
the deepest module name of a node (nodes without a module stack inherit their first user's). `@ops`
restricts a rule to those op types (names as in `aten.<op>`, e.g. `sum.dim_IntList`, `mul`, `rsqrt`).
Configs: `a8w8`, `a16w8`, `a32w8` (int32 activations, for LayerNorm internals), `a8in16out`,
`a16in8out`, `a16inptf8out` (PTF per-channel int8 output), `a16inpc8out`, `a16inpc16out`, `fp32`.
Observers: `histogram` (default for int8), `minmax`.

### Memory-op pass

After annotation, shape / memory ops (view, permute, slice, cat, ...) whose producer is per-channel get
their annotation removed (the per-channel values pass through, the consumer re-quantizes); those whose
producer is int16 / int32 per-tensor get int16 carriers. `PTQ_MEMORY_OP_PASS=off` restores the stock
int8 boundaries (A/B control). The harness prints `memory-op pass: transparent=N int16-carrier=M`.

## Known limits

* Per-channel activation specs (`a16inptf8out`, `a16inpc*`) are accuracy experiments; the Arm backend
  and Vela only take per-tensor activations, so those graphs do not lower yet (the planned rewrite is
  per-tensor Q/DQ + int32 MUL by a per-channel constant).
* Swin's `torch.roll` decomposes to ops the Arm partitioner leaves on the CPU; Vela cost numbers for
  Swin are sums over its partitions.

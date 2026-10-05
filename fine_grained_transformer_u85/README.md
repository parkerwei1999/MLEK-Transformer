# fine_grained_transformer_u85: mixed-precision PTQ of Transformers for the Arm Ethos-U85

Post-training quantization of DeiT / Swin (ImageNet) and Whisper (LibriSpeech) through the stock
ExecuTorch 1.3.1 Arm quantizer, with fake-quant accuracy, TOSA lowering, and NPU latency on the
Corstone-320 FVP. Every extension is a quantizer subclass or a graph pass under `core/`.

## Setup

```bash
git submodule update --init fine_grained_transformer_u85/3rdparty/FQ-ViT      # from the kit root
python_env/build_venv_gpu.sh ~/venv_et131_gpu                    # torch 2.12 cu130, executorch 1.3.1, torchao 0.17, vela 5.1.0 + requirements-gpu.txt
python_env/build_overlay.sh ~/venv_et131_gpu/bin/python3         # ./ovl: transformers 4.57 / openai-whisper / jiwer for whisper (python_env/requirements-overlay.txt)
source ~/venv_et131_gpu/bin/activate
export PYTHONPATH="$PWD/ovl:$PWD"
export PATH="$PATH:<kit>/resources_downloaded/env/bin"             # flatc / vela; appended: its python3 has CPU-only torch
```

`ovl/` goes first on `PYTHONPATH` because transformers < 5 needs older `tokenizers` /
`huggingface_hub` than the executorch venv ships.

Datasets: `--imagenet-dir` (train + val in torchvision ImageFolder layout) and `--librispeech-dir`
(dev-clean for calibration, test-clean for evaluation). Checkpoints download on first use: DeiT from
FQ-ViT's URLs, Whisper from Hugging Face. Swin uses the ImageNet-22k -> 1k fine-tuned checkpoints
(`SWIN_CHECKPOINTS` in `helper/fqvit_models/__init__.py`, sha256-checked), not FQ-ViT's ImageNet-1k ones.

## Quick start

A configuration is named by a recipe (`--recipe`, table below); the same name works in all three entry points.

```bash
# accuracy (fake-quant, GPU): ImageNet top-1 / LibriSpeech WER
python3 vision_eval.py --model-name deit_tiny_patch16_224 --recipe "A8W8 + LN-32" --out-dir out/acc/deit_t
python3 whisper_eval.py --model-id openai/whisper-tiny --recipe "Ours A16W8" --stages int8 --out-dir out/acc/whisper_tiny

# compile only: TOSA partitions + op / dtype report
python3 lower_probe.py --model deit_tiny_patch16_224 --recipe "A8W8 + LN-32" --tosa-only --out-dir out/lower/deit_t

# compile and simulate: NPU latency on the Corstone-320 FVP (1024-MAC Ethos-U85)
./compile_and_profile.sh deit_t_a8w8_ln_32 1024 --model deit_tiny_patch16_224 --recipe "A8W8 + LN-32"
```

Accuracy and latency results can be reproduced with ./reproduce.sh (list / run <filter>)

## Layout

| path | role |
|---|---|
| `vision_eval.py` | DeiT / Swin accuracy: calibrate, convert, ImageNet top-1 / top-5 of the fake-quant graph |
| `whisper_eval.py` | Whisper accuracy: encoder + static-length greedy decoder, WER / CER on LibriSpeech |
| `compile_and_profile.sh` | one configuration end to end: `lower_probe.py` (TOSA) then the FVP, in pieces by default (`--whole` for one graph) |
| `reproduce.sh` | the command behind accuracy and latency result (`list` / `run` with a filter) |
| `lower_probe.py` | one configuration through TOSA (and optionally Ethos-U / Vela): op / dtype report, `.tosa` partitions for the FVP |
| `core/recipes.py` | the named recipes (`--recipe`) |
| `core/quant.py` | `QuantConfig`, `ActObserver`, `--prec-rules` parser, CPU-prepare / CUDA-calibrate / convert |
| `core/mixed_precision_quantizer.py` | `MixedPrecisionQuantizer` (per-module / per-op precision rules) and the memory-op pass |
| `core/newton_layernorm.py` | `NewtonLayerNorm`: LayerNorm with an rsqrt table seed, optional int32 Newton steps, optional dual-range rsqrt |
| `core/ptf_observer.py` | `PTFPerChannelObserver` (FQ-ViT power-of-two per-channel int8), `KMedianObserver` (scale = k x median, saturating) |
| `core/pcs_decompose.py` | lowers symmetric per-channel activation Q/DQ (`a16inpc8symout`) to per-tensor int8 Q/DQ + two int32 MULs |
| `core/slices.py` | splits a model into prologue / repeated blocks / epilogue pieces for per-piece FVP runs |
| `core/lowering.py` | TOSA op / dtype report helpers |
| `helper/fqvit_models/` | adapter over the FQ-ViT submodule (`build_model`) + ImageNet loader |
| `helper/whisper_executorch_wrapper.py`, `helper/whisper_io.py` | Whisper split into exportable encoder / decoder modules; log-mel front end, greedy decoder |
| `helper/data/` | LibriSpeech loader, Whisper calibration / WER protocol |
| `helper/diagnostics/` | calibration-only probes: quantization levels, LayerNorm variance, Q/DQ DAG per block, boundary dumps, activation dumps |
| `fvp/` | Corstone-320 FVP flow: install, build the runner, profile a graph |
| `vela_fix/` | patch to Vela 5.1.0's DRAM byte count (secondary Vela estimates only, see below) |
| `python_env/` | venv + overlay build scripts and their pinned requirements |
| `tests/` | internal unit tests and numerical checks (`tests/README.md`) |
| `3rdparty/FQ-ViT` | git submodule: megvii-research/FQ-ViT @ `5daf591` |

## Recipes

Defined in `core/recipes.py`; a (model, recipe) pair that is not defined there is an error.
`S<i>` / `S<i>-B<j>` name `layers.<i>` / `layers.<i>.blocks.<j>` and count from 0 like the module paths
(so `S2` is the Swin paper's stage 3).

| recipe | models | what it changes |
|---|---|---|
| `Default A8W8`, `Default A16W8` | all | nothing: the stock ExecuTorch Arm quantizer |
| `A8W8 + LN-32` | DeiT, Swin | LayerNorm internals in INT32, LayerNorm input in INT16, output rescaled to consumer's grid |
| `Ours A8W8` | DeiT, Swin | + LN-32 + per-channel symmetric INT8 residual (PCS), evaluated and lowered in its rewritten form |
| `A8W8 + LN-32 + S2-A16` | Swin | LN-32 with all of `layers.2` (stage 3) in INT16, no PCS |
| `Ours A8W8 + S2-B5-FC2-A16` | Swin-T | Ours A8W8 with the fc2 output of block `layers.2.blocks.5` (stage 3's last block) in INT16 |
| `A8W8 + LN-32 + S2-B5-FC2-A16` | Swin-T | the same fc2 output in INT16 on A8W8 + LN-32 (no PCS) |
| `A16W8 + LN-32` | Whisper | LayerNorm internals in INT32 (no Newton step / dual table) |
| `Ours A16W8` | Whisper | A16W8 + GA-LN: LN-32 plus one Newton step (tiny, base, small) or the dual-range rsqrt (medium, large-v3) |
| `Ours A16W8 + INT8-Linear` | Whisper | Ours A16W8 with every Linear except fc2 (the q / k / v / out projections of self- and cross-attention, fc1, logits) plus the conv front end (conv1, conv2, their GELUs) and the conv front end in INT8; the attention matmul outputs, softmax and fc2 stay INT16 |

## Accuracy

`vision_eval.py`: ImageNet val (`--imagenet-dir`), 1000 train images for calibration, 50 000 for
evaluation by default. `whisper_eval.py`: LibriSpeech (`--librispeech-dir`), 200 dev-clean utterances
for calibration, test-clean for evaluation (200 utterances, 100 for large-v3; `--n-eval 2620` = full).
`--stages` of `whisper_eval.py`: `fp32-openai` (openai-whisper `transcribe()`), `fp32-static` (the FP32
wrappers through this repo's static-length decode), `int8` (the quantized wrappers, same decode).
The accuracy runners evaluate the fake-quant graph. Per-channel activations (PCS recipes) are always rewritten to
their lowered form (per-tensor INT8 + INT32 MULs, `core/pcs_decompose.py`) before evaluation, the same graph the lowering produces.

## Lowering

`lower_probe.py --model <name | whisper-<size>-<encoder|decoder>> --recipe ... --tosa-only` writes the
TOSA partitions to `<out>/tosa/*.tosa` and an op / dtype report to `<out>/summary.json`; `--pieces all`
lowers each piece (see below) into `<out>/<piece>/tosa`. Without `--tosa-only` it also runs the Ethos-U
lowering and writes a `.pte`; `--verbose-partition` logs why the Arm partitioner leaves a node on the CPU.
`lower_probe.py` always calibrates with the histogram observer, whatever the recipe's observer; latency
depends on the graph's ops and dtypes, not on the calibrated scales.

## Latency on the Corstone-320 FVP

```bash
fvp/install.sh                     # FVP + Arm GNU toolchain into $U85_WORK/tools (asks to accept the FVP EULA)
fvp/build_runner.sh 1024           # one runner per MAC count: 256 | 512 | 1024 | 2048
./compile_and_profile.sh swin_t_ours_a8w8 1024 --model swin_tiny_patch4_window7_224 --recipe "Ours A8W8"
```

`compile_and_profile.sh <tag> <macs> [--whole] [--lower-only] <lower_probe.py args>` lowers the model
into `$U85_WORK/lower/<tag>` (reused when the same tag runs again with the same arguments; the same tag
with other arguments is an error) and profiles it. Result: `$U85_WORK/cells/<tag>__Z<macs>__pieces.tsv`,
one row per piece plus `TOTAL` (NPU active cycles and ms). `--lower-only` stops after the lowering, for a
GPU host without the FVP; the FVP host then runs the same command without it.

**In pieces (default).** The model is split (`core/slices.py`) into a prologue, one representative per
group of identical blocks, and an epilogue; blocks are grouped only when they are structurally identical
and the same precision rules apply to them. Each piece is calibrated on the inputs it receives inside
the full fp32 model, compiled with stock Vela and run on the FVP on its own; the model latency is
`sum(piece NPU active cycles x repetitions)`, without correcting for overlap between pieces.
`--whole` profiles the whole graph as one cell (`fvp/profile_graph.sh`).

The FVP runner loads the model at run time, so one runner per MAC count serves every model. A cell is
cached under its tag with a hash of its inputs; a tag reused with other inputs is an error.
`KEEP_TFLITE=1` keeps the Vela output. Paths come from `fvp/env.sh` and can be overridden from the
environment: `U85_WORK` (default `out/fvp`), `FVP_HOME`, `ARM_GCC_BIN`, `VELA`, `VELA_INI`, `KIT`.

Vela's own cycle estimate (`lower_probe.py` without `--tosa-only`) is a secondary number. Vela 5.1.0
charges int16 / int32 feature-map DRAM traffic at 1/2 and 1/4 of its bytes;
`vela_fix/vela_5.1.0_bytes_fix.patch` corrects that in a Vela source build.

## Custom configurations

Instead of `--recipe`, a configuration can be given directly: `--quant-config` (`a8w8`, `a16w8`) plus
`--prec-rules 'pattern=config[:observer][@ops];...'`, where `pattern` is a regular expression on module
names (first match wins) and `@ops` limits a rule to some op types. `core/recipes.py` spells out every
recipe in this form. Configs are listed in `QuantConfig` (`core/quant.py`); observers are `histogram`,
`minmax` and `kmedian` (the dual-range LayerNorm's fine table, sized by `--ln-dual-k`).

`--ln-newton-steps N` replaces every LayerNorm by `NewtonLayerNorm` (rsqrt table seed + N INT32 Newton
steps); `--ln-dual-k K` adds the dual-range rsqrt (a fine and a coarse INT16 table blended by a mask).

Memory-op pass: shape / memory ops (view, permute, slice, cat, ...) after a per-channel producer pass
the values through, and after an INT16 / INT32 producer get an INT16 carrier of their own instead of
the stock INT8 boundary. It is always on; the runners print how many ops it touched.
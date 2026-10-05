"""Post-training quantization (PTQ) with real-speech calibration, plus word /
character error rate (WER / CER) evaluation, for the Whisper Ethos-U wrappers
defined in helper/whisper_executorch_wrapper.py.

`examples.arm.aot_arm_compiler` can only calibrate on the single example input,
so this script repeats its quantization recipe (EthosUQuantizer + the default
symmetric config, strict export) with a calibration loop over LibriSpeech.

Measurement protocol: calibrate on
the first N utterances of dev-clean, evaluate on the first N of test-clean,
normalize with Whisper's EnglishTextNormalizer, score with jiwer.

The decoder wrapper has a static length and no KV cache, so greedy decoding
re-runs the whole decoder per generated token and reads the logits at the last
filled position; the causal mask keeps the trailing padding from leaking back.
"""
import argparse
import json
import random
import time
from enum import Enum
from pathlib import Path

import jiwer
import torch
import whisper
from transformers import AutoConfig
from whisper.normalizers import EnglishTextNormalizer

import core.mixed_precision_quantizer as maq
from core.pcs_decompose import rewrite_per_channel_activations
from core.quant import (ActObserver, EthosUQuantizer, QuantConfig, calibrate_and_convert, load_module_from_path,
                        move_graph_module, parse_rules, prepare_on_cpu, quantizer_compile_spec)
from helper import whisper_io as wio

HERE = Path(__file__).resolve().parent
DEC_LEN = 128  # static decoder length (prompt + generated tokens); the exported decoder graph has this many positions


class Stage(Enum):
    FP32_OPENAI = "fp32-openai"  # openai-whisper transcribe() with the settings of helper/data/whisper_protocol.py
    FP32_STATIC = "fp32-static"  # FP32 wrappers through this script's static-length decode
    INT8 = "int8"                # quantized wrappers through the same decode


def score(refs, hyps, normalizer):
    refs_n = [normalizer(r.upper()) for r in refs]
    hyps_n = [normalizer(h.strip().upper()) for h in hyps]
    return jiwer.wer(refs_n, hyps_n) * 100, jiwer.cer(refs_n, hyps_n) * 100


def load_done_hyps(hyp_path: Path) -> dict:
    """Records of an earlier, interrupted run keyed by audio path; a torn last line is dropped."""
    done = {}
    if hyp_path.exists():
        for line in hyp_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["audio"]] = rec
    return done


def evaluate(label: str, transcribe_fn, pairs, load_audio, normalizer, out_dir: Path, resume: bool):
    """transcribe_fn(audio) -> (text, hit_limit, n_steps).

    With resume, utterances already in the hyps file are reused instead of decoded again.
    """
    refs, hyps, n_limit, n_steps = [], [], 0, 0
    t0 = time.time()
    # Long rule strings overflow the 255-byte filename limit: keep the full label in the summary, shorten the file name.
    import hashlib
    fname = label if len(label) <= 120 else label[:80] + "-" + hashlib.md5(label.encode()).hexdigest()[:8]
    hyp_path = out_dir / f"hyps_{fname}.jsonl"
    done = load_done_hyps(hyp_path) if resume else {}
    n_resumed = 0
    # Line buffering puts every decoded utterance on disk right away, so a crash loses at most one line.
    with open(hyp_path, "w", buffering=1) as hyp_file:
        for i, (audio_path, ref) in enumerate(pairs):
            rec = done.get(audio_path)
            if rec is None:
                text, hit_limit, steps = transcribe_fn(load_audio(audio_path))
                rec = {"audio": audio_path, "ref": ref, "hyp": text, "hit_limit": hit_limit, "steps": steps}
            else:
                n_resumed += 1
            refs.append(ref)
            hyps.append(rec["hyp"])
            n_limit += int(rec["hit_limit"])
            n_steps += rec.get("steps", 0)
            hyp_file.write(json.dumps(rec) + "\n")
            if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
                wer, cer = score(refs, hyps, normalizer)
                print(f"  [{label}] {i + 1}/{len(pairs)} WER={wer:.2f}% CER={cer:.2f}% "
                      f"hit_limit={n_limit} ({time.time() - t0:.0f}s)", flush=True)
    wer, cer = score(refs, hyps, normalizer)
    return {"label": label, "n_eval": len(pairs), "wer_pct": wer, "cer_pct": cer,
            "n_hit_static_length": n_limit, "decode_steps": n_steps,
            "eval_time_sec": time.time() - t0, "n_resumed": n_resumed}


def main(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    stages = {Stage(s) for s in args.stages}
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    data = load_module_from_path("librispeech_data", HERE / "helper/data/librispeech_data.py")
    protocol = load_module_from_path("whisper_protocol", HERE / "helper/data/whisper_protocol.py")
    wrapper = load_module_from_path("whisper_executorch_wrapper", HERE / "helper/whisper_executorch_wrapper.py")

    eval_pairs = data.gather_librispeech_files(args.librispeech_dir, "test-clean", args.n_eval)
    cal_pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
    normalizer = EnglishTextNormalizer()
    results = {"args": vars(args), "stages": []}

    def finish(stage_result):
        results["stages"].append(stage_result)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)

    if Stage.FP32_OPENAI in stages:
        # openai-whisper names: tiny / base / small / medium / large-v3 = the HF id without "openai/whisper-"
        model = whisper.load_model(args.model_id.rsplit("/", 1)[-1].removeprefix("whisper-"), device=device)

        def transcribe_openai(audio):
            return protocol.transcribe_whisper_deterministic(model, audio)["text"], False, 0

        finish(evaluate(Stage.FP32_OPENAI.value, transcribe_openai, eval_pairs,
                        data.load_audio_torchaudio, normalizer, out_dir, args.resume))
        del model

    wio.N_MELS = AutoConfig.from_pretrained(args.model_id).num_mel_bins
    if not stages & {Stage.FP32_STATIC, Stage.INT8}:
        return

    encoder, enc_example = wrapper._build(args.model_id, wrapper.WhisperPart.ENCODER, DEC_LEN)
    decoder, dec_example = wrapper._build(args.model_id, wrapper.WhisperPart.DECODER, DEC_LEN)
    if args.ln_newton_steps is not None or args.ln_dual_k is not None:
        from core.newton_layernorm import swap_layernorms, collect_var_quantiles
        steps = args.ln_newton_steps or 0
        cmaps = {}
        if args.ln_dual_k is not None:  # fp32 pass over the calibration utterances (real greedy decode) -> per-LN k x median variance
            encoder.to(device); decoder.to(device)
            greedy_c = wio.GreedyDecoder(args.model_id, DEC_LEN, device)
            def run_c():
                for audio_path, _ in cal_pairs:
                    states = encoder(wio.log_mel(data.load_audio_torchaudio(audio_path), device))
                    greedy_c.decode(decoder, states)
            combined = torch.nn.ModuleDict({"enc": encoder, "dec": decoder})  # one pass collects both
            cm = collect_var_quantiles(combined, run_c, None, k=args.ln_dual_k)
            cmaps = {"encoder": {k[4:]: v for k, v in cm.items() if k.startswith("enc.")},
                     "decoder": {k[4:]: v for k, v in cm.items() if k.startswith("dec.")}}
            encoder.cpu(); decoder.cpu()
            print(f"  dual-range rsqrt: k={args.ln_dual_k}, {len(cm)} LayerNorms, c range {min(cm.values()):.3g}..{max(cm.values()):.3g}", flush=True)
        n_swapped = swap_layernorms(encoder, steps, cmaps.get("encoder")) + swap_layernorms(decoder, steps, cmaps.get("decoder"))
        if args.ln_dual_ablate:
            from core.newton_layernorm import NewtonLayerNorm
            for mod in (encoder, decoder):
                for m in mod.modules():
                    if isinstance(m, NewtonLayerNorm) and m.c is not None:
                        m.dual_ablate = args.ln_dual_ablate
        print(f"  NewtonLayerNorm: {n_swapped} LayerNorms swapped, {steps} step(s), dual k={args.ln_dual_k}", flush=True)
    compile_spec = quantizer_compile_spec()
    quant_config = QuantConfig(args.quant_config)
    act_observer = ActObserver(args.act_observer)
    quantizer_cls = EthosUQuantizer
    if args.prec_rules or args.embedding_bits:  # the embedding table is annotated by MixedPrecisionQuantizer, rules or not
        import functools
        rules = parse_rules(args.prec_rules, act_observer, args.ln_dual_k)
        quantizer_cls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules,
                                          embedding_bits=args.embedding_bits)
    prepared = {}
    if Stage.INT8 in stages:
        t0 = time.time()
        prepared["encoder"] = prepare_on_cpu(encoder, enc_example, compile_spec, quant_config, act_observer, [],
                                             quantizer_cls=quantizer_cls)
        prepared["decoder"] = prepare_on_cpu(decoder, dec_example, compile_spec, quant_config, act_observer, [],
                                             quantizer_cls=quantizer_cls)
        print(f"export+prepare on cpu: {time.time() - t0:.0f}s", flush=True)

    if device != "cpu":
        # Keep GPU float math close to the CPU reference.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    encoder, decoder = encoder.to(device), decoder.to(device)
    if device != "cpu":
        for graph in prepared.values():
            move_graph_module(graph, device)
    enc_example = tuple(t.to(device) for t in enc_example)
    dec_example = tuple(t.to(device) for t in dec_example)
    greedy = wio.GreedyDecoder(args.model_id, DEC_LEN, device)

    def make_transcribe(enc, dec):
        @torch.no_grad()
        def transcribe(audio):
            tokens, hit_limit = greedy.decode(dec, enc(wio.log_mel(audio, device)))
            return greedy.text(tokens), hit_limit, len(tokens) + 1
        return transcribe

    if Stage.FP32_STATIC in stages:
        finish(evaluate(Stage.FP32_STATIC.value, make_transcribe(encoder, decoder), eval_pairs,
                        data.load_audio_torchaudio, normalizer, out_dir, args.resume))

    if Stage.INT8 not in stages:
        return

    t0 = time.time()
    # Real-speech calibration, parallel feed: one teacher-forced decoder pass + one random prefix per utterance.
    mels = [wio.log_mel(data.load_audio_torchaudio(p), device) for p, _ in cal_pairs]
    encoder_q = calibrate_and_convert("encoder", prepared["encoder"], [(m,) for m in mels], device)
    # Per-channel activations are always evaluated in their lowered form (per-tensor INT8 + INT32 MULs), as in
    # vision_eval.py and lower_probe.py; a graph without per-channel activations is left untouched (every Whisper recipe).
    # PTF sites (per-channel zero points) do not lower and stay as annotated (fake-quant only), as in vision_eval.py.
    ptf = []
    if (n_pc := rewrite_per_channel_activations(encoder_q, skipped=ptf)) or ptf:
        print(f"  pcs_decompose[encoder]: {n_pc} per-channel activation sites, {len(ptf)} PTF sites left as fake-quant", flush=True)
    # Decoder calibration inputs: the FP32 model's own transcript (what the
    # decoder is fed at run time) on top of the *quantized* encoder's output
    # (what it will see on device). Each utterance contributes the full
    # sequence and one random prefix, since during decoding most calls see a
    # partially filled buffer.
    dec_batches = []
    with torch.no_grad():
        for mel in mels:
            tokens, _ = greedy.decode(decoder, encoder(mel))
            tokens = greedy.prompt + tokens
            enc_q = encoder_q(mel)
            prefix = rng.randint(len(greedy.prompt), len(tokens))
            dec_batches.append((greedy.padded_ids(tokens), enc_q))
            dec_batches.append((greedy.padded_ids(tokens[:prefix]), enc_q))
    decoder_q = calibrate_and_convert("decoder", prepared["decoder"], dec_batches, device)
    ptf = []
    if (n_pc := rewrite_per_channel_activations(decoder_q, skipped=ptf)) or ptf:
        print(f"  pcs_decompose[decoder]: {n_pc} per-channel activation sites, {len(ptf)} PTF sites left as fake-quant", flush=True)
    print(f"quantized encoder + decoder: {time.time() - t0:.0f}s, decoder calibration batches={len(dec_batches)}", flush=True)

    # The fixed suffix "real-parallel-decoder+encoder" names the calibration setup (real speech, parallel feed,
    # encoder and decoder quantized); the label also names the hyps file that --resume reads.
    label = f"{quant_config.value}-{act_observer.value}-real-parallel-decoder+encoder"
    if args.prec_rules:
        label += f"-rules[{args.prec_rules}]"
    if args.ln_newton_steps is not None:
        label += f"-newton{args.ln_newton_steps}"
    if args.ln_dual_k is not None:
        label += f"-dualk{args.ln_dual_k:g}"
    if args.ln_dual_ablate:
        label += f"-ablate_{args.ln_dual_ablate}"
    if args.embedding_bits:
        label += f"-emb{args.embedding_bits}"
    result = evaluate(label, make_transcribe(encoder_q, decoder_q), eval_pairs,
                      data.load_audio_torchaudio, normalizer, out_dir, args.resume)
    result.update({"calib_mode": "real", "quant_parts": "decoder+encoder",
                   "quant_config": quant_config.value, "device": device,
                   "act_observer": act_observer.value, "calib_feed": "parallel",
                   "n_cal": len(cal_pairs), "cal_split": "dev-clean", "eval_split": "test-clean"})
    finish(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stages", nargs="+", choices=[s.value for s in Stage],
                        default=[s.value for s in Stage])
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=None, help="test-clean utterances; default 100 for large-v3, 200 otherwise")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse utterances already in this stage's hyps file (same --out-dir) instead of "
                             "decoding them again; only safe when the model and rules are unchanged.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant-config", choices=[c.value for c in QuantConfig],
                        default=QuantConfig.A8W8.value)
    parser.add_argument("--act-observer", choices=[ActObserver.HISTOGRAM.value, ActObserver.MINMAX.value],
                        default=ActObserver.HISTOGRAM.value)
    parser.add_argument("--embedding-bits", type=int, default=None, choices=[8, 16],
                        help="Quantize the decoder token-embedding table (per-tensor symmetric) instead of leaving it fp32.")
    parser.add_argument("--ln-dual-k", type=float, default=None, help="dual-range rsqrt (GA-LN dual): fine table sized by K x median per-token variance; the rules must give the fine table and mask the kmedian observer (core/recipes.py)")
    parser.add_argument("--ln-dual-ablate", choices=["coarse_zero", "fine_zero", "coarse_x16"], default=None,
                        help="GA-LN dual ablation (fake-quant only): zero one table's output, or widen the coarse input grid 16x")
    parser.add_argument("--ln-newton-steps", type=int, default=None,
                        help="Swap nn.LayerNorm for NewtonLayerNorm (keepdim formulation): rsqrt table seed + N int32 Newton steps; 0 = swap only.")
    parser.add_argument("--prec-rules", default=None,
                        help='Ordered "regex=a16w8|a8w8|fp32[:observer];..." rules on the deepest module FQN.')
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--recipe", default=None, help="named recipe from core/recipes.py (e.g. \"Ours A8W8\"); sets the quantization flags itself")
    parser.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    import core.recipes as recipes
    args = recipes.parse_args(parser, "whisper", lambda a: "whisper-" + a.model_id.rsplit("whisper-", 1)[-1])
    if args.n_eval is None:
        args.n_eval = 100 if "large-v3" in args.model_id else 200
    main(args)

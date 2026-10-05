"""Does a rule-based mixed-precision graph lower? Whisper-tiny encoder (or DeiT-T) with
--quant-config and --prec-rules: calibrate, convert, then (1) TOSA lowering with the
TOSAPartitioner (op / dtype / TABLE report of the .tosa partitions, CPU-fallback ops) and
(2) the Ethos-U85 flow with EthosUPartitioner + Vela (pass/fail, delegated node counts)."""
import argparse, copy, functools, json, shutil, sys, traceback
from collections import Counter
from pathlib import Path
import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.ethosu.partitioner import EthosUPartitioner
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "helper"))  # fqvit_models: adapter over the official FQ-ViT in 3rdparty/FQ-ViT
import core.lowering as low, core.mixed_precision_quantizer as maq, core.quant as cq
from core.int32_requant import install as install_int32_requant
install_int32_requant()  # INT32-quantized activations lowered at their annotated scales (core/int32_requant.py)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="whisper-tiny-encoder", help="whisper-<size>-<encoder|decoder> or an FQ-ViT vision model name")
ap.add_argument("--target", default="ethos-u85-256", help="ethos-u85-256 | ethos-u55-128 | ethos-u65-256")
ap.add_argument("--vela-flags", default="--verbose-cycle-estimate", help="space-separated extra Vela flags (cost model summary)")
ap.add_argument("--recipe", default=None, help="a named recipe (core/recipes.py NAMES), e.g. 'Ours A8W8'; sets --quant-config, "
                "--prec-rules, --ln-newton-steps, --ln-dual-k, which must then not be given")
ap.add_argument("--quant-config", default="a16w8")
ap.add_argument("--prec-rules", default="")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
ap.add_argument("--n-cal", type=int, default=8)
ap.add_argument("--ln-newton-steps", type=int, default=None, help="swap nn.LayerNorm for NewtonLayerNorm with N Newton steps (0 = swap only)")
ap.add_argument("--ln-dual-k", type=float, default=None, help="dual-range rsqrt (GA-LN dual): fine table sized by K x median per-token variance; the rules must give the fine table and mask the kmedian observer (core/recipes.py)")
ap.add_argument("--embedding-bits", type=int, default=None, choices=[8, 16], help="quantize aten.embedding tables (decoder)")
ap.add_argument("--verbose-partition", action="store_true", default=False, help="log the Arm partitioner's per-node rejection reasons")
ap.add_argument("--pieces", default=None,
                help="lower per piece (core/slices.py) instead of the whole model: 'list' writes <out>/pieces.tsv and exits; "
                     "'all' or comma-separated piece names lower each into <out>/<piece>/ (each piece calibrated on the inputs it "
                     "receives inside the full fp32 model)")
ap.add_argument("--tosa-only", action="store_true", default=False,
                help="lower once with the TOSA partitioner and dump the .tosa partitions; skip the Ethos-U lowering and .pte serialization (run Vela on the .tosa files instead)")
import core.recipes as recipes  # noqa: E402
args = recipes.parse_args(ap, "lower", lambda a: a.model.rsplit("-", 1)[0] if a.model.startswith("whisper-") else a.model)
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
rules = cq.parse_rules(args.prec_rules, cq.ActObserver.HISTOGRAM, args.ln_dual_k)
qcls = (functools.partial(maq.MixedPrecisionQuantizer, rules=rules, embedding_bits=args.embedding_bits)
        if (rules or args.embedding_bits) else cq.EthosUQuantizer)
SYSCFG = {"ethos-u85-256": "Ethos_U85_SYS_DRAM_Low", "ethos-u55-128": "Ethos_U55_High_End_Embedded", "ethos-u65-256": "Ethos_U65_High_End"}
cs = EthosUCompileSpec(args.target, system_config=SYSCFG.get(args.target, "Ethos_U85_SYS_DRAM_Low"), memory_mode="Dedicated_Sram" if "u85" in args.target else "Shared_Sram",
                       extra_flags=args.vela_flags.split() if args.vela_flags else None,
                       config_ini=str(HERE.parent / "scripts/vela/default_vela.ini"))
if args.model.startswith("whisper-"):
    data = cq.load_module_from_path("librispeech_data", HERE / "helper/data/librispeech_data.py")
    wrapper = cq.load_module_from_path("whisper_executorch_wrapper", HERE / "helper/whisper_executorch_wrapper.py")
    size, part = args.model[len("whisper-"):].rsplit("-", 1)  # whisper-<size>-<encoder|decoder>; size may contain '-' (large-v3)
    if part not in ("encoder", "decoder"):
        raise SystemExit(f"--model must be whisper-<size>-<encoder|decoder>, got {args.model}")
    from transformers import AutoConfig
    from helper import whisper_io as wio
    wio.N_MELS = AutoConfig.from_pretrained(f"openai/whisper-{size}").num_mel_bins  # large-v3 uses 128 mel bins
    pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
    mels = [wio.log_mel(data.load_audio_torchaudio(p), DEV) for p, _ in pairs]
    if part == "encoder":
        module, example = wrapper._build(f"openai/whisper-{size}", wrapper.WhisperPart.ENCODER, 128)
        cal = [(m,) for m in mels]
    else:  # decoder: static-length ids + fp32 encoder states; ids from an fp32 greedy decode so observers see real tokens
        encoder, _ = wrapper._build(f"openai/whisper-{size}", wrapper.WhisperPart.ENCODER, 128)
        module, example = wrapper._build(f"openai/whisper-{size}", wrapper.WhisperPart.DECODER, 128)
        greedy = wio.GreedyDecoder(f"openai/whisper-{size}", 128, DEV)
        encoder.to(DEV); module.to(DEV); cal = []
        with torch.no_grad():
            for m in mels:
                states = encoder(m)
                tokens, _ = greedy.decode(module, states)
                cal.append((greedy.padded_ids(list(greedy.prompt) + tokens), states))
        module.cpu(); del encoder
else:
    sys.path.insert(0, str(HERE))
    import fqvit_models
    from fqvit_models import imagenet_data as data
    module = fqvit_models.build_model(args.model)
    example = (torch.randn(1, 3, 224, 224),)
    cal = [(b[:1].to(DEV),) for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=0, data_dir=args.imagenet_dir, model_name=args.model, batch_size=1)]
if args.ln_newton_steps is not None or args.ln_dual_k is not None:
    from core.newton_layernorm import swap_layernorms, collect_var_quantiles
    steps = args.ln_newton_steps or 0
    cmap = None
    if args.ln_dual_k is not None:  # fp32 pass on CPU: a module that has run on CUDA keeps
        # cached CUDA tensors that .cpu() does not move, and prepare_pt2e then rejects the mixed devices
        cal_cpu = [tuple(t.cpu() for t in b) for b in cal]
        cmap = collect_var_quantiles(module, lambda: [module(*b) for b in cal_cpu], None, k=args.ln_dual_k)
    print(f"NewtonLayerNorm: {swap_layernorms(module, steps, cmap)} swapped, {steps} step(s), dual k={args.ln_dual_k}", flush=True)


def lower(module, example, cal, out, name):
    """prepare -> calibrate -> convert -> TOSA (and Ethos-U) lowering of `module`; reports into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    prepared = cq.prepare_on_cpu(module, example, cs, cq.QuantConfig(args.quant_config), cq.ActObserver.HISTOGRAM, [], quantizer_cls=qcls)
    cq.move_graph_module(prepared, DEV)
    with torch.no_grad():
        for b in cal:
            prepared(*b)
    if args.verbose_partition:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("executorch.backends.arm").setLevel(logging.INFO)
    converted = convert_pt2e(prepared)
    from core.pcs_decompose import rewrite_per_channel_activations
    n_pc = rewrite_per_channel_activations(converted)  # always: per-channel activations only lower in the rewritten form
    if n_pc:
        print(f"pcs_decompose: {n_pc} per-channel activation sites rewritten", flush=True)
    cq.move_graph_module(converted, "cpu")
    example_cpu = tuple(t.cpu() for t in example)
    exported = torch.export.export(converted, example_cpu, strict=True)
    n_q = Counter(str(n.args[5]).replace("torch.", "") for n in converted.graph.nodes if low.is_q(n) and "per_tensor" in str(n.target))
    print(f"=== {name} {args.quant_config} rules=[{args.prec_rules}] quantize nodes by dtype: {dict(n_q)}", flush=True)
    result = {"quantize_dtypes": dict(n_q)}

    def report(tag, partitioner, intermediates=None):
        try:
            edge = to_edge_transform_and_lower(exported, partitioner=[partitioner], compile_config=EdgeCompileConfig(_check_ir_validity=False))
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:600]}"
            print(f"  [{tag}] LOWERING FAILED: {msg}", flush=True)
            (out / f"{tag}_traceback.txt").write_text(traceback.format_exc())
            return {"error": msg}
        info = get_delegation_info(edge.exported_program().graph_module)
        df = info.get_operator_delegation_dataframe()
        cpu = df[df["occurrences_in_non_delegated_graphs"] > 0]
        entry = {"delegated_subgraphs": info.num_delegated_subgraphs, "delegated_nodes": info.num_delegated_nodes,
                 "non_delegated_nodes": info.num_non_delegated_nodes,
                 "cpu_ops": dict(zip(cpu["op_type"], cpu["occurrences_in_non_delegated_graphs"].astype(int)))}
        print(f"  [{tag}] partitions={entry['delegated_subgraphs']} delegated={entry['delegated_nodes']} cpu={entry['non_delegated_nodes']} cpu_ops={entry['cpu_ops']}", flush=True)
        if intermediates is not None:
            rep = low.tosa_report(intermediates)
            entry.update({"tosa_ops": dict(rep["ops"]), "tosa_output_dtypes": dict(rep["dtypes"]),
                          "tables": {f"in={k[0]} table={k[1]}[{k[2]}] out={k[3]}": v for k, v in rep["tables"].items()},
                          "matmul": {f"in={k[0]} out={k[1]}": v for k, v in rep["matmul"].items()}})
            print(f"  [{tag}] tosa dtypes={entry['tosa_output_dtypes']} tables={entry['tables']} matmul={entry['matmul']}", flush=True)
            print(f"  [{tag}] tosa ops={entry['tosa_ops']}", flush=True)
        if args.tosa_only:
            return entry  # --tosa-only: the FVP flow takes the .tosa partitions, so no .pte is serialized
        try:
            edge.to_executorch()
            entry["to_executorch"] = "ok"
        except Exception as e:
            entry["to_executorch"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  [{tag}] to_executorch: {entry['to_executorch']}", flush=True)
        return entry

    tosa_cs = TosaCompileSpec(TosaSpecification.create_from_string("TOSA-1.0+INT+int16" + ("+u55" if "u55" in args.target else "")))
    inter = out / "tosa"  # emptied first: a reused --out-dir would otherwise keep stale partitions that the FVP scripts glob
    shutil.rmtree(inter, ignore_errors=True); tosa_cs.dump_intermediate_artifacts_to(str(inter))
    result["tosa"] = report("TOSA", TOSAPartitioner(tosa_cs), inter)
    if not args.tosa_only:  # --tosa-only: cost the dumped .tosa files with Vela / the FVP directly (fvp/profile_graph.sh)
        result["ethosu"] = report("EthosU85+Vela", EthosUPartitioner(cs))
    (out / "summary.json").write_text(json.dumps(result, indent=2))


if args.pieces is None:
    lower(module, example, cal, out, args.model)
else:
    from core.slices import check_layernorms, make_pieces, piece_inputs
    module.eval()
    pieces = make_pieces(module, args.prec_rules)  # after the LayerNorm swap above (see make_pieces)
    (out / "pieces.tsv").write_text("".join(f"{pc.name}\t{pc.reps}\t{','.join(pc.covers)}\n" for pc in pieces))
    print(f"{len(pieces)} pieces: " + ", ".join(f"{pc.name} x{pc.reps}" for pc in pieces), flush=True)
    if args.pieces != "list":
        wanted = None if args.pieces == "all" else args.pieces.split(",")
        unknown = set(wanted or ()) - {pc.name for pc in pieces}
        assert not unknown, f"unknown pieces {sorted(unknown)}; see {out / 'pieces.tsv'}"
        todo = [pc for pc in pieces if wanted is None or pc.name in wanted]
        swapped = args.ln_newton_steps is not None or args.ln_dual_k is not None
        # fp32 pass on CPU for the same reason as the LayerNorm statistics above
        inputs = piece_inputs(module, todo, [tuple(t.cpu() for t in b) for b in cal])
        for pc in todo:
            check_layernorms(pc.module, replaced=swapped)
            pcal = [tuple(t.to(DEV) for t in b) for b in inputs[pc.name]]
            # a copy: pieces share the full model's parameters, and moving one piece to the GPU in place would move
            # a weight another piece also holds (Whisper's proj_out is tied to the prologue's embed_tokens)
            lower(copy.deepcopy(pc.module), inputs[pc.name][0], pcal, out / pc.name, f"{args.model}:{pc.name}")

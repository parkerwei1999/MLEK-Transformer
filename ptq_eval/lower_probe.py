"""Does a rule-based mixed-precision graph lower? Whisper-tiny encoder (or DeiT-T) with
--quant-config and --prec-rules: calibrate, convert, then (1) TOSA lowering with the
TOSAPartitioner (op / dtype / TABLE report of the .tosa partitions, CPU-fallback ops) and
(2) the Ethos-U85 flow with EthosUPartitioner + Vela (pass/fail, delegated node counts)."""
import argparse, functools, json, sys, traceback
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
import whisper_ptq_eval as pe, mask_aware_quantizer as maq, dump_boundaries as db

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="whisper-tiny-encoder", help="whisper-<size>-<encoder|decoder> or an FQ-ViT vision model name")
ap.add_argument("--target", default="ethos-u85-256", help="ethos-u85-256 | ethos-u55-128 | ethos-u65-256")
ap.add_argument("--vela-flags", default="--verbose-cycle-estimate", help="space-separated extra Vela flags (cost model summary)")
ap.add_argument("--quant-config", default="a16w8")
ap.add_argument("--prec-rules", default="")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--n-cal", type=int, default=8)
ap.add_argument("--ln-newton-steps", type=int, default=None, help="swap nn.LayerNorm for NewtonLayerNorm with N Newton steps (0 = swap only)")
ap.add_argument("--ln-dual-q", type=float, default=None, help="dual-range rsqrt: fine table sized by this per-token variance quantile")
ap.add_argument("--embedding-bits", type=int, default=None, choices=[8, 16], help="quantize aten.embedding tables (decoder)")
ap.add_argument("--mask-aware", action="store_true", default=False, help="MaskAwareQuantizer softmax-input rewrite, as the vision accuracy runs use")
ap.add_argument("--pc-rewrite", action="store_true", default=False, help="lower per-channel activation Q/DQ to per-tensor + int32 MULs (pc_rewrite.py)")
ap.add_argument("--verbose-partition", action="store_true", default=False, help="log the Arm partitioner's per-node rejection reasons")
args = ap.parse_args()
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
rules = []
for item in [r for r in args.prec_rules.split(";") if r]:
    rx, cfg = item.rsplit("=", 1); cfg, _, ops = cfg.partition("@"); cfg, _, obs = cfg.partition(":")
    rules.append((rx, None if cfg == "fp32" else pe.QuantConfig(cfg).build(pe.ActObserver(obs or "histogram")), set(ops.split(",")) if ops else None))
qcls = (functools.partial(maq.MixedPrecisionQuantizer, rules=rules, mask_threshold=maq.MASK_THRESHOLD if args.mask_aware else None,
                          embedding_bits=args.embedding_bits) if (rules or args.embedding_bits or args.mask_aware) else pe.EthosUQuantizer)
SYSCFG = {"ethos-u85-256": "Ethos_U85_SYS_DRAM_Low", "ethos-u55-128": "Ethos_U55_High_End_Embedded", "ethos-u65-256": "Ethos_U65_High_End"}
cs = EthosUCompileSpec(args.target, system_config=SYSCFG.get(args.target, "Ethos_U85_SYS_DRAM_Low"), memory_mode="Dedicated_Sram" if "u85" in args.target else "Shared_Sram",
                       extra_flags=args.vela_flags.split() if args.vela_flags else None,
                       config_ini=str(HERE.parent / "scripts/vela/default_vela.ini"))
if args.model.startswith("whisper-"):
    data = pe.load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
    wrapper = pe.load_module_from_path("whisper_executorch_wrapper", HERE / "whisper_executorch_wrapper.py")
    size, part = args.model.split("-")[1:3]  # whisper-<size>-<encoder|decoder>
    pairs = data.gather_librispeech_files("/home/shared/LibriSpeech", "dev-clean", args.n_cal)
    mels = [pe.log_mel(data.load_audio_torchaudio(p), DEV) for p, _ in pairs]
    if part == "encoder":
        module, example = wrapper._build(f"openai/whisper-{size}", wrapper.WhisperPart.ENCODER, 128)
        cal = [(m,) for m in mels]
    else:  # decoder: static-length ids + fp32 encoder states; ids from an fp32 greedy decode so observers see real tokens
        encoder, _ = wrapper._build(f"openai/whisper-{size}", wrapper.WhisperPart.ENCODER, 128)
        module, example = wrapper._build(f"openai/whisper-{size}", wrapper.WhisperPart.DECODER, 128)
        greedy = pe.GreedyDecoder(f"openai/whisper-{size}", 128, DEV)
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
    cal = [(b[:1].to(DEV),) for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=0, data_dir="/home/shared/ImageNet", model_name=args.model, batch_size=1)]
if args.ln_newton_steps is not None or args.ln_dual_q is not None:
    from newton_layernorm import swap_layernorms, collect_var_quantiles
    steps = args.ln_newton_steps or 0
    cmap = None
    if args.ln_dual_q is not None:
        module.to(DEV)
        cmap = collect_var_quantiles(module, lambda: [module(*b) for b in cal], args.ln_dual_q)
        module.cpu()
    print(f"NewtonLayerNorm: {swap_layernorms(module, steps, cmap)} swapped, {steps} step(s), dual={args.ln_dual_q}", flush=True)
prepared = pe.prepare_on_cpu(module, example, cs, pe.QuantConfig(args.quant_config), pe.ActObserver.HISTOGRAM, [], quantizer_cls=qcls)
pe.move_graph_module(prepared, DEV)
with torch.no_grad():
    for b in cal:
        prepared(*b)
if args.verbose_partition:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("executorch.backends.arm").setLevel(logging.INFO)
converted = convert_pt2e(prepared)
if args.pc_rewrite:
    from pc_rewrite import rewrite_per_channel_activations
    print(f"pc_rewrite: {rewrite_per_channel_activations(converted)} per-channel activation sites rewritten", flush=True)
pe.move_graph_module(converted, "cpu")
example_cpu = tuple(t.cpu() for t in example)
exported = torch.export.export(converted, example_cpu, strict=True)
n_q = Counter(str(n.args[5]).replace("torch.", "") for n in converted.graph.nodes if db.is_q(n) and "per_tensor" in str(n.target))
print(f"=== {args.model} {args.quant_config} rules=[{args.prec_rules}] quantize nodes by dtype: {dict(n_q)}", flush=True)
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
        rep = db.tosa_report(intermediates)
        entry.update({"tosa_ops": dict(rep["ops"]), "tosa_output_dtypes": dict(rep["dtypes"]),
                      "tables": {f"in={k[0]} table={k[1]}[{k[2]}] out={k[3]}": v for k, v in rep["tables"].items()},
                      "matmul": {f"in={k[0]} out={k[1]}": v for k, v in rep["matmul"].items()}})
        print(f"  [{tag}] tosa dtypes={entry['tosa_output_dtypes']} tables={entry['tables']} matmul={entry['matmul']}", flush=True)
        print(f"  [{tag}] tosa ops={entry['tosa_ops']}", flush=True)
    try:
        edge.to_executorch()
        entry["to_executorch"] = "ok"
    except Exception as e:
        entry["to_executorch"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(f"  [{tag}] to_executorch: {entry['to_executorch']}", flush=True)
    return entry

tosa_cs = TosaCompileSpec(TosaSpecification.create_from_string("TOSA-1.0+INT+int16" + ("+u55" if "u55" in args.target else "")))
inter = out / "tosa"; tosa_cs.dump_intermediate_artifacts_to(str(inter))
result["tosa"] = report("TOSA", TOSAPartitioner(tosa_cs), inter)
result["ethosu"] = report("EthosU85+Vela", EthosUPartitioner(cs))
(out / "summary.json").write_text(json.dumps(result, indent=2))

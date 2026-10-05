"""Dump the quantization boundaries of Whisper-Tiny encoder block 0.

For each variant (a8w8 / a16w8 x LayerNorm quantized / kept FP32):
  1. prepare_pt2e as whisper_eval.py does, calibrate on a few dev-clean
     utterances, convert_pt2e: the framework fake-quant graph that the WER
     evaluation runs. Every op of block 0 is listed with the Q/DQ spec on its inputs
     and output, so a "boundary" is a Q/DQ pair and an "fp32 island" is an op
     with neither.
  2. lower the converted graph to TOSA (no Vela) and list the integer
     operators, table dtypes and MATMUL dtypes, i.e. what the intermediates
     become on the device path.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
sys.path.insert(0, str(ROOT))
import core.quant as cq  # noqa: E402
from core.lowering import DTYPE_NAMES, OP_NAMES, is_dq, is_q, tosa_report  # noqa: E402,F401
from helper import whisper_io as wio  # noqa: E402

VARIANTS = [
    ("a8w8_ln", cq.QuantConfig.A8W8, []),
    ("a8w8_lnfp32", cq.QuantConfig.A8W8, [cq.KeepFp32.LAYERNORM]),
    ("a16w8_ln", cq.QuantConfig.A16W8, []),
    ("a16w8_lnfp32", cq.QuantConfig.A16W8, [cq.KeepFp32.LAYERNORM]),
]


def qspec(node):
    """Short description of a quantize / dequantize node's parameters."""
    target = str(node.target)
    if "per_tensor" in target:
        scale, zp, _, _, dtype = node.args[1:6]
        return f"{str(dtype).replace('torch.', '')} s={scale:.3g} zp={zp}"
    if "per_channel" in target:
        dtype = node.args[6]
        return f"{str(dtype).replace('torch.', '')} per-channel"
    return target


def short(node):
    return str(node.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")


def module_path(node):
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return ""
    return list(stack.values())[-1][0]


def category(node):
    path = module_path(node)
    leaf = path.rsplit(".", 1)[-1]
    if leaf.endswith("layer_norm"):
        return "LN-self" if leaf.startswith("self") else "LN-final"
    if leaf in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"):
        return leaf
    if leaf == "self_attn":
        return "attn"
    if "gelu" in short(node):
        return "gelu"
    if short(node).startswith("add"):
        return "resadd"
    return "block"


def describe_inputs(node):
    parts = []
    for arg in node.all_input_nodes:
        if is_dq(arg):
            parts.append(f"DQ[{qspec(arg)}]")
        elif arg.op in ("get_attr", "placeholder"):
            parts.append(arg.op)
        else:
            parts.append(f"fp32<{short(arg)}>")
    return ", ".join(parts)


def describe_output(node):
    users = list(node.users)
    if users and all(is_q(u) for u in users):
        return "Q[" + qspec(users[0]) + "]"
    if any(is_q(u) for u in users):
        return "mixed:" + "/".join(("Q" if is_q(u) else short(u)) for u in users)
    return "fp32->" + "/".join(short(u) for u in users) if users else "output"


def dump_block(gm, block_prefix: str):
    lines, rows = [], []
    for node in gm.graph.nodes:
        if node.op != "call_function" or is_q(node) or is_dq(node):
            continue
        if block_prefix not in module_path(node):
            continue
        row = {"cat": category(node), "op": short(node),
               "in": describe_inputs(node), "out": describe_output(node)}
        rows.append(row)
        lines.append(f"[{row['cat']:8s}] {row['op']:22s} in: {row['in']:70s} -> {row['out']}")
    return lines, rows




def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = cq.load_module_from_path("librispeech_data", ROOT / "helper/data/librispeech_data.py")
    wrapper = cq.load_module_from_path("whisper_executorch_wrapper", ROOT / "helper/whisper_executorch_wrapper.py")
    cal_pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
    mels = [wio.log_mel(data.load_audio_torchaudio(p), args.device) for p, _ in cal_pairs]
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    tosa_spec = TosaSpecification.create_from_string(args.tosa_spec)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name, quant_config, keep_fp32 in VARIANTS:
        if args.variants and name not in args.variants:
            continue
        print(f"=== {name}", flush=True)
        encoder, example = wrapper._build(args.model_id, wrapper.WhisperPart.ENCODER, 128)
        compile_spec = cq.EthosUCompileSpec(args.target, system_config=args.system_config,
                                            memory_mode=args.memory_mode)
        prepared = cq.prepare_on_cpu(encoder, example, compile_spec, quant_config,
                                     cq.ActObserver.HISTOGRAM, keep_fp32)
        cq.move_graph_module(prepared, args.device)
        with torch.no_grad():
            for mel in mels:
                prepared(mel)
        converted = convert_pt2e(prepared)
        lines, rows = dump_block(converted, "layers.0")
        (out / f"{name}_block0.txt").write_text("\n".join(lines) + "\n")
        n_q = sum(1 for n in converted.graph.nodes if is_q(n))
        print(f"  fake-quant graph: {n_q} quantize nodes, block0 ops listed: {len(rows)}", flush=True)

        entry = {"quantize_nodes": n_q, "block0": rows}
        cq.move_graph_module(converted, "cpu")
        example_cpu = tuple(t.cpu() for t in example)
        exported = torch.export.export(converted, example_cpu, strict=True)
        tosa_cs = TosaCompileSpec(tosa_spec)
        intermediates = out / f"{name}_tosa"
        tosa_cs.dump_intermediate_artifacts_to(str(intermediates))
        try:
            edge = to_edge_transform_and_lower(
                exported, partitioner=[TOSAPartitioner(tosa_cs)],
                compile_config=EdgeCompileConfig(_check_ir_validity=False))
        except Exception as error:  # a lowering failure is a result in itself
            entry["lowering_error"] = f"{type(error).__name__}: {str(error)[:300]}"
            print(f"  TOSA lowering failed: {entry['lowering_error']}", flush=True)
        else:
            info = get_delegation_info(edge.exported_program().graph_module)
            df = info.get_operator_delegation_dataframe()
            cpu_ops = df[df["occurrences_in_non_delegated_graphs"] > 0]
            entry["cpu_ops"] = dict(zip(cpu_ops["op_type"], cpu_ops["occurrences_in_non_delegated_graphs"].astype(int)))
            report = tosa_report(intermediates)
            entry.update({
                "delegated_subgraphs": info.num_delegated_subgraphs,
                "delegated_nodes": info.num_delegated_nodes,
                "non_delegated_nodes": info.num_non_delegated_nodes,
                "tosa_ops": dict(report["ops"]),
                "tosa_output_dtypes": dict(report["dtypes"]),
                "tables": {f"in={k[0]} table={k[1]}[{k[2]}] out={k[3]}": v for k, v in report["tables"].items()},
                "matmul": {f"in={k[0]} out={k[1]}": v for k, v in report["matmul"].items()},
            })
            print(f"  TOSA: {report['files']} partition(s), delegated {info.num_delegated_nodes} / "
                  f"cpu {info.num_non_delegated_nodes} nodes; tables {dict(report['tables'])}; "
                  f"matmul {dict(report['matmul'])}", flush=True)
        summary[name] = entry
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-cal", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    parser.add_argument("--target", default="ethos-u85-256")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Low")
    parser.add_argument("--memory-mode", default="Dedicated_Sram")
    parser.add_argument("--tosa-spec", default="TOSA-1.0+INT+int16")
    parser.add_argument("--variants", nargs="*", choices=[v[0] for v in VARIANTS], default=[],
                        help="Subset of variants to run; results merge into an existing summary.json.")
    main(parser.parse_args())

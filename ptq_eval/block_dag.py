"""Dump one block's ops with input/output quantization under a precision-rule
set, so the int8 <-> int16 hand-offs are visible."""
import functools
import sys
from pathlib import Path
import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

HERE = Path(__file__).resolve().parent
W = HERE
sys.path.insert(0, str(W))
import dump_boundaries as db  # noqa: E402
import whisper_ptq_eval as pe  # noqa: E402
import mask_aware_quantizer as maq  # noqa: E402

model_name, block, rules_str = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
sys.path.insert(0, str(HERE))
import fqvit_models  # noqa: E402
from fqvit_models import imagenet_data as data  # noqa: E402
model = fqvit_models.build_model(model_name)
cs = pe.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
rules = []
for item in [r for r in rules_str.split(";") if r]:
    rx, cfg = item.rsplit("=", 1)
    cfg, _, ops = cfg.partition("@")   # optional op-type filter: a32w8@sub,mul,sum.dim_IntList
    cfg, _, obs = cfg.partition(":")
    rules.append((rx, None if cfg == "fp32" else pe.QuantConfig(cfg).build(pe.ActObserver(obs or "histogram")),
                  set(ops.split(",")) if ops else None))
qcls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules) if rules else pe.EthosUQuantizer
prepared = pe.prepare_on_cpu(model, (torch.randn(10, 3, 224, 224),), cs, pe.QuantConfig.A8W8, pe.ActObserver.HISTOGRAM, [], quantizer_cls=qcls)
cal = data._load_imgs(20, None, "train", shuffle=True, seed=0, data_dir="/home/shared/ImageNet", model_name=model_name, batch_size=10)
with torch.no_grad():
    for b in cal:
        prepared(b)
gm = convert_pt2e(prepared)

def qspec(n):
    """Like db.qspec but resolves per-channel scales from the graph (PTF: layer scale x 2^alpha_c)."""
    if "per_channel" in str(n.target):
        scales = getattr(gm, n.args[1].target) if isinstance(n.args[1], torch.fx.Node) else None
        dtype = str(n.args[6]).replace("torch.", "")
        if scales is None:
            return f"{dtype} per-channel"
        base = scales.min().item()
        alpha = torch.log2(scales / base).round()
        hist = {int(a): int((alpha == a).sum()) for a in alpha.unique()}
        return f"{dtype} per-ch axis={n.args[3]} s_base={base:.3g} alpha={hist}"
    return db.qspec(n)


def src(n):
    while isinstance(n, torch.fx.Node) and (db.is_dq(n) or db.is_q(n)):
        n = n.args[0]
    return n.name if isinstance(n, torch.fx.Node) else str(n)

print(f"=== {model_name} {block} rules=[{rules_str}]")
for n in gm.graph.nodes:
    if n.op != "call_function" or db.is_q(n) or db.is_dq(n) or block not in db.module_path(n):
        continue
    leaf = db.module_path(n).split(block, 1)[-1].lstrip(".") or "<block>"
    ins = ", ".join(f"{src(a)}:[{qspec(a)}]" if db.is_dq(a) else f"{src(a)}:fp32" for a in n.all_input_nodes if a.op != "get_attr")
    out = db.describe_output(n)
    users = [u for u in n.users if db.is_q(u)]
    if users and "per_channel" in str(users[0].target):
        out = "Q[" + qspec(users[0]) + "]"
    print(f"  {n.name:14s} {leaf:9s} {db.short(n):15s} <- {ins[:60]:60s} -> {out[:160]}")

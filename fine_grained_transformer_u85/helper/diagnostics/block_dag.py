"""Dump one block's ops with input/output quantization under a precision-rule
set, so the int8 <-> int16 hand-offs are visible."""
import functools
import sys
from pathlib import Path
import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
W = HERE
sys.path.insert(0, str(W))
import dump_boundaries as db  # noqa: E402
import core.quant as cq  # noqa: E402
import core.mixed_precision_quantizer as maq  # noqa: E402

import argparse  # noqa: E402
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("model_name"); ap.add_argument("block"); ap.add_argument("rules_str", nargs="?", default="")
ap.add_argument("--ln-newton-steps", type=int, default=None, help="swap LayerNorms for NewtonLayerNorm with N Newton steps")
args = ap.parse_args()
model_name, block, rules_str = args.model_name, args.block, args.rules_str
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))  # fqvit_models: adapter over the official FQ-ViT in 3rdparty/FQ-ViT
import fqvit_models  # noqa: E402
from fqvit_models import imagenet_data as data  # noqa: E402
model = fqvit_models.build_model(model_name)
if args.ln_newton_steps is not None:
    from core.newton_layernorm import swap_layernorms
    print(f"NewtonLayerNorm: {swap_layernorms(model, args.ln_newton_steps)} swapped")
cs = cq.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
rules = cq.parse_rules(rules_str, cq.ActObserver.HISTOGRAM, None)
qcls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules) if rules else cq.EthosUQuantizer
prepared = cq.prepare_on_cpu(model, (torch.randn(10, 3, 224, 224),), cs, cq.QuantConfig.A8W8, cq.ActObserver.HISTOGRAM, [], quantizer_cls=qcls)
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

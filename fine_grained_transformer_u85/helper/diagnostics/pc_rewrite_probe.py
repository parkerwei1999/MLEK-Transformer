"""Print the converted-graph pattern around per-channel ACTIVATION quantize/dequantize nodes (DeiT-T, a16inpc8symout residual)."""
import functools, sys, torch
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))  # fqvit_models: adapter over the official FQ-ViT in 3rdparty/FQ-ViT
import core.quant as cq
import core.mixed_precision_quantizer as maq
import fqvit_models
from fqvit_models import imagenet_data as data

model = fqvit_models.build_model("deit_tiny_patch16_224")
cs = cq.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
VLN32 = 'norm\\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\\d?$=a16w8'
rules = cq.parse_rules(f"blocks\\.\\d+$=a16inpc8symout@add;{VLN32}", cq.ActObserver.HISTOGRAM, None)
qcls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules)
prepared = cq.prepare_on_cpu(model, (torch.randn(10, 3, 224, 224),), cs, cq.QuantConfig.A8W8, cq.ActObserver.HISTOGRAM, [], quantizer_cls=qcls)
cal = data._load_imgs(20, None, "train", shuffle=True, seed=0, data_dir="/home/shared/ImageNet", model_name="deit_tiny_patch16_224", batch_size=10)
with torch.no_grad():
    for b in cal:
        prepared(b)
gm = convert_pt2e(prepared)

def short(n):
    return f"{n.name}<{str(n.target).replace('torch.ops.', '')}>" if isinstance(n, torch.fx.Node) else repr(n)[:40]

shown = 0
for n in gm.graph.nodes:
    if n.op == "call_function" and "quantize_per_channel" in str(n.target) and "dequantize" not in str(n.target):
        src = n.args[0]
        if src.op == "get_attr":
            continue  # weight
        print(f"--- Q node {n.name} target={n.target}")
        print("    args:", [short(a) for a in n.args])
        for a in n.args[1:3]:
            if isinstance(a, torch.fx.Node) and a.op == "get_attr":
                t = getattr(gm, a.target); print(f"    {a.target}: shape={tuple(t.shape)} dtype={t.dtype} min={t.min().item():.4g} max={t.max().item():.4g}")
        print("    producer:", short(src), "| producer args:", [short(a) for a in src.args])
        for u in n.users:
            print("    DQ:", short(u), "target=", u.target, "| DQ users:", [short(x) for x in u.users])
            for x in u.users:
                print("        ->", short(x), "args:", [short(a) for a in x.args][:4])
        shown += 1
        if shown >= 2:
            break
print("total activation per-channel Q nodes:", sum(1 for n in gm.graph.nodes if n.op == "call_function" and "quantize_per_channel" in str(n.target) and "dequantize" not in str(n.target) and n.args[0].op != "get_attr"))

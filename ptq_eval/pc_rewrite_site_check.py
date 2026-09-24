"""Isolate one per-channel site: capture x and the original dequantize_per_channel output, recompute the rewrite path by hand."""
import functools, sys, torch
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe, mask_aware_quantizer as maq, fqvit_models
from fqvit_models import imagenet_data as data
from pc_rewrite import Q, DQ, Q_PC, DQ_PC, I32, F_SCALE, G_SCALE

model = fqvit_models.build_model("deit_tiny_patch16_224")
cs = pe.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
rules = []
for item in 'blocks\\.\\d+$=a16inpc8symout@add;norm\\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\\d?$=a16w8'.split(";"):
    rx, cfg = item.rsplit("=", 1); cfg, _, ops = cfg.partition("@")
    rules.append((rx, pe.QuantConfig(cfg).build(pe.ActObserver.HISTOGRAM), set(ops.split(",")) if ops else None))
prepared = pe.prepare_on_cpu(model, (torch.randn(10, 3, 224, 224),), cs, pe.QuantConfig.A8W8, pe.ActObserver.HISTOGRAM, [],
                             quantizer_cls=functools.partial(maq.MixedPrecisionQuantizer, rules=rules))
cal = list(data._load_imgs(30, None, "train", shuffle=True, seed=0, data_dir="/home/shared/ImageNet", model_name="deit_tiny_patch16_224", batch_size=10))
with torch.no_grad():
    for b in cal[:2]:
        prepared(b)
gm = convert_pt2e(prepared)
sites = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is Q_PC and n.args[0].op != "get_attr"]
captured = {}
class Cap(torch.fx.Interpreter):
    def run_node(self, n):
        out = super().run_node(n)
        if n in sites or (n.op == "call_function" and n.target is DQ_PC):
            captured[n.name] = out.detach().clone()
        if n.op == "call_function" and n.target is Q_PC and n in sites:
            captured[n.name + "_x"] = self.env[n.args[0]].detach().clone()
        return out
with torch.no_grad():
    Cap(gm).run(cal[2])
for site in sites[:3]:
    x = captured[site.name + "_x"]
    scales = getattr(gm, site.args[1].target).float(); axis, qmin, qmax, dtype = site.args[3:7]
    dq_node = [u for u in site.users if u.target is DQ_PC][0]
    ref = captured[dq_node.name]
    s_base = scales.min().item(); shape = [1] * x.ndim; shape[axis] = scales.numel()
    f = (torch.round((s_base / scales) / F_SCALE) * F_SCALE).reshape(shape)
    g = (torch.round((scales / s_base) / G_SCALE) * G_SCALE).reshape(shape)
    s_x = s_base / 256
    x_dq = DQ(Q(x, s_x, 0, *I32, torch.int32), s_x, 0, *I32, torch.int32)
    t_dq = DQ(Q(x_dq * f, s_base, 0, qmin, qmax, dtype), s_base, 0, qmin, qmax, dtype)
    y = t_dq * g
    y_dq = DQ(Q(y, s_x, 0, *I32, torch.int32), s_x, 0, *I32, torch.int32)
    d = (y_dq - ref).abs()
    codes_ref = torch.round(x / scales.reshape(shape)).clamp(qmin, qmax)
    codes_new = torch.round(x_dq * f / s_base).clamp(qmin, qmax)
    flips = (codes_ref != codes_new).float().mean().item()
    print(f"{site.name}: s_base={s_base:.4g} s_max={scales.max():.4g} |x|max={x.abs().max():.3f} ref-vs-rewrite max|d|={d.max():.4g} "
          f"mean|d|={d.mean():.3g} (max|d|/s_max={d.max()/scales.max():.3f}) code flips={flips:.5f}  "
          f"x_dq-vs-x max={((x_dq-x).abs().max()):.3g}")

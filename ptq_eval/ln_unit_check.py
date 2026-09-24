"""Isolate the LayerNorm: fake-quant one LN (rules as in the harness) on synthetic sink-like inputs and measure the per-token
output error vs fp32 LayerNorm, for plain / Newton / dual variants. No decoder, no WER."""
import functools, sys, torch
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe, mask_aware_quantizer as maq
from newton_layernorm import NewtonLayerNorm, collect_var_quantiles

torch.manual_seed(0)
d, T = 768, 128
ln = torch.nn.LayerNorm(d)
with torch.no_grad():
    ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.2, 0.2)
def make_batch(n_utt=8):
    xs = []
    for _ in range(n_utt):
        scale = 0.19 * torch.exp(0.3 * torch.randn(1, T, 1))  # var lognormal around 0.036 (p99 ~ 2x median)
        x = torch.randn(1, T, d) * scale
        x[0, 1, :4] += 270.0                                # sink token: var ~ 4*270^2/768 ~ 380
        xs.append(x)
    return xs
cal, test = make_batch(16), make_batch(4)
rules = 'ln\\.fine$=a32w8@clamp;ln\\.fine$=a16w8e16;ln$=a32w8@sub,mul,sum.dim_IntList,add,clamp;ln$=a16w8'
def run(variant, steps, q, k=None):
    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.ln = torch.nn.LayerNorm(d); self.ln.load_state_dict(ln.state_dict())
        def forward(self, x): return self.ln(x)
    m = Wrap()
    cmap = collect_var_quantiles(m, lambda: [m(x) for x in cal], q or 0.9, k=k) if (q or k) else None
    if variant != "stock":
        m.ln = NewtonLayerNorm.from_layernorm(m.ln, steps, c=(cmap["ln"] if cmap else None))
    rl = []
    for item in rules.split(";"):
        rx, cfg = item.rsplit("=", 1); cfg, _, ops = cfg.partition("@")
        rl.append((rx, pe.QuantConfig(cfg).build(pe.ActObserver.MINMAX), set(ops.split(",")) if ops else None))
    cs = pe.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
    prep = pe.prepare_on_cpu(m, (cal[0],), cs, pe.QuantConfig.A16W8, pe.ActObserver.MINMAX, [], quantizer_cls=functools.partial(maq.MixedPrecisionQuantizer, rules=rl))
    with torch.no_grad():
        for x in cal: prep(x)
    gm = convert_pt2e(prep)
    errs = []
    with torch.no_grad():
        for x in test:
            ref = ln(x); out = gm(x)
            e = (out - ref).norm(dim=-1) / ref.norm(dim=-1)     # per-token relative error
            errs.append(e[0])
    e = torch.cat(errs)
    return f"{variant:8s} steps={steps} q={q} k={k}: per-token rel err median={e.median():.4f} p95={e.kthvalue(int(0.95*e.numel())).values:.4f} p99={e.kthvalue(int(0.99*e.numel())).values:.4f} max={e.max():.3f} sink err={e.view(4,T)[:,1].mean():.4f}"
for v, s, q, k in (("stock", 0, None, None), ("newton", 1, None, None), ("dual", 0, 0.9, None), ("dual", 0, None, 4), ("dual", 0, None, 8), ("dual", 1, None, 4)):
    print(run(v, s, q, k), flush=True)

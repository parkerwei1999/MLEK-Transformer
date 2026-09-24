"""LayerNorm whose 1/sqrt(var) is a coarse table seed refined by Newton steps in int32.

Whisper-small's LN-i32 gap (4.21% vs fp32 2.27%) is the int16 grid in front of the rsqrt TABLE:
after the attention-sink token appears, the per-token variance spans 0.025 .. 390 in one tensor,
so the int16 per-tensor grid (step var_max / 32767 ~ 0.0126) leaves ordinary tokens 2-3 levels
and their 1/sqrt(var) comes out ~10% off (`ln_scales_*.txt` from --dump-ln-scales).

Here rsqrt(q16(var)) is only the seed y0; each Newton step
    y <- y * (1.5 - 0.5 * var * y * y)
uses the int32 variance (its own calibrated scale, ~0.4% resolution for ordinary tokens) and is
built from mul / sub only, so under the LN-i32 rules it runs in int32 and lowers to TOSA MUL / SUB /
RESCALE, all in Vela's supported set. One step turns a relative seed error e into 1.5 e^2
(12% -> 2%), two steps into <0.1%.

The module subclasses nn.LayerNorm and keeps the same parameters and name, so FQN rules
(`layer_norm$=...`) and `KeepFp32.LAYERNORM` keep matching. The mean is written as sum * 1/N so
the graph has the same op set as the Arm decomposition (sum.dim_IntList, mul, sub, add, rsqrt).
"""
import torch
from torch import nn


class _FineRsqrt(nn.Module):
    """rsqrt(clamp(v, max=c)) as its own submodule, so precision rules can give the fine table's int16 input
    a finer eps floor (`...fine$=a16w8e16`) than the default a16w8 observer (eps 2^-12 caps the grid at 8.0)."""

    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def forward(self, v):
        return torch.rsqrt(torch.clamp(v, max=self.c))


class NewtonLayerNorm(nn.LayerNorm):
    """keepdim LayerNorm with (a) N Newton steps on 1/sqrt(var) and (b) an optional dual-range rsqrt:

    dual range (c = per-LN quantile of the per-token variance, set at calibration): a fine table
    y_f = rsqrt(clamp(v, max=c)) whose int16 input grid is sized by c (ordinary tokens get thousands of
    levels, sink tokens saturate) and the coarse table y_c = rsqrt(v) sized by the max (right for the
    sink), blended with a mask m = clamp(5 v / c - 4, 0, 1) (0 up to v = 0.8 c, 1 from c): y = y_f + m (y_c - y_f).
    c must sit below the sink fraction of tokens (decoder sinks are ~1 per utterance, i.e. a few percent), so q ~ 0.9. Every op is
    add / sub / mul / clamp / rsqrt, i.e. TOSA ADD/SUB/MUL/CLAMP/TABLE + RESCALE.
    """

    def __init__(self, normalized_shape, eps=1e-5, newton_steps=2, c=None):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=True)
        self.newton_steps = newton_steps
        self.c = None if c is None else float(c)
        if self.c is not None:  # mask ramps from 0 at v = 0.8 c to 1 at v = c (fine table exact below c)
            self.fine = _FineRsqrt(self.c)
            self.register_buffer("mask_gain", torch.tensor(5.0 / self.c))
            self.register_buffer("mask_bias", torch.tensor(4.0))
        n = 1
        for d in self.normalized_shape:
            n *= d
        self.register_buffer("inv_n", torch.tensor(1.0 / n))
        self.register_buffer("c15", torch.tensor(1.5))
        self.register_buffer("c05", torch.tensor(0.5))

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm, newton_steps: int, c=None):
        m = cls(tuple(ln.normalized_shape), eps=ln.eps, newton_steps=newton_steps, c=c)
        with torch.no_grad():
            m.weight.copy_(ln.weight)
            m.bias.copy_(ln.bias)
        return m

    def forward(self, x, *_args, **_kwargs):  # FQ-ViT LayerNorms are called with extra quantizer args (unused when quant=False)
        mean = x.sum(-1, keepdim=True) * self.inv_n
        xc = x - mean
        var = (xc * xc).sum(-1, keepdim=True) * self.inv_n + self.eps
        if self.c is None:
            y = torch.rsqrt(var)  # seed through the int16 table (grid sized by the max token)
        else:
            y_coarse = torch.rsqrt(var)
            y_fine = self.fine(var)  # int16 grid sized by c (rule the submodule to a16w8e16); sinks saturate here
            m = torch.clamp(var * self.mask_gain - self.mask_bias, 0.0, 1.0)  # 0 for v <= 0.8 c, 1 for v >= c
            y = y_fine + m * (y_coarse - y_fine)
        for _ in range(self.newton_steps):
            y = y * (self.c15 - (var * y * y) * self.c05)
        return xc * y * self.weight + self.bias


def swap_layernorms(module: nn.Module, newton_steps: int, cmap=None) -> int:
    """Replace every nn.LayerNorm under `module` in place (same attribute name); returns the count.
    cmap: {module name: c} from collect_var_quantiles enables the dual-range rsqrt."""
    count = 0
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.LayerNorm) and not isinstance(child, NewtonLayerNorm) and name:
            parent_name, _, attr = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            c = cmap.get(name) if cmap else None
            setattr(parent, attr, NewtonLayerNorm.from_layernorm(child, newton_steps, c=c))
            count += 1
    return count


def collect_var_quantiles(module: nn.Module, run_fn, q: float, k: float | None = None) -> dict:
    """fp32 calibration pass: per LayerNorm (by name) the fine-table ceiling c of the per-token input variance:
    the q-quantile, or k * median when k is given (a quantile lands inside the bulk when the bulk is tight and the
    sink fraction is a few percent; k * median with k ~ 4-8 keeps every ordinary token on the fine table).
    run_fn() must drive `module` over the calibration data (hooks collect the variances)."""
    stats, hooks = {}, []
    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            stats.setdefault(name, []).append(x.var(-1, unbiased=False).reshape(-1).cpu())
        return fn
    for name, child in module.named_modules():
        if isinstance(child, nn.LayerNorm) and name:
            hooks.append(child.register_forward_hook(hook(name)))
    with torch.no_grad():
        run_fn()
    for h in hooks:
        h.remove()
    out = {}
    for name, v in stats.items():
        v = torch.cat(v)
        out[name] = (k * v.median().item()) if k else v.quantile(q).item()
    return out


if __name__ == "__main__":  # fp32 self-check: Newton LN == nn.LayerNorm
    torch.manual_seed(0)
    ln = nn.LayerNorm(768)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
    x = torch.randn(4, 20, 768) * torch.logspace(-2, 2, 20).view(1, 20, 1)
    x[:, 3, :5] += 500.0  # a sink-like row
    for steps in (0, 1, 2):
        m = NewtonLayerNorm.from_layernorm(ln, steps)
        err = (m(x) - ln(x)).abs().max().item()
        print(f"newton_steps={steps}: max|diff| vs nn.LayerNorm = {err:.2e}")
    c = x.var(-1, unbiased=False).quantile(0.995).item()
    m = NewtonLayerNorm.from_layernorm(ln, 0, c=c)
    print(f"dual-range c={c:.3g}: max|diff| vs nn.LayerNorm = {(m(x) - ln(x)).abs().max().item():.2e}")

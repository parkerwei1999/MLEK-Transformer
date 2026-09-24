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


class NewtonLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, newton_steps=2):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=True)
        self.newton_steps = newton_steps
        n = 1
        for d in self.normalized_shape:
            n *= d
        self.register_buffer("inv_n", torch.tensor(1.0 / n))
        self.register_buffer("c15", torch.tensor(1.5))
        self.register_buffer("c05", torch.tensor(0.5))

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm, newton_steps: int):
        m = cls(ln.normalized_shape, eps=ln.eps, newton_steps=newton_steps)
        with torch.no_grad():
            m.weight.copy_(ln.weight)
            m.bias.copy_(ln.bias)
        return m

    def forward(self, x):
        mean = x.sum(-1, keepdim=True) * self.inv_n
        xc = x - mean
        var = (xc * xc).sum(-1, keepdim=True) * self.inv_n + self.eps
        y = torch.rsqrt(var)  # coarse seed through the int16 table
        for _ in range(self.newton_steps):
            y = y * (self.c15 - (var * y * y) * self.c05)
        return xc * y * self.weight + self.bias


def swap_layernorms(module: nn.Module, newton_steps: int) -> int:
    """Replace every nn.LayerNorm under `module` in place (same attribute name); returns the count."""
    count = 0
    for name, child in list(module.named_modules()):
        if type(child) is nn.LayerNorm and name:
            parent_name, _, attr = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, attr, NewtonLayerNorm.from_layernorm(child, newton_steps))
            count += 1
    return count


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

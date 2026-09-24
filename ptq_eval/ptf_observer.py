"""Per-channel power-of-two-factor (PTF, FQ-ViT) int8 activation observer for PT2E.

One layer-wise scale s and one integer factor alpha_c per channel so that channel c is
quantized with scale s * 2**alpha_c (asymmetric, per-channel zero point). The massive
channels get a large factor, ordinary channels keep their resolution, and the integer
LayerNorm reads codes back through a per-channel shift (or, on Ethos-U, a MUL by a
constant vector). Channel axis = last dim.
"""
import torch
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver


class PTFPerChannelObserver(PerChannelMinMaxObserver):
    def __init__(self, ch_axis=2, dtype=torch.int8, qscheme=torch.per_channel_affine, quant_min=-128,
                 quant_max=127, eps=2 ** -12, max_alpha=12, **kwargs):
        super().__init__(ch_axis=ch_axis, dtype=dtype, qscheme=qscheme, quant_min=quant_min,
                         quant_max=quant_max, eps=eps, **kwargs)
        import os  # PTF_MAX_ALPHA overrides the alpha clamp (A/B on the clipping ceiling base * 2^max_alpha * 127)
        self.max_alpha = int(os.environ.get("PTF_MAX_ALPHA", max_alpha))

    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        levels = float(self.quant_max - self.quant_min)
        ranges = (max_val - min_val).clamp(min=self.eps)
        per_ch = ranges / levels                      # the scale each channel would want
        base = per_ch.min().clamp(min=self.eps)       # layer-wise scale s
        alpha = torch.log2(per_ch / base).ceil().clamp(0, self.max_alpha)
        scale = base * torch.pow(2.0, alpha)
        zp = (self.quant_min - torch.round(min_val / scale)).clamp(self.quant_min, self.quant_max)
        return scale.to(torch.float32), zp.to(torch.int32)


class KMedianObserver(MinMaxObserver):
    """Per-tensor symmetric scale = k * median(|x|) / quant_max instead of max / quant_max.

    For the fine rsqrt table of the dual-range LayerNorm: the per-token variance tensor is dominated by a
    few sink tokens (max / median up to 1e5), so a MinMax grid leaves ordinary tokens a handful of levels.
    Sizing the grid by k * median gives ordinary tokens thousands of levels; the sink tokens saturate in the
    quantize (= the int32 -> int16 RESCALE on the device), and the coarse table handles them.
    """

    def __init__(self, k=4.0, max_samples=1_000_000, **kwargs):
        super().__init__(**kwargs)
        self.k = float(k)
        self.max_samples = max_samples
        self.register_buffer("samples", torch.zeros(0))

    def forward(self, x):
        super().forward(x)
        v = x.detach().abs().reshape(-1).float()  # stay on the observer's device: convert_pt2e asserts a single device
        if v.numel() > 8192:
            v = v[torch.randint(0, v.numel(), (8192,), device=v.device)]
        self.samples = torch.cat([self.samples.to(v.device), v])[-self.max_samples:]
        return x

    def calculate_qparams(self):
        if self.samples.numel() == 0:
            return super().calculate_qparams()
        c = self.k * self.samples.median().item()
        dev = self.samples.device
        scale = torch.tensor([max(c / self.quant_max, self.eps)], dtype=torch.float32, device=dev)
        return scale, torch.zeros(1, dtype=torch.int64, device=dev)

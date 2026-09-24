"""Lower per-channel ACTIVATION quantization to per-tensor quantization plus per-channel int32 MULs.

The Arm backend and Vela take per-tensor activations only (IFM scaling is per-tensor on Ethos-U),
so a converted graph with `quantize_per_channel` on an activation (PTF / PCS residual outputs)
cannot be delegated. With symmetric per-channel scales s_c and base s = min_c s_c, the code
q_c = round(x / s_c) equals round((x * s / s_c) / s), i.e. a per-tensor int8 quantization of
x * f_c with f_c = s / s_c <= 1, and the dequantized value q_c * s_c equals (q_c * s) * g_c with
g_c = s_c / s. Both factors are per-channel constant vectors, so each side becomes one elementwise
MUL by a constant (int32 on the NPU: TOSA MUL + RESCALE, in Vela's supported set) around an
ordinary per-tensor int8 tensor: the int8 codes move through memory ops untouched and the
expansion happens once at each consumer, which is the semantics the fake-quant experiments assume.

Rewrite, on the graph produced by convert_pt2e (before export / to_edge):

    x -> quantize_per_channel -> dequantize_per_channel -> consumers
becomes
    x -> Q32 -> DQ32 -> mul(f) -> Q8(s) -> DQ8(s) -> mul(g) -> Q32 -> DQ32 -> consumers

with f, g stored as int32 buffers read through per-tensor dequantize (scales 2^-24 and 2^-16),
and the two int32 activation grids at s / 2^bits (adaptive, up to 2^-16). Zero points must be zero (a16inpc8symout).
"""
import torch

Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
Q_PC = torch.ops.quantized_decomposed.quantize_per_channel.default
DQ_PC = torch.ops.quantized_decomposed.dequantize_per_channel.default
I32 = (-2147483647, 2147483647)
F_SCALE, G_SCALE = 2 ** -24, 2 ** -16


def rewrite_per_channel_activations(gm: torch.fx.GraphModule, x_bits_below: int | None = None) -> int:
    """x_bits_below: int32 activation grid = s_base / 2^bits. None = adaptive: as fine as 2^-16 while the largest
    code 127 * 2^bits * (s_max / s_base) stays under 2^22 (int32 headroom); s_base/2^8 costs DeiT-B ~1 pt."""
    import math
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not Q_PC or node.args[0].op == "get_attr":
            continue  # weights keep their per-channel quantization
        x, scale_n, zp_n, axis, qmin, qmax, dtype = node.args[:7]
        scales = getattr(gm, scale_n.target).detach().float()
        zps = getattr(gm, zp_n.target)
        if int(zps.abs().max()) != 0:
            raise ValueError(f"{node.name}: per-channel zero points are not zero; use a symmetric per-channel config")
        dq_nodes = [u for u in node.users if u.target is DQ_PC]
        s_base = scales.min().item()
        ndim = x.meta["val"].ndim if "val" in x.meta else axis + 1
        shape = [1] * ndim
        shape[axis] = scales.numel()
        f_q = torch.round((s_base / scales) / F_SCALE).to(torch.int32).reshape(shape)
        g_q = torch.round((scales / s_base) / G_SCALE).to(torch.int32).reshape(shape)
        gm.register_buffer(f"{node.name}_f", f_q)
        gm.register_buffer(f"{node.name}_g", g_q)
        bits = x_bits_below if x_bits_below is not None else max(4, min(16, 22 - math.ceil(math.log2(scales.max().item() / s_base))))
        s_x = s_base / 2 ** bits
        with gm.graph.inserting_before(node):
            f_dq = gm.graph.call_function(DQ, (gm.graph.get_attr(f"{node.name}_f"), F_SCALE, 0, *I32, torch.int32))
            x_q = gm.graph.call_function(Q, (x, s_x, 0, *I32, torch.int32))
            x_dq = gm.graph.call_function(DQ, (x_q, s_x, 0, *I32, torch.int32))
            t = gm.graph.call_function(torch.ops.aten.mul.Tensor, (x_dq, f_dq))
            t_q = gm.graph.call_function(Q, (t, s_base, 0, qmin, qmax, dtype))
            t_dq = gm.graph.call_function(DQ, (t_q, s_base, 0, qmin, qmax, dtype))
            g_dq = gm.graph.call_function(DQ, (gm.graph.get_attr(f"{node.name}_g"), G_SCALE, 0, *I32, torch.int32))
            y = gm.graph.call_function(torch.ops.aten.mul.Tensor, (t_dq, g_dq))
            y_q = gm.graph.call_function(Q, (y, s_x, 0, *I32, torch.int32))
            y_dq = gm.graph.call_function(DQ, (y_q, s_x, 0, *I32, torch.int32))
        for new in (f_dq, x_q, x_dq, t, t_q, t_dq, g_dq, y, y_q, y_dq):
            new.meta["nn_module_stack"] = dict(node.meta.get("nn_module_stack", {}))
        # The TOSA partitioner's CheckArmQuantized only delegates ops carrying the Arm quantizer's marker.
        for new in (t, y):
            custom = dict(new.meta.get("custom", {}))
            custom["_arm_annotation_info"] = {"quantized": True}
            new.meta["custom"] = custom
        for dq in dq_nodes:
            dq.replace_all_uses_with(y_dq)
            gm.graph.erase_node(dq)
        gm.graph.erase_node(node)
        count += 1
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return count

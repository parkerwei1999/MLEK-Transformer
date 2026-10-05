"""Lower INT32-quantized activations to TOSA at the scales the quantized model uses.

ExecuTorch 1.3.1's InsertRescaleInt32Pass (backends/arm/_passes/insert_rescales_pass.py:317, :356, :405) only rescales
INT8 / INT16 operands and outputs. An op whose operands or output carry INT32 qparams (the `a32w8` rules: LN-32
internals, GA-LN, the PCS rewrite's mul(f) / mul(g)) is emitted with no rescale at all: a MUL's result stays on the
product grid s_a * s_b while every consumer assumes the annotated output scale, an INT32 output that should be
re-quantized is not, and the Q8 of the PCS rewrite (INT32 operands, INT8 output) is never materialised.

Int32RequantPass runs right before InsertRescaleInt32Pass and lowers every add / sub / mul / sum.dim_IntList that
touches INT32 qparams itself, so the integer graph realises the annotated scale of every tensor:
  * INT8 / INT16 operands go to INT32 through a RESCALE (as the Arm pass does);
  * MUL: the 64-bit product is brought onto the output grid inside the MUL (TOSA MUL `shift`, int32 only);
    an INT8 / INT16 output gets its RESCALE from the actual product scale;
  * ADD / SUB: both operands are put on the output grid first; SUM accumulates on the output grid;
  * every INT32 -> INT32 change of scale is an INT32 MUL by a fixed-point constant with a shift
    (x * m >> s), never an INT32 -> INT32 RESCALE (Arm's fuse_consecutive_rescales_pass.py:50-53: Vela mishandles it);
  * a DQ(int32) -> Q(int32) pair left between two INT32 tensors becomes that same MUL.
The TOSA MUL shift is serialised by MulShiftVisitor (the stock MulVisitor, operators/op_mul.py:52, always writes 0).
Call install() before lowering (lower_probe.py does).
"""
import math

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.insert_rescales_pass import InsertRescaleInt32Pass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.export.graph_signature import InputKind

MUL, ADD, SUB = exir_ops.edge.aten.mul.Tensor, exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.sub.Tensor
SUM = exir_ops.edge.aten.sum.dim_IntList
RESCALE = exir_ops.backend.tosa.RESCALE.default
SHIFT_KEY = "int32_requant_mul_shift"  # node.meta key read by MulShiftVisitor
I32_MAX = 2 ** 31 - 1
HEADROOM = 2 ** 30  # largest INT32 code an intermediate may reach (one bit of margin)


def _q32(scale: float) -> QuantArgs:
    return QuantArgs(scale=scale, zp=0, qmin=-I32_MAX - 1, qmax=I32_MAX, dtype=torch.int32)


def _scale(q: QuantArgs) -> float:
    return float(q.get_scale_per_tensor())


def _zp(q: QuantArgs) -> int:
    return int(q.get_zp_per_tensor())


def _fixed_point(ratio: float) -> tuple[int, int]:
    """(m, s) with m * 2^-s ~= ratio, m < 2^30, 0 <= s <= 62 (TOSA MUL int32 shift range)."""
    assert ratio > 0, ratio
    s = min(62, max(0, 29 - math.floor(math.log2(ratio))))
    m = round(ratio * 2 ** s)
    assert 0 < m < 2 ** 31, (ratio, m, s)
    return m, s


class Int32RequantPass(ArmPass):
    _passes_required_after = set()

    def __init__(self, exported_program):
        super().__init__()
        self.exported_program = exported_program
        self.report = []  # (node, action) for debugging / the DAG check

    # -- graph helpers -------------------------------------------------------------------------------------------
    def _const(self, graph, ref: torch.fx.Node, value: int) -> torch.fx.Node:
        name = f"b_int32_requant_{len(self.report)}_{abs(value)}"
        rank = len(ref.meta["val"].shape)
        with graph.inserting_before(next(iter(graph.nodes))):
            return create_constant_placeholder(self.exported_program, graph, name, InputKind.BUFFER,
                                               torch.full((1,) * rank, value, dtype=torch.int32), persistent_buffer=True)

    def _mul32(self, graph, x: torch.fx.Node, ratio: float, s_in: float, from_node) -> torch.fx.Node:
        """INT32 x on grid s_in -> INT32 on grid s_in / ratio, as MUL(x, m) >> s."""
        m, s = _fixed_point(ratio)
        c = self._const(graph, x, m)
        with graph.inserting_after(x):
            y = create_node(graph, MUL, (x, c), from_node=from_node)
        y.meta["input_qparams"] = {0: _q32(s_in), 1: _q32(1.0 / m)}  # the constant m stands for the real value 1
        y.meta["output_qparams"] = {0: _q32(s_in / ratio)}
        y.meta[SHIFT_KEY] = s
        y.meta["val"] = x.meta["val"]
        self.report.append((y.name, f"requant x{ratio:.6g} (m={m}, shift={s})"))
        return y

    def _rescale_to32(self, graph, x: torch.fx.Node, q: QuantArgs, s_new: float, from_node) -> torch.fx.Node:
        """INT8 / INT16 x -> INT32 on grid s_new (a RESCALE, the standard input conversion)."""
        with graph.inserting_after(x):
            r = create_node(graph, RESCALE, (x, torch.int32, [_scale(q) / s_new], _zp(q), 0), from_node=from_node)
        r.meta["val"] = x.meta["val"]
        self.report.append((r.name, f"rescale {q.dtype}->int32 x{_scale(q) / s_new:.6g}"))
        return r

    def _operand_to(self, graph, node, i: int, q: QuantArgs, s_new: float) -> None:
        """Put operand i of node on INT32 grid s_new (no-op when it already is)."""
        x = node.args[i]
        if q.dtype == torch.int32:
            assert _zp(q) == 0, f"{node.name}: INT32 operand with zero point {_zp(q)}"
            if abs(_scale(q) / s_new - 1) < 1e-9:
                return
            y = self._mul32(graph, x, _scale(q) / s_new, _scale(q), node)
        else:
            y = self._rescale_to32(graph, x, q, s_new, node)
        node.replace_input_with(x, y)  # both operands of a square share x

    def _output_to(self, graph, node, s_actual: float, out_q: QuantArgs) -> None:
        """node's INT32 result sits on s_actual; bring it to out_q (INT32 MUL or RESCALE to INT8 / INT16)."""
        users = list(node.users)
        if out_q.dtype == torch.int32:
            if abs(s_actual / _scale(out_q) - 1) < 1e-9:
                return
            y = self._mul32(graph, node, s_actual / _scale(out_q), s_actual, node)
        else:
            with graph.inserting_after(node):
                y = create_node(graph, RESCALE, (node, out_q.dtype, [s_actual / _scale(out_q)], 0, _zp(out_q)),
                                from_node=node)
            y.meta["val"] = node.meta["val"]
            self.report.append((y.name, f"rescale int32->{out_q.dtype} x{s_actual / _scale(out_q):.6g}"))
        for u in users:
            u.replace_input_with(node, y)

    # -- per-op lowering -----------------------------------------------------------------------------------------
    def _mul(self, graph, node, in_q, out_q) -> None:
        for i, q in list(in_q.items()):
            if q.dtype != torch.int32:  # INT8 / INT16 operand: to INT32 at its own scale (Arm's MUL convention)
                if i == 1 and node.args[1] is node.args[0]:
                    continue
                self._operand_to(graph, node, i, q, _scale(q))
        p = math.prod(_scale(q) for q in in_q.values())  # integer product grid
        s_o = _scale(out_q)
        if out_q.dtype == torch.int32:
            k = max(0, math.ceil(math.log2(s_o / p) - 1e-9))  # p * 2^k >= s_o: the shifted product fits like the output
        else:
            bound = s_o * max(abs(out_q.qmin - _zp(out_q)), abs(out_q.qmax - _zp(out_q)))
            k = max(0, math.ceil(math.log2(bound / (p * HEADROOM))))
        node.meta[SHIFT_KEY] = k
        self.report.append((node.name, f"mul shift {k}"))
        node.meta["input_qparams"] = {i: _q32(_scale(q)) for i, q in in_q.items()}
        node.meta["output_qparams"] = {0: _q32(p * 2 ** k)}
        self._output_to(graph, node, p * 2 ** k, out_q)

    def _add_sub(self, graph, node, in_q, out_q) -> None:
        if out_q.dtype == torch.int32:
            grid = _scale(out_q)
        else:  # INT8 / INT16 result: accumulate on a grid with headroom for the INT32 operands
            grid = max(2 * max(_scale(q) for q in in_q.values() if q.dtype == torch.int32),
                       2 * max(_scale(q) for q in in_q.values()) / 2 ** 12)
        for i, q in list(in_q.items()):
            if i == 1 and node.args[1] is node.args[0]:
                continue
            self._operand_to(graph, node, i, q, grid)
        node.meta["input_qparams"] = {i: _q32(grid) for i in in_q}
        node.meta["output_qparams"] = {0: _q32(grid)}
        self._output_to(graph, node, grid, out_q)

    def _sum(self, graph, node, in_q, out_q) -> None:
        """Accumulate on a grid where no partial sum leaves the INT32 headroom; the result then goes to out_q.
        No extra op where it can be avoided: an INT32 operand that is a requantize of an INT8 / INT16 tensor
        (Q(int32) after DQ(int8 / int16)) has that narrower range, and the change of grid is folded into that
        requantize (it lowers to the INT16 -> INT32 RESCALE that exists anyway); a result whose users are all ops this
        pass lowers is handed to them on the accumulation grid instead of through a requant op."""
        q = in_q[0]
        x = node.args[0]
        n = math.prod(x.meta["val"].shape[d] for d in node.args[1])
        square = getattr(x, "target", None) is MUL and x.args[0] is x.args[1]  # partial sums of squares <= the total
        narrow = (q.dtype == torch.int32 and x.target in Q_OPS and len(x.users) == 1 and isinstance(x.args[0], torch.fx.Node)
                  and x.args[0].target in DQ_OPS and x.args[0].args[5] in (torch.int8, torch.int16))
        if square and out_q.dtype == torch.int32:
            grid = _scale(out_q)  # the total fits out_q, so every partial sum does too
        else:
            if narrow:  # the INT8 / INT16 tensor behind the requantize bounds the value
                dq = x.args[0]
                peak = float(dq.args[1]) * max(abs(int(dq.args[3]) - int(dq.args[2])), abs(int(dq.args[4]) - int(dq.args[2])))
            else:
                peak = _scale(q) * (HEADROOM if q.dtype == torch.int32 else max(abs(q.qmin - _zp(q)), abs(q.qmax - _zp(q))))
            grid = max(n * peak / HEADROOM, _scale(out_q) if out_q.dtype == torch.int32 else 0.0)
        if narrow and abs(_scale(q) / grid - 1) > 1e-9:
            x.update_arg(1, grid)  # requantize straight onto the accumulation grid
            self.report.append((x.name, f"requantize onto sum grid {grid:.6g} (was {_scale(q):.6g})"))
        else:
            self._operand_to(graph, node, 0, q, grid)
        node.meta["input_qparams"] = {0: _q32(grid)}
        node.meta["output_qparams"] = {0: _q32(grid)}
        users = list(node.users)
        if (out_q.dtype == torch.int32 and abs(grid / _scale(out_q) - 1) > 1e-9 and users
                and all(u.target in (MUL, ADD, SUB, SUM) and "input_qparams" in u.meta for u in users)):
            for u in users:  # the users are lowered later by this pass and take the accumulation grid as their input scale
                for i, a in enumerate(u.args):
                    if a is node and i in u.meta["input_qparams"]:
                        u.meta["input_qparams"][i] = _q32(grid)
            self.report.append((node.name, f"sum result handed on at grid {grid:.6g}"))
            return
        self._output_to(graph, node, grid, out_q)

    def _dq_q_pair(self, graph, q_node) -> bool:
        """DQ(int32, s1) -> Q(int32, s2) between two INT32 tensors: one INT32 MUL instead of a later INT32 RESCALE.
        The DQ may feed other quantizes too (e.g. an INT16 copy); those keep it."""
        dq = q_node.args[0]
        if not (isinstance(dq, torch.fx.Node) and dq.target in DQ_OPS and q_node.args[5] == torch.int32
                and dq.args[5] == torch.int32):
            return False
        s1, s2 = float(dq.args[1]), float(q_node.args[1])
        assert int(dq.args[2]) == 0 and int(q_node.args[2]) == 0, f"{q_node.name}: INT32 zero points"
        y = self._mul32(graph, dq.args[0], s1 / s2, s1, q_node)
        q_node.replace_all_uses_with(y)
        graph.erase_node(q_node)
        if not dq.users:
            graph.erase_node(dq)
        return True

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target in Q_OPS:
                modified |= self._dq_q_pair(graph, node)
                continue
            in_q = node.meta.get("input_qparams") or {}
            out_q = (node.meta.get("output_qparams") or {}).get(0)
            if not in_q or out_q is None:
                continue
            if not (any(q.dtype == torch.int32 for q in in_q.values()) or out_q.dtype == torch.int32):
                continue
            if node.target is MUL:
                self._mul(graph, node, in_q, out_q)
            elif node.target in (ADD, SUB):
                self._add_sub(graph, node, in_q, out_q)
            elif node.target is SUM:
                self._sum(graph, node, in_q, out_q)
            elif node.target in InsertRescaleInt32Pass.included_targets:
                raise NotImplementedError(f"{node.name} ({node.target}): INT32 qparams on an op this pass does not lower")
            else:
                continue
            modified = True
        if modified:
            graph_module = super().call(graph_module).graph_module
            graph_module.recompile()
        return PassResult(graph_module, modified)


def _mul_shift_visitor():
    import tosa_serializer as ts  # the module the stock visitors use
    from executorch.backends.arm.operators.op_mul import MulVisitor

    class MulShiftVisitor(MulVisitor):
        """The stock MUL visitor with the TOSA int32 `shift` taken from node.meta[SHIFT_KEY] (stock: always 0)."""

        def define_node(self, node, tosa_graph, inputs, output) -> None:
            shift = int(node.meta.get(SHIFT_KEY, 0))
            if shift == 0:
                return super().define_node(node, tosa_graph, inputs, output)
            assert all(t.dtype == ts.DType.INT32 for t in (*inputs, output)), f"{node.name}: shift needs INT32 MUL"
            tosa_graph.addConst([1], ts.DType.INT8, [shift], name=f"{output.name}_shift")
            attr = ts.TosaSerializerAttribute()
            attr.MulAttribute()
            self._serialize_operator(node, tosa_graph, ts.Op.MUL,
                                     [inputs[0].name, inputs[1].name, f"{output.name}_shift"], [output.name], attr)

    return MulShiftVisitor


_installed = False


def install() -> None:
    """Put Int32RequantPass in front of InsertRescaleInt32Pass in every ArmPassManager and register MulShiftVisitor."""
    global _installed
    if _installed:
        return
    from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
    from executorch.backends.arm.operators.node_visitor import register_node_visitor

    stock = ArmPassManager._configure_pass_insertions

    def configure(self, exported_program):
        stock(self, exported_program)
        self.insert_passes_before(InsertRescaleInt32Pass, [Int32RequantPass(exported_program)])

    ArmPassManager._configure_pass_insertions = configure
    register_node_visitor(_mul_shift_visitor())
    _installed = True

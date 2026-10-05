"""Helpers on converted graphs and lowered TOSA files, used by the lowering entry point and the
diagnostics.
"""
from collections import Counter
from pathlib import Path

from tosa import DType, Op
from tosa.TosaGraph import TosaGraph

OP_NAMES = {v: k for k, v in vars(Op.Op).items() if not k.startswith("_")}
DTYPE_NAMES = {v: k for k, v in vars(DType.DType).items() if not k.startswith("_")}


def is_q(node):
    target = str(node.target)
    return node.op == "call_function" and "quantize_per" in target and "dequantize" not in target


def is_dq(node):
    return node.op == "call_function" and "dequantize_per" in str(node.target)



def tosa_report(intermediates: Path):
    report = {"files": 0, "ops": Counter(), "tables": [], "matmul": [], "dtypes": Counter()}
    for path in sorted(intermediates.glob("*.tosa")):
        report["files"] += 1
        buf = path.read_bytes()
        block = TosaGraph.GetRootAs(buf, 0).Regions(0).Blocks(0)
        tensors = {}
        for i in range(block.TensorsLength()):
            t = block.Tensors(i)
            tensors[t.Name().decode()] = (DTYPE_NAMES.get(t.Type()),
                                          [t.Shape(j) for j in range(t.ShapeLength())])
        for i in range(block.OperatorsLength()):
            op = block.Operators(i)
            name = OP_NAMES.get(op.Op(), str(op.Op()))
            report["ops"][name] += 1
            ins = [op.Inputs(j).decode() for j in range(op.InputsLength())]
            outs = [op.Outputs(j).decode() for j in range(op.OutputsLength())]
            if name not in ("CONST", "CONST_SHAPE"):
                report["dtypes"][tensors[outs[0]][0]] += 1
            if name == "TABLE":
                report["tables"].append((tensors[ins[0]][0], tensors[ins[1]][0],
                                         tensors[ins[1]][1][0], tensors[outs[0]][0]))
            if name == "MATMUL":
                report["matmul"].append((tensors[ins[0]][0], tensors[outs[0]][0]))
    report["tables"] = Counter(report["tables"])
    report["matmul"] = Counter(report["matmul"])
    return report

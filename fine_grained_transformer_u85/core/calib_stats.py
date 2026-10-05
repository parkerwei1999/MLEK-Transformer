"""Calibrate once, apply to any MinMax vision recipe (vision_eval.py --calib-stats).

Only for MinMaxObserver / PerChannelMinMaxObserver, not HistogramObserver or KMedianObserver.
"""
import copy

import torch
from torch import nn

from torchao.quantization.pt2e import (FixedQParamsObserver, HistogramObserver, MinMaxObserver, NoopObserver,
                                       ObserverBase, PerChannelMinMaxObserver, PlaceholderObserver)

from core.quant import ActObserver, EthosUQuantizer, QuantConfig, move_graph_module

CH_AXIS = 2

class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mn = self.mx = self.ch_min = self.ch_max = None

    def forward(self, x):
        if not (isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel()):
            return x
        x = x.detach()
        mn, mx = torch.aminmax(x)
        self.mn = mn if self.mn is None else torch.minimum(self.mn, mn)
        self.mx = mx if self.mx is None else torch.maximum(self.mx, mx)
        if x.ndim > CH_AXIS:
            dims = [d for d in range(x.ndim) if d != CH_AXIS]
            cmn, cmx = x.amin(dim=dims), x.amax(dim=dims)
            self.ch_min = cmn if self.ch_min is None else torch.minimum(self.ch_min, cmn)
            self.ch_max = cmx if self.ch_max is None else torch.maximum(self.ch_max, cmx)
        return x


def collect(module, example_inputs, compile_spec, batches, device) -> dict:
    """Collect min/max values for each node in the transformed graph."""
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(QuantConfig.A8W8.build(ActObserver.MINMAX, None))
    # a copy: the exported graph shares the module's parameters, and moving it to `device` would move the caller's
    # model off the CPU, where prepare_pt2e must run
    exported = torch.export.export(copy.deepcopy(module), example_inputs, strict=True)

    # We need transformation for decomposed graph
    gm = quantizer.transform_for_annotation(exported.module(check_guards=False))

    recorders = {}
    for n in list(gm.graph.nodes):
        if n.op not in ("placeholder", "call_function"):
            continue
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor) and not val.is_floating_point():
            continue
        name = f"_calib_rec_{len(recorders)}"
        gm.add_module(name, _Recorder())
        with gm.graph.inserting_after(n):
            gm.graph.call_module(name, (n,))  # no users: recorded, not replacing anything
        recorders[n.name] = name
    gm.recompile()
    move_graph_module(gm, device)
    with torch.no_grad():
        for b in batches:
            gm(*[t.to(device) for t in b])
    nodes = {}
    for node, name in recorders.items():
        r = getattr(gm, name)
        if r.mn is None:
            continue  # never saw a float tensor
        nodes[node] = {"min": r.mn.cpu(), "max": r.mx.cpu(),
                       "ch_min": None if r.ch_min is None else r.ch_min.cpu(),
                       "ch_max": None if r.ch_max is None else r.ch_max.cpu()}
    return {"nodes": nodes, "ch_axis": CH_AXIS}


def _observed(n, gm):
    """The node whose value an observer call sees: walk back through observer calls (they pass values through)."""
    while n.op == "call_module" and isinstance(getattr(gm, n.target, None), ObserverBase):
        n = n.args[0]
    return n


def _attr(gm, target):
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _set(buf, value):
    buf.resize_(value.shape).copy_(value.to(buf.device))


def apply(prepared, stats) -> dict:
    """Apply min/max values collected by `collect()` to the observers in a prepared graph."""
    nodes = stats["nodes"]
    groups, names = {}, {}  # keyed by the observer OBJECT: a shared spec registers one observer under several names
    for n in prepared.graph.nodes:
        if n.op == "call_module" and isinstance(getattr(prepared, n.target, None), ObserverBase):
            obj = getattr(prepared, n.target)
            groups.setdefault(id(obj), []).append(_observed(n.args[0], prepared)); names.setdefault(id(obj), n.target)
    counts = {"activation": 0, "shared": 0, "constant": 0, "skipped": 0}
    with torch.no_grad():
        for key, srcs in groups.items():
            target = names[key]; obs = getattr(prepared, target)
            srcs = list(dict.fromkeys(srcs))
            if isinstance(obs, (FixedQParamsObserver, PlaceholderObserver, NoopObserver)):
                counts["skipped"] += 1
                continue
            if isinstance(obs, HistogramObserver) or type(obs).__name__ == "KMedianObserver":
                raise NotImplementedError(f"{target}: {type(obs).__name__} cannot be restored from min / max; "
                                          "calib_stats covers MinMax recipes only (calibrate normally instead)")
            if all(s.op == "get_attr" for s in srcs):
                for s in srcs:  # min / max of a constant: one observation is enough
                    obs(_attr(prepared, s.target))
                counts["constant"] += 1
                continue
            missing = [s.name for s in srcs if s.op != "get_attr" and s.name not in nodes]
            if missing:
                raise KeyError(f"{target}: no recorded state for {missing} (graph differs from the collected one)")
            states = []
            for s in srcs:
                if s.op == "get_attr":  # a constant sharing an observer with activations
                    r = _Recorder(); r(_attr(prepared, s.target)); states.append({"min": r.mn.cpu(), "max": r.mx.cpu(),
                        "ch_min": None if r.ch_min is None else r.ch_min.cpu(), "ch_max": None if r.ch_max is None else r.ch_max.cpu()})
                else:
                    states.append(nodes[s.name])

            # Handles per-channel or per-tensor min/max.
            if isinstance(obs, PerChannelMinMaxObserver):
                assert obs.ch_axis == stats["ch_axis"], f"{target}: per-channel axis {obs.ch_axis} != recorded {stats['ch_axis']}"
                _set(obs.min_val, torch.stack([st["ch_min"] for st in states]).amin(0))
                _set(obs.max_val, torch.stack([st["ch_max"] for st in states]).amax(0))
            elif isinstance(obs, MinMaxObserver):
                _set(obs.min_val, torch.stack([st["min"] for st in states]).amin())
                _set(obs.max_val, torch.stack([st["max"] for st in states]).amax())
            else:
                raise NotImplementedError(f"{target}: observer type {type(obs).__name__}")
            counts["shared" if len(states) > 1 else "activation"] += 1
    return counts

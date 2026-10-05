"""core.slices: pieces compose back to the full model, blocks group as expected, and exported pieces
keep the original module paths that precision rules match on.
Run: python tests/test_slices.py   (or pytest)"""
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))
from core.slices import check_layernorms, make_pieces  # noqa: E402
from core.newton_layernorm import swap_layernorms  # noqa: E402

torch.manual_seed(0)


def _vision(name):
    import fqvit_models
    return fqvit_models.build_model(name, pretrained=False)


def _whisper(part):
    from helper import whisper_executorch_wrapper as w
    return w._build("openai/whisper-tiny", w.WhisperPart(part), 16)


def _one_piece_per_block(full, paths):
    """A rule naming each block makes every block its own group, so the pieces cover the model in order."""
    return make_pieces(full, [re.escape(p) + r"\.=a8w8" for p in paths])


def _deepest_paths(module, example):
    ep = torch.export.export(module, example, strict=True)
    out = []
    for n in ep.graph.nodes:
        stack = n.meta.get("nn_module_stack")
        if stack:
            out.append((list(stack.values())[-1][0], str(n.target)))
    return out


def test_deit_compose_and_group():
    full = _vision("deit_tiny_patch16_224")
    x = torch.randn(2, 3, 224, 224)
    blocks = [f"blocks.{i}" for i in range(len(full.blocks))]
    pcs = _one_piece_per_block(full, blocks)
    with torch.no_grad():
        y = pcs[0].module(x)
        for pc in pcs[1:-1]:
            y = pc.module(y)
        y = pcs[-1].module(y)
        ref = full(x)
    assert torch.allclose(y, ref, atol=1e-5, rtol=0), (y - ref).abs().max()
    grouped = make_pieces(full, [r"norm\d?$=a32w8@sub,mul,sum.dim_IntList,add"])
    assert [(p.name, p.reps) for p in grouped] == [("prologue", 1), ("blocks.0", 12), ("epilogue", 1)]


def test_swin_compose_and_group():
    full = _vision("swin_tiny_patch4_window7_224")
    x = torch.randn(2, 3, 224, 224)
    blocks = [f"layers.{s}.blocks.{j}" for s, l in enumerate(full.layers) for j in range(len(l.blocks))]
    pcs = _one_piece_per_block(full, blocks)
    with torch.no_grad():
        y = pcs[0].module(x)
        for pc in pcs[1:-1]:
            y = pc.module(y)
        y = pcs[-1].module(y)
        ref = full(x)
    assert torch.allclose(y, ref, atol=1e-4, rtol=0), (y - ref).abs().max()
    got = [(p.name, p.reps) for p in make_pieces(full, [r"layers\.2\.=a16w8"])]
    # stages 1-3: plain + shifted window block; stage 4 (7x7 = one window): shift disabled, one group
    assert got == [("prologue", 1), ("layers.0.blocks.0", 1), ("layers.0.blocks.1", 1), ("layers.0.downsample", 1),
                   ("layers.1.blocks.0", 1), ("layers.1.blocks.1", 1), ("layers.1.downsample", 1),
                   ("layers.2.blocks.0", 3), ("layers.2.blocks.1", 3), ("layers.2.downsample", 1),
                   ("layers.3.blocks.0", 2), ("epilogue", 1)], got
    # a rule that names one block splits it out of its group
    got = [(p.name, p.reps) for p in make_pieces(full, [r"layers\.2\.blocks\.5\.=a16w8"]) if p.name.startswith("layers.2.b")]
    assert got == [("layers.2.blocks.0", 3), ("layers.2.blocks.1", 2), ("layers.2.blocks.5", 1)], got
    # exported block keeps its full-model path
    paths = {p for p, _ in _deepest_paths(make_pieces(full)[8].module, (torch.randn(1, 196, 384),))}
    assert any(p.startswith("layers.2.blocks.1") for p in paths), sorted(paths)[:5]


def test_whisper_encoder_compose_and_paths():
    full, (mel,) = _whisper("encoder")
    full.eval()
    layers = [f"encoder.layers.{i}" for i in range(len(full.encoder.layers))]
    pcs = _one_piece_per_block(full, layers)
    with torch.no_grad():
        h = pcs[0].module(mel)
        for pc in pcs[1:-1]:
            h = pc.module(h)
        h = pcs[-1].module(h)
        ref = full(mel)
    assert torch.allclose(h, ref, atol=1e-4, rtol=0), (h - ref).abs().max()
    # the front GELUs must sit in module `encoder`, where the rule `encoder$=a8w8@gelu` finds them
    gelus = [p for p, t in _deepest_paths(make_pieces(full)[0].module, (mel,)) if "gelu" in t]
    assert gelus and all(p == "encoder" for p in gelus), gelus


def test_whisper_decoder_compose():
    full, (ids, enc) = _whisper("decoder")
    full.eval()
    layers = [f"decoder.layers.{i}" for i in range(len(full.decoder.layers))]
    pcs = _one_piece_per_block(full, layers)
    with torch.no_grad():
        h = pcs[0].module(ids)
        for pc in pcs[1:-1]:
            h = pc.module(h, enc)
        out = pcs[-1].module(h)
        ref = full(ids, enc)
    assert torch.allclose(out, ref, atol=1e-4, rtol=0), (out - ref).abs().max()
    assert [(p.name, p.reps) for p in make_pieces(full)] == [("prologue", 1), ("decoder.layers.0", 4), ("epilogue", 1)]


def test_layernorm_swap_order_and_rule_string():
    # swap on the full model before slicing, or on each piece after slicing: every piece fully replaced
    for order in ("full_then_slice", "slice_then_piece"):
        full = _vision("deit_tiny_patch16_224")
        if order == "full_then_slice":
            swap_layernorms(full, 1)
            pcs = make_pieces(full)
        else:
            pcs = make_pieces(full)
            for p in pcs:
                swap_layernorms(p.module, 1)
        for p in pcs:
            check_layernorms(p.module, replaced=True)
    # swap on the full model AFTER slicing misses the top-level `norm` held by the epilogue piece
    full = _vision("deit_tiny_patch16_224")
    pcs = make_pieces(full)
    swap_layernorms(full, 1)
    try:
        check_layernorms(pcs[-1].module, replaced=True)
        raise AssertionError("missed swap not detected")
    except AssertionError as e:
        assert "not replaced" in str(e), e
    # the --prec-rules string and its item list group the same way
    s = r"layers\.2\.blocks\.5\.=a16w8;norm\d?$=a32w8@sub,mul"
    full = _vision("swin_tiny_patch4_window7_224")
    assert [(p.name, p.reps) for p in make_pieces(full, s)] == [(p.name, p.reps) for p in make_pieces(full, s.split(";"))]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

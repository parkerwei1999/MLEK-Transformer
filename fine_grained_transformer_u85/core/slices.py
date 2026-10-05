"""Split a model into prologue / repeated pieces / epilogue for latency profiling.

Each piece is lowered, compiled and simulated on its own; the model latency is
    sum over pieces of (piece latency x repetitions),
without any correction for overlap between pieces (stated as part of the experiment setup).

A piece is a skeleton module that keeps the ORIGINAL module paths: the original submodule objects
are grafted at their full-model paths (weights shared, not copied), so precision rules, which match
the deepest module path of each op (e.g. `layers\\.2\\.blocks\\.5\\.`, `encoder$=a8w8@gelu`),
select the same tensors as in the full graph. Functional code that the full model runs inside a
module's forward (Whisper's conv/GELU front inside `encoder`) runs inside a scope module at that
same path.

Repeated blocks are grouped when they are structurally identical AND the same precision rules
match them (a rule naming one block, e.g. `layers\\.2\\.blocks\\.5\\.`, puts that block in a group
of its own). Swin blocks are further keyed by stage and shift size.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn


class _Scope(nn.Module):
    """Container at an original module path; with `fn` it runs that part of the original forward."""

    def __init__(self, fn: Callable = None):
        super().__init__()
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn(self, *args, **kwargs)


@dataclass
class Piece:
    name: str                  # "prologue", "epilogue" or the representative path, e.g. "layers.2.blocks.1"
    module: nn.Module          # path-preserving skeleton; forward(*inputs)
    reps: int                  # occurrences in the full model
    covers: list = field(default_factory=list)  # full-model paths this piece stands for
    hook_path: str = ""        # full-model module whose input feeds this piece ("" = model input)
    input_kwargs: tuple = ()   # keyword inputs of hook_path that are also piece inputs (after the first positional)


def _resolve(node: nn.Module, path: str) -> nn.Module:
    """Attribute / ModuleDict walk (traceable, keeps the access path for export's nn_module_stack)."""
    for p in path.split("."):
        node = node[p] if isinstance(node, nn.ModuleDict) else getattr(node, p)
    return node


def _graft(root: nn.Module, full: nn.Module, path: str, scope_fn: Callable = None) -> None:
    """Register full's module at `path` under root (intermediate containers created on the way).
    With scope_fn, the module at `path` is a new _Scope instead (its children are grafted separately)."""
    parts = path.split(".")
    node = root
    for i, p in enumerate(parts[:-1]):
        child = node[p] if isinstance(node, nn.ModuleDict) and p in node else node._modules.get(p)
        if child is None:
            child = nn.ModuleDict() if parts[i + 1].isdigit() else _Scope()
            if isinstance(node, nn.ModuleDict):
                node[p] = child
            else:
                node.add_module(p, child)
        node = child
    leaf = _Scope(scope_fn) if scope_fn is not None else full.get_submodule(path)
    if isinstance(node, nn.ModuleDict):
        node[parts[-1]] = leaf
    else:
        node.add_module(parts[-1], leaf)


def _skeleton(full: nn.Module, paths, fn: Callable, params=(), buffers=(), scopes=None) -> _Scope:
    """Root skeleton running fn(root, *inputs); `paths` grafted, `scopes` {path: fn} created as scopes
    (graft a scope before its children), root-level `params` / `buffers` shared by name."""
    root = _Scope(fn)
    for sp, sfn in (scopes or {}).items():
        _graft(root, full, sp, scope_fn=sfn)
    for p in paths:
        _graft(root, full, p)
    for n in params:
        root.register_parameter(n, getattr(full, n))
    for n in buffers:
        root.register_buffer(n, getattr(full, n), persistent=False)
    return root


def _rule_signature(full: nn.Module, path: str, rules) -> tuple:
    """Which rules match which submodule of `path`, expressed relative to `path` (for grouping)."""
    sig = []
    for name, _ in full.named_modules():
        if name == path or name.startswith(path + "."):
            rel = name[len(path):]
            hit = tuple(i for i, rx in enumerate(rules) if re.search(rx, name))
            if hit:
                sig.append((rel, hit))
    return tuple(sig)


def _group(full: nn.Module, paths, key: Callable, rules):
    """[(representative path, [paths])] for paths with equal key(path) and equal rule signature."""
    groups = {}
    for p in paths:
        groups.setdefault((key(p), _rule_signature(full, p, rules)), []).append(p)
    return [(ps[0], ps) for ps in groups.values()]


def _block_out(out):
    return out[0] if isinstance(out, tuple) else out


def _deit(full, rules):
    def pro(r, x):
        if r.input_quant:
            x = r.qact_input(x)
        x = r.patch_embed(x)
        x = torch.cat((r.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = r.qact_embed(x)
        x = x + r.qact_pos(r.pos_embed)
        return r.pos_drop(r.qact1(x))

    def epi(r, x):
        x = r.qact2(r.norm(x, None, None)[:, 0])
        return r.act_out(r.head(r.pre_logits(x)))

    pro_mods = ["patch_embed", "qact_embed", "qact_pos", "qact1", "pos_drop"] + (["qact_input"] if full.input_quant else [])
    prologue = _skeleton(full, pro_mods, pro, params=("cls_token", "pos_embed"))
    prologue.input_quant = full.input_quant
    pieces = [Piece("prologue", prologue, 1, ["prologue"], "")]
    blocks = [f"blocks.{i}" for i in range(len(full.blocks))]
    for rep, covers in _group(full, blocks, lambda p: "block", rules):
        m = _skeleton(full, [rep], lambda r, x, rep=rep: _resolve(r, rep)(x, None))
        pieces.append(Piece(rep, m, len(covers), covers, rep))
    epilogue = _skeleton(full, ["norm", "qact2", "pre_logits", "head", "act_out"], epi)
    return pieces + [Piece("epilogue", epilogue, 1, ["epilogue"], "norm")]


def _swin(full, rules):
    def pro(r, x):
        if r.input_quant:
            x = r.qact_input(x)
        x = r.patch_embed(x)
        if r.absolute_pos_embed is not None:
            x = r.qact1(x + r.absolute_pos_embed)
        return r.pos_drop(x)

    def epi(r, x):
        x = r.qact2(r.norm(x, None, None))
        x = r.qact3(r.avgpool(x.transpose(1, 2)))
        return r.act_out(r.head(torch.flatten(x, 1)))

    pro_mods = ["patch_embed", "pos_drop"] + (["qact_input"] if full.input_quant else []) \
        + (["qact1"] if full.absolute_pos_embed is not None else [])
    prologue = _skeleton(full, pro_mods, pro)
    prologue.input_quant = full.input_quant
    prologue.absolute_pos_embed = full.absolute_pos_embed  # None (ape=False) or the shared Parameter
    pieces = [Piece("prologue", prologue, 1, ["prologue"], "")]
    for s, layer in enumerate(full.layers):
        blocks = [f"layers.{s}.blocks.{j}" for j in range(len(layer.blocks))]
        key = lambda p: (p.split(".blocks.")[0], full.get_submodule(p).shift_size)
        for rep, covers in _group(full, blocks, key, rules):
            m = _skeleton(full, [rep], lambda r, x, rep=rep: _resolve(r, rep)(x, None))
            pieces.append(Piece(rep, m, len(covers), covers, rep))
        if layer.downsample is not None:
            ds = f"layers.{s}.downsample"
            m = _skeleton(full, [ds], lambda r, x, ds=ds: _resolve(r, ds)(x, None))
            pieces.append(Piece(ds, m, 1, [ds], ds))
    epilogue = _skeleton(full, ["norm", "qact2", "avgpool", "qact3", "head", "act_out"], epi)
    return pieces + [Piece("epilogue", epilogue, 1, ["epilogue"], "norm")]


def _whisper_encoder(full, rules):
    """full = whisper_executorch_wrapper.WhisperEncoder (full.encoder is the HF WhisperEncoder)."""
    def front(scope, mel):  # runs inside the scope at path `encoder`, like HF WhisperEncoder.forward
        x = nn.functional.gelu(scope.conv1(mel))
        x = nn.functional.gelu(scope.conv2(x)).permute(0, 2, 1)
        pos = torch.arange(scope.embed_positions.num_embeddings, device=x.device)
        return x + scope.embed_positions(pos)

    prologue = _skeleton(full, ["encoder.conv1", "encoder.conv2", "encoder.embed_positions"],
                         lambda r, mel: r.encoder(mel), scopes={"encoder": front})
    pieces = [Piece("prologue", prologue, 1, ["prologue"], "")]
    layers = [f"encoder.layers.{i}" for i in range(len(full.encoder.layers))]
    for rep, covers in _group(full, layers, lambda p: "layer", rules):
        m = _skeleton(full, [rep], lambda r, h, rep=rep: _block_out(_resolve(r, rep)(h, attention_mask=None, layer_head_mask=None)))
        pieces.append(Piece(rep, m, len(covers), covers, rep))
    epilogue = _skeleton(full, ["encoder.layer_norm"], lambda r, h: r.encoder.layer_norm(h))
    return pieces + [Piece("epilogue", epilogue, 1, ["epilogue"], "encoder.layer_norm")]


def _whisper_decoder(full, rules):
    """full = whisper_executorch_wrapper.WhisperDecoder (full.decoder is the HF WhisperDecoder)."""
    def front(scope, ids):  # token + position embedding, inside the scope at path `decoder`
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = scope.embed_tokens(ids)
        return x + scope.embed_positions(ids, position_ids=pos)

    prologue = _skeleton(full, ["decoder.embed_tokens", "decoder.embed_positions"],
                         lambda r, ids: r.decoder(ids), scopes={"decoder": front})
    pieces = [Piece("prologue", prologue, 1, ["prologue"], "")]
    # The mask a decoder layer receives is whatever the HF decoder derives from the wrapper's causal
    # mask; record it from one full forward so each block piece sees exactly that tensor.
    seen = {}

    def record(mod, args, kwargs):  # must return None (a returned value would replace the inputs)
        seen.setdefault("mask", kwargs.get("attention_mask"))

    h = full.decoder.layers[0].register_forward_pre_hook(record, with_kwargs=True)
    try:
        with torch.no_grad():
            d, n_ctx, dm = full.causal_mask.shape[-1], full.decoder.config.max_source_positions, full.decoder.config.d_model
            full(torch.zeros(1, d, dtype=torch.long), torch.zeros(1, n_ctx, dm))
    finally:
        h.remove()
    layer_mask = seen.get("mask")
    assert layer_mask is not None, "decoder layer got no attention_mask keyword; the HF call convention changed"
    layers = [f"decoder.layers.{i}" for i in range(len(full.decoder.layers))]
    for rep, covers in _group(full, layers, lambda p: "layer", rules):
        fn = lambda r, h, enc, rep=rep: _block_out(_resolve(r, rep)(
            h, attention_mask=r.layer_attention_mask, encoder_hidden_states=enc, use_cache=False))
        m = _skeleton(full, [rep], fn)
        m.register_buffer("layer_attention_mask", layer_mask, persistent=False)
        pieces.append(Piece(rep, m, len(covers), covers, rep, input_kwargs=("encoder_hidden_states",)))
    epilogue = _skeleton(full, ["decoder.layer_norm", "proj_out"], lambda r, h: r.proj_out(r.decoder.layer_norm(h)))
    return pieces + [Piece("epilogue", epilogue, 1, ["epilogue"], "decoder.layer_norm")]


_FAMILIES = {"VisionTransformer": _deit, "SwinTransformer": _swin,
             "WhisperEncoder": _whisper_encoder, "WhisperDecoder": _whisper_decoder}


def make_pieces(full: nn.Module, rules=()) -> list:
    """Pieces of `full` (eval mode, fp32). `rules`: the recipe's precision rules, either the
    `--prec-rules` string ('regex=config[@ops];...') or a list of its items; only the regexes are used.

    LayerNorm replacement (newton_layernorm.swap_layernorms) must run on the full model BEFORE this
    call or on each piece AFTER it, never on the full model after it: swap_layernorms rebinds the
    attribute on the parent, and top-level modules (e.g. the final `norm`) have their own parent in a
    piece, so a later swap on the full model does not reach them. check_layernorms() catches that.
    """
    fam = _FAMILIES.get(type(full).__name__)
    assert fam is not None, f"no slicing for {type(full).__name__}; known: {sorted(_FAMILIES)}"
    if isinstance(rules, str):
        rules = [r for r in rules.split(";") if r]
    return fam(full, [r.split("=", 1)[0] for r in rules])


def check_layernorms(piece_module: nn.Module, replaced: bool) -> None:
    """Every LayerNorm of the piece is replaced (replaced=True, the recipe swaps LayerNorms) or none is.
    Checked per recipe, not per piece: a piece holding a single LayerNorm (the epilogue) cannot tell a
    missed swap from a recipe without one."""
    from core.newton_layernorm import NewtonLayerNorm
    lns = [(n, isinstance(m, NewtonLayerNorm)) for n, m in piece_module.named_modules() if isinstance(m, nn.LayerNorm)]
    wrong = [n for n, is_new in lns if is_new != replaced]
    assert not wrong, f"LayerNorms {'not ' if replaced else ''}replaced in this piece: {wrong}"


def piece_inputs(full: nn.Module, pieces, batches) -> dict:
    """Per piece, the inputs it receives inside the full fp32 model on `batches` (list of input tuples):
    forward pre-hooks on each piece's hook_path. Used as calibration data for the piece."""
    got = {pc.name: [] for pc in pieces}
    hooks = []
    for pc in pieces:
        if not pc.hook_path:
            continue
        def pre(mod, args, kwargs, pc=pc):  # returns None: the inputs are recorded, not replaced
            got[pc.name].append((args[0],) + tuple(kwargs[k] for k in pc.input_kwargs))
        hooks.append(full.get_submodule(pc.hook_path).register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with torch.no_grad():
            for b in batches:
                full(*b)
                for pc in pieces:
                    if not pc.hook_path:
                        got[pc.name].append(b[:1])  # prologue: the model input (decoder: token ids only)
    finally:
        for h in hooks:
            h.remove()
    return got

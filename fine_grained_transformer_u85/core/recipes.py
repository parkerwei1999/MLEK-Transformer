"""Named quantization recipes: one name resolves, per model, to the flags of the accuracy runners and
lower_probe.py (--quant-config, --prec-rules, observer, LayerNorm swap). Per-channel activations are
always rewritten to their lowered form by every runner, so no recipe flag is needed for that.

Names: `<start> + <LayerNorm variant> + <extras>`.
  Default A8W8 / Default A16W8   the stock Arm quantizer, no rules (Default LN)
  A8W8 + LN-32                   Default A8W8 with LN-32 only (vision ladder rung)
  Ours A8W8                      Default A8W8 + LN-32 + PCS (per-channel symmetric int8 residual, evaluated and lowered in its rewritten form)
  A8W8 + LN-32 + S2-A16          Default A8W8 + LN-32 + all of layers.2 (stage 3) in INT16 (Swin, no PCS)
  A8W8 + LN-32 + S2-B5-FC2-A16   A8W8 + LN-32 with the fc2 output of block layers.2.blocks.5 in INT16 (Swin-T)
  Ours A8W8 + S2-B5-FC2-A16      Ours A8W8 (LN-32 + PCS) with that same fc2 output in INT16 (Swin-T)
  Ours A8W8 + S2-A16             Ours A8W8 (LN-32 + PCS) with all of layers.2 (stage 3) in INT16, its residual adds
                                 and LayerNorms included (Swin; tables: "PCS + S2-A16")
  Ours A8W8 + S2-B5-A16          Ours A8W8 with the submodules of block layers.2.blocks.5 in INT16; the block's own
                                 residual adds stay PCS (Swin-T)
  (S<i>[-B<j>] = layers.<i>[.blocks.<j>], 0-based like the module paths)
  A16W8 + LN-32                  Default A16W8 with LN-32 only (Whisper ladder rung below GA-LN)
  Ours A16W8                     Default A16W8 + GA-LN (Newton-1, or dual k=8 from Whisper-medium up)
  Ours A16W8 + INT8-Linear       Ours A16W8 with every Linear except fc2 (attention projections, fc1, logits) plus the
                                 conv front end and its GELU in INT8 (Whisper); the attention matmul outputs,
                                 softmax and fc2 stay INT16 (Q, K, V come from INT8 projections)
A (model, recipe) pair without a definition here is an error, not a fallback.
"""
import argparse
from dataclasses import dataclass, replace

# LayerNorm with int32 internals and int16 in / out (LN-32); vision models name them norm / norm1 / norm2
_VISION_LN32 = r"norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8"
_WHISPER_LN32 = r"layer_norm$=a32w8@sub,mul,sum.dim_IntList,add;layer_norm$=a16w8"
# GA-LN dual-range rsqrt: fine table and mask observed with k x median (k = ln_dual_k)
_WHISPER_DUAL = (r"layer_norm\.fine$=a32w8@mul;layer_norm\.fine$=a16w8e16:kmedian;layer_norm\.mask$=a16w8e16:kmedian;"
                 + _WHISPER_LN32)
_VISION_PCS = r"blocks\.\d+$=a16inpc8symout@add"  # residual adds: per-channel symmetric int8 output
_SWIN_S2_A16 = r"layers\.2\.=a16w8"
_SWIN_T_S2_B5_FC2 = r"layers\.2\.blocks\.5\.mlp\.fc2$=a8in16out"  # Swin-T stage 3 has 6 blocks; int8 in, int16 out
_SWIN_T_S2_B5 = r"layers\.2\.blocks\.5\.=a16w8"  # trailing dot: the block's submodules, not its residual adds
_WHISPER_INT8 = r"conv[12]$=a8w8;encoder$=a8w8@gelu;(q_proj|k_proj|v_proj|out_proj)$=a8w8;proj_out$=a8w8;fc1$=a8w8"

DEIT = ("deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224")
SWIN = ("swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224")
WHISPER = ("whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3")
WHISPER_DUAL_SIZES = ("whisper-medium", "whisper-large-v3")  # Newton steps are not enough for the larger models, so they use the dual-range table


@dataclass(frozen=True)
class Recipe:
    quant_config: str
    prec_rules: str
    act_observer: str
    ln_newton_steps: int | None
    ln_dual_k: float | None

    def lower_probe_args(self) -> list:
        """The lower_probe.py flags of this recipe (lower_probe always calibrates with its histogram observer)."""
        a = ["--quant-config", self.quant_config]
        if self.prec_rules:
            a += ["--prec-rules", self.prec_rules]
        if self.ln_newton_steps is not None:
            a += ["--ln-newton-steps", str(self.ln_newton_steps)]
        if self.ln_dual_k is not None:
            a += ["--ln-dual-k", str(self.ln_dual_k)]
        return a


    def runner_args(self, runner: str) -> list:
        """Flags for `runner`: "lower" (lower_probe.py), "vision" (vision_eval.py) or "whisper" (whisper_eval.py).
        Per-channel activations (PCS) are rewritten to their lowered form by every runner, so accuracy and
        latency describe the same graph."""
        if runner == "lower":
            return self.lower_probe_args()
        a = (["--quant-configs", self.quant_config, "--act-observers", self.act_observer]
             if runner == "vision" else ["--quant-config", self.quant_config, "--act-observer", self.act_observer])
        if self.prec_rules:
            a += ["--prec-rules", self.prec_rules]
        if self.ln_newton_steps is not None:
            a += ["--ln-newton-steps", str(self.ln_newton_steps)]
        if self.ln_dual_k is not None:
            a += ["--ln-dual-k", str(self.ln_dual_k)]
        return a


# flags a recipe sets, per runner; giving one of them together with --recipe is an error
_OWNED = {
    "lower": ("--quant-config", "--prec-rules", "--ln-newton-steps", "--ln-dual-k"),
    "vision": ("--quant-configs", "--act-observers", "--prec-rules", "--ln-newton-steps", "--ln-dual-k"),
    "whisper": ("--quant-config", "--act-observer", "--prec-rules", "--ln-newton-steps", "--ln-dual-k"),  # --ln-dual-ablate is a debug flag layered on top
}


def parse_args(parser: argparse.ArgumentParser, runner: str, model_of) -> argparse.Namespace:
    """parser.parse_args(), with `--recipe NAME` expanded into the recipe's flags for `runner`.
    `model_of(args)` gives the recipe's model key (FQ-ViT name or whisper-<size>)."""
    import sys
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.recipe is None:
        return args
    given = [f for f in _OWNED[runner] if any(a == f or a.startswith(f + "=") for a in argv)]
    assert not given, f"--recipe sets these flags itself: {given}"
    args = parser.parse_args(argv + resolve(args.recipe, model_of(args)).runner_args(runner))
    print(f"recipe {args.recipe!r} ({model_of(args)}): {' '.join(resolve(args.recipe, model_of(args)).runner_args(runner))}", flush=True)
    return args


_DEFAULT = Recipe("a8w8", "", "histogram", None, None)


def _whisper_ga_ln(model: str, base: Recipe, extra_rules: str) -> Recipe:
    if model in WHISPER_DUAL_SIZES:
        return replace(base, prec_rules=";".join(r for r in (extra_rules, _WHISPER_DUAL) if r), ln_newton_steps=0, ln_dual_k=8.0)
    return replace(base, prec_rules=";".join(r for r in (extra_rules, _WHISPER_LN32) if r), ln_newton_steps=1)


def resolve(name: str, model: str) -> Recipe:
    """`model`: an FQ-ViT model name or whisper-<size> (encoder and decoder share the recipe)."""
    known = DEIT + SWIN + WHISPER
    assert model in known, f"unknown model {model!r}; known: {known}"
    whisper = model in WHISPER
    obs = "minmax" if whisper else "histogram"  # Whisper recipes calibrate with MinMax
    if name == "Default A8W8":
        return replace(_DEFAULT, act_observer=obs)
    if name == "Default A16W8":
        return replace(_DEFAULT, quant_config="a16w8", act_observer=obs)
    if name == "A8W8 + LN-32" and not whisper:  # rung between Default A8W8 and Ours A8W8, same flags as Ours minus PCS
        return replace(_DEFAULT, prec_rules=_VISION_LN32)
    if name == "Ours A8W8" and not whisper:
        return replace(_DEFAULT, prec_rules=f"{_VISION_PCS};{_VISION_LN32}")
    if name == "A8W8 + LN-32 + S2-A16" and model in SWIN:
        return replace(_DEFAULT, prec_rules=f"{_SWIN_S2_A16};{_VISION_LN32}")
    if name == "A8W8 + LN-32 + S2-B5-FC2-A16" and model == "swin_tiny_patch4_window7_224":
        return replace(_DEFAULT, prec_rules=f"{_SWIN_T_S2_B5_FC2};{_VISION_LN32}")
    if name == "Ours A8W8 + S2-B5-FC2-A16" and model == "swin_tiny_patch4_window7_224":
        return replace(_DEFAULT, prec_rules=f"{_SWIN_T_S2_B5_FC2};{_VISION_PCS};{_VISION_LN32}")
    if name == "Ours A8W8 + S2-A16" and model in SWIN:  # rules apply first match: stage 3 takes a16w8 before PCS / LN-32
        return replace(_DEFAULT, prec_rules=f"{_SWIN_S2_A16};{_VISION_PCS};{_VISION_LN32}")
    if name == "Ours A8W8 + S2-B5-A16" and model == "swin_tiny_patch4_window7_224":
        return replace(_DEFAULT, prec_rules=f"{_SWIN_T_S2_B5};{_VISION_PCS};{_VISION_LN32}")
    if name == "A16W8 + LN-32" and whisper:
        return replace(_DEFAULT, quant_config="a16w8", act_observer="minmax", prec_rules=_WHISPER_LN32)
    if name == "Ours A16W8" and whisper:
        return _whisper_ga_ln(model, replace(_DEFAULT, quant_config="a16w8", act_observer="minmax"), "")
    if name == "Ours A16W8 + INT8-Linear" and whisper:
        return _whisper_ga_ln(model, replace(_DEFAULT, quant_config="a16w8", act_observer="minmax"), _WHISPER_INT8)
    raise AssertionError(f"recipe {name!r} is not defined for {model}")


NAMES = ("Default A8W8", "Default A16W8", "A8W8 + LN-32", "Ours A8W8", "A8W8 + LN-32 + S2-A16", "A8W8 + LN-32 + S2-B5-FC2-A16", "Ours A8W8 + S2-B5-FC2-A16", "Ours A8W8 + S2-A16", "Ours A8W8 + S2-B5-A16", "A16W8 + LN-32", "Ours A16W8", "Ours A16W8 + INT8-Linear")

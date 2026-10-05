"""Every LayerNorm of the transformed graph gets each role the expected number of times (the Arm decomposition
computes the mean twice: once for x - mean, once inside var), for the vision models and the Whisper encoder / decoder.

    python3 tests/ln_role_check.py
"""
import copy, sys
from collections import Counter
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "helper"))
import core.quant as cq  # noqa: E402
from core.mixed_precision_quantizer import LN_ROLES, _deepest_fqn, _ln_role  # noqa: E402

EXPECTED = {"mean": 4, "center": 2, "square": 1, "sumsq": 1, "var": 1, "var_eps": 1, "rsqrt": 1, "xhat": 1, "gamma": 1, "beta": 1}


def graphs():
    import fqvit_models
    for name in ("deit_tiny_patch16_224", "swin_tiny_patch4_window7_224"):
        yield name, fqvit_models.build_model(name).eval(), (torch.randn(2, 3, 224, 224),)
    wrapper = cq.load_module_from_path("whisper_executorch_wrapper", ROOT / "helper/whisper_executorch_wrapper.py")
    for part in (wrapper.WhisperPart.ENCODER, wrapper.WhisperPart.DECODER):
        m, ex = wrapper._build("openai/whisper-tiny", part, 128)
        yield f"whisper-tiny {part.name.lower()}", m, ex


def main():
    q = cq.EthosUQuantizer(cq.quantizer_compile_spec()); q.set_global(cq.QuantConfig("a8w8").build(cq.ActObserver.MINMAX, None))
    ok = True
    for name, m, ex in graphs():
        lns = {n for n, mod in m.named_modules() if isinstance(mod, torch.nn.LayerNorm)}
        g = q.transform_for_annotation(torch.export.export(copy.deepcopy(m), ex, strict=True).module(check_guards=False))
        per = {ln: Counter() for ln in lns}
        for n in g.graph.nodes:
            f = _deepest_fqn(n)
            if f in per and (r := _ln_role(n)):
                per[f][r] += 1
        bad = {ln: dict(c) for ln, c in per.items() if dict(c) != EXPECTED}
        ok &= not bad
        print(f"{'OK  ' if not bad else 'DIFF'} {name}: {len(lns)} LayerNorms" + (f"; e.g. {next(iter(bad.items()))}" if bad else ""))
    print("ALL ROLES FOUND" if ok else "ROLE MISMATCH")


if __name__ == "__main__":
    main()

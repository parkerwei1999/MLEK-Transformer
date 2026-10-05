"""FQ-ViT's own fully quantized PTQ on this project's checkpoints (the "FQ-ViT native" baseline).

Runs the upstream script 3rdparty/FQ-ViT/test_quant.py (git submodule, megvii-research/FQ-ViT @ 5daf591) unmodified,
through runpy. The only change is the one helper/fqvit_models makes for the fp32 path: Swin models load the
ImageNet-22k -> 1k fine-tuned checkpoints (SWIN_CHECKPOINTS) after construction instead of upstream's ImageNet-1k ones.
DeiT is untouched. Calibration sampling, the quantized model and the evaluation are upstream's.

usage (arguments exactly as upstream test_quant.py):
  python3 helper/fqvit_native.py swin_tiny /home/shared/ImageNet --quant --ptf --lis --quant-method minmax
  python3 helper/fqvit_native.py deit_tiny /home/shared/ImageNet                       (fp32 reference)
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fqvit_models import FQVIT_DIR, load_swin_checkpoint  # noqa: E402  (also puts the FQ-ViT checkout on sys.path)
import models  # noqa: E402  the upstream package; test_quant.py's `from models import *` reads its namespace


def _with_our_checkpoint(name: str):
    upstream = getattr(models, name)

    def build(pretrained=False, **kwargs):
        model = upstream(pretrained=False, **kwargs)
        if pretrained:
            load_swin_checkpoint(model, name)
        return model

    build.__name__ = upstream.__name__  # test_quant.py prints it
    return build


for _name in ("swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224"):
    setattr(models, _name, _with_our_checkpoint(_name))

if __name__ == "__main__":
    sys.argv = [str(FQVIT_DIR / "test_quant.py")] + sys.argv[1:]
    runpy.run_path(str(FQVIT_DIR / "test_quant.py"), run_name="__main__")

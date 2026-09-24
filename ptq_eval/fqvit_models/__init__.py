# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# Vendored from the int-fq-vit fork of FQ-ViT (Apache-2.0) so the PTQ harness imports the exact
# model definitions behind the FQ-ViT receipts without a submodule. See VERSION.md for the pinned
# commit and per-file digests; the only edit is config.py's import made package-relative.
from .ptq import *
from .swin_quant import *
from .vit_quant import *
from .config import Config  # noqa: E402

SOURCE_COMMIT = "0efb5563f4f8fe388585e43ff0ffff6d507fd6a0"


def build_model(model_name: str, pretrained: bool = True):
    """The fp32 network exactly as FQ-ViT's test_quant.py builds it (quant / calibrate off)."""
    import sys
    constructor = getattr(sys.modules[__name__], model_name)
    return constructor(pretrained=pretrained, quant=False, calibrate=False, cfg=Config()).eval()

"""Adapter over the official FQ-ViT (git submodule `3rdparty/FQ-ViT`, megvii-research/FQ-ViT, Apache-2.0)
for the fp32 path this project uses (quant=False, calibrate=False): `build_model()` plus the ImageNet
loader in `imagenet_data`.

One deliberate deviation from upstream: Swin uses the ImageNet-22k -> 1k fine-tuned checkpoints
(`SWIN_CHECKPOINTS`), loaded after construction; upstream's pretrained=True loads the ImageNet-1k ones.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

FQVIT_DIR = Path(__file__).resolve().parents[2] / "3rdparty" / "FQ-ViT"
UPSTREAM_COMMIT = "5daf5915c715dc21d6afcd64e661652030e640ab"
SWIN_CHECKPOINTS = {  # size -> (url, sha256); the file names carry no hash, so torch.hub's check_hash cannot check them
    "tiny": ("https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth",
             "9b6405a1ce3c3258eb3f735b4aa1142dd0c052abcf3a092a0f4d51c2dd31ca03"),
    "small": ("https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth",
              "4f7acd1b78afc4d9d30bfaf1ba39f9e0260b5ee10d6539c02a2df3049c086631"),
    "base": ("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth",
             "a5348473e69277d69db915362bef760b5fd04362057bd59c7fe064d431532eab"),
}

# Upstream imports its package as top-level `models` (config.py does `from models import ...`).
if str(FQVIT_DIR) not in sys.path:
    sys.path.insert(0, str(FQVIT_DIR))
import models as _models  # noqa: E402

# config.py is loaded under a private name so it cannot shadow or be shadowed by another `config` module.
_spec = importlib.util.spec_from_file_location("_fqvit_config", FQVIT_DIR / "config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
Config = _config.Config


def build_model(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    """The fp32 network as FQ-ViT's test_quant.py builds it (quant / calibrate off), Swin weights overridden."""
    constructor = getattr(_models, model_name)
    swin = model_name.startswith("swin_")
    model = constructor(pretrained=pretrained and not swin, quant=False, calibrate=False, cfg=Config())
    if swin and pretrained:
        load_swin_checkpoint(model, model_name)
    return model.eval()


def load_swin_checkpoint(model: torch.nn.Module, model_name: str) -> None:
    """Load the ImageNet-22k -> 1k fine-tuned weights of SWIN_CHECKPOINTS into a constructed Swin model (sha256-checked)."""
    url, sha256 = SWIN_CHECKPOINTS[model_name.split("_")[1]]
    torch.hub.load_state_dict_from_url(url, map_location="cpu")  # download into the hub cache
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == sha256, f"{path}: sha256 {digest}, expected {sha256}"
    model.load_state_dict(torch.load(path, map_location="cpu")["model"], strict=True)

"""Shared ImageNet data loading — SSoT for calibration/eval loaders and the FQ-ViT
calibration-set-size constant. Extracted from probe_fq_apo_cossim.py; that module
re-exports these names so existing `from probe_fq_apo_cossim import ...` callers keep
working."""
import math
import os

import torch

# DataLoader workers + default file_descriptor sharing exhausts FDs at large n_cal
# ("Too many open files"); file_system strategy is the canonical fix (mirrors
# acc_apo_fq_agree.py).
torch.multiprocessing.set_sharing_strategy("file_system")

# FQ-ViT official calibration set size = calib_iter(10) x calib_batchsize(100) = 1000
# images (test_quant.py:33,36). This is the FQ-ViT-specific count needed to reproduce
# test_quant.py accuracy — NOT novella's 1024 (tvm/.../novella/utils.py uses 1024). All
# acc/probe scripts default to this so the n_cal=256 calibration-undersize trap cannot
# silently recur. SSoT — do not re-introduce a local 256/32 default in any script.
NUM_IMAGE_CAL_FQ_VIT = 1000
CAL_BATCH_SIZE_FQ_VIT = 100


def imagenet_preprocess(model_name=None, crop_pct=None):
    """Return the official FQ-ViT preprocessing tuple for one model family."""
    family = model_name.split("_", 1)[0] if model_name else None
    if family == "vit":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        default_crop_pct = 0.9
    elif family == "deit":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        default_crop_pct = 0.875
    elif family in ("swin", None):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        default_crop_pct = 0.9
    else:
        raise ValueError(f"unsupported ImageNet model family: {family!r}")
    return mean, std, default_crop_pct if crop_pct is None else crop_pct


def _imagenet_transform(model_name=None, crop_pct=None, *,
                        transforms_module, bicubic):
    """Build the shared FQ-ViT ImageNet transform and resolved crop percent."""
    mean, std, crop_pct = imagenet_preprocess(model_name, crop_pct)
    transform = transforms_module.Compose([
        transforms_module.Resize(
            int(math.floor(224 / crop_pct)), interpolation=bicubic),
        transforms_module.CenterCrop(224),
        transforms_module.ToTensor(),
        transforms_module.Normalize(mean, std),
    ])
    return transform, crop_pct


class _DL:
    """Minimal (data,label) loader over a materialized image list (apo calibrate)."""
    def __init__(self, imgs):
        self.imgs = imgs
    def __iter__(self):
        for d in self.imgs:
            yield d, torch.zeros(d.shape[0], dtype=torch.long)


def _take_image_batches(loader, n):
    """Materialize exactly ``n`` images while preserving DataLoader batches."""
    out = []
    loaded = 0
    for x, _ in loader:
        if loaded >= n:
            break
        take = min(x.shape[0], n - loaded)
        out.append(x[:take])
        loaded += take
    return out


def _load_imgs(n, crop_pct, split, shuffle, seed=0, *, data_dir,
               model_name=None, batch_size=1):
    from PIL import Image  # local import: keep --help usable without torchvision
    from torchvision import datasets, transforms
    d = os.path.join(data_dir, split)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"ImageNet split not found: {d}")
    tf, crop_pct = _imagenet_transform(
        model_name, crop_pct, transforms_module=transforms,
        bicubic=Image.BICUBIC)
    ds = datasets.ImageFolder(d, tf)
    ld = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=4,
        drop_last=(split == "train"),
        generator=torch.Generator().manual_seed(seed))
    print(f"[data] REAL ImageNet {d} crop={crop_pct} n={n} "
          f"batch_size={batch_size} shuffle={shuffle}")
    return _take_image_batches(ld, n)


def val_loader(crop_pct, bs, *, deterministic_order, pin_memory=False, num_workers=8,
               data_dir, model_name=None):
    """Shared ImageNet val DataLoader — SSoT for the acc_* / run_* eval harnesses.

    deterministic_order=True  -> shuffle=False: every model sees the IDENTICAL image
                                 sequence (paired agreement / bit-match comparisons that
                                 align predictions image-by-image).
    deterministic_order=False -> shuffle=True (seed 0): a representative random subset
                                 when only the first n_val images are consumed (smoke /
                                 <50k runs), NOT the first few (alphabetical) classes.
    The shuffle flag is intentionally divergent across callers (do NOT collapse it); this
    SSoT shares only the transform + dataset wiring."""
    from PIL import Image  # local import: keep --help usable without torchvision
    from torchvision import datasets, transforms
    d = os.path.join(data_dir, "val")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"ImageNet val not found: {d}")
    tf, _crop_pct = _imagenet_transform(
        model_name, crop_pct, transforms_module=transforms,
        bicubic=Image.BICUBIC)
    ds = datasets.ImageFolder(d, tf)
    if deterministic_order:
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                           num_workers=num_workers, pin_memory=pin_memory)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory,
                                       generator=torch.Generator().manual_seed(0))

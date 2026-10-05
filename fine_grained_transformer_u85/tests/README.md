# tests

```bash
python3 -m pytest -q tests/test_slices.py   # core.slices: pieces compose to the full model, grouping, LayerNorm swap order (CPU)
python3 tests/ln_unit_check.py              # one fake-quantized LayerNorm (Default / Newton / dual) vs fp32, per-token error
python3 tests/pcs_decompose_check.py           # PCS rewrite: converted DeiT-T outputs with vs without the rewrite
python3 tests/pcs_decompose_site_check.py      # PCS rewrite at one site, recomputed by hand
```

Run from `fine_grained_transformer_u85/` with the `PYTHONPATH` of the top-level README. The `*_check.py` scripts
print reports rather than assert and read ImageNet from `/home/shared/ImageNet`.

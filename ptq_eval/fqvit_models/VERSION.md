# Vendored FQ-ViT model package

Source: int-fq-vit fork of FQ-ViT (upstream https://github.com/megvii-research/FQ-ViT, Apache-2.0), commit `0efb5563f4f8fe388585e43ff0ffff6d507fd6a0`, copied 2026-09-24.
Files: `models/` -> this package root, `config.py`, `experiments/imagenet_data.py`. Only edit: `config.py` line 1 `from models import BIT_TYPE_DICT` -> `from .ptq.bit_type import BIT_TYPE_DICT`; `__init__.py` gained `Config`, `SOURCE_COMMIT`, `build_model`.
The fork's changes vs upstream are apo-int hooks on the quantized path (`quant=True`); the fp32 path used here (`quant=False`) is upstream's.

sha256 of the source files at that commit:

```
df7898755d467d4ac979d975062269dc3ae67488ef7b99f5c61c3a3000264d23  config.py
26e096af8177513e9c04844d6f7322bf30b8228718966b96b83b5a4de2696c2c  experiments/imagenet_data.py
d56839395b7e501622d926b1495a8d2ab1186fcd8f63b0d51de1e3d8df6e5e89  models/__init__.py
92f220de455c3c464bad4227bcc54224b546818bd1a75d424ec40e8a7ff65231  models/layers_quant.py
87e4324091e116733008cfefdb583de7774ae16326f3881e74916817cec6580a  models/ptq/__init__.py
1ea9804e01fb3edad923f6c59049cb0e0dd8078c3ca61a3105a48a8e4e1bee87  models/ptq/bit_type.py
e8d641bf61789aa7a7682a0174c919e0238d1d30408b40568115625155a41460  models/ptq/layers.py
0bb0fa9e0edc0c2431deb1e40f13e2fa43420635db594972b80f808aa6480398  models/ptq/observer/__init__.py
bc9782248efa49fc12ad37f5f4b34e81a9fe2ee10678603bad9688ff39b6f64c  models/ptq/observer/base.py
610bab766be5b19837ac4ece5ca941d3f376af0c20c32f5640f9f7ab21e4c722  models/ptq/observer/build.py
2d6a67c793baf7a69f7cad39b38ab7009571c002c3bf81158516844c7861e65a  models/ptq/observer/ema.py
ee09ade2f2ef6cb4b0423b25273080ba0c4238fde6eacef0d6d9fd967f029e0a  models/ptq/observer/minmax.py
dfe287f2f966fd27f4e4a53b7de30f7025547586755cd3c0ac37718084700015  models/ptq/observer/omse.py
0765d74294c4dc36e718c4021668e319a9c38a67063abec2fe95530f62f31610  models/ptq/observer/percentile.py
d23193ceaf09e232ba46330a9b326230c7c8181702eb993fae0c69c5a1b07b1a  models/ptq/observer/ptf.py
18ce52cb7bd6beb7695e3e06e124691a1233c8db0103b7d34547b86e2670e28a  models/ptq/observer/utils.py
305c47c530435e46f6055cb1e39e382fe04fd96046d39077b41d8382d6626d7e  models/ptq/quantizer/__init__.py
9f73ea1e0a6780ad3128710967458918bc743e80fab9c60e7ca19f071405debe  models/ptq/quantizer/base.py
f774aaa1d726a4e4fed551453e775fadfbff1a84a05510de2feb3d2452a11e9c  models/ptq/quantizer/build.py
fc1fa911b911d7ce382f14ba86692cf10c1b830c4cd8ad2aa8328015df7896a5  models/ptq/quantizer/log2.py
ae0e2310ea623d7aa43ca145045378d2ec38f31ea4f06b8b47d43b8d5080bf29  models/ptq/quantizer/uniform.py
e08062b82979cba9e966bcbb017950d63daab895cdceeceb27414958364ced56  models/swin_quant.py
b41e66058991d74b939cc7d79eba55762241df6e85faa7833d61ccbb050d1e99  models/utils.py
ca5816aef50e8b4ba97848c11d62b5b7db11b5f5c15885da7ee30c7429f8c9be  models/vit_quant.py
```

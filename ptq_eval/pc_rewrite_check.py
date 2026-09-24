"""Numerical check of pc_rewrite: converted DeiT-T graph (a16inpc8symout residual, LN-i32) with and without the rewrite."""
import copy, functools, sys, torch
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe, mask_aware_quantizer as maq, fqvit_models
from fqvit_models import imagenet_data as data
from pc_rewrite import rewrite_per_channel_activations

model = fqvit_models.build_model("deit_tiny_patch16_224")
cs = pe.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
rules = []
for item in 'blocks\\.\\d+$=a16inpc8symout@add;norm\\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\\d?$=a16w8'.split(";"):
    rx, cfg = item.rsplit("=", 1); cfg, _, ops = cfg.partition("@")
    rules.append((rx, pe.QuantConfig(cfg).build(pe.ActObserver.HISTOGRAM), set(ops.split(",")) if ops else None))
prepared = pe.prepare_on_cpu(model, (torch.randn(10, 3, 224, 224),), cs, pe.QuantConfig.A8W8, pe.ActObserver.HISTOGRAM, [],
                             quantizer_cls=functools.partial(maq.MixedPrecisionQuantizer, rules=rules))
cal = list(data._load_imgs(30, None, "train", shuffle=True, seed=0, data_dir="/home/shared/ImageNet", model_name="deit_tiny_patch16_224", batch_size=10))
with torch.no_grad():
    for b in cal[:2]:
        prepared(b)
gm = convert_pt2e(prepared)
with torch.no_grad():
    ref = gm(cal[2])
for bits in (8, 12, 16):
    gm2 = copy.deepcopy(gm)
    n = rewrite_per_channel_activations(gm2, x_bits_below=bits)
    with torch.no_grad():
        new = gm2(cal[2])
    diff = (ref - new).abs()
    print(f"x_bits_below={bits}: rewritten sites={n}  logits |ref| max={ref.abs().max():.3f}  max|diff|={diff.max():.4g}  mean|diff|={diff.mean():.4g}  "
          f"argmax agree={(ref.argmax(1) == new.argmax(1)).float().mean():.2f}")
kinds = {}
for node in gm2.graph.nodes:
    if node.op == "call_function":
        k = str(node.target).replace("torch.ops.", ""); kinds[k] = kinds.get(k, 0) + 1
print("per-channel act Q left:", sum(1 for nd in gm2.graph.nodes if nd.op == "call_function" and "quantize_per_channel" in str(nd.target) and "dequantize" not in str(nd.target) and nd.args[0].op != "get_attr"))
print("mul.Tensor nodes:", kinds.get("aten.mul.Tensor", 0), "| int32 Q nodes:", sum(1 for nd in gm2.graph.nodes if nd.op == "call_function" and nd.target is torch.ops.quantized_decomposed.quantize_per_tensor.default and nd.args[5] is torch.int32))

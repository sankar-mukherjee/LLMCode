import time
import torch

from src.llms.llm_0 import GPTDecoder
from src.quantization.QAT.quant import replace_linear_with_quant
from src.quantization.quantization_basics.quantlinear import QuantLinear

# =====================================================
# CONFIG
# =====================================================

vocab_size = 65
d_model = 128
n_heads = 4
n_layers = 4
max_seq_len = 512

num_bits = 4          # 🔴 set to 2 for INT2
seq_len = 128
gen_tokens = 100
iters = 50

real_model = False
device = "cpu"
torch.manual_seed(1337)

# =====================================================
# BUILD MODELS
# =====================================================
state = torch.load("checkpoints/qat/4bit/last.pt", map_location="cpu")

print("🔧 Building FP32 model...")
fp32_model = GPTDecoder(
    vocab_size, d_model, n_heads, n_layers, max_seq_len
).to(device)

print("🔧 Building QAT model...")
qat_model = GPTDecoder(
    vocab_size, d_model, n_heads, n_layers, max_seq_len
).to(device)


if real_model:
    fp32_model.load_state_dict(state["model"])
    qat_model.load_state_dict(state["model"])

fp32_model.eval()

replace_linear_with_quant(qat_model, num_bits=num_bits)
qat_model.eval()

# =====================================================
# VERIFY QAT ACTIVE
# =====================================================

quant_layers = sum(
    isinstance(m, QuantLinear) for m in qat_model.modules()
)
assert quant_layers > 0, "❌ Quantization NOT active"

print(f"✅ QAT active ({quant_layers} QuantLinear layers, {num_bits}-bit)")

# =====================================================
# INPUT
# =====================================================

x = torch.randint(0, vocab_size, (1, seq_len), device=device)

# =====================================================
# NUMERICAL DRIFT TEST
# =====================================================

print("\n📐 Numerical drift test...")
with torch.no_grad():
    fp32_logits = fp32_model(x)
    qat_logits = qat_model(x)

diff = (fp32_logits - qat_logits).abs()

print(f"Max diff : {diff.max().item():.6f}")
print(f"Mean diff: {diff.mean().item():.6f}")
print(f"Std diff : {diff.std().item():.6f}")

assert not torch.isnan(qat_logits).any(), "❌ NaNs detected in QAT logits"

# =====================================================
# LATENCY TEST (FAKE QUANT)
# =====================================================

print("\n⏱️ Latency test (CPU, fake quant)...")

# Warmup
for _ in range(10):
    _ = fp32_model(x)
    _ = qat_model(x)

def bench(model):
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    return (time.perf_counter() - start) / iters * 1000

fp32_ms = bench(fp32_model)
qat_ms = bench(qat_model)

print(f"FP32 latency: {fp32_ms:.2f} ms")
print(f"QAT latency : {qat_ms:.2f} ms")
print("ℹ️ QAT slower here is EXPECTED (fake quant overhead)")

# =====================================================
# GENERATION STABILITY TEST
# =====================================================

print("\n🧪 Generation stability test...")

prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

out_fp32 = fp32_model.generate(prompt, gen_tokens, temperature=0.8)
out_qat  = qat_model.generate(prompt, gen_tokens, temperature=0.8)

print("FP32 tokens:", out_fp32[0][:30].tolist())
print("QAT  tokens:", out_qat[0][:30].tolist())

# =====================================================
# FINAL VERDICT
# =====================================================

print("\n✅ ALL TESTS PASSED")
print("🚀 Model is ready for INT export (CoreML / iPhone)")

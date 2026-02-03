import torch
import torch.nn.functional as F
import os
from quant import replace_linear_with_quant

from src.llms.llm_0 import GPTDecoder

torch.manual_seed(1337)

# ---------------------------------------------------
# Hyperparameters
# ---------------------------------------------------

batch_size = 32
block_size = 128
max_iters = 2000
eval_interval = 500

d_model = 128
n_heads = 4
n_layers = 4
max_seq_len = 512

# qunatization
quant_per_channel = False
num_bits = 4          # 4 = INT4, 8 = INT8
qat_start_step = 200  # warmup

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "input.txt")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join([itos[i] for i in t])

data = encode(text)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x, y

# ---------------------------------------------------
# Model
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTDecoder(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------------------------------------
# Checkpointing
# ---------------------------------------------------

ckpt_dir = f"checkpoints/qat/{num_bits}bit"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "last.pt")
save_interval = 500

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
start_step = 0
qat_enabled = False

if os.path.exists(ckpt_path):
    print(f"📦 Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    qat_enabled = ckpt["qat_enabled"]

    if qat_enabled:
        replace_linear_with_quant(model, num_bits=num_bits, per_channel=quant_per_channel)
        print("🔥 QAT restored from checkpoint")
        for m in model.modules():
            if "QuantLinear" in str(type(m)):
                print("✅ Quantized layer found")
                break



for step in range(start_step, max_iters):
    model.train()

    if (not qat_enabled) and step == qat_start_step:
        replace_linear_with_quant(model, num_bits=num_bits, per_channel=quant_per_channel)
        qat_enabled = True
        print(f"🔥 QAT ENABLED ({num_bits}-bit)")
        for m in model.modules():
            if "QuantLinear" in str(type(m)):
                print("✅ Quantized layer found")
                break


    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch("val")
            xv, yv = xv.to(device), yv.to(device)
            val_loss = F.cross_entropy(
                model(xv).view(-1, vocab_size),
                yv.view(-1)
            )

        print(
            f"step {step} | "
            f"train {loss.item():.3f} | "
            f"val {val_loss.item():.3f} | "
            f"qat={qat_enabled}"
        )

    if step % save_interval == 0:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "qat_enabled": qat_enabled,
                "num_bits": num_bits,
            },
            ckpt_path,
        )


# ---------------------------------------------------
# Text Generation
# ---------------------------------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=400, temperature=0.8)
print(decode(out[0].tolist()))

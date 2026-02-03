import torch

from src.llms.llm_0 import GPTDecoder

import torch.nn.functional as F
import os
torch.manual_seed(1337)


# hyperparameters
batch_size = 32
block_size = 128
max_iters = 2000
eval_interval = 500
save_interval = 1000

d_model=128
n_heads=4
n_layers=4
max_seq_len=512

#1. Build a Minimal Character Dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('vocab_size: ', vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join([itos[i] for i in t])

data = encode(text)

# train / val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#2. BATCHING
def get_batch(split, batch_size, block_size):
    data_src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x, y


#3. INITIALIZE THE MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    max_seq_len=max_seq_len,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# 4. TRAINING LOOP
ckpt_dir = "checkpoints/normal"
ckpt_path = os.path.join(ckpt_dir, "last.pt")
os.makedirs(ckpt_dir, exist_ok=True)

start_step = 0

if os.path.exists(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1

for step in range(start_step, max_iters):
    model.train()

    x, y = get_batch("train", batch_size, block_size)
    x, y = x.to(device), y.to(device)

    logits = model(x)                      # (B, T, vocab)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch("val", batch_size, block_size)
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_logits = model(x_val)
            val_loss = F.cross_entropy(
                val_logits.view(-1, vocab_size),
                y_val.view(-1)
            )

        print(
            f"step {step} | "
            f"train loss {loss.item():.3f} | "
            f"val loss {val_loss.item():.3f}"
        )

        # ---- SAVE CHECKPOINT ----
        if step % save_interval == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                },
                ckpt_path,
            )


# 5. GENERATE TEXT
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8)

print('---------------------------------------------------------------')
print(decode(generated[0].tolist()))
print('---------------------------------------------------------------')

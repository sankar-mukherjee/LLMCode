import os
import time
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.llms.gpt_basic import GPTDecoder


@dataclass(frozen=True)
class TrainConfig:
    # Data / batching
    data_path: str = "data/input.txt"
    batch_size: int = 32
    block_size: int = 128

    # Training
    max_iters: int = 5000
    eval_interval: int = 500
    save_interval: int = 1000
    learning_rate: float = 3e-4
    seed: int = 1337

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 512

    # Checkpointing
    ckpt_dir: str = "checkpoints/normal"
    ckpt_name: str = "last.pt"
    delete_ckpt_on_start: bool = True

    # Logging
    log_dir: str = "output/logs"
    log_name: str = "train_log.csv"
    append_logs: bool = False
    enable_activation_logging: bool = False

    # Run identifier (used in log/ckpt filenames). If empty, derived from decoder module.
    run_name: str = ""

    # Generation
    gen_tokens: int = 500
    gen_temperature: float = 0.8


def load_text(path: str) -> str:
    # Read the full training corpus as a single string.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    # Build character-level vocabulary and lookup tables.
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    # Convert a string to token ids.
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(tokens: torch.Tensor, itos: dict) -> str:
    # Convert token ids back to a string.
    return "".join([itos[i] for i in tokens])


def split_data(data: torch.Tensor, train_ratio: float = 0.9):
    # Split tokenized data into train/val segments.
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor,
              batch_size: int, block_size: int):
    # Sample a random batch of subsequences.
    data_src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i + block_size] for i in ix])
    y = torch.stack([data_src[i + 1:i + block_size + 1] for i in ix])
    return x, y


def build_model(vocab_size: int, cfg: TrainConfig, device: str) -> GPTDecoder:
    # Initialize the GPT decoder model.
    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
    )
    return model.to(device)


def setup_checkpointing(cfg: TrainConfig, run_name: str):
    # Prepare checkpoint directory and full path (namespaced by run).
    ckpt_file = f"{run_name}_{cfg.ckpt_name}" if run_name else cfg.ckpt_name
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_file)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.delete_ckpt_on_start and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return ckpt_path


def maybe_load_checkpoint(ckpt_path: str, model: GPTDecoder,
                          optimizer: torch.optim.Optimizer, device: str):
    # Resume from the last checkpoint if it exists.
    start_step = 0
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
    return start_step


def compute_loss(logits: torch.Tensor, targets: torch.Tensor,
                 vocab_size: int) -> torch.Tensor:
    # Cross-entropy over flattened time and batch dimensions.
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
    )


def evaluate(model: GPTDecoder, train_data: torch.Tensor, val_data: torch.Tensor,
             cfg: TrainConfig, device: str, vocab_size: int):
    # Run a single validation step and return loss.
    model.eval()
    with torch.no_grad():
        x_val, y_val = get_batch("val", train_data, val_data, cfg.batch_size, cfg.block_size)
        x_val, y_val = x_val.to(device), y_val.to(device)
        val_logits = model(x_val)
        val_loss = compute_loss(val_logits, y_val, vocab_size)
    return val_loss


def save_checkpoint(ckpt_path: str, step: int, model: GPTDecoder,
                    optimizer: torch.optim.Optimizer,
                    train_loss: float, val_loss: float):
    # Persist training state for resume.
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        ckpt_path,
    )


def setup_logging(cfg: TrainConfig, run_name: str):
    # Create log directory and return log file path (namespaced by run).
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = f"{run_name}_{cfg.log_name}" if run_name else cfg.log_name
    log_path = os.path.join(cfg.log_dir, log_file)
    mode = "a" if cfg.append_logs else "w"
    if mode == "w" or not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "timestamp,decoder,step,train_loss,val_loss,grad_l2,"
                "step_time_ms,tokens_per_second,forward_time_ms,backward_time_ms,"
                "max_memory_allocated_mb,max_memory_reserved_mb,memory_per_token_mb,"
                "attention_time_ratio,grad_max,grad_std,grad_norm_over_param_norm,"
                "activation_mean,activation_std,activation_std_by_layer,"
                "first_token_latency_ms,generation_tokens_per_sec,decode_tokens_per_sec_by_len,generated_length\n"
            )
    return log_path


def log_step(log_path: str, run_name: str, step: int,
             train_loss: float, val_loss: float, grad_l2: float,
             step_time_ms: float, tokens_per_second: float,
             forward_time_ms: float, backward_time_ms: float,
             max_memory_allocated_mb: float, max_memory_reserved_mb: float,
             memory_per_token_mb: float, attention_time_ratio: float,
             grad_max: float, grad_std: float, grad_norm_over_param_norm: float,
             activation_mean: float, activation_std: float, activation_std_by_layer: str,
             first_token_latency_ms: float, generation_tokens_per_sec: float,
             decode_tokens_per_sec_by_len: str, generated_length: int):
    # Append a single training record for comparison across runs.
    ts = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{ts},{run_name},{step},{train_loss},{val_loss},{grad_l2},"
            f"{step_time_ms},{tokens_per_second},{forward_time_ms},{backward_time_ms},"
            f"{max_memory_allocated_mb},{max_memory_reserved_mb},{memory_per_token_mb},"
            f"{attention_time_ratio},{grad_max},{grad_std},{grad_norm_over_param_norm},"
            f"{activation_mean},{activation_std},{activation_std_by_layer},"
            f"{first_token_latency_ms},{generation_tokens_per_sec},"
            f"{decode_tokens_per_sec_by_len},{generated_length}\n"
        )


def compute_grad_stats(model: GPTDecoder):
    # Compute gradient L2, max, std, and grad_norm_over_param_norm.
    total_sq = 0.0
    grad_max = 0.0
    count = 0
    sum_vals = 0.0
    sumsq_vals = 0.0
    param_sq = 0.0

    for p in model.parameters():
        param_sq += p.data.norm(2).item() ** 2
        if p.grad is None:
            continue
        g = p.grad.data.float().view(-1)
        if g.numel() == 0:
            continue
        total_sq += g.norm(2).item() ** 2
        grad_max = max(grad_max, g.abs().max().item())
        count += g.numel()
        sum_vals += g.sum().item()
        sumsq_vals += g.pow(2).sum().item()

    grad_l2 = total_sq ** 0.5
    if count > 1:
        mean = sum_vals / count
        grad_std = max(sumsq_vals / count - mean ** 2, 0.0) ** 0.5
    else:
        grad_std = float("nan")
    param_norm = param_sq ** 0.5
    grad_norm_over_param_norm = grad_l2 / param_norm if param_norm > 0 else float("nan")
    return grad_l2, grad_max, grad_std, grad_norm_over_param_norm


def attach_activation_hook(model: GPTDecoder, enabled: bool):
    # Track activation stats for final block and per-layer std (if enabled).
    stats = {"mean": float("nan"), "std": float("nan"), "std_by_layer": []}
    if not enabled:
        return stats, []
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        return stats, []

    handles = []

    def _last_block_hook(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        out = out.detach().float()
        stats["mean"] = out.mean().item()
        stats["std"] = out.std().item()

    def _per_layer_hook(layer_idx):
        def _hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            out = out.detach().float()
            while len(stats["std_by_layer"]) <= layer_idx:
                stats["std_by_layer"].append(float("nan"))
            stats["std_by_layer"][layer_idx] = out.std().item()
        return _hook

    handles.append(model.blocks[-1].register_forward_hook(_last_block_hook))
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(_per_layer_hook(i)))
    return stats, handles


def train_loop(model: GPTDecoder, optimizer: torch.optim.Optimizer,
               train_data: torch.Tensor, val_data: torch.Tensor,
               cfg: TrainConfig, device: str, vocab_size: int, ckpt_path: str,
               log_path: str, decoder_name: str, run_name: str,
               activation_stats: dict):
    # Main optimization loop with periodic eval/checkpoint.
    start_step = maybe_load_checkpoint(ckpt_path, model, optimizer, device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for step in range(start_step, cfg.max_iters):
        model.train()
        x, y = get_batch("train", train_data, val_data, cfg.batch_size, cfg.block_size)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

            step_start.record()
            fwd_start.record()
            logits = model(x)
            fwd_end.record()
            loss = compute_loss(logits, y, vocab_size)

            bwd_start.record()
            loss.backward()
            bwd_end.record()
            optimizer.step()
            step_end.record()
            torch.cuda.synchronize()

            forward_time_ms = fwd_start.elapsed_time(fwd_end)
            backward_time_ms = bwd_start.elapsed_time(bwd_end)
            step_time_ms = step_start.elapsed_time(step_end)
        else:
            step_start = time.perf_counter()
            fwd_start = time.perf_counter()
            logits = model(x)
            loss = compute_loss(logits, y, vocab_size)
            forward_time_ms = (time.perf_counter() - fwd_start) * 1000.0

            bwd_start = time.perf_counter()
            loss.backward()
            backward_time_ms = (time.perf_counter() - bwd_start) * 1000.0
            optimizer.step()
            step_time_ms = (time.perf_counter() - step_start) * 1000.0

        grad_l2, grad_max, grad_std, grad_norm_over_param_norm = compute_grad_stats(model)

        tokens = cfg.batch_size * cfg.block_size
        tokens_per_second = tokens / (step_time_ms / 1000.0) if step_time_ms > 0 else float("nan")
        attention_time_ratio = 1.0 if forward_time_ms > 0 else float("nan")

        if use_cuda:
            max_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            max_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            memory_per_token_mb = max_memory_allocated_mb / tokens
        else:
            max_memory_allocated_mb = float("nan")
            max_memory_reserved_mb = float("nan")
            memory_per_token_mb = float("nan")

        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, train_data, val_data, cfg, device, vocab_size)
            print(
                f"step {step} | "
                f"train loss {loss.item():.3f} | "
                f"val loss {val_loss.item():.3f} | "
                f"grad l2 {grad_l2:.3f} | "
                f"decoder {decoder_name} | "
                f"run {run_name}"
            )
            log_step(
                log_path,
                run_name,
                step,
                loss.item(),
                val_loss.item(),
                grad_l2,
                step_time_ms,
                tokens_per_second,
                forward_time_ms,
                backward_time_ms,
                max_memory_allocated_mb,
                max_memory_reserved_mb,
                memory_per_token_mb,
                attention_time_ratio,
                grad_max,
                grad_std,
                grad_norm_over_param_norm,
                activation_stats["mean"],
                activation_stats["std"],
                "|".join(f"{v:.6f}" for v in activation_stats["std_by_layer"]) if activation_stats["std_by_layer"] else "",
                float("nan"),
                float("nan"),
                "",
                -1,
            )

            if step % cfg.save_interval == 0:
                save_checkpoint(
                    ckpt_path,
                    step,
                    model,
                    optimizer,
                    loss.item(),
                    val_loss.item(),
                )


def generate_sample(model: GPTDecoder, cfg: TrainConfig, device: str,
                    itos: dict, log_path: str, run_name: str):
    # Sample text from the trained model.
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # First-token latency (measure on a cloned context without changing RNG state).
    first_token_latency_ms = float("nan")
    if cfg.gen_tokens > 0:
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if use_cuda else None

        if use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            _ = model.generate(context.clone(), max_new_tokens=1, temperature=cfg.gen_temperature)
            end_evt.record()
            torch.cuda.synchronize()
            first_token_latency_ms = start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            _ = model.generate(context.clone(), max_new_tokens=1, temperature=cfg.gen_temperature)
            first_token_latency_ms = (time.perf_counter() - t0) * 1000.0

        torch.set_rng_state(cpu_state)
        if use_cuda and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

    def _supports_kv_cache():
        try:
            import inspect
            return "kv_cache" in inspect.signature(model.forward).parameters
        except Exception:
            return False

    def _measure_decode_tokens_per_sec():
        # Measure tokens/sec as generated length increases (uses cloned context).
        tokens_per_sec = []
        if cfg.gen_tokens <= 0:
            return tokens_per_sec
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if use_cuda else None
        input_ids = context.clone()
        kv_cache = [None] * len(model.blocks) if _supports_kv_cache() and hasattr(model, "blocks") else None
        t0 = time.perf_counter()
        for _ in range(cfg.gen_tokens):
            if kv_cache is not None:
                out = model(input_ids[:, -1:], kv_cache=kv_cache)
                if isinstance(out, (tuple, list)):
                    logits, kv_cache = out
                else:
                    logits = out
            else:
                out = model(input_ids[:, -1:])
                logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits[:, -1, :] / cfg.gen_temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            generated_len = len(tokens_per_sec) + 1
            tokens_per_sec.append(generated_len / elapsed if elapsed > 0 else float("nan"))
        torch.set_rng_state(cpu_state)
        if use_cuda and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        return tokens_per_sec

    # Full generation timing.
    if use_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        generated = model.generate(
            context,
            max_new_tokens=cfg.gen_tokens,
            temperature=cfg.gen_temperature,
        )
        end_evt.record()
        torch.cuda.synchronize()
        gen_time_ms = start_evt.elapsed_time(end_evt)
    else:
        t0 = time.perf_counter()
        generated = model.generate(
            context,
            max_new_tokens=cfg.gen_tokens,
            temperature=cfg.gen_temperature,
        )
        gen_time_ms = (time.perf_counter() - t0) * 1000.0

    generation_tokens_per_sec = (
        cfg.gen_tokens / (gen_time_ms / 1000.0) if gen_time_ms > 0 else float("nan")
    )
    decode_tokens_per_sec_by_len = _measure_decode_tokens_per_sec()
    decode_tokens_per_sec_by_len_str = "|".join(f"{v:.6f}" for v in decode_tokens_per_sec_by_len)

    log_step(
        log_path,
        run_name,
        -1,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "",
        first_token_latency_ms,
        generation_tokens_per_sec,
        decode_tokens_per_sec_by_len_str,
        cfg.gen_tokens,
    )

    print("---------------------------------------------------------------")
    print(decode(generated[0].tolist(), itos))
    print("---------------------------------------------------------------")


def main():
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text(cfg.data_path)
    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)
    print("vocab_size: ", vocab_size)

    data = encode(text, stoi)
    train_data, val_data = split_data(data)

    model = build_model(vocab_size, cfg, device)
    decoder_name = model.__class__.__name__
    decoder_module = model.__class__.__module__.split(".")[-1]
    run_name = cfg.run_name or decoder_module
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ckpt_path = setup_checkpointing(cfg, run_name)
    log_path = setup_logging(cfg, run_name)
    activation_stats, hook = attach_activation_hook(model, cfg.enable_activation_logging)

    train_loop(
        model,
        optimizer,
        train_data,
        val_data,
        cfg,
        device,
        vocab_size,
        ckpt_path,
        log_path,
        decoder_name,
        run_name,
        activation_stats,
    )
    if hook:
        for h in hook:
            h.remove()
    generate_sample(model, cfg, device, itos, log_path, run_name)


if __name__ == "__main__":
    main()

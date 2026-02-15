"""
Plot training logs for quick comparison across GPTDecoder variants.

Reads CSV logs produced by src/llms/train.py:
timestamp,decoder,step,train_loss,val_loss,grad_l2
"""

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime


def read_logs(log_paths):
    # Aggregate metrics by decoder name across multiple log files.
    data = defaultdict(list)
    for path in log_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                decoder = row["decoder"]
                data[decoder].append(
                    {
                        "timestamp": datetime.fromisoformat(row["timestamp"]),
                        "step": int(row["step"]),
                        "train_loss": float(row["train_loss"]),
                        "tokens_per_second": float(row.get("tokens_per_second", "nan")),
                        "step_time_ms": float(row.get("step_time_ms", "nan")),
                        "forward_time_ms": float(row.get("forward_time_ms", "nan")),
                        "backward_time_ms": float(row.get("backward_time_ms", "nan")),
                        "attention_time_ratio": float(row.get("attention_time_ratio", "nan")),
                        "max_memory_allocated_mb": float(row.get("max_memory_allocated_mb", "nan")),
                        "memory_per_token_mb": float(row.get("memory_per_token_mb", "nan")),
                        "grad_l2": float(row.get("grad_l2", "nan")),
                        "grad_norm_over_param_norm": float(row.get("grad_norm_over_param_norm", "nan")),
                        "grad_max": float(row.get("grad_max", "nan")),
                        "grad_std": float(row.get("grad_std", "nan")),
                        "activation_std": float(row.get("activation_std", "nan")),
                        "activation_std_by_layer": row.get("activation_std_by_layer", ""),
                        "first_token_latency_ms": float(row.get("first_token_latency_ms", "nan")),
                        "generation_tokens_per_sec": float(row.get("generation_tokens_per_sec", "nan")),
                        "decode_tokens_per_sec_by_len": row.get("decode_tokens_per_sec_by_len", ""),
                        "generated_length": int(row.get("generated_length", "-1")) if row.get("generated_length") else -1,
                    }
                )
    # Ensure each decoder series is step-sorted.
    for decoder in data:
        data[decoder].sort(key=lambda r: r["step"])
    return data


def plot_series(data, out_dir):
    # Single figure with three overlayed plots for key efficiency metrics.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for decoder, rows in data.items():
        # Use only training rows (skip generation row with step < 0).
        train_rows = [r for r in rows if r["step"] >= 0]
        if not train_rows:
            continue
        train_rows.sort(key=lambda r: r["timestamp"])

        t0 = train_rows[0]["timestamp"]
        elapsed_s = [(r["timestamp"] - t0).total_seconds() for r in train_rows]

        tokens_per_step = []
        for r in train_rows:
            if r["step_time_ms"] > 0 and r["tokens_per_second"] == r["tokens_per_second"]:
                tokens_per_step.append(r["tokens_per_second"] * (r["step_time_ms"] / 1000.0))
            else:
                tokens_per_step.append(0.0)
        cumulative_tokens = []
        total = 0.0
        for n in tokens_per_step:
            total += n
            cumulative_tokens.append(total)

        loss_vals = [r["train_loss"] for r in train_rows]
        if len(loss_vals) > 1:
            axes[0].plot(elapsed_s[1:], loss_vals[1:], ".-",
                         markersize=4.5, label=decoder)
        else:
            axes[0].plot(elapsed_s, loss_vals, ".-",
                         markersize=4.5, label=decoder)
        axes[1].plot(elapsed_s, cumulative_tokens, ".-", markersize=4.5, label=decoder)
        axes[2].plot([r["step"] for r in train_rows],
                     [r["tokens_per_second"] for r in train_rows],
                     ".-", markersize=4.5, label=decoder)

    axes[0].set_title("Loss vs Wall-Clock Time")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_xlabel("Elapsed Time (s)")

    axes[1].set_title("Tokens Processed vs Time")
    axes[1].set_ylabel("Cumulative Tokens")
    axes[1].set_xlabel("Elapsed Time (s)")

    axes[2].set_title("Tokens/sec vs Step")
    axes[2].set_ylabel("Tokens / sec")
    axes[2].set_xlabel("Step")

    for ax in axes:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

    axes[0].legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "metrics_efficiency.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Additional plots for systems and gradient metrics.
    fig2, axes2 = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

    for decoder, rows in data.items():
        train_rows = [r for r in rows if r["step"] >= 0]
        if not train_rows:
            continue
        steps = [r["step"] for r in train_rows]

        fwd = [r["forward_time_ms"] for r in train_rows]
        bwd = [r["backward_time_ms"] for r in train_rows]
        step_t = [r["step_time_ms"] for r in train_rows]
        other = [max(s - f - b, 0.0) if s == s and f == f and b == b else float("nan")
                 for s, f, b in zip(step_t, fwd, bwd)]

        axes2[0].plot(steps, fwd, ".-", markersize=4.5, label=f"{decoder} fwd")
        axes2[0].plot(steps, bwd, ".-", markersize=4.5, label=f"{decoder} bwd")
        axes2[0].plot(steps, other, ".-", markersize=4.5, label=f"{decoder} other")

        axes2[1].plot(steps,
                      [r["attention_time_ratio"] for r in train_rows],
                      ".-", markersize=4.5, label=decoder)
        axes2[2].plot(steps,
                      [r["max_memory_allocated_mb"] for r in train_rows],
                      ".-", markersize=4.5, label=decoder)
        axes2[3].plot(steps,
                      [r["memory_per_token_mb"] for r in train_rows],
                      ".-", markersize=4.5, label=decoder)
        axes2[4].plot(steps,
                      [r["grad_l2"] for r in train_rows],
                      ".-", markersize=4.5, label=decoder)

    axes2[0].set_title("Step Time Breakdown")
    axes2[0].set_ylabel("Time (ms)")
    axes2[0].legend(loc="best")

    axes2[1].set_title("Attention Time Ratio")
    axes2[1].set_ylabel("attention_time / forward_time")

    axes2[2].set_title("Peak GPU Memory vs Step")
    axes2[2].set_ylabel("MB")

    axes2[3].set_title("Memory per Token")
    axes2[3].set_ylabel("MB / token")

    axes2[4].set_title("Gradient Norm vs Step")
    axes2[4].set_ylabel("grad_l2")
    axes2[4].set_xlabel("Step")

    for ax in axes2:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, "metrics_systems.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)

    # Gradient-focused plots.
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axr = axes3[1].twinx()
    for decoder, rows in data.items():
        train_rows = [r for r in rows if r["step"] >= 0]
        if not train_rows:
            continue
        steps = [r["step"] for r in train_rows]
        axes3[0].plot(steps,
                      [r["grad_norm_over_param_norm"] for r in train_rows],
                      ".-", markersize=4.5, label=decoder)
        axes3[1].plot(steps, [r["grad_max"] for r in train_rows],
                      ".-", markersize=4.5, label=f"{decoder} grad_max")
        axr.plot(steps, [r["grad_std"] for r in train_rows],
                 ".-", markersize=4.5, label=f"{decoder} grad_std")

    axes3[0].set_title("Grad Norm / Param Norm")
    axes3[0].set_ylabel("||g|| / ||θ||")
    axes3[1].set_title("Gradient Max & Std (Dual Axis)")
    axes3[1].set_ylabel("grad_max")
    axr.set_ylabel("grad_std")
    axes3[1].set_xlabel("Step")
    for ax in axes3:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
    axes3[0].legend(loc="best")
    fig3.tight_layout()
    out_path3 = os.path.join(out_dir, "metrics_gradients.png")
    fig3.savefig(out_path3, dpi=150)
    plt.close(fig3)

    # Activation plots.
    fig4, axes4 = plt.subplots(2, 1, figsize=(12, 10))
    for decoder, rows in data.items():
        train_rows = [r for r in rows if r["step"] >= 0]
        if not train_rows:
            continue
        steps = [r["step"] for r in train_rows]
        axes4[1].plot(steps,
                      [r["activation_std"] for r in train_rows],
                      ".-", markersize=4.5, label=decoder)
        # Use the most recent activation std by layer for depth plot.
        last_layer_str = ""
        for r in reversed(train_rows):
            if r["activation_std_by_layer"]:
                last_layer_str = r["activation_std_by_layer"]
                break
        if last_layer_str:
            layer_vals = [float(v) for v in last_layer_str.split("|") if v]
            axes4[0].plot(list(range(len(layer_vals))), layer_vals, ".-",
                          markersize=4.5, label=decoder)

    axes4[0].set_title("Activation Std vs Depth")
    axes4[0].set_ylabel("activation std")
    axes4[0].set_xlabel("layer index")
    axes4[1].set_title("Activation Std vs Step")
    axes4[1].set_ylabel("activation std")
    axes4[1].set_xlabel("step")
    for ax in axes4:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
    axes4[0].legend(loc="best")
    fig4.tight_layout()
    out_path4 = os.path.join(out_dir, "metrics_activation.png")
    fig4.savefig(out_path4, dpi=150)
    plt.close(fig4)

    # Generation plots.
    fig5, axes5 = plt.subplots(2, 1, figsize=(12, 8))
    run_names = []
    latencies = []
    for decoder, rows in data.items():
        gen_rows = [r for r in rows if r["step"] < 0]
        if not gen_rows:
            continue
        row = gen_rows[-1]
        run_names.append(decoder)
        latencies.append(row["first_token_latency_ms"])

        seq = row["decode_tokens_per_sec_by_len"]
        if seq:
            vals = [float(v) for v in seq.split("|") if v]
            axes5[0].plot(list(range(1, len(vals) + 1)), vals, ".-",
                          markersize=4.5, label=decoder)

    axes5[0].set_title("Decode Tokens/sec vs Generated Length")
    axes5[0].set_ylabel("tokens/sec")
    axes5[0].set_xlabel("generated tokens")
    axes5[0].legend(loc="best")

    axes5[1].set_title("First-Token Latency")
    axes5[1].set_ylabel("ms")
    axes5[1].bar(run_names, latencies)

    for ax in axes5:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
    fig5.tight_layout()
    out_path5 = os.path.join(out_dir, "metrics_generation.png")
    fig5.savefig(out_path5, dpi=150)
    plt.close(fig5)

    # Scaling plots (loss reduction per GPU-hour, tokens/sec per GB).
    fig6, axes6 = plt.subplots(2, 1, figsize=(12, 8))
    run_names = []
    tps_per_gb = []
    for decoder, rows in data.items():
        train_rows = [r for r in rows if r["step"] >= 0]
        if not train_rows:
            continue
        train_rows.sort(key=lambda r: r["timestamp"])
        t0 = train_rows[0]["timestamp"]
        base_loss = train_rows[0]["train_loss"]
        gpu_hours = [(r["timestamp"] - t0).total_seconds() / 3600.0 for r in train_rows]
        loss_red = [base_loss - r["train_loss"] for r in train_rows]
        axes6[0].plot(gpu_hours, loss_red, ".-", markersize=4.5, label=decoder)

        vals = []
        for r in train_rows:
            mem_gb = r["max_memory_allocated_mb"] / 1024.0
            if mem_gb > 0 and r["tokens_per_second"] == r["tokens_per_second"]:
                vals.append(r["tokens_per_second"] / mem_gb)
        if vals:
            run_names.append(decoder)
            tps_per_gb.append(sum(vals) / len(vals))

    axes6[0].set_title("Loss Reduction per GPU-Hour")
    axes6[0].set_ylabel("loss decrease")
    axes6[0].set_xlabel("GPU hours")
    axes6[0].legend(loc="best")

    axes6[1].set_title("Tokens/sec per GB")
    axes6[1].set_ylabel("tokens/sec/GB")
    axes6[1].bar(run_names, tps_per_gb)

    for ax in axes6:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
    fig6.tight_layout()
    out_path6 = os.path.join(out_dir, "metrics_scaling.png")
    fig6.savefig(out_path6, dpi=150)
    plt.close(fig6)


def main():
    parser = argparse.ArgumentParser(description="Plot GPT training logs.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more CSV log files produced by train.py",
    )
    parser.add_argument(
        "--out",
        default="output/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    data = read_logs(args.logs)
    if not data:
        raise RuntimeError("No log data found in provided files.")
    plot_series(data, args.out)
    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()

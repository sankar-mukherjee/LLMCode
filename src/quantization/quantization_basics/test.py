import torch
import torch.nn as nn
from src.quantization.quantization_basics.simple_quantization import PerTensorQuant, PerChannelQuant
from quantlinear import QuantLinear


def drift_metrics(fp32, int8):
    diff = fp32 - int8
    return {
        "mae": diff.abs().mean().item(),
        "max": diff.abs().max().item(),
        "rel_l2": (diff.norm() / fp32.norm()).item(),
    }


def test_linear_drift(quantizer, batch=32, in_f=1024, out_f=256):
    torch.manual_seed(0)

    X = torch.randn(batch, in_f)
    W = torch.randn(out_f, in_f)

    # FP32 reference
    Y_fp32 = X @ W.T

    # INT8 path (weight-only)
    W_q = quantizer.quantize(W)
    W_hat = quantizer.dequantize(W_q)
    Y_int8 = X @ W_hat.T

    return drift_metrics(Y_fp32, Y_int8)


def test_quant_linear_drift(ql):
    torch.manual_seed(0)

    x = torch.randn(32, 1024)

    fp = nn.Linear(1024, 256)

    # Copy weights
    ql.weight.data.copy_(fp.weight.data)
    ql.bias.data.copy_(fp.bias.data)

    y_fp32 = fp(x)
    y_int8 = ql(x)

    return drift_metrics(y_fp32, y_int8)


if __name__ == "__main__":
    for num_bit in (8, 4):
        print("num_bits:", num_bit)

        # Per-tensor
        pt = PerTensorQuant(num_bits=num_bit)
        pt_metrics = test_linear_drift(pt)

        # Per-channel
        pc = PerChannelQuant(dim=0, num_bits=num_bit)
        pc_metrics = test_linear_drift(pc)

        print("Per-Tensor Drift:", pt_metrics)
        print("Per-Channel Drift:", pc_metrics)

        #
        ql = QuantLinear(1024, 256, num_bits=num_bit, per_channel=False)
        ql_metrics = test_quant_linear_drift(ql)
        print("Per-Tensor QuantLinear Drift:", ql_metrics)
        ql = QuantLinear(1024, 256, num_bits=num_bit, per_channel=True)
        ql_metrics = test_quant_linear_drift(ql)
        print("Per-Channel QuantLinear Drift:", ql_metrics)
        print('-----------------------------------------')

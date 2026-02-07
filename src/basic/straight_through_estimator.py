# - Straight-Through Estimator (STE)
#     - **Custom Autograd (Quantization):** Implement a **Straight-Through Estimator (STE)** in PyTorch to 
#           simulate INT8/FP8 quantization effects during fine-tuning.
#     - **Custom Autograd Function:** Define a torch.autograd.Function with a manual backward pass 
#           (e.g., implementing a Straight-Through Estimator for quantization).

# https://chatgpt.com/share/69845dda-f8b4-800f-83eb-2f9be9cda8c8

import torch

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin=-128, qmax=127):
        # Save for backward if needed (not required for STE)
        ctx.save_for_backward(x)

        # Fake quantization
        x_int = torch.clamp(torch.round(x / scale), qmin, qmax)
        x_q = x_int * scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class STEFP8Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exp_bits=4, man_bits=3):
        # Simulate FP8 via log-domain clamping
        max_exp = 2 ** (exp_bits - 1) - 1

        mantissa_scale = 2 ** man_bits
        sign = torch.sign(x)
        abs_x = torch.abs(x) + 1e-8

        exp = torch.floor(torch.log2(abs_x))
        exp = torch.clamp(exp, -max_exp, max_exp)

        mant = abs_x / (2 ** exp)
        mant_q = torch.round(mant * mantissa_scale) / mantissa_scale

        x_q = sign * mant_q * (2 ** exp)
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE
        return grad_output.clone(), None, None


class QuantLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, scale=0.02):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.scale = scale

    def forward(self, x):
        qw = STEQuantize.apply(self.weight, self.scale)
        return torch.nn.functional.linear(x, qw, self.bias)

class MaskedSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin=-128, qmax=127):
        x_int = torch.round(x / scale)
        mask = (x_int >= qmin) & (x_int <= qmax)
        ctx.save_for_backward(mask)
        return torch.clamp(x_int, qmin, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None, None, None

# Quantization Logic: Implement a simple function to quantize weights from float32 to int8 and handle the resulting scaling factor.

import torch

class PerTensorQuant:
    def __init__(self, num_bits=8, eps=1e-8):
        self.num_bits = num_bits
        self.eps = eps
        self.scale = None

        self.qmax = 2 ** (num_bits - 1) - 1 # for 8bit its 127.0
        self.qmin = -self.qmax

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        max_val = w.abs().max()
        self.scale = torch.clamp(max_val / self.qmax, min=self.eps)

        w_q = torch.round(w / self.scale)
        return torch.clamp(w_q, self.qmin, self.qmax).to(torch.int8)

    def dequantize(self, w_q: torch.Tensor) -> torch.Tensor:
        return w_q.float() * self.scale

    def fake_quant(self, w):
        max_val = w.abs().max()
        scale = torch.clamp(max_val / self.qmax, min=self.eps)

        w_q = torch.round(w / scale).clamp(self.qmin, self.qmax)
        w_hat = w_q * scale

        # Straight-Through Estimator
        return w + (w_hat - w).detach()


class PerChannelQuant:
    def __init__(self, dim=0, num_bits=8, eps=1e-8):
        self.dim = dim
        self.num_bits = num_bits
        self.eps = eps
        self.scale = None

        self.qmax = 2 ** (num_bits - 1) - 1 # for 8bit its 127.0
        self.qmin = -self.qmax

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(d for d in range(w.ndim) if d != self.dim)
        max_val = w.abs().amax(dim=reduce_dims)
        self.scale = torch.clamp(max_val / self.qmax, min=self.eps)

        shape = [1] * w.ndim
        shape[self.dim] = -1
        scale = self.scale.view(*shape)

        w_q = torch.round(w / scale)
        return torch.clamp(w_q, self.qmin, self.qmax).to(torch.int8)

    def dequantize(self, w_q: torch.Tensor) -> torch.Tensor:
        shape = [1] * w_q.ndim
        shape[self.dim] = -1
        return w_q.float() * self.scale.view(*shape)

    def fake_quant(self, w):
        reduce_dims = tuple(d for d in range(w.ndim) if d != self.dim)
        max_val = w.abs().amax(dim=reduce_dims)
        scale = torch.clamp(max_val / self.qmax, min=self.eps)

        shape = [1] * w.ndim
        shape[self.dim] = -1
        scale = scale.view(*shape)

        w_q = torch.round(w / scale).clamp(self.qmin, self.qmax)
        w_hat = w_q * scale

        # Straight-Through Estimator
        return w + (w_hat - w).detach()

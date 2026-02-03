import torch
import torch.nn as nn
import torch.nn.functional as F
from src.qunatization_basics.simple_qunatization import PerChannelQuant, PerTensorQuant


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_bits=8, per_channel=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if per_channel:
            self.quantizer = PerChannelQuant(dim=0, num_bits=num_bits)
        else:
            self.quantizer = PerTensorQuant(num_bits=num_bits)

    def forward(self, x):
        w = self.quantizer.fake_quant(self.weight)
        return F.linear(x, w, self.bias)


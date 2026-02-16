import torch
import torch.nn.functional as F
import math

# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu(x):
    """ReLU activation: max(0, x)"""
    return torch.clamp(x, min=0)


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + e^(-x))"""
    return 1 / (1 + torch.exp(-x))

def gelu(x):
    """Fast GELU (tanh approximation)"""
    t = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1 + torch.tanh(t))

def silu(x):
    return x * sigmoid(x)

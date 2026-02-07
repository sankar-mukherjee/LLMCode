# - **numerically stable Softmax**
#     - Implement the **forward pass of Softmax** for a batch of inputs.
#     - Derive and implement the **backward pass of Softmax**.

# https://chatgpt.com/share/6984597a-03d4-800f-a19e-7296fbd626cb

import torch

def softmax_forward(x):
    """
    x: (B, C)
    returns:
      s: softmax output (B, C)
      cache: values needed for backward
    """
    x_shifted = x - x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    s = exp_x / exp_x.sum(dim=1, keepdim=True)
    return s, s  # cache softmax itself

def softmax_backward(dout, s):
    """
    dout: upstream gradient dL/ds (B, C)
    s: softmax output from forward (B, C)
    returns:
      dx: gradient w.r.t input x (B, C)
    """
    # dot = sum_j (dout_j * s_j)
    dot = torch.sum(dout * s, dim=1, keepdim=True)

    dx = s * (dout - dot)
    return dx


x = torch.randn(4, 5, requires_grad=True)

s, cache = softmax_forward(x)
loss = s.sum()
loss.backward()

# Manual backward
dout = torch.ones_like(s)
dx_manual = softmax_backward(dout, cache)

print(torch.allclose(x.grad, dx_manual, atol=1e-6))

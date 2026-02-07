# - ## 2. Activation Functions
#     - 2.1 Implementation
#     - Implement **ReLU**, **Sigmoid**, and **GeLU**.
#     - Write their **derivatives explicitly** (no PyTorch autograd).

#     - 2.2 Theory & Tradeoffs
#     - Why does Sigmoid cause vanishing gradients?
#     - Why is GeLU preferred in Transformers?
#     - Compare ReLU vs Leaky ReLU vs GeLU in terms of gradient flow.

# https://chatgpt.com/share/69845ae7-d760-800f-8869-9eefe30d62b0

import torch
import math

class ReLU:
    def forward(self, x):
        self.x = x
        return torch.clamp(x, min=0)

    def backward(self, grad_out):
        grad_x = grad_out.clone()
        grad_x[self.x <= 0] = 0
        return grad_x

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + torch.exp(-x))
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1 - self.out)


class GELU:
    def forward(self, x):
        self.x = x
        self.t = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        self.tanh_t = torch.tanh(self.t)
        return 0.5 * x * (1 + self.tanh_t)

    def backward(self, grad_out):
        sech2 = 1 - self.tanh_t ** 2
        dt_dx = math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * self.x**2)

        grad = (
            0.5 * (1 + self.tanh_t)
            + 0.5 * self.x * sech2 * dt_dx
        )
        return grad_out * grad

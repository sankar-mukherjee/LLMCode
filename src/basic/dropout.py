# - **Custom Dropout:** Implement a Dropout layer that scales the output by $1/(1-p)$ during training to ensure 
# the expected value remains consistent during inference (Inverted Dropout).
    # - Implement a **Dropout layer class** with mask reuse in backward pass.
    # - Explain how **Dropout** behaves differently during training vs inference and implement both passes.
    # - Where does Dropout hurt performance?
# https://chatgpt.com/share/69843e43-06ec-800f-ba68-76b73f8eafee

import torch
import torch.nn as nn

class NormalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask
        else:
            return x * (1.0 - self.p)


class InvertedDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        # Bernoulli mask
        mask = (torch.rand_like(x) > self.p).float()

        # Inverted dropout scaling
        return x * mask / (1.0 - self.p)

class Dropout:
    def __init__(self, p=0.5):
        assert 0.0 <= p < 1.0
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        # Bernoulli mask
        self.mask = (torch.rand_like(x) > self.p).float()

        # Inverted dropout scaling
        return x * self.mask / (1.0 - self.p)

    def backward(self, grad_output):
        if not self.training or self.p == 0.0:
            return grad_output

        # Mask reuse is critical
        return grad_output * self.mask / (1.0 - self.p)


x = torch.ones(100000)
drop = InvertedDropout(p=0.2)
drop.train()

y = drop(x)
print(y.mean().item())  # ≈ 1.0

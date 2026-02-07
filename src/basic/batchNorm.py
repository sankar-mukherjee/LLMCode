# - **Batch Normalization (Forward):** Code the forward pass of Batch Norm by calculating mini-batch mean and variance, and applying learnable $\gamma$ and $\beta$ parameters.
#     - Write the **forward and backward pass of Batch Normalization** (training mode).
#     - 1.2 Full Layer Class
#     - Implement a **BatchNorm layer class from scratch** (no autograd).
#     - Forward
#     - Backward
#     - Running mean / variance
#     - Follow-ups**
#     - Why is BatchNorm unstable for very small batch sizes?
#     - How does LayerNorm fix this?

# https://chatgpt.com/share/69844319-68e0-800f-be19-ba92a875fec9

import torch

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """
    x:     (N, D) torch.Tensor
    gamma: (D,)
    beta:  (D,)
    """
    # Mini-batch statistics
    mean = x.mean(dim=0)                      # (D,)
    var = x.var(dim=0, unbiased=False)        # (D,)

    std = torch.sqrt(var + eps)
    x_hat = (x - mean) / std                  # (N, D)
    out = gamma * x_hat + beta

    cache = (x, x_hat, mean, var, std, gamma, eps)
    return out, cache


def batchnorm_backward(dout, cache):
    """
    dout: upstream gradient (N, D)
    """
    x, x_hat, mean, var, std, gamma, eps = cache
    N, D = x.shape

    # Gradients for affine params
    dbeta = dout.sum(dim=0)                   # (D,)
    dgamma = (dout * x_hat).sum(dim=0)        # (D,)

    # Gradient w.r.t normalized input
    dx_hat = dout * gamma                     # (N, D)

    # Gradient w.r.t input
    dx = (1.0 / N) * (1.0 / std) * (
        N * dx_hat
        - dx_hat.sum(dim=0)
        - x_hat * (dx_hat * x_hat).sum(dim=0)
    )

    return dx, dgamma, dbeta


class BatchNorm:
    def __init__(self, dim, momentum=0.9, eps=1e-5, device="cpu"):
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)

        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

        self.momentum = momentum
        self.eps = eps
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            out = self.gamma * x_hat + self.beta

            # Update running statistics
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * var
            )

            self.cache = (x, x_hat, mean, var, torch.sqrt(var + self.eps), self.gamma)
            return out

        else:
            # Inference mode
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x, x_hat, mean, var, std, gamma = self.cache
        N, D = x.shape

        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)

        dx_hat = dout * gamma

        dx = (1.0 / N) * (1.0 / std) * (
            N * dx_hat
            - dx_hat.sum(dim=0)
            - x_hat * (dx_hat * x_hat).sum(dim=0)
        )

        return dx, dgamma, dbeta

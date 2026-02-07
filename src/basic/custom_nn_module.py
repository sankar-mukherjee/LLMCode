# *   **Custom Module Design:** Define a `torch.nn.Module` with a custom `forward` pass. 
# Be prepared to explain how the computation graph is built and how `autograd` tracks gradients.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        # Trainable parameters registered automatically
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: Tensor [batch, in_dim]
        """
        h = self.fc1(x)        # affine transform
        h = F.relu(h)          # non-linearity
        y = self.fc2(h)        # affine transform
        return y


model = MyMLP(4, 8, 2)

x = torch.randn(5, 4)
y = model(x)

loss = y.sum()
loss.backward()


# 60-Second Answer (Strong Technical Signal)
# Autograd tracks gradients by building a define-by-run computation graph during the forward pass.
# Each tensor that has requires_grad=True records the operation that produced it via a grad_fn.
# When we call backward() on a scalar loss, autograd performs a reverse topological traversal of this graph, 
# computes local derivatives for each operation, and uses the chain rule to propagate gradients back to leaf tensors, 
# accumulating them in .grad.
# 
# Optionally add:
# This is reverse-mode AD, which is efficient when you have many parameters and a scalar loss.

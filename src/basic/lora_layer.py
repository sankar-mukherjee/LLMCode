# **LoRA from Scratch:** Since you adapted Whisper with LoRA, be prepared to implement a `LoRALayer` that wraps a `nn.Linear`, 
# including the $A$ and $B$ matrix initializations and the scaling factor.

# https://chatgpt.com/share/69845c0c-062c-800f-93cd-55bbd42ff9c1

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias=True,
    ):
        super().__init__()

        # Base (frozen) linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # LoRA params
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(lora_dropout)

            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        # Key LoRA initialization trick
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.linear(x)

        if self.r > 0:
            lora_update = (
                self.dropout(x)
                @ self.lora_A.T
                @ self.lora_B.T
            )
            result = result + self.scaling * lora_update

        return result

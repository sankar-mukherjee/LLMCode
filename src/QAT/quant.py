import torch.nn as nn
from src.qunatization_basics.quantlinear import QuantLinear


def replace_linear_with_quant(module, num_bits=4, per_channel=True):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):

            ql = QuantLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                num_bits=num_bits,
                per_channel=per_channel,
            ).to(child.weight.device)

            ql.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ql.bias.data.copy_(child.bias.data)

            setattr(module, name, ql)
        else:
            replace_linear_with_quant(child, num_bits, per_channel)

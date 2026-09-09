"""Positional encoding of point and ray features."""

import torch
from torch import nn

def posenc(x, L_embed, factor=2.0, without_self=False, mult_factor=1.0):
    if without_self:
        rets = []
    else:
        rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(factor**i * x * mult_factor))
    # To make sure the dimensions of the same meaning are together
    return torch.flatten(torch.stack(rets, -1), start_dim=-2, end_dim=-1)


class PoseEnc(nn.Module):
    def __init__(self, factor=2.0, mult_factor=1.0):
        super(PoseEnc, self).__init__()
        self.factor = factor
        self.mult_factor = mult_factor

    def forward(self, x, L_embed, without_self=False):
        return posenc(x, L_embed, self.factor, without_self, self.mult_factor)

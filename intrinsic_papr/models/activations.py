"""Activation layers selectable from the configuration."""

from torch import nn

def activation_func(act_type="leakyrelu", neg_slope=0.2, inplace=True):
    act_type = act_type.lower()
    if act_type == "none":
        layer = nn.Identity()
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "tanh":
        layer = nn.Tanh()
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )

    return layer


class SoftplusActivation(nn.Module):
    def __init__(self, c1=1, c2=1, c3=0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def forward(self, x):
        return self.c1 * nn.functional.softplus(self.c2 * x + self.c3)

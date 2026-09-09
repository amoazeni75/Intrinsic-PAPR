import torch
from torch import nn

from .activations import activation_func


class MLP(nn.Module):
    def __init__(self, inp_dim=2, num_layers=3, num_channels=128, out_dim=2, act_type="leakyrelu",
                 last_act_type="none", skip_layers=[], half_layers=[]):
        super(MLP, self).__init__()
        self.skip_layers = skip_layers
        layers = [nn.Identity()]
        for i in range(num_layers):
            cur_inp = inp_dim if i == 0 else num_channels
            cur_out = out_dim if i == num_layers - 1 else num_channels
            if (i+1) in half_layers:
                cur_out = cur_out // 2
            if i in half_layers:
                cur_inp = cur_inp // 2
            if i in self.skip_layers:
                cur_inp += inp_dim
            layers.append(nn.Linear(cur_inp, cur_out))
            layers.append(activation_func(act_type=act_type))
        layers[-1] = activation_func(act_type=last_act_type)
        assert len(layers) == 2 * num_layers + 1
        self.model = nn.ModuleList(layers)

    def forward(self, x):
        skip_layers = [i*2+1 for i in self.skip_layers]
        inp = x
        for i, layer in enumerate(self.model):
            if i in skip_layers:
                x = torch.cat([x, inp], dim=-1)
            x = layer(x)
        return x

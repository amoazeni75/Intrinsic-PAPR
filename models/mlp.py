import torch
from torch import nn
from torch import autocast
from .utils import activation_func


def get_mapping_mlp(args, use_amp=False, amp_dtype=torch.float16):
    return MappingMLP(args.mapping_mlp, inp_dim=args.shading_code_dim, out_dim=args.mapping_mlp.out_dim, use_amp=use_amp, amp_dtype=amp_dtype)


class MLP(nn.Module):
    def __init__(self, inp_dim=2, num_layers=3, num_channels=128, out_dim=2, act_type="leakyrelu", last_act_type="none",
                 a=1., b=1., trainable=False, skip_layers=[], half_layers=[]):
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
            layers.append(activation_func(act_type=act_type,
                          num_channels=cur_out, a=a, b=b, trainable=trainable))
        layers[-1] = activation_func(act_type=last_act_type,
                                     num_channels=out_dim, a=a, b=b, trainable=trainable)
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


class MappingMLP(nn.Module):
    def __init__(self, args, inp_dim=2, out_dim=2, use_amp=False, amp_dtype=torch.float16):
        super(MappingMLP, self).__init__()
        self.args = args
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim,
                         act_type=args.act, last_act_type=args.last_act)
        print("Mapping MLP:\n", self.model)

    def forward(self, x):

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            return self.model(x)

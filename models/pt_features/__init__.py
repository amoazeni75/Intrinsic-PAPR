import torch
from torch import nn


class PointFeatures(nn.Module):
    def __init__(self, args, num_pts):
        super(PointFeatures, self).__init__()
        self.args = args
        self.num_pts = num_pts

        # One can add more types of point features here
        if self.args.type == "learnable":
            self.features = nn.Parameter(torch.randn(num_pts, args.dim), requires_grad=True)

        else:
            raise NotImplementedError(
                'point feature type [{:s}] is not supported'.format(self.args.type))

    def forward(self, points):
        if self.args.type == "learnable":
            features = self.features

        return features

# credit: https://github.com/isl-org/MiDaS (MidasNet_small architecture)
# credit: https://github.com/compphoto/Intrinsic (ordinal shading pipeline)
# Extended here with AdaIN style modulation for ambiguity-aware (cIMLE) training.
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)



class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _calc_same_pad(i, k, s, d):
    """Added by Chris"""
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(
    x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    """Added by Chris"""
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class Conv2dSame(nn.Conv2d):
    """Added by Chris
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        """Added by Chris"""
        return conv2d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False, in_chan=3):
    """Modified by Chris to add in_chan"""
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
    )

    if in_chan != 3:
        efficientnet.conv_stem = Conv2dSame(
            in_chan, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
        )

    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_encoder(
    backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    exportable=True,
    in_chan=3,
):
    """Added by Chris: in_chan argument, which is used by _make_pretrained_resnext101_wsl and _make_pretrained_efficientnet_lite3"""
    if backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(
            use_pretrained,
            exportable=exportable,
            in_chan=in_chan,
        )
        scratch = _make_scratch(
            [32, 48, 136, 384], features, groups=groups, expand=expand
        )  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


class AdaIn(nn.Module):
    def __init__(self, latent_size, out_channels):
        super(AdaIn, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_channels * 2),
        )

    def forward(self, x, latent, mean_shift=0.0, var_shift=0.0):
        style = self.mlp(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]

        shape2 = [1, x.size(1)] + (x.dim() - 2) * [1]
        mean_shift = mean_shift.view(shape2)
        var_shift = var_shift.view(shape2)


        mean = style[:, 1] - mean_shift.cuda()
        var = style[:, 0] + 1.0 - var_shift.cuda()


        x = x * (var) + mean

        return x


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


class MidasNet_small(BaseModel):
    """Network for monocular depth estimation."""

    def __init__(
        self,
        activation="sigmoid",
        pretrained=False,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        channels_last=False,
        align_corners=True,
        blocks={"expand": True},
        input_channels=3,
        output_channels=1,
        out_bias=0,
        use_cIMLE_pretrained=False,
        AdaIn_latent_size=32,
    ):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(MidasNet_small, self).__init__()
        self.debug = True

        # cIMLE
        self.use_cIMLE_pretrained = use_cIMLE_pretrained
        self.AdaIn_latent_size = AdaIn_latent_size

        self.out_chan = output_channels

        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks["expand"] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        self.pretrained, self.scratch = _make_encoder(
            self.backbone,
            features,
            pretrained,
            in_chan=input_channels,
            groups=self.groups,
            expand=self.expand,
            exportable=exportable,
        )

        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            features4,
            self.scratch.activation,
            deconv=False,
            bn=False,
            expand=self.expand,
            align_corners=align_corners,
        )
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            features3,
            self.scratch.activation,
            deconv=False,
            bn=False,
            expand=self.expand,
            align_corners=align_corners,
        )
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            features2,
            self.scratch.activation,
            deconv=False,
            bn=False,
            expand=self.expand,
            align_corners=align_corners,
        )
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            features1,
            self.scratch.activation,
            deconv=False,
            bn=False,
            align_corners=align_corners,
        )

        if activation == "sigmoid":
            output_act = nn.Sigmoid()
        if activation == "tanh":
            output_act = nn.Tanh()
        if activation == "none":
            output_act = nn.Identity()

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(
                features,
                features // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.groups,
            ),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act,
        )
        self.scratch.output_conv[-2].bias = torch.nn.Parameter(
            torch.ones(output_channels) * out_bias
        )

        if self.use_cIMLE_pretrained:
            print("Decoder_cIMLE with AdaIn v2")
            ## Noise1
            self.style_mod0 = AdaIn(self.AdaIn_latent_size, out_channels=32)
            self.style_mod0_meanshift = torch.zeros(32)
            self.style_mod0_varshift = torch.zeros(32)

            ## Noise2
            self.style_mod1 = AdaIn(self.AdaIn_latent_size, out_channels=48)
            self.style_mod1_meanshift = torch.zeros(48)
            self.style_mod1_varshift = torch.zeros(48)

            ## Noise3
            self.style_mod2 = AdaIn(self.AdaIn_latent_size, out_channels=136)
            self.style_mod2_meanshift = torch.zeros(136)
            self.style_mod2_varshift = torch.zeros(136)

            ## Noise4
            self.style_mod3 = AdaIn(self.AdaIn_latent_size, out_channels=384)
            self.style_mod3_meanshift = torch.zeros(384)
            self.style_mod3_varshift = torch.zeros(384)

    def forward_no_CIMLE(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last == True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        if self.debug:
            print(
                f"Input to self.pretrained.layer1 has shape: {x.shape}"
            )  # torch.Size([1, 3, W, H])
        layer_1 = self.pretrained.layer1(x)
        if self.debug:
            print(
                f"Output from self.pretrained.layer1 has shape: {layer_1.shape}"
            )  # torch.Size([1, 32, W/4, H/4])

        if self.debug:
            print(
                f"Input to self.pretrained.layer2 has shape: {layer_1.shape}"
            )  # torch.Size([1, 32, W/4, H/4])
        layer_2 = self.pretrained.layer2(layer_1)
        if self.debug:
            print(
                f"Output from self.pretrained.layer2 has shape: {layer_2.shape}"
            )  # torch.Size([1, 48, W/8, H/8])

        if self.debug:
            print(
                f"Input to self.pretrained.layer3 has shape: {layer_2.shape}"
            )  # torch.Size([1, 48, W/8, H/8])
        layer_3 = self.pretrained.layer3(layer_2)
        if self.debug:
            print(
                f"Output from self.pretrained.layer3 has shape: {layer_3.shape}"
            )  # torch.Size([1, 136, W/16, H/16])

        if self.debug:
            print(
                f"Input to self.pretrained.layer4 has shape: {layer_3.shape}"
            )  # torch.Size([1, 136, W/16, H/16])
        layer_4 = self.pretrained.layer4(layer_3)
        if self.debug:
            print(
                f"Output from self.pretrained.layer4 has shape: {layer_4.shape}"
            )  # torch.Size([1, 384, W/32, H/32])

        if self.debug:
            print(
                f"Input to self.scratch.layer1_rn has shape: {layer_1.shape}"
            )  # torch.Size([1, 32, W/4, H/4])
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        if self.debug:
            print(
                f"Output from self.scratch.layer1_rn has shape: {layer_1_rn.shape}"
            )  # torch.Size([1, 64, W/4, H/4])

        if self.debug:
            print(
                f"Input to self.scratch.layer2_rn has shape: {layer_2.shape}"
            )  # torch.Size([1, 48, W/8, H/8])
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        if self.debug:
            print(
                f"Output from self.scratch.layer2_rn has shape: {layer_2_rn.shape}"
            )  # torch.Size([1, 128, W/8, H/8])

        if self.debug:
            print(
                f"Input to self.scratch.layer3_rn has shape: {layer_3.shape}"
            )  # torch.Size([1, 136, W/16, H/16])
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.debug:
            print(
                f"Output from self.scratch.layer3_rn has shape: {layer_3_rn.shape}"
            )  # torch.Size([1, 256, W/16, H/16])

        if self.debug:
            print(
                f"Input to self.scratch.layer4_rn has shape: {layer_4.shape}"
            )  # torch.Size([1, 384, W/32, H/32])
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        if self.debug:
            print(
                f"Output from self.scratch.layer4_rn has shape: {layer_4_rn.shape}"
            )  # torch.Size([1, 512, W/32, H/32])

        if self.debug:
            print(
                f"Input to self.scratch.refinenet4 has shape: {layer_4_rn.shape}"
            )  # torch.Size([1, 512, W/32, H/32])
        path_4 = self.scratch.refinenet4(layer_4_rn)
        if self.debug:
            print(f"Output from self.scratch.refinenet4 has shape: {path_4.shape}")

        if self.debug:
            print(
                f"Input to self.scratch.refinenet3 has shape: {path_4.shape}, {layer_3_rn.shape}"
            )  # torch.Size([1, 256, W/16, H/16])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        if self.debug:
            print(f"Output from self.scratch.refinenet3 has shape: {path_3.shape}")

        if self.debug:
            print(
                f"Input to self.scratch.refinenet2 has shape: {path_3.shape}, {layer_2_rn.shape}"
            )  # torch.Size([1, 128, W/8, H/8])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        if self.debug:
            print(f"Output from self.scratch.refinenet2 has shape: {path_2.shape}")

        if self.debug:
            print(
                f"Input to self.scratch.refinenet1 has shape: {path_2.shape}, {layer_1_rn.shape}"
            )  # torch.Size([1, 64, W/4, H/4])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        if self.debug:
            print(f"Output from self.scratch.refinenet1 has shape: {path_1.shape}")

        if self.debug:
            print(
                f"Input to self.scratch.output_conv has shape: {path_1.shape}"
            )  # torch.Size([1, 64, W/4, H/4])
        out = self.scratch.output_conv(path_1)
        if self.debug:
            print(f"Output from self.scratch.output_conv has shape: {out.shape}")

        print("X, Min: ", x.min())
        print("X, Max: ", x.max())
        print("X, Mean: ", x.mean())
        print("X, Std: ", x.std())
        print(f"Output shape: {out.shape}")
        print("Output Min: ", out.min())
        print("Output Max: ", out.max())
        print("Output Mean: ", out.mean())
        print("Output Std: ", out.std())
        return out

    def forward_cIMLE(self, x, latent):
        """Forward pass.

        Args:
            x (tensor): input data (image)
            latent (tensor): latent vector

        Returns:
            tensor: depth
        """
        if self.channels_last == True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_1 = self.style_mod0(
            layer_1, latent, self.style_mod0_meanshift, self.style_mod0_varshift
        )

        layer_2 = self.pretrained.layer2(layer_1)
        layer_2 = self.style_mod1(
            layer_2, latent, self.style_mod1_meanshift, self.style_mod1_varshift
        )

        layer_3 = self.pretrained.layer3(layer_2)
        layer_3 = self.style_mod2(
            layer_3, latent, self.style_mod2_meanshift, self.style_mod2_varshift
        )

        layer_4 = self.pretrained.layer4(layer_3)
        layer_4 = self.style_mod3(
            layer_4, latent, self.style_mod3_meanshift, self.style_mod3_varshift
        )

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return out

    def forward(self, x, latent=None, mean_shift=0.0, var_shift=0.0):
        if latent is None:
            return self.forward_no_CIMLE(x)
        else:
            return self.forward_cIMLE(x, latent)

    def decompose_EffiNet(self):
        self.pretrained.layer1_1 = nn.Sequential(self.pretrained.layer1[0])
        self.pretrained.layer1_2 = nn.Sequential(
            self.pretrained.layer1[1],
            self.pretrained.layer1[2],
            self.pretrained.layer1[3],
            self.pretrained.layer1[4],
        )
        del self.pretrained.layer1

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        self.style_mod0_meanshift = mean0
        self.style_mod0_varshift = var0
        self.style_mod1_meanshift = mean1
        self.style_mod1_varshift = var1
        self.style_mod2_meanshift = mean2
        self.style_mod2_varshift = var2
        self.style_mod3_meanshift = mean3
        self.style_mod3_varshift = var3

    def get_adain_init_act(self, x, z):
        x = self.pretrained.layer1(x)
        x = self.style_mod0(x, z, self.style_mod0_meanshift, self.style_mod0_varshift)
        adain0 = x

        x = self.pretrained.layer2(x)
        x = self.style_mod1(x, z, self.style_mod1_meanshift, self.style_mod1_varshift)
        adain1 = x

        x = self.pretrained.layer3(x)
        x = self.style_mod2(x, z, self.style_mod2_meanshift, self.style_mod2_varshift)
        adain2 = x

        x = self.pretrained.layer4(x)
        x = self.style_mod3(x, z, self.style_mod3_meanshift, self.style_mod3_varshift)
        adain3 = x

        return adain0, adain1, adain2, adain3

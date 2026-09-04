import torch

from .unet import SmallUNet


def get_generator(args, in_c, out_c, use_amp=False, amp_dtype=torch.float16):
    if args.type == "snp-small-unet":
        opt = args.snp_small_unet
        return SmallUNet(
            in_c,
            out_c,
            bilinear=opt.bilinear,
            single=opt.single,
            norm=opt.norm,
            last_act=opt.last_act,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            affine_layer=opt.affine_layer,
            channel_size_multiplier=float(opt.channel_factor),
        )
    else:
        raise NotImplementedError(
            "generator type [{:d}] is not supported".format(args.type)
        )

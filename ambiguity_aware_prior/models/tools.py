import argparse
import random
from glob import glob
from pathlib import Path

import imageio
import kornia.morphology as kn_morph
import numpy as np
import torch
from chrislib.general import round_32, uninvert
from chrislib.loss import ImageDerivative, resize_aa
from chrislib.resolution_util import optimal_resize
from intrinsic.ordinal_util import base_resize, equalize_predictions
from PIL import Image
from skimage.transform import resize
from torch.optim import Adam
from .midas_net_small import MidasNet_small
from altered_midas.midas_net import MidasNet

STAGE = 1


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batchsize",
        type=int,
        default=16,
        help="training batch size",  # 16 ~ 12GB, each step ~ 0.2 sec
    )
    parser.add_argument("--base_lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
        "--clip",
        type=int,
        default=15,
        help="maximum shading value (larger values will be masked)",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of dataloader workers"
    )
    parser.add_argument("--epochs", type=int, default=5000, help="number of epochs")
    parser.add_argument(
        "--train_iters",
        type=int,
        default=500,
        help="number of training iterations between evaluations",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device to train on")
    parser.add_argument(
        "--cache_prob",
        type=float,
        default=0.0,
        help="probability of using a cached datapoint when generating a batch",
    )
    parser.add_argument(
        "--cache_length",
        type=int,
        default=300,
        help="max number of cached datapoints to store in RAM at any time",
    )

    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument(
        "--debug_path",
        type=str,
        default="debug.jpg",
        help="path to save debug image while training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="whether or not to run torch.compile for training the model",
    )

    parser.add_argument(
        "--qual_eval_dir",
        type=str,
        default="qual_eval/",
        help="directory of images for qualitative evaluation",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="checkpoint to load in order to resume training",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="saved_weights/",
        help="path to save weights",
    )
    parser.add_argument(
        "--ordinal_path",
        type=str,
        default="",
        help="path to ordinal model, used for qualitative evaluation",
    )
    parser.add_argument(
        "--MID_PATH",
        type=str,
        default=None,
        required=True,
        help="path to the MID dataset",
    )
    parser.add_argument(
        "--HYPERSIM_PATH",
        type=str,
        default=None,
        required=True,
        help="path to the Hypersim dataset",
    )
    parser.add_argument(
        "--is_subset",
        action="store_true",
        help="use a subset of the dataset for faster training",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="name of the experiment, used for saving weights",
    )
    parser.add_argument(
        "--d_latent",
        type=int,
        default=32,
        help="dimension of the latent vector for AdaIN",
    )
    parser.add_argument(
        "--use_IMLE",
        action="store_true",
        help="use the IMLE pretrained model",
    )
    parser.add_argument(
        "--IMLE_num_samples",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--refresh_z",
        default=10,
        type=int,
        help="Number of epochs of when to recache z",
    )
    parser.add_argument("--mlp_lr", default=0.0001, type=float)
    parser.add_argument("--mlp_lr2", default=0.0001, type=float)
    parser.add_argument("--pretrain_mlp", action="store_true")
    parser.add_argument("--pretrain_steps", default=31, type=int)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--only_output_adain_init", action="store_true")
    parser.add_argument("--num_data_for_AdaIN_init", default=0, type=int)
    parser.add_argument("--IMLE_mini_batch_size", type=int, default=20)
    parser.add_argument("--i_print_step", type=int, default=100)
    parser.add_argument("--i_eval_step", type=int, default=100)
    parser.add_argument("--AdaIN_init_batch", type=int, default=1)
    parser.add_argument("--result_dir_on_edge_node", type=str, default="")

    args = parser.parse_args()
    return args


def load_image_for_eval(path):
    img = imageio.imread(path)
    new_w, new_h = img.shape[1], img.shape[0]
    img = Image.fromarray(img).resize((new_w, new_h))
    img = (np.array(img) / 255.0).astype(np.float32)
    # check if it has an alpha channel
    if img.shape[-1] == 4:
        alpha_channel = img[..., 3]
    else:
        alpha_channel = None
    img = img[..., :3]

    return img, alpha_channel


def save_image_for_eval(img, alpha_channel, img_name):
    if alpha_channel is not None:
        img = np.dstack([img, alpha_channel])
    img = Image.fromarray((np.clip(img, 0, 1) * 255.0).astype(np.uint8))
    img.save(img_name)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def tone_map_image(image, gamma=2.2):
    """
    Tone map the image using gamma correction
    image: [B, H, W, C]
    gamma: float
    """
    image[image < 0] = 0
    return image ** (1.0 / gamma)


def run_gray_pipeline(
    models,
    img_arr,
    resize_conf=0.0,
    base_size=384,
    maintain_size=False,
    linear=False,
    device="cuda",
    lstsq_p=0.0,
    inputs="all",
    z_code=None,
):
    """Runs the complete pipeline for grayscale shading and albedo prediction

    params:
        models (dict): models dictionary returned by load_models()
        img_arr (np.array): RGB input image as numpy array between 0-1
        resize_conf (float) optional: confidence to use for resizing (between 0-1) if None maintain
            original size (default None)
        base_size (int) optional: size of the base resolution estimation (default 384)
        maintain_size (bool) optional: whether or not the results match the input image size
            (default False)
        linear (bool) optional: whether or not the input image is already linear (default False)
        device (str) optional: string representing device to use for pipeline (default "cuda")
        lstsq_p (float) optional: subsampling factor for computing least-squares fit
            when matching the scale of base and full estimations (default 0.0)
        inputs (str) optional: network inputs ("full", "base", "rgb", "all") the rgb image is
            always included (default "all")

    returns:
        results (dict): a result dictionary with albedo, shading and potentially ordinal estimations
    """
    results = {}

    orig_h, orig_w, _ = img_arr.shape

    # if no confidence value set, just round original size to 32 for model input
    if resize_conf is None:
        img_arr = resize(
            img_arr, (round_32(orig_h), round_32(orig_w)), anti_aliasing=True
        )

    # if a the confidence is an int, just rescale image so that the large side
    # of the image matches the specified integer value
    elif isinstance(resize_conf, int):
        scale = resize_conf / max(orig_h, orig_w)
        img_arr = resize(
            img_arr,
            (round_32(orig_h * scale), round_32(orig_w * scale)),
            anti_aliasing=True,
        )

    # if the confidence is a float use the optimal resize code from Miangoleh et al.
    elif isinstance(resize_conf, float):
        img_arr = optimal_resize(img_arr, conf=resize_conf)

    fh, fw, _ = img_arr.shape

    # if the image is in sRGB we do simple linearization using gamma=2.2
    if not linear:
        lin_img = img_arr**2.2
    else:
        lin_img = img_arr

    with torch.no_grad():
        # ordinal shading estimation --------------------------

        # resize image for base and full estimations and send through ordinal net
        base_input = base_resize(lin_img, base_size)
        full_input = lin_img

        base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(device).float()
        full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(device).float()

        base_out = models["ord_model"](base_input.unsqueeze(0)).squeeze(0)
        full_out = models["ord_model"](full_input.unsqueeze(0)).squeeze(0)

        # the ordinal estimations come out of the model with a channel dim
        base_out = base_out.permute(1, 2, 0).cpu().numpy()
        full_out = full_out.permute(1, 2, 0).cpu().numpy()

        base_out = resize(base_out, (fh, fw))

        # if we are using all inputs, we scale the input estimations using the base estimate
        if inputs == "all":
            ord_base, ord_full = equalize_predictions(
                lin_img, base_out, full_out, p=lstsq_p
            )
        else:
            ord_base, ord_full = base_out, full_out
        # ------------------------------------------------------

        # ordinal shading to real shading ----------------------
        inp = torch.from_numpy(lin_img).permute(2, 0, 1).to(device)
        ord_base_t = torch.from_numpy(ord_base).permute(2, 0, 1).to(device)
        ord_full_t = torch.from_numpy(ord_full).permute(2, 0, 1).to(device)

        # combine the base and full ordinal estimations w/ the input image
        # NOTE: this is just for ablation studies provided in the paper
        if inputs == "full":
            combined = torch.cat((inp, ord_full_t), 0).unsqueeze(0)
        elif inputs == "base":
            combined = torch.cat((inp, ord_base_t), 0).unsqueeze(0)
        elif inputs == "rgb":
            combined = inp.unsqueeze(0)
        else:
            combined = torch.cat((inp, ord_base_t, ord_full_t), 0).unsqueeze(0)

        inv_shd = models["iid_model"](combined, z_code).squeeze(1)

        # the shading comes out in the inverse space so undo it
        shd = uninvert(inv_shd)
        alb = inp / shd
        # ------------------------------------------------------

    # put all the outputs into a dictionary to return
    inv_shd = inv_shd.squeeze(0).detach().cpu().numpy()
    alb = alb.permute(1, 2, 0).detach().cpu().numpy()

    if maintain_size:
        ord_base = resize(base_out, (orig_h, orig_w), anti_aliasing=True)
        ord_full = resize(full_out, (orig_h, orig_w), anti_aliasing=True)

        inv_shd = resize(inv_shd, (orig_h, orig_w), anti_aliasing=True)
        alb = resize(alb, (orig_h, orig_w), anti_aliasing=True)

    results["ord_full"] = ord_full
    results["ord_base"] = ord_base

    results["gry_shd"] = inv_shd
    results["gry_alb"] = alb
    results["image"] = img_arr
    results["lin_img"] = lin_img

    return results


def run_pipeline(
    models,
    img_arr,
    stage=4,
    resize_conf=0.0,
    base_size=384,
    maintain_size=True,
    linear=False,
    device="cuda",
    z_code=None,
):

    results = run_gray_pipeline(
        models,
        img_arr,
        resize_conf=resize_conf,
        linear=linear,
        device=device,
        base_size=base_size,
        maintain_size=maintain_size,
        z_code=z_code,
    )

    if stage == 1:
        return results


def qual_eval(
    ord_model, iid_model, args, results_path, epoch, i_step, num_samples_per_image=5
):

    # this function willl run some test images through the pipeline
    # using an existing pre-trained ordinal model. The results are
    # returned as tiled images (albedo, shading) for visualization

    img_paths = glob(f"{args.qual_eval_dir}/*")
    iid_model.eval()

    model_dict = {}
    model_dict["ord_model"] = ord_model
    model_dict["iid_model"] = iid_model
    log_al_images = []
    avg_diff = 0.0
    counter_diff = 0
    diff_maps = []
    for i_eval, path in enumerate(img_paths):
        print("evaluating:", path)
        img_arr, alpha_channel = load_image_for_eval(path)
        image_name = Path(path).stem
        previous_image = None
        for s_index in range(num_samples_per_image):
            z_code = torch.randn(1, args.d_latent).to(args.device)
            result = run_pipeline(
                model_dict,
                img_arr,
                resize_conf=0.0,
                maintain_size=True,
                linear=False,
                device="cuda",
                stage=STAGE,
                z_code=z_code,
            )

            shd = result["gry_shd"]
            shd = uninvert(shd)
            alb = result["gry_alb"]

            shd = tone_map_image(shd)
            alb = tone_map_image(alb)

            if previous_image is not None:
                diff = np.abs(previous_image - alb)
                avg_diff += diff.sum()
                counter_diff += 1
                diff_maps.append(
                    Image.fromarray((np.clip(diff, 0, 1) * 255.0).astype(np.uint8))
                )  # [B, H, W, C] and [0, 1]

            previous_image = alb

            # save debug image
            save_image_for_eval(
                alb,
                alpha_channel=alpha_channel,
                img_name=f"{results_path}/{image_name}_alb_epoch_{epoch}_step_{i_step}_sample_{s_index + 1}.png",
            )
            save_image_for_eval(
                shd,
                alpha_channel=alpha_channel,
                img_name=f"{results_path}/{image_name}_shd_epoch_{epoch}_step_{i_step}_samlpe_{s_index + 1}.png",
            )
            if i_eval == 0:
                alb = np.dstack([alb, alpha_channel])
                log_al_images.append(
                    Image.fromarray((np.clip(alb, 0, 1) * 255.0).astype(np.uint8))
                )
                shd = np.dstack([shd, alpha_channel])

    # tile the qualitative images side by side: 1 row, num_samples_per_image columns
    tiled_alb = np.concatenate(log_al_images, axis=1)
    tiled_diff = np.concatenate(diff_maps, axis=1)
    return {
        "albedo": tiled_alb,
        "avg_diff_images": avg_diff / counter_diff,
        "diff_images": tiled_diff,
    }


def load_cIMLE_model(cIMLE_iid_model_checkpoint):
    model = MidasNet_small(
        activation="sigmoid",
        exportable=False,
        input_channels=5,
        output_channels=1,
        use_cIMLE_pretrained=True,
    )
    ord_model = MidasNet(
        activation="sigmoid",
        input_channels=3,
        output_channels=1,
    )
    # load ordinal model
    try:
        combined_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/compphoto/Intrinsic/releases/download/v1.0/final_weights.pt",
            map_location="cuda",
            progress=True,
        )
    except:
        combined_dict = torch.load(
            "./checkpoints/final_weights.pt", map_location="cuda"
        )
    ord_state_dict = combined_dict["ord_state_dict"]
    ord_model.load_state_dict(ord_state_dict, strict=True)

    # load cIMLE model
    compiled_dict = torch.load(cIMLE_iid_model_checkpoint, map_location="cuda")
    model.load_state_dict(compiled_dict, strict=True)

    model = model.to("cuda")
    ord_model = ord_model.to("cuda")
    model.eval()
    ord_model.eval()

    model_dict = {
        "ord_model": ord_model,
        "iid_model": model,
    }

    return model_dict


def load_model(model, ord_model, args):
    if args.checkpoint != "":
        if "paper_weights" in args.checkpoint or args.checkpoint == "rendered_only":
            if args.checkpoint == "paper_weights-online":
                combined_dict = torch.hub.load_state_dict_from_url(
                    "https://github.com/compphoto/Intrinsic/releases/download/v1.0/final_weights.pt",
                    map_location="cuda",
                    progress=True,
                )
            elif args.checkpoint == "paper_weights-offline":
                combined_dict = torch.load(
                    "./checkpoints/final_weights.pt", map_location=args.device
                )
            elif args.checkpoint == "rendered_only":
                combined_dict = torch.hub.load_state_dict_from_url(
                    "https://github.com/compphoto/Intrinsic/releases/download/v1.0/rendered_only_weights.pt",
                    map_location="cuda",
                    progress=True,
                )
            ord_state_dict = combined_dict["ord_state_dict"]
            ord_model.load_state_dict(ord_state_dict)

            iid_state_dict = combined_dict["iid_state_dict"]
            model.load_state_dict(
                iid_state_dict, strict=False if args.use_IMLE else True
            )
        else:
            compiled_dict = torch.load(f"{args.checkpoint}", map_location=args.device)
            remove_prefix = "_orig_mod."
            model_dict = {
                k[len(remove_prefix) :] if k.startswith(remove_prefix) else k: v
                for k, v in compiled_dict.items()
            }

            model.load_state_dict(model_dict)
            ord_model.load_state_dict(
                torch.load(args.ordinal_path, map_location="cuda")
            )
        return model, ord_model
    else:
        return model, ord_model


##############################################
###### Dataset utils for cIMLE implementation
##############################################
class ModelOptimizer_AdaIn(object):
    def __init__(self, model, base_lr, mlp_lr, fixed_backbone=False):
        super(ModelOptimizer_AdaIn, self).__init__()
        encoder_decoder_params = []
        encoder_decoder_params_names = []
        nograd_param_names = []

        mlp_params = []
        mlp_params_names = []

        for key, value in model.named_parameters():
            if value.requires_grad:
                if "style" in key:
                    mlp_params.append(value)
                    mlp_params_names.append(key)
                else:
                    encoder_decoder_params.append(value)
                    encoder_decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder_decoder = base_lr
        lr_mlp = mlp_lr

        if not fixed_backbone:
            print("Joint backbone.")
            net_params = [
                {
                    "params": encoder_decoder_params,
                    "lr": lr_encoder_decoder,
                },
                {
                    "params": mlp_params,
                    "lr": lr_mlp,
                },
            ]
        else:
            print("Fixed backbone.")
            net_params = [
                {"params": mlp_params, "lr": lr_mlp},
            ]

        self.optimizer = Adam(net_params)
        self.model = model

    def optim(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


def load_mean_var_adain(fname, device):
    input_dict = np.load(fname, allow_pickle=True)

    mean0 = input_dict.item().get("mean0")
    mean1 = input_dict.item().get("mean1")
    mean2 = input_dict.item().get("mean2")
    mean3 = input_dict.item().get("mean3")

    var0 = input_dict.item().get("var0")
    var1 = input_dict.item().get("var1")
    var2 = input_dict.item().get("var2")
    var3 = input_dict.item().get("var3")

    mean0 = torch.from_numpy(mean0).to(device=device)
    mean1 = torch.from_numpy(mean1).to(device=device)
    mean2 = torch.from_numpy(mean2).to(device=device)
    mean3 = torch.from_numpy(mean3).to(device=device)
    var0 = torch.from_numpy(var0).to(device=device)
    var1 = torch.from_numpy(var1).to(device=device)
    var2 = torch.from_numpy(var2).to(device=device)
    var3 = torch.from_numpy(var3).to(device=device)

    return mean0, var0, mean1, var1, mean2, var2, mean3, var3


def lp_loss(pred, grnd, mask, p=2, keepdim=False):
    """Performs a regular LP loss where P is specified. Can be used to
    compute both MSE (p=2) and L1 (p=1) loss functions

    params:
        pred (torch.Tensor): network prediction tensor (B x C x H x W)
        grnd (torch.Tensor): ground truth tensor (B x C x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x 1 x H x W)
        p (int) optional: degree of L norm (default 2)

    returns:
        the mean LP loss between pixels in prediction and ground truth
        if keepdim is True, the loss is returned as a tensor of shape (B,)
        else, the loss is a scalar
    """
    if p == 1:
        lp_term = torch.nn.functional.l1_loss(pred, grnd, reduction="none") * mask
    else:
        lp_term = torch.nn.functional.mse_loss(pred, grnd, reduction="none") * mask

    if keepdim:
        per_batch_loss = lp_term.sum(dim=[1, 2, 3]) / (
            mask.sum(dim=[1, 2, 3]) * lp_term.shape[1]
        )
        return per_batch_loss
    else:
        return lp_term.sum() / (mask.sum() * lp_term.shape[1])


class MSGLoss:
    """Multi-scale Gradient Loss implementation

    params:
        scales (int) optional: TODO (default 4)
        taps (list) optional: TODO (default [1,1,1,1])
        k_size (list) optional: TODO (default [3,3,3,3])
        device (str) optional: TODO (default None)
    """

    def __init__(self, scales=4, taps=[1, 1, 1, 1], k_size=[3, 3, 3, 3], device=None):
        """Create an instance of MSGLoss.

        params:
            scales (int) optional: TODO (default 4)
            taps (list) optional: TODO (default [1,1,1,1])
            k_size (list) optional: TODO (default [3,3,3,3])
            device (str) optional: TODO (default None)
        """
        self.n_scale = scales
        self.taps = taps
        self.k_size = k_size
        self.device = device

        assert (
            len(self.taps) == self.n_scale
        ), "number of scales and number of taps must be the same"
        assert (
            len(self.k_size) == self.n_scale
        ), "number of scales and number of kernels must be the same"

        self.imgDerivative = ImageDerivative()

        self.erod_kernels = [torch.ones(2 * t + 1, 2 * t + 1) for t in self.taps]

        if self.device is not None:
            self.to_device(self.device)

    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            device (str): TODO
        """
        self.imgDerivative.to_device(device)
        self.device = device
        self.erod_kernels = [kernel.to(device) for kernel in self.erod_kernels]

    def __call__(self, output, target, mask=None, keepdim=False):
        """TODO DESCRIPTION

        params:
            output (TODO): TODO
            target (TODO): TODO
            mask (TODO) optional: TODO (default None)

        returns:
            (TODO): TODO
        """
        return self.forward(output, target, mask, keepdim)

    def forward(self, output, target, mask, keepdim=False):
        """TODO DESCRIPTION

        params:
            output is th predicted by the model, shape (B x C x H x W)
            target is the ground truth, shape (B x C x H x W)
            mask is the mask denoting valid pixels, shape (B x 1 x H x W)

        returns:
            scalar loss value if keepdim is False, else a tensor of shape (B,)
        """
        diff = output - target

        if mask is None:
            mask = torch.ones(diff.shape[0], 1, diff.shape[2], diff.shape[3])
            mask = mask.to(self.device)

        if keepdim:
            # To accumulate loss for each batch element
            loss_per_batch = torch.zeros(diff.shape[0], device=self.device)
        else:
            loss = 0
        for i in range(self.n_scale):
            # resize with antialias
            mask_resized = torch.floor(resize_aa(mask, i) + 0.001)

            # erosion to mask out pixels that are effected by unkowns
            mask_resized = kn_morph.erosion(mask_resized, self.erod_kernels[i])
            diff_resized = resize_aa(diff, i)

            # compute grads
            grad_mag = self.gradient_mag(diff_resized, i)

            # mean over channels
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

            # average the per pixel diffs
            temp = mask_resized * grad_mag
            mask_sum = torch.sum(mask_resized, dim=[1, 2, 3])  # Sum over H and W

            # Calculate loss
            if keepdim:
                # Calculate loss per batch element
                loss_per_batch += torch.sum(temp, dim=[1, 2, 3]) / (
                    mask_sum * grad_mag.shape[1] + 1e-8
                )
            else:
                if mask_sum.sum() != 0:
                    loss += torch.sum(temp) / (mask_sum.sum() * grad_mag.shape[1])

        if keepdim:
            # Average across scales for each batch element
            return loss_per_batch / self.n_scale
        else:
            # Average across scales globally
            return loss / self.n_scale

    def gradient_mag(self, diff, scale):
        """TODO DESCRIPTION

        params:
            diff (TODO): TODO
            scale (TODO): TODO

        returns:
            grad_magnitude (TODO): TODO
        """
        # B x C x H x W
        grad_x, grad_y = self.imgDerivative(diff, self.taps[scale])

        # B x C x H x W
        grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-8)

        return grad_magnitude

import os
import random

import imageio
import Imath
import numpy as np
import OpenEXR
import scipy
import torch
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import KDTree
from torch import nn


def update_log_statistics(inp_dict, names, values):
    for name, value in zip(names, values):
        if name not in inp_dict and value is not None:
            inp_dict[f"{name}_min"] = value.min().item()
            inp_dict[f"{name}_max"] = value.max().item()
            inp_dict[f"{name}_mean"] = value.mean().item()
            inp_dict[f"{name}_std"] = value.std().item()
    return inp_dict


trans_t = lambda t: np.asarray(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


rot_phi = lambda phi: np.asarray(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


rot_theta = lambda th: np.asarray(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


rot_beta = lambda th: np.asarray(
    [
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def add_points_knn(
    coords,
    influ_scores,
    add_num,
    k,
    comb_type="mean",
    sample_type="random",
    sample_k=10,
    point_features=None,
    last_coord_grad=None,
    acc_coord_grad=None,
    acc_coord_grad_norm=None,
    grad_cnt=None,
    move_scale=1.0,
    hybrid_weight=0.5,
):
    """
    Add points to the point cloud by kNN
    """
    kdtree = KDTree(coords)
    N = coords.shape[0]

    # Step 1: Determine where to add points
    if N == 0:
        return None, 0, None, None
    if N <= add_num and "random" in comb_type:
        inds = np.random.choice(N, add_num, replace=True)
        query_coords = coords[inds, :]
    elif N <= add_num:
        query_coords = coords
        inds = list(range(N))
    else:
        if sample_type == "random":
            inds = np.random.choice(N, add_num, replace=False)
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-std":
            assert k >= 2
            nns_dists, nns_inds = kdtree.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.std(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-mean":
            assert k >= 2
            nns_dists, nns_inds = kdtree.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.mean(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-max":
            assert k >= 2
            nns_dists, nns_inds = kdtree.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.max(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-min":
            assert k >= 2
            nns_dists, nns_inds = kdtree.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.min(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-max":
            inds = np.argsort(influ_scores.squeeze())[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-min":
            inds = np.argsort(influ_scores.squeeze())[:add_num]
            query_coords = coords[inds, :]
        elif sample_type == "last-coord-grad-max":
            inds = np.argsort(last_coord_grad.abs().sum(-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-max":
            inds = np.argsort(acc_coord_grad.abs().sum(-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-cnt-max":
            inds = np.argsort(acc_coord_grad.abs().sum(-1) / (grad_cnt + 1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max":
            inds = np.argsort(acc_coord_grad_norm)[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-cnt-max":
            inds = np.argsort(acc_coord_grad_norm / (grad_cnt + 1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max-hybrid-top-knn-std":
            inds_a = np.argsort(acc_coord_grad_norm)
            ranks_a = np.zeros_like(inds_a)
            ranks_a[inds_a] = np.arange(len(inds_a))

            assert k >= 2
            nns_dists, nns_inds = kdtree.query(coords, k=sample_k + 1)
            nns_dists = nns_dists[:, 1:]
            inds_b = np.argsort(nns_dists.std(axis=-1))
            ranks_b = np.zeros_like(inds_b)
            ranks_b[inds_b] = np.arange(len(inds_b))

            ranks = hybrid_weight * ranks_a + (1 - hybrid_weight) * ranks_b
            inds = np.argsort(ranks)[-add_num:]
            query_coords = coords[inds, :]
        else:
            raise NotImplementedError

    # Step 2: Add points by kNN
    new_features = None
    if comb_type == "duplicate":
        noise = np.random.randn(3).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        noise *= k
        new_coords = query_coords + noise
        new_influ_scores = influ_scores[inds, :]
        if point_features is not None:
            new_features = point_features[inds, :]
    else:
        nns_dists, nns_inds = kdtree.query(query_coords, k=k + 1)
        nns_dists = nns_dists.astype(np.float32)
        nns_dists = nns_dists[:, 1:]
        nns_inds = nns_inds[:, 1:]
        if comb_type == "mean":
            new_coords = coords[nns_inds, :].mean(axis=-2)  # (Nq, k, 3) -> (Nq, 3)
            new_influ_scores = influ_scores[nns_inds, :].mean(axis=-2)
            if point_features is not None:
                new_features = point_features[nns_inds, :].mean(axis=-2)
        elif comb_type == "random":
            rnd_w = np.random.uniform(0, 1, (query_coords.shape[0], k)).astype(
                np.float32
            )
            rnd_w /= rnd_w.sum(axis=-1, keepdims=True)
            new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            new_influ_scores = (
                influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)
            ).sum(axis=-2)
            if point_features is not None:
                new_features = (
                    point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)
                ).sum(axis=-2)
        elif comb_type == "random-softmax":
            rnd_w = np.random.randn(query_coords.shape[0], k).astype(np.float32)
            rnd_w = scipy.special.softmax(rnd_w, axis=-1)
            new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            new_influ_scores = (
                influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)
            ).sum(axis=-2)
            if point_features is not None:
                new_features = (
                    point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)
                ).sum(axis=-2)
        elif comb_type == "weighted":
            new_coords = (
                coords[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
            ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            new_influ_scores = (
                influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
            ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (
                    point_features[nns_inds, :]
                    * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
                ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        elif comb_type == "along-last-coord-grad":
            new_coords = query_coords + last_coord_grad[inds, :] * move_scale
            nns_dists, nns_inds = kdtree.query(new_coords, k=k)
            nns_dists = nns_dists.astype(np.float32)
            new_influ_scores = (
                influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
            ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (
                    point_features[nns_inds, :]
                    * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
                ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        elif comb_type == "along-acc-coord-grad":
            new_coords = query_coords + acc_coord_grad[inds, :] * move_scale
            nns_dists, nns_inds = kdtree.query(new_coords, k=k)
            nns_dists = nns_dists.astype(np.float32)
            new_influ_scores = (
                influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
            ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (
                    point_features[nns_inds, :]
                    * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)
                ).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        else:
            raise NotImplementedError
    return new_coords, len(new_coords), new_influ_scores, new_features


def cam_to_world(coords, c2w, vector=True):
    """
    coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
    c2w: [N, 4, 4] or [4, 4]
    """
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    if coords.ndim == 5:
        assert c2w.ndim == 2
        B, H, W, N, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 1, 1, 4, 4), -1
        )  # [B, H, W, N, 3]
    elif coords.ndim == 4:
        assert c2w.ndim == 3
        N, H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(N, 1, 1, 4, 4), -1
        )  # [N, H, W, 4]
    elif coords.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 4, 4), -1
        )  # [H, W, 4]
    elif coords.ndim == 2:
        assert c2w.ndim == 2
        K, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 4, 4), -1
        )  # [K, 4]
    else:
        raise ValueError("Wrong dimension of coords")
    return transformed_coords[..., :3]


def world_to_cam(coords, c2w, vector=True):
    """
    coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
    c2w: [N, 4, 4] or [4, 4]
    """
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    w2c = torch.inverse(c2w)
    if coords.ndim == 5:
        assert w2c.ndim == 2
        B, H, W, N, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * w2c.reshape(1, 1, 1, 1, 4, 4), -1
        )  # [B, H, W, N, 3]
    elif coords.ndim == 4:
        assert w2c.ndim == 3
        N, H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * w2c.reshape(N, 1, 1, 4, 4), -1
        )  # [N, H, W, 4]
    elif coords.ndim == 3:
        assert w2c.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * w2c.reshape(1, 1, 4, 4), -1
        )  # [H, W, 4]
    elif coords.ndim == 2:
        assert w2c.ndim == 2
        K, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * w2c.reshape(1, 4, 4), -1
        )  # [K, 4]
    else:
        raise ValueError("Wrong dimension of coords")
    return transformed_coords[..., :3]


def activation_func(
    act_type="leakyrelu",
    neg_slope=0.2,
    inplace=True,
    num_channels=128,
    a=1.0,
    b=1.0,
    trainable=False,
):
    act_type = act_type.lower()
    if act_type == "none":
        layer = nn.Identity()
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_channels)
    elif act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "tanh":
        layer = nn.Tanh()
    elif act_type == "sigmoid":
        layer = nn.Sigmoid()
    elif act_type == "gelu":
        layer = nn.GELU()
    elif "softplus" in act_type:
        a, b, c = [float(i) for i in act_type.split("_")[1:]]
        print("Softplus activation: a={:.2f}, b={:.2f}, c={:.2f}".format(a, b, c))
        layer = SoftplusActivation(a, b, c)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )

    return layer


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


def normalize_vector(x, eps=0.0):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def create_learning_rate_fn(optimizer, max_steps, args, debug=False):
    """Create learning rate schedule."""
    if args.type == "none":
        return None

    if args.warmup > 0:
        warmup_start_factor = 1e-16
    else:
        warmup_start_factor = 1.0

    warmup_fn = lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup,
        verbose=debug,
    )

    if args.type == "linear":
        decay_fn = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max_steps - args.warmup,
            verbose=debug,
        )
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "cosine":
        cosine_steps = max(max_steps - args.warmup, 1)
        decay_fn = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, verbose=debug
        )
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "cosine-stop":
        cosine_steps = max(min(args.stop, max_steps) - args.warmup, 1)
        decay_fn = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, verbose=debug
        )
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "cosine-hlfperiod":
        cosine_steps = max(max_steps - args.warmup, 1) * 2
        decay_fn = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, verbose=debug
        )
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    else:
        raise NotImplementedError

    schedule_fn = lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones, verbose=debug
    )

    return schedule_fn


class SoftplusActivation(nn.Module):
    def __init__(self, c1=1, c2=1, c3=0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def forward(self, x):
        return self.c1 * nn.functional.softplus(self.c2 * x + self.c3)


def extract_features_from_feature_map(features_map, features_dim, side):
    """
    features_map: [B, H,W, C]
    features_dim: int: the size of the slice of the feature map that we need to extract
    side: str: "left" or "right"; it's the location of the starting point of the slice
    """
    features_map_dim = features_map.shape[-1]
    assert features_map_dim >= features_dim
    if side == "right":
        start_index = features_map_dim - features_dim
    elif side == "left":
        start_index = 0
    else:
        raise ValueError("Invalid side")
    end_index = start_index + features_dim
    return features_map[..., start_index:end_index]


def shift_scale_imgage(img, min_val, max_val):
    """
    Shift and scale the image to the range [min_val, max_val]
    image: [B, H, W, C]
    min_val: float
    max_val: float
    """
    img = (img - min_val) / (max_val - min_val)
    return img


def inv_shift_scale_imgage(img, min_val, max_val):
    """
    Inverse shift and scale the image to the range [0, 1]
    image: [B, H, W, C]
    min_val: float
    max_val: float
    """
    img = img * (max_val - min_val) + min_val
    return img


def preprocess_postproces_images_pipeline(
    img,
    pipline,
    eps,
    min_val,
    max_val,
    white_bg_value,
    supervision_scaler=None,
    alpha_channel=None,
    clamp_min=0.0,
    clamp_max=1.0,
):
    eps = float(eps)
    if img is None:
        return None
    if len(pipline) == 0:
        return img
    result = img
    for process in pipline:
        if process == "normalize_after_painting_bg_white":
            result = result / white_bg_value
        elif process == "inv_normalize_after_painting_bg_white":
            result = result * white_bg_value
        elif process == "log+eps":
            if isinstance(result, np.ndarray):
                result = np.log(result + eps)
            else:
                result = torch.log(result + eps)
        elif process == "normalize":
            result = shift_scale_imgage(
                img=result,
                min_val=min_val,
                max_val=max_val,
            )
        elif process == "inv_normalize":
            result = inv_shift_scale_imgage(
                img=result,
                min_val=min_val,
                max_val=max_val,
            )
        elif process == "exp-eps":
            if isinstance(result, np.ndarray):
                result = np.exp(result)
            else:
                result = torch.exp(result)
            result = result - eps
        elif process == "tone_map":
            result = tone_map_image(result)
        elif process == "scale_supervision_fg":
            result = result * supervision_scaler * alpha_channel
        elif process == "clamp":
            if isinstance(result, np.ndarray):
                result = np.clip(result, clamp_min, clamp_max)
            else:
                result = torch.clamp(result, clamp_min, clamp_max)
        elif "white_bg" in process:
            if result.shape[-1] == 4 or alpha_channel is not None:
                if alpha_channel is not None:
                    if isinstance(result, np.ndarray):
                        _alpha_channel = alpha_channel[..., None]
                    else:
                        _alpha_channel = alpha_channel[..., :1]
                    result = (
                        result[..., :3] * _alpha_channel
                        + (1.0 - _alpha_channel) * white_bg_value
                    )
                else:
                    result = (
                        result[..., :3] * result[..., -1:]
                        + (1.0 - result[..., -1:]) * white_bg_value
                    )
            else:
                raise ValueError("The image should have 4 channels")
        else:
            raise ValueError(f"Invalid preprocessing step: {process}")
    return result


def calculate_shading_from_albedo_and_rendered_image(albedo, rendered_img, epsilon):
    copy_albedo = np.copy(albedo)
    mask_albedo = copy_albedo == 0
    copy_albedo[mask_albedo] = epsilon
    shading_channels = rendered_img / copy_albedo
    shading_channels[mask_albedo] = 0
    return shading_channels


def retrieve_raw_from_rgb(rgb_img):
    assert rgb_img.shape[-1] == 3, "The image should have 3 channels"
    # we do inverse gamma correction
    raw_img = invert_tone_map_image(rgb_img)
    return raw_img


def retrieve_rgb_from_raw(raw_img):
    # we do gamma correction
    rgb_img = tone_map_image(raw_img)
    return rgb_img


def read_exr_with_alpha(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exr_file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the color channels and alpha channel as 32-bit floats
    channels = ["R", "G", "B", "A"]  # Include alpha channel
    channel_data = [exr_file.channel(c, pt) for c in channels]

    # Convert the strings to numpy arrays
    channel_arrays = [np.frombuffer(cd, dtype=np.float32) for cd in channel_data]
    for ca in channel_arrays:
        ca.shape = (size[1], size[0])  # Numpy arrays have (row, col) structure

    alpha_channel = channel_arrays.pop()  # Remove the alpha channel from the list
    return (
        channel_arrays,
        alpha_channel,
        size,
    )  # Returns list of numpy arrays for R, G, B, A and the size


def load_resize_normal_image(
    image_path,
    scene_config,
    img_type,
    convert_image_to_raw_space,
    force_convert_image_to_raw_space_white_bg,
    pre_post_processing_steps,
    resize_w=None,
    resize_h=None,
    factor=None,
    constant_bg=1.0,
    alpha_channel=None,
    debug=False,
    force_to_load_alpha_channel=False,
):
    """
    Load a png or raw image file stored as a numpy array
    returns the image as a numpy array, float32, [0,1]
    """
    if scene_config is not None and debug:
        print("*" * 50)
        print(f"Loading image: {image_path}")
        print(f"Image type: {img_type}")
        print(f"Convert image to raw space: {convert_image_to_raw_space}")
        print(
            f"Force convert image to raw space white bg: {force_convert_image_to_raw_space_white_bg}"
        )
        print(f"Background init_scale: {constant_bg}")
        print("*" * 50)
    if img_type == "shading":
        render_raw, _, _, _ = load_resize_normal_image(
            image_path.replace("_shading", ""),
            scene_config=None,
            img_type="render",
            convert_image_to_raw_space=(
                True if ".png" in image_path else convert_image_to_raw_space
            ),
            force_convert_image_to_raw_space_white_bg=force_convert_image_to_raw_space_white_bg,
            pre_post_processing_steps=pre_post_processing_steps,
            debug=debug,
        )
        albedo_raw, _, _, _ = load_resize_normal_image(
            image_path.replace("_shading", "_albedo"),
            scene_config=None,
            img_type="albedo",
            convert_image_to_raw_space=(
                True if ".png" in image_path else convert_image_to_raw_space
            ),
            force_convert_image_to_raw_space_white_bg=force_convert_image_to_raw_space_white_bg,
            pre_post_processing_steps=pre_post_processing_steps,
            debug=debug,
        )
        image = calculate_shading_from_albedo_and_rendered_image(
            albedo=albedo_raw,
            rendered_img=render_raw,
            epsilon=1e-6,
        )
    else:
        # if the image extension is .exr, load it as a numpy array
        if image_path.endswith(".npy"):
            image = np.load(image_path).astype(np.float32)
            if img_type == "render" or force_to_load_alpha_channel:
                if image.shape[-1] == 4:
                    alpha_channel = image[..., -1]
                else:
                    alpha_channel = np.ones_like(image[..., 0])
            image = image[..., :3]
        elif image_path.endswith(".exr"):
            image_channels, image_alpha, _ = read_exr_with_alpha(image_path)
            if img_type == "render" or force_to_load_alpha_channel:
                alpha_channel = image_alpha
            image = np.stack(image_channels, axis=-1)
        elif image_path.endswith(".png"):
            image = imageio.imread(image_path)
            if resize_w is not None and resize_h is not None and factor is not None:
                new_w = resize_w // factor
                new_h = resize_h // factor
            elif resize_w is not None and resize_h is not None and factor is None:
                new_w = resize_w
                new_h = resize_h
            elif resize_w is None and resize_h is None and factor is not None:
                H, W = image.shape[:2]
                new_w = W // factor
                new_h = H // factor
            else:
                H, W = image.shape[:2]
                new_w = W
                new_h = H
            image = Image.fromarray(image).resize((new_w, new_h))
            image = (np.array(image) / 255.0).astype(np.float32)
            if img_type == "render":
                if image.shape[-1] == 4:
                    alpha_channel = image[..., -1]
                else:
                    alpha_channel = np.ones_like(image[..., 0])
            image = image[..., :3]

    if convert_image_to_raw_space:
        print("Converting the image to raw space")
        if force_convert_image_to_raw_space_white_bg:
            image = (
                image[..., :3] * alpha_channel[..., None]
                + (1.0 - alpha_channel[..., None]) * 1
            )
        image = retrieve_raw_from_rgb(image)

    original_H = image.shape[0]
    original_W = image.shape[1]

    # preprocessing steps
    if scene_config is not None and len(pre_post_processing_steps) != 0:
        image = preprocess_postproces_images_pipeline(
            img=image,
            pipline=pre_post_processing_steps,
            eps=scene_config.models.predict_in_log_space_eps,
            min_val=getattr(scene_config.dataset, "min_{}_log".format(img_type), None),
            max_val=getattr(scene_config.dataset, "max_{}_log".format(img_type), None),
            white_bg_value=constant_bg,
            alpha_channel=alpha_channel,
        )

    return image, alpha_channel, original_H, original_W


def cacluate_rgb_from_albedo_and_shading(albedo, shading, scene_config):
    # albedo and shading are in log space and unbounded -> inv_trans -> log_pred_raw -> add together
    log_pred_raw_albedo = inv_shift_scale_imgage(
        img=albedo,
        min_val=scene_config.dataset.min_albedo_log,
        max_val=scene_config.dataset.max_albedo_log,
    )
    log_pred_raw_shading = inv_shift_scale_imgage(
        img=shading,
        min_val=scene_config.dataset.min_shading_log,
        max_val=scene_config.dataset.max_shading_log,
    )
    log_pred_raw_rgb = log_pred_raw_albedo + log_pred_raw_shading
    normal_log_pred_rgb = shift_scale_imgage(
        img=log_pred_raw_rgb,
        min_val=scene_config.dataset.min_render_log,
        max_val=scene_config.dataset.max_render_log,
    )
    return normal_log_pred_rgb


def tone_map_image(image, gamma=2.2):
    """
    Tone map the image using gamma correction
    image: [B, H, W, C]
    gamma: float
    """
    image[image < 0] = 0
    return image ** (1.0 / gamma)


def invert_tone_map_image(image, gamma=2.2):
    """
    Invert tone map the image using gamma correction
    image: [B, H, W, C]
    gamma: float
    """
    return image**gamma


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_training_loss(
    scene_manager,
    render_pred_patch_pred_space,
    render_gt_patch_pred_space,
    albedo_pred_patch_pred_space=None,
    albedo_gt_patch_pred_space=None,
    shading_pred_patch_pred_space=None,
    shading_gt_patch_pred_space=None,
    clip=False,
    log_dictioanry=None,
    add_to_log_dictioanry=False,
    phase="train",
):

    render_pred_patch_rgb_space = None
    render_gt_patch_rgb_space = None
    albedo_pred_patch_rgb_space = None
    albedo_gt_patch_rgb_space = None

    render_loss_patch_pred_space = None
    render_loss_patch_rgb_space = None
    albedo_loss_patch_pred_space = None
    albedo_loss_patch_rgb_space = None
    albedo_loss_patch_pred_space_cIMLE = None
    albedo_loss_patch_rgb_space_cIMLE = None
    # Kept so the return signature of this function is unchanged; the shading
    # branch of the model was removed, so these are always None.
    shading_loss_patch_pred_space = None
    shading_loss_patch_rgb_space = None

    def is_in_space_carving_loss_iters(current_iter, iters_list):
        for start, end in iters_list:
            if start <= current_iter <= end:
                return True
        return False

    def get_space_carving_loss(pred, GTs):

        B, H, W, C = pred.shape
        N_Samples = GTs.shape[1]

        # Expand pred to match the shape of GTs for broadcasting
        pred_expanded = pred.unsqueeze(1)  # Shape: [B, 1, H, W, C]

        # Compute the difference between pred and GTs
        diff = pred_expanded - GTs  # Shape: [B, N_Samples, H, W, C]

        # Compute the L2 norm along the channel dimension
        distances = torch.norm(diff, p=2, dim=-1)  # Shape: [B, N_Samples, H, W]

        # Find the minimum distance over N_Samples for each pixel
        min_distances, _ = torch.min(distances, dim=1)  # Shape: [B, H, W]

        # Compute the total loss by summing over all pixels and batches
        total_loss = min_distances.sum()

        # Average the loss over the number of pixels and batch size
        avg_loss = total_loss / (B * H * W)

        return avg_loss

    def get_pred_and_rgb_losses(
        img_pred_patch_pred_space,
        img_gt_patch_pred_space,
        img_type,
        loss_fn,
        calculate_rgb_space_loss,
        loss_weight,
    ):
        pred_space_loss = None
        rgb_space_loss = None
        img_pred_in_rgb_space = None
        img_gt_in_rgb_space = None

        pred_space_loss = (
            loss_fn(
                (
                    torch.clamp(img_pred_patch_pred_space, 0, 1)
                    if clip
                    else img_pred_patch_pred_space
                ),
                (
                    torch.clamp(img_gt_patch_pred_space, 0, 1)
                    if clip
                    else img_gt_patch_pred_space
                ),
            )
            * loss_weight
        )
        if calculate_rgb_space_loss:
            img_pred_in_rgb_space = preprocess_postproces_images_pipeline(
                img=img_pred_patch_pred_space,
                pipline=scene_manager.scene_config.training.loss_func_preprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset, f"min_{img_type}_log", None
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset, f"max_{img_type}_log", None
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    f"{img_type}_init_scale",
                    None,
                ),
            )
            img_gt_in_rgb_space = preprocess_postproces_images_pipeline(
                img=img_gt_patch_pred_space,
                pipline=scene_manager.scene_config.training.loss_func_preprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset, f"min_{img_type}_log", None
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset, f"max_{img_type}_log", None
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    f"{img_type}_init_scale",
                    None,
                ),
            )
            rgb_space_loss = (
                loss_fn(
                    (
                        torch.clamp(img_pred_in_rgb_space, 0, 1)
                        if clip
                        else img_pred_in_rgb_space
                    ),
                    (
                        torch.clamp(img_gt_in_rgb_space, 0, 1)
                        if clip
                        else img_gt_in_rgb_space
                    ),
                )
                * loss_weight
            )
        return (
            pred_space_loss,
            rgb_space_loss,
            img_pred_in_rgb_space,
            img_gt_in_rgb_space,
        )

    current_iter = scene_manager.step

    if scene_manager.scene_config.models.include_rgb_loss:
        (
            render_loss_patch_pred_space,
            render_loss_patch_rgb_space,
            render_pred_patch_rgb_space,
            render_gt_patch_rgb_space,
        ) = get_pred_and_rgb_losses(
            render_pred_patch_pred_space,
            render_gt_patch_pred_space,
            "render",
            scene_manager.render_loss_fn,
            scene_manager.scene_config.models.include_loss_in_original_space,
            loss_weight=scene_manager.scene_config.models.rgb_loss_weight,
        )
    if scene_manager.scene_config.models.include_albedo_loss:
        if (
            scene_manager.scene_config.training.albedo_space_carving_loss.use
            and is_in_space_carving_loss_iters(
                current_iter,
                scene_manager.scene_config.training.albedo_space_carving_loss.iters,
            )
        ):
            if scene_manager.scene_config.models.include_loss_in_pred_space:
                albedo_loss_patch_pred_space_cIMLE = (
                    get_space_carving_loss(
                        pred=albedo_pred_patch_pred_space,
                        GTs=albedo_gt_patch_pred_space,
                    )
                    * scene_manager.scene_config.models.albedo_loss_weight
                )
            if scene_manager.scene_config.models.include_loss_in_original_space:
                albedo_loss_patch_rgb_space_cIMLE = (
                    get_space_carving_loss(
                        albedo_pred_patch_rgb_space,
                        albedo_gt_patch_rgb_space,
                    )
                    * scene_manager.scene_config.models.albedo_loss_weight
                )
            if scene_manager.args.debug:
                print(
                    "CURRENT ITER: {}, using albedo_space_carving_loss".format(
                        current_iter
                    )
                )
        else:
            (
                albedo_loss_patch_pred_space,
                albedo_loss_patch_rgb_space,
                albedo_pred_patch_rgb_space,
                albedo_gt_patch_rgb_space,
            ) = get_pred_and_rgb_losses(
                albedo_pred_patch_pred_space,
                albedo_gt_patch_pred_space[
                    :, 0, :, :
                ],  # it doesn't matter which samples we use as gt for the original loss
                "albedo",
                scene_manager.albedo_loss_fn,
                scene_manager.scene_config.models.include_loss_in_original_space,
                loss_weight=scene_manager.scene_config.models.albedo_loss_weight,
            )
            if scene_manager.args.debug:
                print("CURRENT ITER: {}, using MSE loss function".format(current_iter))
    else:
        albedo_gt_patch_pred_space = None

    total_loss = 0
    if render_loss_patch_pred_space is not None:
        total_loss += render_loss_patch_pred_space * float(
            scene_manager.scene_config.models.weight_loss_in_pred_space
        )
    if render_loss_patch_rgb_space is not None:
        total_loss += render_loss_patch_rgb_space * float(
            scene_manager.scene_config.models.weight_loss_render_in_original_space
        )
    if albedo_loss_patch_pred_space is not None:
        total_loss += albedo_loss_patch_pred_space * float(
            scene_manager.scene_config.models.weight_loss_in_pred_space
        )
    if albedo_loss_patch_rgb_space is not None:
        total_loss += albedo_loss_patch_rgb_space * float(
            scene_manager.scene_config.models.weight_loss_albedo_in_original_space
        )

    if albedo_loss_patch_pred_space_cIMLE is not None:
        total_loss += albedo_loss_patch_pred_space_cIMLE

    if albedo_loss_patch_rgb_space_cIMLE is not None:
        total_loss += albedo_loss_patch_rgb_space_cIMLE

    if add_to_log_dictioanry:
        log_dictioanry = update_log_statistics(
            inp_dict=log_dictioanry,
            values=[
                render_pred_patch_pred_space,
                render_pred_patch_rgb_space,
                render_gt_patch_pred_space,
                render_gt_patch_rgb_space,
                render_loss_patch_pred_space,
                render_loss_patch_rgb_space,
                albedo_pred_patch_pred_space,
                albedo_pred_patch_rgb_space,
                albedo_gt_patch_pred_space,
                albedo_gt_patch_rgb_space,
                albedo_loss_patch_pred_space,
                albedo_loss_patch_rgb_space,
                albedo_loss_patch_pred_space_cIMLE,
                albedo_loss_patch_rgb_space_cIMLE,
            ],
            names=[
                f"{phase}_render_pred_patch_pred_space",
                f"{phase}_render_pred_patch_rgb_space",
                f"{phase}_render_gt_patch_pred_space",
                f"{phase}_render_gt_patch_rgb_space",
                f"{phase}_render_loss_patch_pred_space_per_step",
                f"{phase}_render_loss_patch_rgb_space_per_step",
                f"{phase}_albedo_pred_patch_pred_space",
                f"{phase}_albedo_pred_patch_rgb_space",
                f"{phase}_albedo_gt_patch_pred_space",
                f"{phase}_albedo_gt_patch_rgb_space",
                f"{phase}_albedo_loss_patch_pred_space_per_step",
                f"{phase}_albedo_loss_patch_rgb_space_per_step",
                f"{phase}_albedo_loss_patch_pred_space_cIMLE",
                f"{phase}_albedo_loss_patch_rgb_space_cIMLE",
            ],
        )

    return (
        total_loss,
        render_loss_patch_pred_space,
        render_loss_patch_rgb_space,
        albedo_loss_patch_pred_space,
        albedo_loss_patch_rgb_space,
        shading_loss_patch_pred_space,
        shading_loss_patch_rgb_space,
        albedo_loss_patch_pred_space_cIMLE,
        albedo_loss_patch_rgb_space_cIMLE,
        log_dictioanry,
    )


def write_a_text_on_image(image, text, font_size=18):
    if text is None:
        return image
    # convert the image to PIL image if it is not
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    # pick a bold font
    font = ImageFont.truetype("FreeMono.ttf", font_size)
    draw.text((30, 10), text, font=font, fill=(0, 0, 0))
    return image


def make_img_bg_transparent(img, white_values=1.0, bg_value=0):
    # img is a numpy array, [H, W, C]
    alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    img = img[:, :, :3]
    mask_r = img[:, :, 0] == white_values
    mask_g = img[:, :, 1] == white_values
    mask_b = img[:, :, 2] == white_values
    mask = np.logical_and(mask_r, np.logical_and(mask_g, mask_b))
    alpha_channel[mask] = 0

    # update the bg value
    img[mask] = bg_value

    return img, alpha_channel


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def radius_func(angle, a, b):
    theta = (angle - (36 - 180)) * np.pi / 180
    return a * b / np.sqrt(a * a * np.sin(theta) ** 2 + b * b * np.cos(theta) ** 2)


def get_render_poses(scene="Barn"):
    stride = 120
    parameters = {
        "Ignatius": [1.7, 1.7, -87.0],
        "Truck": [2.5, 1.5, 91.0],
        "Caterpillar": [2.2, 2.2, -89.0],
        "Family": [0.9, 0.9, -91.0],
        "Barn": [2.5, 2.5, 88.0],
        "Character": [1.2, 1.2, -105.0],
        "Fountain": [1.2, 1.2, -105.0],
        "Jade": [1.2, 1.2, -105.0],
        "maneki": [1, -1, -30],
        "lego": [1, 4, -30],
    }
    factors = {
        "Ignatius": 1.0,
        "Truck": 1.0,
        "Caterpillar": 25.0,
        "Family": 35.0,
        "Barn": 1.0,
        "Character": 1.0,
        "Fountain": 1.0,
        "Jade": 1.0,
        "maneki": 40.0,
        "lego": 10.0,
    }
    a, b, phi = parameters[scene]
    a *= factors[scene]
    b *= factors[scene]

    if scene == "maneki":
        stride = 20
        render_poses = np.stack(
            [
                pose_spherical(90, angle, 40)
                for angle in np.linspace(-70, -100, stride + 1)[:-1]
            ],
            0,
        )
    elif scene == "lego":
        render_poses = np.stack(
            [
                pose_spherical(angle, phi, b)
                for angle in np.linspace(-180, 180, stride + 1)
            ],
            0,
        )
    else:
        render_poses = np.stack(
            [
                pose_spherical(angle, phi, radius_func(angle, a, b))
                for angle in np.linspace(-180, 180, stride + 1)[:-1]
            ],
            0,
        )

    return torch.tensor(render_poses, dtype=torch.float32)

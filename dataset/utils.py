import math

import numpy as np
import torch

from .load_mip360 import load_mip360_data
from .load_nerfsyn import load_blender_data
from .load_t2 import load_t2_data


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
        _, H, W, _ = coords.shape
        N = c2w.shape[0]
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

    c2w = torch.inverse(c2w)
    if coords.ndim == 5:
        assert c2w.ndim == 2
        B, H, W, N, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 1, 1, 4, 4), -1
        )  # [B, H, W, N, 3]
    elif coords.ndim == 4:
        assert c2w.ndim == 3
        _, H, W, _ = coords.shape
        N = c2w.shape[0]
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


def get_rays(H, W, focal_x, focal_y, c2w, fineness=1, coord="world"):
    N = c2w.shape[0]
    width = torch.linspace(
        0, W / focal_x, steps=int(W / fineness) + 1, dtype=torch.float32
    )
    height = torch.linspace(
        0, H / focal_y, steps=int(H / fineness) + 1, dtype=torch.float32
    )
    y, x = torch.meshgrid(height, width)
    pixel_size_x = width[1] - width[0]
    pixel_size_y = height[1] - height[0]
    x = (x - W / focal_x / 2 + pixel_size_x / 2)[:-1, :-1]
    y = -(y - H / focal_y / 2 + pixel_size_y / 2)[:-1, :-1]
    # [H, W, 3], vectors, since the camera is at the origin
    dirs_d = torch.stack([x, y, -torch.ones_like(x)], -1)
    if coord == "world":
        rays_d = cam_to_world(dirs_d.unsqueeze(0), c2w)  # [N, H, W, 3]
        rays_o = c2w[:, :3, -1]  # [N, 3]
    elif coord == "cam":
        rays_d = dirs_d.reshape(1, H, W, 3).repeat(N, 1, 1, 1)  # [N, H, W, 3]
        rays_o = torch.zeros(N, 3, dtype=torch.float32)
    return rays_o, rays_d / torch.norm(rays_d, dim=-1, keepdim=True)


def extract_patches(imgs, albedos, rays_o, rays_d, dataset_args, alpha_channel):
    patch_opt = dataset_args.patches
    N, H, W, C = imgs.shape
    _, num_cIMLE_samples, _, _, _ = albedos.shape

    if patch_opt.type == "continuous":
        num_patches_H = math.ceil(
            (H - patch_opt.overlap) / (patch_opt.height - patch_opt.overlap)
        )
        num_patches_W = math.ceil(
            (W - patch_opt.overlap) / (patch_opt.width - patch_opt.overlap)
        )
        num_patches = num_patches_H * num_patches_W
        rayd_patches = np.zeros(
            (N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32
        )
        rayo_patches = np.zeros((N, num_patches, 3), dtype=np.float32)
        img_patches = np.zeros(
            (N, num_patches, patch_opt.height, patch_opt.width, C), dtype=np.float32
        )
        if albedos is not None:
            albedo_patches = np.zeros(
                (
                    N,
                    num_cIMLE_samples,
                    num_patches,
                    patch_opt.height,
                    patch_opt.width,
                    C,
                ),
                dtype=np.float32,
            )
        else:
            albedo_patches = None

        if alpha_channel is not None:
            alpha_channel_patches = np.zeros(
                (N, num_patches, patch_opt.height, patch_opt.width, 1), dtype=np.float32
            )

        for i in range(N):
            n_patch = 0
            for start_height in range(
                0, H - patch_opt.overlap, patch_opt.height - patch_opt.overlap
            ):
                for start_width in range(
                    0, W - patch_opt.overlap, patch_opt.width - patch_opt.overlap
                ):
                    end_height = min(start_height + patch_opt.height, H)
                    end_width = min(start_width + patch_opt.width, W)
                    start_height = end_height - patch_opt.height
                    start_width = end_width - patch_opt.width
                    rayd_patches[i, n_patch, :, :] = rays_d[
                        i, start_height:end_height, start_width:end_width
                    ]
                    rayo_patches[i, n_patch, :] = rays_o[i, :]
                    img_patches[i, n_patch, :, :] = imgs[
                        i, start_height:end_height, start_width:end_width
                    ]
                    if albedos is not None:
                        albedo_patches[i, :, n_patch, :, :] = albedos[
                            i, :, start_height:end_height, start_width:end_width
                        ]
                    else:
                        albedo_patches = None

                    if alpha_channel is not None:
                        alpha_channel_patches[i, n_patch, :, :] = alpha_channel[
                            i, start_height:end_height, start_width:end_width
                        ]

                    n_patch += 1

    elif patch_opt.type == "random":
        num_patches = patch_opt.max_patches
        rayd_patches = np.zeros(
            (N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32
        )
        rayo_patches = np.zeros((N, num_patches, 3), dtype=np.float32)
        img_patches = np.zeros(
            (N, num_patches, patch_opt.height, patch_opt.width, C), dtype=np.float32
        )
        albedo_patches = np.zeros(
            (N, num_cIMLE_samples, num_patches, patch_opt.height, patch_opt.width, C),
            dtype=np.float32,
        )
        alpha_channel_patches = np.zeros(
            (N, num_patches, patch_opt.height, patch_opt.width, 1), dtype=np.float32
        )

        for i in range(N):
            for n_patch in range(num_patches):
                start_height = np.random.randint(0, H - patch_opt.height)
                start_width = np.random.randint(0, W - patch_opt.width)
                end_height = start_height + patch_opt.height
                end_width = start_width + patch_opt.width
                rayd_patches[i, n_patch, :, :] = rays_d[
                    i, start_height:end_height, start_width:end_width
                ]
                rayo_patches[i, n_patch, :] = rays_o[i, :]
                img_patches[i, n_patch, :, :] = imgs[
                    i, start_height:end_height, start_width:end_width
                ]
                if albedos is not None:
                    albedo_patches[i, :, n_patch, :, :] = albedos[
                        i, :, start_height:end_height, start_width:end_width
                    ]
                else:
                    albedo_patches = None

                if alpha_channel is not None:
                    alpha_channel_patches[i, n_patch, :, :] = alpha_channel[
                        i, start_height:end_height, start_width:end_width
                    ]
                else:
                    alpha_channel_patches = None

    return (
        img_patches,
        albedo_patches,
        rayd_patches,
        rayo_patches,
        num_patches,
        alpha_channel_patches,
    )


def load_meta_data(
    dataset_args,
    scene_config,
    mode,
    use_albedo,
):
    """
    0 -----------> W
    |
    |
    |
    ⬇
    H
    [H, W, 4]
    """
    image_paths = None
    albedo_paths = None

    if dataset_args.type == "synthetic":
        (
            images,
            albedos,
            poses,
            hwf,
            image_paths,
            albedo_paths,
            alpha_channels,
        ) = load_blender_data(
            scene_config=scene_config,
            split=mode,
            use_albedo=use_albedo,
            dataset_args=dataset_args,
        )
        print("Loaded blender", images.shape, hwf, dataset_args.path)

        H, W, focal = hwf
        hwf = [H, W, focal, focal]

    elif dataset_args.type == "t2":
        (
            images,
            albedos,
            poses,
            hwf,
            image_paths,
            albedo_paths,
            alpha_channels,
        ) = load_t2_data(
            scene_config=scene_config,
            split=mode,
            use_albedo=use_albedo,
            dataset_args=dataset_args,
        )
        print("Loaded t2", images.shape, hwf, dataset_args.path)

    elif dataset_args.type == "mip360":
        (
            images,
            albedos,
            poses,
            hwf,
            image_paths,
            albedo_paths,
            alpha_channels,
        ) = load_mip360_data(
            scene_config=scene_config,
            split=mode,
            use_albedo=use_albedo,
            dataset_args=dataset_args,
        )
        print("Loaded mip360", images.shape, hwf, dataset_args.path)

    else:
        raise ValueError("Unknown dataset type: {}".format(dataset_args.type))

    H, W, focal_x, focal_y = hwf

    images = torch.from_numpy(images).float()
    alpha_channels = torch.from_numpy(alpha_channels).float()
    if use_albedo:
        albedos = torch.from_numpy(albedos).float()
    poses = torch.from_numpy(poses).float()

    return (
        images,
        albedos,
        poses,
        H,
        W,
        focal_x,
        focal_y,
        image_paths,
        albedo_paths,
        alpha_channels,
    )


def rgb2norm(img):
    norm_vec = np.stack(
        [
            img[..., 0] * 2.0 / 255.0 - 1.0,
            img[..., 1] * 2.0 / 255.0 - 1.0,
            img[..., 2] * 2.0 / 255.0 - 1.0,
            img[..., 3] / 255.0,
        ],
        axis=-1,
    )
    return norm_vec


def find_proj_coord(pc, c2w, W, focal):
    points_cam = (
        np.linalg.inv(c2w) @ np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=-1).T
    )

    points_cam = points_cam[:3, :].T

    # Project points onto image plane
    points_img = points_cam[:, :2] / points_cam[:, 2:]

    # Convert points to pixel coordinates
    points_px = points_img * np.array([W, W]) / (W / focal)

    points_px[:, 0] += W / 2
    points_px[:, 1] += W / 2
    points_px[:, 0] = W - points_px[:, 0]

    return points_px

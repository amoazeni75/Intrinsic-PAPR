import numpy as np
import torch
from torch.utils.data import Dataset

from models.utils import load_resize_normal_image

from .utils import extract_patches, get_rays, load_meta_data


class RINDataset(Dataset):
    """Ray Image Normal Dataset"""

    def __init__(
        self,
        dataset_args,
        scene_config,
        mode,
        use_albedo,
        debug,
    ):
        self.use_albedo = use_albedo
        if use_albedo:
            print("\033[91m" + "Using albedo as input" + "\033[0m")
        self.dataset_args = dataset_args
        self.scene_config = scene_config
        self.debug = debug
        (
            images,
            albedos,
            c2w,
            H,
            W,
            focal_x,
            focal_y,
            image_paths,
            albedo_paths,
            alpha_channels,
        ) = load_meta_data(
            dataset_args=self.dataset_args,
            scene_config=self.scene_config,
            mode=mode,
            use_albedo=use_albedo,
        )
        num_imgs = len(image_paths)

        if "num_views" in dataset_args and num_imgs > dataset_args.num_views:
            select_inds = np.random.choice(
                num_imgs, dataset_args.num_views, replace=False
            )
            if dataset_args.read_offline:
                images = images[select_inds]
                alpha_channels = alpha_channels[select_inds]
                if use_albedo:
                    albedos = albedos[select_inds]
                c2w = c2w[select_inds]
            image_paths = [image_paths[i] for i in select_inds]
            if use_albedo:
                albedo_paths = [albedo_paths[i] for i in select_inds]
            num_imgs = dataset_args.num_views
            print("Select %d images from %d images" % (num_imgs, len(image_paths)))

        self.num_imgs = num_imgs
        coord_scale = dataset_args.coord_scale
        if coord_scale != 1:
            scaling_matrix = torch.tensor(
                [
                    [coord_scale, 0, 0, 0],
                    [0, coord_scale, 0, 0],
                    [0, 0, coord_scale, 0],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
            c2w = torch.matmul(scaling_matrix, c2w)
        print("c2w: ", c2w.shape)

        focal_x = focal_x / dataset_args.rays.focal_factor
        focal_y = focal_y / dataset_args.rays.focal_factor
        self.H = H
        self.W = W
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.c2w = c2w  # (N, 4, 4)
        self.image_paths = image_paths
        self.albedo_paths = albedo_paths
        self.images = images  # (N, H, W, C) or None
        self.albedos = albedos  # (N, n_samples, H, W, C) or None
        self.mode = mode

        # alpha_channels is (N, H, W) and is a tensor -> reshape to (N, H, W, 1)
        self.alpha_channels = alpha_channels.unsqueeze(-1)

        if self.dataset_args.read_offline:
            rays_o, rays_d = get_rays(
                H, W, focal_x, focal_y, c2w, coord=dataset_args.rays.cam_world
            )
            self.rayd = rays_d  # (N, H, W, 3)
            self.rayo = rays_o  # (N, 3)

        if (
            self.dataset_args.extract_patch == True
            and self.dataset_args.extract_online == False
            and self.dataset_args.read_offline == True
        ):
            (
                img_patches,
                albedo_patches,
                rayd_patches,
                rayo_patches,
                num_patches,
                alpha_channel_patches,
            ) = extract_patches(
                images,
                albedos,
                rays_o,
                rays_d,
                self.dataset_args,
                alpha_channels,
            )
            # (N, n_patches, patch_height, patch_width, C) or None
            self.img_patches = img_patches
            self.albedo_patches = albedo_patches
            # (N, n_patches, patch_height, patch_width, 3)
            self.rayd_patches = rayd_patches
            self.rayo_patches = rayo_patches  # (N, n_patches, 3)
            self.alpha_channel_patches = alpha_channel_patches
            self.num_patches = num_patches

    def _read_image_albedo_from_path(self, image_idx):
        image_path = self.image_paths[image_idx]
        image, alpha_channel, _, _ = load_resize_normal_image(
            image_path=image_path,
            scene_config=self.scene_config,
            img_type="render",
            convert_image_to_raw_space=self.scene_config.dataset.convert_image_to_raw_space,
            force_convert_image_to_raw_space_white_bg=self.scene_config.dataset.force_convert_image_to_raw_space_white_bg,
            resize_w=self.W,
            resize_h=self.H,
            constant_bg=self.scene_config.geoms.background.render_init_scale,
            pre_post_processing_steps=self.dataset_args.render_GT_preprocessing,
        )
        assert "white_bg" in self.dataset_args.render_GT_preprocessing, (
            "white_bg" + " must be in the preprocessing steps for the render image"
        )

        if self.use_albedo:
            albedo_path = self.albedo_paths[image_idx]

            if (
                "albedo_space_carving_loss" in self.scene_config.training
                and self.scene_config.training.albedo_space_carving_loss.use
                and self.mode == "train"
            ):
                albedos = []
                for i_sample in range(
                    self.scene_config.training.albedo_space_carving_loss.num_samples
                ):
                    # we don't know the extension. we need to take out the name and extension, then add _sample_i to it
                    _albedo_path = albedo_path.split(".")
                    albedo_path_sample = (
                        _albedo_path[0]
                        + "_sample_"
                        + str(i_sample)
                        + "."
                        + _albedo_path[1]
                    )
                    albedo, _, _, _ = load_resize_normal_image(
                        image_path=albedo_path_sample,
                        scene_config=self.scene_config,
                        convert_image_to_raw_space=self.scene_config.dataset.convert_image_to_raw_space,
                        force_convert_image_to_raw_space_white_bg=self.scene_config.dataset.force_convert_image_to_raw_space_white_bg,
                        img_type="albedo",
                        resize_w=self.W,
                        resize_h=self.H,
                        constant_bg=self.scene_config.geoms.background.albedo_init_scale,
                        alpha_channel=alpha_channel,
                        pre_post_processing_steps=self.dataset_args.albedo_GT_preprocessing,
                    )
                    albedos.append(albedo)
                # we need to stack the albedos to become (num_samples, H, W, C)
                albedos = np.stack(albedos)
            else:
                albedo, _, _, _ = load_resize_normal_image(
                    image_path=albedo_path,
                    scene_config=self.scene_config,
                    convert_image_to_raw_space=self.scene_config.dataset.convert_image_to_raw_space,
                    force_convert_image_to_raw_space_white_bg=self.scene_config.dataset.force_convert_image_to_raw_space_white_bg,
                    img_type="albedo",
                    resize_w=self.W,
                    resize_h=self.H,
                    constant_bg=self.scene_config.geoms.background.albedo_init_scale,
                    alpha_channel=alpha_channel,
                    pre_post_processing_steps=self.dataset_args.albedo_GT_preprocessing,
                )
                albedos = np.array([albedo])
            assert (
                "white_bg" in self.dataset_args.albedo_GT_preprocessing
                or "white_bg" in self.scene_config.training.GT_albedo_preprocessing
            ), "white_bg must be in the preprocessing steps for the render image"

        else:
            albedos = None

        image = torch.from_numpy(image).float()
        if self.use_albedo:
            albedos = torch.from_numpy(albedos).float()  # (num_samples, H, W, C)
        alpha_channel = (
            torch.from_numpy(alpha_channel).float().reshape(self.H, self.W, 1)
        )

        rayo, rayd = get_rays(
            self.H,
            self.W,
            self.focal_x,
            self.focal_y,
            self.c2w[image_idx : image_idx + 1],
            coord=self.dataset_args.rays.cam_world,
        )

        return image, albedos, rayo, rayd, alpha_channel

    def __len__(self):
        if (
            self.dataset_args.extract_patch == True
            and self.dataset_args.extract_online == False
            and self.dataset_args.read_offline == True
        ):
            return self.num_imgs * self.num_patches
        else:
            return self.num_imgs

    def __getitem__(self, idx):
        if (
            self.dataset_args.extract_patch == True
            and self.dataset_args.extract_online == False
            and self.dataset_args.read_offline == True
        ):
            img_idx = idx // self.num_patches
            patch_idx = idx % self.num_patches
            return (
                img_idx,
                patch_idx,
                (
                    self.img_patches[img_idx, patch_idx]
                    if self.img_patches is not None
                    else 0
                ),
                (
                    self.albedo_patches[img_idx, patch_idx]
                    if self.albedo_patches is not None
                    else 0
                ),
                self.rayd_patches[img_idx, patch_idx],
                self.rayo_patches[img_idx, patch_idx],
                self.alpha_channel_patches[img_idx, patch_idx],
            )

        elif (
            self.dataset_args.extract_patch == True
            and self.dataset_args.extract_online == True
        ):
            img_idx = idx
            dataset_args = self.dataset_args
            self.dataset_args.patches.max_patches = 1
            self.dataset_args.patches.type = "random"
            if self.dataset_args.read_offline:
                (
                    img_patches,
                    albedo_patches,
                    rayd_patches,
                    rayo_patches,
                    _,
                    alpha_channel_patches,
                ) = extract_patches(
                    self.images[img_idx : img_idx + 1],
                    self.albedos[img_idx : img_idx + 1] if self.use_albedo else None,
                    self.rayo[img_idx : img_idx + 1],
                    self.rayd[img_idx : img_idx + 1],
                    dataset_args,
                    self.alpha_channels[img_idx : img_idx + 1],
                )
            else:
                image, albedo, rayo, rayd, alpha_channel = (
                    self._read_image_albedo_from_path(img_idx)
                )
                (
                    img_patches,
                    albedo_patches,
                    rayd_patches,
                    rayo_patches,
                    _,
                    alpha_channel_patches,
                ) = extract_patches(
                    image[None, ...],
                    albedo[None, ...] if self.use_albedo else None,
                    rayo,
                    rayd,
                    dataset_args,
                    alpha_channel[None, ...],
                )

            return (
                img_idx,
                0,
                img_patches[0, 0] if img_patches is not None else 0,
                albedo_patches[0, :, 0] if albedo_patches is not None else 0,
                rayd_patches[0, 0],
                rayo_patches[0, 0],
                alpha_channel_patches[0, 0],
            )
        else:
            if self.dataset_args.read_offline:
                return (
                    idx,
                    0,
                    self.images[idx] if self.images is not None else 0,
                    self.albedos[idx] if self.albedos is not None else 0,
                    self.rayd[idx],
                    self.rayo[idx],
                    self.alpha_channels[idx],
                )
            else:
                image, albedo, rayo, rayd, alpha_channel = (
                    self._read_image_albedo_from_path(idx)
                )
                return (
                    idx,
                    0,
                    image,
                    albedo,
                    rayd.squeeze(0),
                    rayo.squeeze(0),
                    alpha_channel,
                )

    def get_full_img(self, img_idx):
        if self.dataset_args.read_offline:
            return (
                self.images[img_idx].unsqueeze(0) if self.images is not None else None,
                self.albedos[img_idx].unsqueeze(0) if self.use_albedo else None,
                self.rayd[img_idx].unsqueeze(0),
                self.rayo[img_idx].unsqueeze(0),
                self.alpha_channels[img_idx].unsqueeze(0),
            )
        else:
            image, albedo, rayo, rayd, alpha_channel = (
                self._read_image_albedo_from_path(img_idx)
            )
            return (
                image[None, ...],
                albedo[None, ...] if self.use_albedo else None,
                rayd,
                rayo,
                alpha_channel[None, ...],
            )

    def get_c2w(self, img_idx):
        return self.c2w[img_idx]


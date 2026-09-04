import json
import os
import re

import numpy as np

from models.utils import load_resize_normal_image


def _albedo_shading_dir(images_dir, suffix, method):
    """images_4 -> images_4_albedo<method> / images_4_shading<method>."""
    return f"{images_dir}_{suffix}{method}"


def load_mip360_data(
    scene_config,
    split,
    use_albedo,
    dataset_args,
):
    """Load a Mip-NeRF 360 scene in the instant-ngp / COLMAP layout.

    Expected on disk, for a scene directory `basedir` and `dataset.factor` N:

        basedir/transforms.json          camera poses + intrinsics
        basedir/images_N/                the RGB frames
        basedir/images_N_albedo<method>/ albedo maps from the 2D prior
        basedir/images_N_meta/           raw_statistics_eps_<eps>.json

    Mip-NeRF 360 has no train/test split on disk. Every frame is returned for
    every split; which views are held out is decided by the config, matching the
    convention used for the other real-world scenes.
    """
    basedir = dataset_args.path
    with open(os.path.join(basedir, "transforms.json"), "r") as fp:
        meta = json.load(fp)

    original_w = int(meta["w"])
    fl_x = float(meta["fl_x"])
    fl_y = float(meta["fl_y"])

    factor = int(dataset_args.factor) if dataset_args.factor else 1
    images_dir = f"images_{factor}" if factor > 1 else "images"
    method = scene_config.dataset.get(
        "{}_albedo_extraction_method".format(split), ""
    )

    poses, images, alpha_channels, image_paths = [], [], [], []
    albedos, albedo_paths = ([], []) if use_albedo else (None, None)

    for i, frame in enumerate(meta["frames"]):
        rel_path = re.sub(r"images(_\d+)?/", f"{images_dir}/", frame["file_path"])
        img_path = os.path.abspath(os.path.join(basedir, rel_path))
        base_name = os.path.splitext(os.path.basename(rel_path))[0]

        poses.append(np.array(frame["transform_matrix"]))
        image_paths.append(img_path)

        if use_albedo:
            albedo_path = os.path.abspath(
                os.path.join(
                    basedir,
                    _albedo_shading_dir(images_dir, "albedo", method),
                    base_name + ".{}".format(scene_config.dataset.image_file_format),
                )
            )
            albedo_paths.append(albedo_path)

        if not (dataset_args.read_offline or i == 0):
            continue

        img, alpha_channel, _, _ = load_resize_normal_image(
            image_path=img_path,
            scene_config=scene_config,
            img_type="render",
            convert_image_to_raw_space=dataset_args.convert_image_to_raw_space,
            force_convert_image_to_raw_space_white_bg=dataset_args.force_convert_image_to_raw_space_white_bg,
            constant_bg=scene_config.geoms.background.render_init_scale,
            pre_post_processing_steps=dataset_args.render_GT_preprocessing,
        )
        images.append(img)
        # Real captures have no alpha channel; every pixel is foreground.
        if alpha_channel is None:
            alpha_channel = np.ones(img.shape[:2] + (1,), dtype=np.float32)
        alpha_channels.append(alpha_channel)

        if use_albedo:
            alb, _, _, _ = load_resize_normal_image(
                image_path=albedo_path,
                scene_config=scene_config,
                img_type="albedo",
                convert_image_to_raw_space=dataset_args.convert_image_to_raw_space,
                force_convert_image_to_raw_space_white_bg=dataset_args.force_convert_image_to_raw_space_white_bg,
                constant_bg=scene_config.geoms.background.albedo_init_scale,
                alpha_channel=alpha_channel,
                pre_post_processing_steps=dataset_args.albedo_GT_preprocessing,
            )
            albedos.append(np.array([alb]))

    poses = np.array(poses).astype(np.float32)
    images = np.array(images).astype(np.float32)
    alpha_channels = np.array(alpha_channels).astype(np.float32)
    if use_albedo:
        albedos = np.array(albedos).astype(np.float32)

    H, W = images[0].shape[:2]
    # transforms.json records the intrinsics at full resolution; rescale them to
    # whatever images_<factor> actually holds.
    scale = original_w / float(W)
    focal_x = fl_x / scale
    focal_y = fl_y / scale

    return (
        images,
        albedos,
        poses,
        [H, W, focal_x, focal_y],
        image_paths,
        albedo_paths,
        alpha_channels,
    )

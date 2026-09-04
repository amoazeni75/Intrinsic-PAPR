import json
import os

import numpy as np

from models.utils import load_resize_normal_image


def load_blender_data(
    scene_config,
    split,
    use_albedo,
    dataset_args,
):
    basedir = dataset_args.path
    with open(os.path.join(basedir, f"transforms_{split}.json"), "r") as fp:
        meta = json.load(fp)

    print(
        "\033[91m"
        + f"File format we are using is: {scene_config.dataset.image_file_format}"
        + "\033[0m"
    )

    poses = []
    images = []
    alpha_channels = []
    image_paths = []

    if use_albedo:
        albedos = []
        albedo_paths = []
    else:
        albedos = None
        albedo_paths = None

    for i, frame in enumerate(meta["frames"]):
        img_path = os.path.abspath(
            os.path.join(
                basedir,
                frame["file_path"]
                + ".{}".format(scene_config.dataset.image_file_format),
            )
        )
        # albedo path is the same as the img_path with one difference: ../{split}/ -> ../{split}_albedo/
        if use_albedo:
            if "maneki" in img_path:
                _split = "rgb"
            else:
                _split = split
            albedo_path = os.path.abspath(
                os.path.join(
                    basedir,
                    frame["file_path"].replace(
                        f"{_split}/",
                        f"{_split}_albedo{getattr(scene_config.dataset, '{}_albedo_extraction_method'.format(split))}/",
                    )
                    + ".{}".format(scene_config.dataset.image_file_format),
                )
            )
        else:
            albedo_path = None

        poses.append(np.array(frame["transform_matrix"]))
        image_paths.append(img_path)
        if use_albedo:
            albedo_paths.append(albedo_path)

        if dataset_args.read_offline or i == 0:
            img, alpha_channel, _, _ = load_resize_normal_image(
                image_path=img_path,
                scene_config=scene_config,
                img_type="render",
                convert_image_to_raw_space=dataset_args.convert_image_to_raw_space,
                force_convert_image_to_raw_space_white_bg=dataset_args.force_convert_image_to_raw_space_white_bg,
                constant_bg=scene_config.geoms.background.render_init_scale,
                pre_post_processing_steps=dataset_args.render_GT_preprocessing,
            )
            assert "white_bg" in dataset_args.render_GT_preprocessing, (
                "white_bg" + " must be in the preprocessing steps for the render image"
            )
            images.append(img)
            alpha_channels.append(alpha_channel)
            if use_albedo:
                if (
                    scene_config.training.albedo_space_carving_loss.use
                    and split == "train"
                ):
                    albedo_samples = []
                    for i_sample in range(
                        scene_config.training.albedo_space_carving_loss.num_samples
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
                        alb, _, _, _ = load_resize_normal_image(
                            image_path=albedo_path_sample,
                            scene_config=scene_config,
                            convert_image_to_raw_space=dataset_args.convert_image_to_raw_space,
                            force_convert_image_to_raw_space_white_bg=dataset_args.force_convert_image_to_raw_space_white_bg,
                            img_type="albedo",
                            constant_bg=scene_config.geoms.background.albedo_init_scale,
                            alpha_channel=alpha_channel,
                            pre_post_processing_steps=dataset_args.albedo_GT_preprocessing,
                        )
                        albedo_samples.append(alb)
                    # we need to stack the albedos to become (num_samples, H, W, C)
                    albedo_samples = np.stack(albedo_samples)
                else:
                    alb, _, _, _ = load_resize_normal_image(
                        image_path=albedo_path,
                        scene_config=scene_config,
                        convert_image_to_raw_space=dataset_args.convert_image_to_raw_space,
                        force_convert_image_to_raw_space_white_bg=dataset_args.force_convert_image_to_raw_space_white_bg,
                        img_type="albedo",
                        constant_bg=scene_config.geoms.background.albedo_init_scale,
                        alpha_channel=alpha_channel,
                        pre_post_processing_steps=dataset_args.albedo_GT_preprocessing,
                    )
                    albedo_samples = np.array([alb])
            assert (
                "white_bg" in dataset_args.albedo_GT_preprocessing
                or "white_bg" in scene_config.training.GT_albedo_preprocessing
            ), "white_bg must be in the preprocessing steps for the render image"

            albedos.append(albedo_samples)


    poses = np.array(poses).astype(np.float32)
    images = np.array(images).astype(np.float32)
    alpha_channels = np.array(alpha_channels).astype(np.float32)
    if use_albedo:
        albedos = np.array(albedos).astype(np.float32)

    H, W = images[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    return (
        images,
        albedos,
        poses,
        [H, W, focal],
        image_paths,
        albedo_paths,
        alpha_channels,
    )

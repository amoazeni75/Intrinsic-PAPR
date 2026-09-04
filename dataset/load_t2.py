import os

import numpy as np

from models.utils import *

blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_intrinsics(filepath):
    try:
        intrinsic = np.loadtxt(filepath).astype(np.float32)[:3, :3]
        return intrinsic
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, "r") as file:
        f, cx, cy, _ = map(float, file.readline().split())
    fy = fx = f

    # Build the intrinsic matrices
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0, 1]])
    return intrinsic


def load_t2_data(
    scene_config,
    split,
    use_albedo,
    dataset_args,
):
    basedir = dataset_args.path
    assert dataset_args.factor == 1
    colordir = os.path.join(basedir, "rgb")
    albedodir = None

    train_albedo_paths = None
    test_albedo_paths = None

    if use_albedo:
        albedodir = os.path.join(basedir, "rgb_albedo")
    posedir = os.path.join(basedir, "pose")
    train_image_paths = [
        f
        for f in os.listdir(colordir)
        if os.path.isfile(os.path.join(colordir, f))
        and f.startswith("0")
        and f.endswith(scene_config.dataset.image_file_format)
    ]
    if use_albedo:
        train_albedo_paths = [
            f
            for f in os.listdir(albedodir)
            if os.path.isfile(os.path.join(albedodir, f))
            and f.startswith("0")
            and f.endswith(scene_config.dataset.image_file_format)
        ]
    test_image_paths = [
        f
        for f in os.listdir(colordir)
        if os.path.isfile(os.path.join(colordir, f))
        and f.startswith("1")
        and f.endswith(scene_config.dataset.image_file_format)
    ]
    if use_albedo:
        test_albedo_paths = [
            f
            for f in os.listdir(albedodir)
            if os.path.isfile(os.path.join(albedodir, f))
            and f.startswith("1")
            and f.endswith(scene_config.dataset.image_file_format)
        ]

    if split == "train":
        image_paths = train_image_paths
        albedo_paths = train_albedo_paths
    elif split == "test":
        image_paths = test_image_paths
        albedo_paths = test_albedo_paths
    else:
        raise ValueError("Unknown split: {}".format(split))

    image_paths = sorted(image_paths, key=lambda x: int(x.split(".")[0].split("_")[-1]))
    if use_albedo:
        albedo_paths = sorted(
            albedo_paths, key=lambda x: int(x.split(".")[0].split("_")[-1])
        )

    images = []
    alpha_channels = []
    albedos = None
    if use_albedo:
        albedos = []

    poses = []
    out_image_paths = []
    out_albedo_paths = None
    if use_albedo:
        out_albedo_paths = []

    intrinsic = get_intrinsics(os.path.join(basedir, "intrinsics.txt"))
    fx, _, cx = intrinsic[0]
    _, fy, cy = intrinsic[1]

    for i, img_path in enumerate(image_paths):
        image_path = os.path.abspath(os.path.join(colordir, img_path))
        out_image_paths.append(image_path)
        if use_albedo:
            out_albedo_paths.append(
                os.path.join(
                    f"{albedodir}{getattr(scene_config.dataset, '{}_albedo_extraction_method'.format(split), None)}",
                    img_path,
                )
            )

        if dataset_args.read_offline or i == 0:
            render_img, alpha_channel, _, _ = load_resize_normal_image(
                image_path=os.path.join(colordir, img_path),
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
            images.append(render_img)
            alpha_channels.append(alpha_channel)
            if use_albedo:
                if (
                    "albedo_space_carving_loss" in scene_config.training
                    and scene_config.training.albedo_space_carving_loss.use
                    and split == "train"
                ):
                    albedo_samples = []
                    for i_sample in range(
                        scene_config.training.albedo_space_carving_loss.num_samples
                    ):
                        # we don't know the extension. we need to take out the name and extension, then add _sample_i to it
                        _albedo_path = img_path.split(".")
                        _albedo_path = (
                            _albedo_path[0]
                            + "_sample_"
                            + str(i_sample)
                            + "."
                            + _albedo_path[1]
                        )
                        albedo_load_path_sample = os.path.join(
                            f"{albedodir}{getattr(scene_config.dataset, '{}_albedo_extraction_method'.format(split), None)}",
                            _albedo_path,
                        )
                        alb, _, _, _ = load_resize_normal_image(
                            image_path=albedo_load_path_sample,
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
                    albedo_load_path = out_albedo_paths[i]
                    alb, _, _, _ = load_resize_normal_image(
                        image_path=albedo_load_path,
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


        pose_path = os.path.join(
            posedir,
            img_path.replace(f".{scene_config.dataset.image_file_format}", ".txt"),
        )
        pose = np.loadtxt(pose_path).astype(np.float32)
        pose = pose @ blender2opencv
        poses.append(pose)

    images = np.stack(images, 0)
    alpha_channels = np.stack(alpha_channels, 0)
    if use_albedo:
        albedos = np.stack(albedos, 0)
    poses = np.stack(poses, 0)

    realH, realW = images.shape[1:3]
    fx = fx * (realW / scene_config.dataset.original_image_width)
    fy = fy * (realH / scene_config.dataset.original_image_height)

    return (
        images,
        albedos,
        poses,
        [realH, realW, fx, fy],
        out_image_paths,
        out_albedo_paths,
        alpha_channels,
    )

"""Reading the per-scene dataset statistics written by the extraction step.

The statistics record the min and max of each image type in log space, which
the normalisation stages of the image pipelines need.
"""

import json
import os


def load_dataset_statistics(scene_config, dataset_path):
    eps = float(scene_config["models"]["predict_in_log_space_eps"])
    eps = "{:.0e}".format(eps)
    image_space = "rgb" if scene_config["models"]["predict_rgb_in_log_space"] else "raw"
    extra_prefix = ""
    if scene_config["dataset"]["convert_image_to_raw_space"]:
        if scene_config["dataset"]["force_convert_image_to_raw_space_white_bg"]:
            extra_prefix = "reconstructed_white_bg_"
        else:
            extra_prefix = "reconstructed_transparent_bg_"
        # print with red color we are using reconstructed images
        print(
            "The reconstructed RAW image statistics are used for the dataset statistics."
            ""
        )
    # Mip-NeRF 360 scenes keep their frames in a resolution subdirectory, so the
    # statistics sit next to it as <scene>/images_<factor>_meta rather than
    # <scene>_meta like the object-centric and Tanks & Temples layouts.
    if scene_config["dataset"].get("type") == "mip360":
        factor = int(scene_config["dataset"].get("factor", 1) or 1)
        images_dir = f"images_{factor}" if factor > 1 else "images"
        meta_dir = os.path.join(dataset_path, images_dir + "_meta")
    else:
        meta_dir = dataset_path + "_meta"

    stat_file = os.path.join(
        meta_dir,
        f"{extra_prefix}{image_space}_statistics_eps_{eps}{scene_config['dataset']['train_albedo_extraction_method']}.json",
    )

    if not os.path.exists(stat_file):
        raise FileNotFoundError(
            f"Dataset statistics not found at {stat_file}.\n"
            "Run the albedo/shading extraction step for this scene first; it writes "
            "raw_statistics_eps_<eps>.json into the scene's _meta directory."
        )

    # return the statistics
    with open(stat_file, "r") as f:
        stats = json.load(f)
    return stats

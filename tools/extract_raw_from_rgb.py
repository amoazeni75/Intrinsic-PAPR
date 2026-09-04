import json
import math
import os
import sys

import numpy as np
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import *

# log shift and scale min-max
min_render_log = math.inf
max_render_log = -math.inf
min_albedo_log = math.inf
max_albedo_log = -math.inf


def save_statistics(dataset_root, eps, force_white_bg):
    global min_render_log, max_render_log
    global min_albedo_log, max_albedo_log
    # we need to save the statistics to a json file
    res = {
        "min_render_log": float(min_render_log),
        "max_render_log": float(max_render_log),
        "min_albedo_log": float(min_albedo_log),
        "max_albedo_log": float(max_albedo_log),
    }
    eps = f"{eps:.0e}"
    extra_name = "_white_bg" if force_white_bg else "_transparent_bg"
    with open(
        f"{dataset_root}_meta/reconstructed{extra_name}_raw_statistics_eps_{eps}.json",
        "w",
    ) as f:
        json.dump(res, f)


def load_pngs_and_calculate_raw(dataset_root, db_parts, force_white_bg, num_views, eps):
    global min_render_log, max_render_log
    global min_albedo_log, max_albedo_log

    for part in db_parts:
        for i in tqdm.tqdm(range(num_views), desc="Progress"):
            rgb_render_image, _, _, _ = load_resize_normal_image(
                image_path=os.path.join(dataset_root, part, f"r_{i}.png"),
                scene_config=None,
                img_type="render",
                pre_post_processing_steps=["white_bg"] if force_white_bg else None,
            )
            rgb_albedo_image, _, _, _ = load_resize_normal_image(
                image_path=os.path.join(dataset_root, f"{part}_albedo", f"r_{i}.png"),
                scene_config=None,
                img_type="albedo",
                pre_post_processing_steps=["white_bg"] if force_white_bg else None,
            )
            # calculate the raw render and albedo
            raw_render = retrieve_raw_from_rgb(rgb_render_image)
            raw_albedo = retrieve_raw_from_rgb(rgb_albedo_image)

            # calculate the log
            log_raw_render = np.log(raw_render + eps)
            log_raw_albedo = np.log(raw_albedo + eps)

            # update the min-max
            min_render_log = min(min_render_log, log_raw_render.min())
            max_render_log = max(max_render_log, log_raw_render.max())
            min_albedo_log = min(min_albedo_log, log_raw_albedo.min())
            max_albedo_log = max(max_albedo_log, log_raw_albedo.max())


def main():
    global min_render_log, max_render_log
    global min_albedo_log, max_albedo_log

    num_views = 100
    eps = 1e-3
    force_white_bg = False

    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "./data/nerf_synthetic/lego"
    db_parts = ["train", "test"]

    load_pngs_and_calculate_raw(
        dataset_root=dataset_root,
        db_parts=db_parts,
        force_white_bg=force_white_bg,
        num_views=num_views,
        eps=eps,
    )

    save_statistics(dataset_root, eps, force_white_bg)


if __name__ == "__main__":
    main()

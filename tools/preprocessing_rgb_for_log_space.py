import json
import math
import os
import sys

import numpy as np
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.utils import load_resize_normal_image, shift_scale_imgage

# log shift and scale
min_rgb_render_log = np.array([math.inf, math.inf, math.inf])
max_rgb_render_log = np.array([-math.inf, -math.inf, -math.inf])

min_rgb_albedo_log = np.array([math.inf, math.inf, math.inf])
max_rgb_albedo_log = np.array([-math.inf, -math.inf, -math.inf])


def update_statistics(
    log_rendered_channels,
    log_albedo_channels,
):
    global min_rgb_render_log, max_rgb_render_log
    global min_rgb_albedo_log, max_rgb_albedo_log

    for i in range(3):
        min_rgb_render_log[i] = min(
            min_rgb_render_log[i], np.min(log_rendered_channels[:, :, i])
        )
        max_rgb_render_log[i] = max(
            max_rgb_render_log[i], np.max(log_rendered_channels[:, :, i])
        )

        min_rgb_albedo_log[i] = min(
            min_rgb_albedo_log[i], np.min(log_albedo_channels[:, :, i])
        )
        max_rgb_albedo_log[i] = max(
            max_rgb_albedo_log[i], np.max(log_albedo_channels[:, :, i])
        )


def compare_print_statistics(
    rendered_images, log_rendered_images, albedo_images, log_albedo_images
):
    normalized_log_rendered_images = shift_scale_imgage(
        img=log_rendered_images,
        min_val=np.min(min_rgb_render_log),
        max_val=np.max(max_rgb_render_log),
    )
    normalized_log_albedo_images = shift_scale_imgage(
        img=log_albedo_images,
        min_val=np.min(min_rgb_albedo_log),
        max_val=np.max(max_rgb_albedo_log),
    )

    print("stats")
    print(
        "min_r_render_log: {:.6f}, min_g_render_log: {:.6f}, min_b_render_log: {:.6f}, min_render_log: {:.6f}".format(
            min_rgb_render_log[0],
            min_rgb_render_log[1],
            min_rgb_render_log[2],
            np.min(min_rgb_render_log),
        )
    )
    print(
        "max_r_render_log: {:.6f}, max_g_render_log: {:.6f}, max_b_render_log: {:.6f}, max_render_log: {:.6f}".format(
            max_rgb_render_log[0],
            max_rgb_render_log[1],
            max_rgb_render_log[2],
            np.max(max_rgb_render_log),
        )
    )

    print(
        "min_r_albedo_log: {:.6f}, min_g_albedo_log: {:.6f}, min_b_albedo_log: {:.6f}, min_albedo_log: {:.6f}".format(
            min_rgb_albedo_log[0],
            min_rgb_albedo_log[1],
            min_rgb_albedo_log[2],
            np.min(min_rgb_albedo_log),
        )
    )
    print(
        "max_r_albedo_log: {:.6f}, max_g_albedo_log: {:.6f}, max_b_albedo_log: {:.6f}, max_albedo_log: {:.6f}".format(
            max_rgb_albedo_log[0],
            max_rgb_albedo_log[1],
            max_rgb_albedo_log[2],
            np.max(max_rgb_albedo_log),
        )
    )

    print(
        "RGB rendered images: min:{:.6f}, max:{:.6f}, mean_red:{:.6f}, mean_green:{:.6f}, mean_blue:{:.6f}, std_red:{:.6f}, std_green:{:.6f}, std_blue:{:.6f}".format(
            np.min(rendered_images),
            np.max(rendered_images),
            np.mean(rendered_images[:, :, :, 0]),
            np.mean(rendered_images[:, :, :, 1]),
            np.mean(rendered_images[:, :, :, 2]),
            np.std(rendered_images[:, :, :, 0]),
            np.std(rendered_images[:, :, :, 1]),
            np.std(rendered_images[:, :, :, 2]),
        )
    )
    print(
        "Normalized Log RGB rendered images: min:{:.6f}, max:{:.6f}, mean_red:{:.6f}, mean_green:{:.6f}, mean_blue:{:.6f}, std_red:{:.6f}, std_green:{:.6f}, std_blue:{:.6f}".format(
            np.min(normalized_log_rendered_images),
            np.max(normalized_log_rendered_images),
            np.mean(normalized_log_rendered_images[:, :, :, 0]),
            np.mean(normalized_log_rendered_images[:, :, :, 1]),
            np.mean(normalized_log_rendered_images[:, :, :, 2]),
            np.std(normalized_log_rendered_images[:, :, :, 0]),
            np.std(normalized_log_rendered_images[:, :, :, 1]),
            np.std(normalized_log_rendered_images[:, :, :, 2]),
        )
    )
    print(
        "RGB albedo images: min:{:.6f}, max:{:.6f}, mean_red:{:.6f}, mean_green:{:.6f}, mean_blue:{:.6f}, std_red:{:.6f}, std_green:{:.6f}, std_blue:{:.6f}".format(
            np.min(albedo_images),
            np.max(albedo_images),
            np.mean(albedo_images[:, :, :, 0]),
            np.mean(albedo_images[:, :, :, 1]),
            np.mean(albedo_images[:, :, :, 2]),
            np.std(albedo_images[:, :, :, 0]),
            np.std(albedo_images[:, :, :, 1]),
            np.std(albedo_images[:, :, :, 2]),
        )
    )
    print(
        "Normalized Log RGB albedo images: min:{:.6f}, max:{:.6f}, mean_red:{:.6f}, mean_green:{:.6f}, mean_blue:{:.6f}, std_red:{:.6f}, std_green:{:.6f}, std_blue:{:.6f}".format(
            np.min(normalized_log_albedo_images),
            np.max(normalized_log_albedo_images),
            np.mean(normalized_log_albedo_images[:, :, :, 0]),
            np.mean(normalized_log_albedo_images[:, :, :, 1]),
            np.mean(normalized_log_albedo_images[:, :, :, 2]),
            np.std(normalized_log_albedo_images[:, :, :, 0]),
            np.std(normalized_log_albedo_images[:, :, :, 1]),
            np.std(normalized_log_albedo_images[:, :, :, 2]),
        )
    )


def calculate_statistics(db_root, db_parts, num_views, eps):
    rendered_images = []
    log_rendered_images = []
    albedo_images = []
    log_albedo_images = []
    for part in db_parts:
        print("*" * 50)
        print(f"Processing {part} set")

        for i in tqdm.tqdm(range(num_views), desc="Progress"):
            render_image_rgb, alpha_channel, _, _ = load_resize_normal_image(
                image_path=os.path.join(db_root, f"{part}", f"r_{i}.png"),
                constant_bg=1.0,
                all_args=None,
                img_type="render",
                pre_post_processing_steps=["white_bg"],
            )
            rendered_images.append(render_image_rgb)
            albedo_image_rgb, _, _, _ = load_resize_normal_image(
                image_path=os.path.join(db_root, f"{part}_albedo", f"r_{i}.png"),
                constant_bg=1.0,
                all_args=None,
                img_type="albedo",
                alpha_channel=alpha_channel,
                pre_post_processing_steps=["white_bg"],
            )
            albedo_images.append(albedo_image_rgb)

            # calculate the log of images
            render_image_rgb_log = np.log(render_image_rgb + eps)
            log_rendered_images.append(render_image_rgb_log)
            albedo_image_rgb_log = np.log(albedo_image_rgb + eps)
            log_albedo_images.append(albedo_image_rgb_log)

            # update statistics
            update_statistics(
                log_rendered_channels=render_image_rgb_log,
                log_albedo_channels=albedo_image_rgb_log,
            )

    return (
        np.array(rendered_images),
        np.array(log_rendered_images),
        np.array(albedo_images),
        np.array(log_albedo_images),
    )


def save_statistics(dataset_root, eps):
    global min_rgb_render_log, max_rgb_render_log
    global min_rgb_albedo_log, max_rgb_albedo_log
    # create the directory if it does not exist
    os.makedirs(dataset_root, exist_ok=True)
    # we need to save the statistics to a json file
    res = {
        "min_render_log": float(np.min(min_rgb_render_log)),
        "max_render_log": float(np.max(max_rgb_render_log)),
        "min_albedo_log": float(np.min(min_rgb_albedo_log)),
        "max_albedo_log": float(np.max(max_rgb_albedo_log)),
    }
    with open(f"{dataset_root}/rgb_statistics_eps_{eps:.0e}.json", "w") as f:
        json.dump(res, f)


def main():
    eps = 0.4

    num_views = 100
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "./data/nerf_synthetic/lego"
    save_root = dataset_root.rstrip("/") + "_meta/"
    db_parts = ["train", "test"]

    rendered_images, log_rendered_images, albedo_images, log_albedo_images = (
        calculate_statistics(dataset_root, db_parts, num_views, eps=eps)
    )
    compare_print_statistics(
        rendered_images, log_rendered_images, albedo_images, log_albedo_images
    )
    save_statistics(dataset_root=save_root, eps=eps)


if __name__ == "__main__":
    main()

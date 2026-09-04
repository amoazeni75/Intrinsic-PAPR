import json
import math
import os
import sys


import matplotlib.pyplot as plt
import numpy as np

import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.utils import *

# log shift and scale min-max
min_render_log = math.inf
max_render_log = -math.inf
min_albedo_log = math.inf
max_albedo_log = -math.inf
min_shading_log = math.inf
max_shading_log = -math.inf


def get_k_smallest_values(array, k):
    flat_array = array.flatten()
    flat_array.sort()
    first_min = flat_array[0]
    counter_smallest = 1
    if k == counter_smallest:
        return first_min
    for i in range(1, len(flat_array)):
        if flat_array[i] != first_min:
            counter_smallest += 1
            if k == counter_smallest:
                return flat_array[i]
            else:
                first_min = flat_array[i]


def print_statistics(
    log_raw_render,
    log_raw_albedo,
    log_raw_shading,
    raw_render_image,
    raw_albedo_image,
    raw_shading_images,
    sort=False,
):
    global min_render_log, max_render_log
    global min_albedo_log, max_albedo_log
    global min_shading_log, max_shading_log

    min_render_log = log_raw_render.min()
    min_render = raw_render_image.min()
    max_render_log = log_raw_render.max()
    max_render = raw_render_image.max()
    mean_render_log = log_raw_render.mean()
    mean_render = raw_render_image.mean()
    std_render_log = log_raw_render.std()
    std_render = raw_render_image.std()

    min_albedo_log = log_raw_albedo.min()
    min_albedo = raw_albedo_image.min()
    max_albedo_log = log_raw_albedo.max()
    max_albedo = raw_albedo_image.max()
    mean_albedo_log = log_raw_albedo.mean()
    mean_albedo = raw_albedo_image.mean()
    std_albedo_log = log_raw_albedo.std()
    std_albedo = raw_albedo_image.std()

    min_shading_log = log_raw_shading.min()
    min_shading = raw_shading_images.min()
    max_shading_log = log_raw_shading.max()
    max_shading = raw_shading_images.max()
    mean_shading_log = log_raw_shading.mean()
    mean_shading = raw_shading_images.mean()
    std_shading_log = log_raw_shading.std()
    std_shading = raw_shading_images.std()
    print("*" * 50)
    print("Log Rendered images statistics:")
    print(
        "Min: {:.9f}, Second Min: {}, Third Min: {}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            min_render_log,
            get_k_smallest_values(log_raw_render, 2) if sort else "N/A",
            get_k_smallest_values(log_raw_render, 3) if sort else "N/A",
            max_render_log,
            mean_render_log,
            std_render_log,
        )
    )
    print("Raw Rendered images statistics:")
    print(
        "Min: {:.9f}, Second Min: {}, Third Min: {}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            min_render,
            get_k_smallest_values(raw_render_image, 2) if sort else "N/A",
            get_k_smallest_values(raw_render_image, 3) if sort else "N/A",
            max_render,
            mean_render,
            std_render,
        )
    )

    print("Log Albedo images statistics:")
    print(
        "Min: {:.9f}, Second Min: {}, Third Min: {}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            min_albedo_log,
            get_k_smallest_values(log_raw_albedo, 2) if sort else "N/A",
            get_k_smallest_values(log_raw_albedo, 3) if sort else "N/A",
            max_albedo_log,
            mean_albedo_log,
            std_albedo_log,
        )
    )
    print("Raw Albedo images statistics:")
    print(
        "Min: {:.9f}, Second Min: {}, Third Min: {}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            min_albedo,
            get_k_smallest_values(raw_albedo_image, 2) if sort else "N/A",
            get_k_smallest_values(raw_albedo_image, 3) if sort else "N/A",
            max_albedo,
            mean_albedo,
            std_albedo,
        )
    )

    print("Log Shading images statistics:")
    print(
        "Min: {:.9f}, Second Min: {}, Third Min: {}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            min_shading_log,
            get_k_smallest_values(log_raw_shading, 2) if sort else "N/A",
            get_k_smallest_values(log_raw_shading, 3) if sort else "N/A",
            max_shading_log,
            mean_shading_log,
            std_shading_log,
        )
    )
    print("Raw Shading images statistics:")
    print(
        "Min: {:.9f}, Second Min: {}, Third Min: {}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            min_shading,
            get_k_smallest_values(raw_shading_images, 2) if sort else "N/A",
            get_k_smallest_values(raw_shading_images, 3) if sort else "N/A",
            max_shading,
            mean_shading,
            std_shading,
        )
    )


def restore_numpy_normalized_raw_image(
    min_val, max_val, eps, file_data=None, filename=None
):
    if file_data is not None:
        numpy_data = file_data
    elif filename is not None:
        numpy_data = np.load(filename)
    else:
        raise ValueError("Invalid input for restoring the numpy image.")

    result = preprocess_postproces_images_pipeline(
        img=numpy_data,
        min_val=min_val,
        max_val=max_val,
        eps=eps,
        pipline=["inv_normalize", "exp-eps", "tone_map", "clamp"],
        white_bg_value=1.0,
    )

    # PIL image instance
    final_result = Image.fromarray((result * 255).astype("uint8"))

    return final_result  # PIL image object


def save_as_numpy_with_alpha(
    channels, filename="final_image_with_alpha.npy", is_bg_transparent=True
):
    # We assume that channels values are in the range [0, 1]
    rgba = channels
    assert (
        min(rgba.reshape(-1)) >= 0 and max(rgba.reshape(-1)) <= 1
    ), "Invalid range of values for saving as numpy.: min: {}, max: {}".format(
        min(rgba.reshape(-1)), max(rgba.reshape(-1))
    )
    np.save(filename, rgba)


def load_raw_render_albedo_images(db_root, db_parts, num_views, use_sort=False):
    raw_render_images = []
    raw_albedo_images = []
    for part in db_parts:
        # first we need to check if shading directory exists, if not we create it
        shading_dir = os.path.join(db_root, f"{part}_shading")
        if not os.path.exists(shading_dir):
            os.makedirs(shading_dir)

        print("*" * 50)
        print(f"Processing {part} set")

        for i in tqdm.tqdm(range(num_views), desc="Progress"):
            render_channels, render_alpha, _ = read_exr_with_alpha(
                os.path.join(db_root, part, f"r_{i}.exr")
            )
            albedo_channels, _, _ = read_exr_with_alpha(
                os.path.join(db_root, f"{part}_albedo", f"r_{i}.exr")
            )
            raw_render_images.append(np.dstack(render_channels))
            raw_albedo_images.append(np.dstack(albedo_channels))

            # remove the numpy and png files if they exist
            if os.path.exists(os.path.join(db_root, part, f"r_{i}.npy")):
                os.remove(os.path.join(db_root, part, f"r_{i}.npy"))
            if os.path.exists(os.path.join(db_root, part, f"r_{i}.png")):
                os.remove(os.path.join(db_root, part, f"r_{i}.png"))
            if os.path.exists(os.path.join(db_root, f"{part}_albedo", f"r_{i}.npy")):
                os.remove(os.path.join(db_root, f"{part}_albedo", f"r_{i}.npy"))
            if os.path.exists(os.path.join(db_root, f"{part}_albedo", f"r_{i}.png")):
                os.remove(os.path.join(db_root, f"{part}_albedo", f"r_{i}.png"))
            if os.path.exists(os.path.join(db_root, f"{part}_shading", f"r_{i}.npy")):
                os.remove(os.path.join(db_root, f"{part}_shading", f"r_{i}.npy"))
            if os.path.exists(os.path.join(db_root, f"{part}_shading", f"r_{i}.png")):
                os.remove(os.path.join(db_root, f"{part}_shading", f"r_{i}.png"))

    raw_render_images = np.array(raw_render_images)
    raw_albedo_images = np.array(raw_albedo_images)

    print("Raw rendered images statistics:")
    print(
        "Min: {}, Second Min: {}, Third Min: {}, Max: {}, Mean: {}, Std: {}".format(
            np.min(raw_render_images),
            get_k_smallest_values(raw_render_images, 2) if use_sort else "N/A",
            get_k_smallest_values(raw_render_images, 3) if use_sort else "N/A",
            np.max(raw_render_images),
            np.mean(raw_render_images),
            np.std(raw_render_images),
        )
    )
    print("Raw albedo images statistics:")
    print(
        "Min: {}, Second Min: {}, Third Min: {}, Max: {}, Mean: {}, Std: {}".format(
            np.min(raw_albedo_images),
            get_k_smallest_values(raw_albedo_images, 2) if use_sort else "N/A",
            get_k_smallest_values(raw_albedo_images, 3) if use_sort else "N/A",
            np.max(raw_albedo_images),
            np.mean(raw_albedo_images),
            np.std(raw_albedo_images),
        )
    )

    return raw_render_images, raw_albedo_images


def paint_white_background(image, alpha, max_value, is_strict_alpha=False):
    # we need to paint the background with white color which is the max value
    if is_strict_alpha:
        mask = alpha <= 0.5
        image[mask] = max_value
    else:
        # in this case, the color is set to the max value if the alpha is exactly 1
        image = image * alpha + max_value * (1 - alpha)
    return image


def calculate_shadings_for_statistics(
    raw_render_image, raw_albedo_image, eps, use_sort=False
):
    raw_shadings = calculate_shading_from_albedo_and_rendered_image(
        albedo=raw_albedo_image, rendered_img=raw_render_image, epsilon=eps
    )

    print("Shading images statistics:")
    print(
        "Min: {}, Second Min: {}, Third Min: {}, Max: {}, Mean: {}, Std: {}".format(
            np.min(raw_shadings),
            get_k_smallest_values(raw_shadings, 2) if use_sort else "N/A",
            get_k_smallest_values(raw_shadings, 3) if use_sort else "N/A",
            np.max(raw_shadings),
            np.mean(raw_shadings),
            np.std(raw_shadings),
        )
    )

    return raw_shadings


def plot_compare_targets(
    normalized_log_render,
    normalized_log_albedo,
    normalized_log_shading,
    raw_render,
    raw_albedo,
    raw_shading,
):
    # we need to plot the images side by side
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(normalized_log_render)
    axs[0, 0].set_title("Normalized log Rendered Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(normalized_log_albedo)
    axs[0, 1].set_title("Normalized log Albedo Image")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(normalized_log_shading)
    axs[0, 2].set_title("Normalized log Shading Image")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(raw_render)
    axs[1, 0].set_title("Raw Rendered Image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(raw_albedo)
    axs[1, 1].set_title("Raw Albedo Image")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(raw_shading)
    axs[1, 2].set_title("Raw Shading Image")
    axs[1, 2].axis("off")

    plt.savefig("compare_targets.png")


def preprocess_images(
    db_root,
    db_parts,
    num_views,
    render_eps,
    albedo_eps,
    shading_eps,
    is_bg_transparent=True,
    is_strict_alpha=False,
    save_numpy=False,
    save_png=False,
):
    if not save_numpy and not save_png:
        print("Nothing to save, please select at least one option.")
        return
    count = 0
    for part in db_parts:
        # first we need to check if shading directory exists, if not we create it
        shading_dir = os.path.join(db_root, f"{part}_shading")
        if not os.path.exists(shading_dir):
            os.makedirs(shading_dir)

        print("*" * 50)
        print(f"Processing {part} set")

        # use tqdm to show progress bar
        # for i in range(num_views):
        for i in tqdm.tqdm(range(num_views), desc="Progress"):

            # here is the pipeline for each image
            # 1: load the raw rendered and albedo images
            # 2: calculate the raw shading
            # 3: calculate the log of the images
            # 4: background value would be the max of each image type
            # 5: normalize the log of the images
            # 6: do tone mapping on the raw images
            # 7: save the png images and their log as numpy

            # 1: load the raw rendered and albedo images
            raw_render, render_alpha, _ = read_exr_with_alpha(
                os.path.join(db_root, part, f"r_{i}.exr")
            )
            raw_render = np.dstack(raw_render)
            raw_albedo, _, _ = read_exr_with_alpha(
                os.path.join(db_root, f"{part}_albedo", f"r_{i}.exr")
            )
            raw_albedo = np.dstack(raw_albedo)
            # 2: calculate the raw shading
            raw_shading = calculate_shading_from_albedo_and_rendered_image(
                albedo=raw_albedo, rendered_img=raw_render, epsilon=albedo_eps
            )

            # 3: calculate the log of the images
            log_raw_render = np.log(raw_render + render_eps)
            log_raw_albedo = np.log(raw_albedo + albedo_eps)
            log_raw_shading = np.log(raw_shading + shading_eps)

            # 4: background value would be the max of each image type
            if not is_bg_transparent:
                log_raw_render_w_bg = [
                    paint_white_background(
                        image=channel,
                        alpha=render_alpha,
                        max_value=max_render_log,
                        is_strict_alpha=is_strict_alpha,
                    )
                    for channel in log_raw_render
                ]
                log_raw_albedo_w_bg = [
                    paint_white_background(
                        image=channel,
                        alpha=render_alpha,
                        max_value=max_albedo_log,
                        is_strict_alpha=is_strict_alpha,
                    )
                    for channel in log_raw_albedo
                ]
                log_raw_shading_w_bg = [
                    paint_white_background(
                        image=channel,
                        alpha=render_alpha,
                        max_value=max_shading_log,
                        is_strict_alpha=is_strict_alpha,
                    )
                    for channel in log_raw_shading
                ]
                # for the raw images, we will use inv log of the max value
                raw_render = [
                    paint_white_background(
                        image=np.copy(channel),
                        alpha=render_alpha,
                        max_value=np.exp(max_render_log),
                        is_strict_alpha=is_strict_alpha,
                    )
                    for channel in raw_render
                ]
                raw_albedo = [
                    paint_white_background(
                        image=np.copy(channel),
                        alpha=render_alpha,
                        max_value=np.exp(max_albedo_log),
                        is_strict_alpha=is_strict_alpha,
                    )
                    for channel in raw_albedo
                ]
                raw_shading = [
                    paint_white_background(
                        image=np.copy(channel),
                        alpha=render_alpha,
                        max_value=np.exp(max_shading_log),
                        is_strict_alpha=is_strict_alpha,
                    )
                    for channel in raw_shading
                ]
            else:
                log_raw_render_w_bg = log_raw_render
                log_raw_albedo_w_bg = log_raw_albedo
                log_raw_shading_w_bg = log_raw_shading

            # 5: normalize the log of the images
            norm_log_raw_render = shift_scale_imgage(
                img=log_raw_render_w_bg, min_val=min_render_log, max_val=max_render_log
            )
            if is_bg_transparent:
                # norm_log_raw_render is w x h x 3 => we need to add the alpha channel
                norm_log_raw_render = np.dstack(
                    (norm_log_raw_render, render_alpha[:, :, np.newaxis])
                )

            norm_log_raw_albedo = shift_scale_imgage(
                log_raw_albedo_w_bg, min_val=min_albedo_log, max_val=max_albedo_log
            )
            if is_bg_transparent:
                norm_log_raw_albedo = np.dstack(
                    (norm_log_raw_albedo, render_alpha[:, :, np.newaxis])
                )

            norm_log_raw_shading = shift_scale_imgage(
                log_raw_shading_w_bg, min_val=min_shading_log, max_val=max_shading_log
            )
            if is_bg_transparent:
                norm_log_raw_shading = np.dstack(
                    (norm_log_raw_shading, render_alpha[:, :, np.newaxis])
                )

            # 6: saving the norm_log_raw images
            if save_numpy:
                save_as_numpy_with_alpha(
                    norm_log_raw_render,
                    os.path.join(db_root, part, f"r_{i}.npy"),
                    is_bg_transparent=is_bg_transparent,
                )
                save_as_numpy_with_alpha(
                    norm_log_raw_albedo,
                    os.path.join(db_root, f"{part}_albedo", f"r_{i}.npy"),
                    is_bg_transparent=is_bg_transparent,
                )
                save_as_numpy_with_alpha(
                    norm_log_raw_shading,
                    os.path.join(db_root, f"{part}_shading", f"r_{i}.npy"),
                    is_bg_transparent=is_bg_transparent,
                )

            # 7: to save pngs, we will load the numpy, and run the postprocessing pipeline
            if save_png:
                rgb_render = restore_numpy_normalized_raw_image(
                    filename=(
                        os.path.join(db_root, part, f"r_{i}.npy")
                        if save_numpy
                        else None
                    ),
                    file_data=norm_log_raw_render,
                    min_val=min_render_log,
                    max_val=max_render_log,
                    eps=render_eps,
                )
                rgb_render.save(os.path.join(db_root, part, f"r_{i}.png"))
                rgb_albedo = restore_numpy_normalized_raw_image(
                    filename=(
                        os.path.join(db_root, f"{part}_albedo", f"r_{i}.npy")
                        if save_numpy
                        else None
                    ),
                    file_data=norm_log_raw_albedo,
                    min_val=min_albedo_log,
                    max_val=max_albedo_log,
                    eps=albedo_eps,
                )
                rgb_albedo.save(os.path.join(db_root, f"{part}_albedo", f"r_{i}.png"))
                rgb_shading = restore_numpy_normalized_raw_image(
                    filename=(
                        os.path.join(db_root, f"{part}_shading", f"r_{i}.npy")
                        if save_numpy
                        else None
                    ),
                    file_data=norm_log_raw_shading,
                    min_val=min_shading_log,
                    max_val=max_shading_log,
                    eps=shading_eps,
                )
                rgb_shading.save(os.path.join(db_root, f"{part}_shading", f"r_{i}.png"))
                if count == 0:
                    plot_compare_targets(
                        normalized_log_render=norm_log_raw_render,
                        normalized_log_albedo=norm_log_raw_albedo,
                        normalized_log_shading=norm_log_raw_shading,
                        raw_render=rgb_render,
                        raw_albedo=rgb_albedo,
                        raw_shading=rgb_shading,
                    )
                count += 1


def save_statistics(dataset_root, eps):
    # we need to save the statistics to a json file
    res = {
        "min_render_log": float(min_render_log),
        "max_render_log": float(max_render_log),
        "min_albedo_log": float(min_albedo_log),
        "max_albedo_log": float(max_albedo_log),
        "min_shading_log": float(min_shading_log),
        "max_shading_log": float(max_shading_log),
    }
    # Must match scene_manager.load_dataset_statistics: the file lives in the
    # "<scene>_meta" sibling directory and the epsilon is formatted as e.g. 1e-03.
    meta_dir = dataset_root.rstrip("/") + "_meta"
    os.makedirs(meta_dir, exist_ok=True)
    eps_str = "{:.0e}".format(float(eps))
    stats_path = os.path.join(meta_dir, f"raw_statistics_eps_{eps_str}.json")
    with open(stats_path, "w") as f:
        json.dump(res, f)
    print(f"Saved statistics to {stats_path}")


def main():
    global min_render_log, max_render_log
    global min_albedo_log, max_albedo_log
    global min_shading_log, max_shading_log

    num_views = 100
    max_bg_constant = 1
    is_bg_transparent = True
    is_strict_alpha = False
    save_numpy = True
    save_pngs = True
    render_eps = 1e-3
    albedo_eps = 1e-3
    shading_eps = 1e-3
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "./data/nerf_synthetic/lego"
    db_parts = ["train", "test"]

    raw_render_image, raw_albedo_image = load_raw_render_albedo_images(
        db_root=dataset_root, db_parts=db_parts, num_views=num_views
    )
    raw_shading_images = calculate_shadings_for_statistics(
        raw_render_image, raw_albedo_image, shading_eps
    )
    log_raw_render = np.log(raw_render_image + render_eps)
    log_raw_albedo = np.log(raw_albedo_image + albedo_eps)
    log_raw_shading = np.log(raw_shading_images + shading_eps)

    print_statistics(
        log_raw_render,
        log_raw_albedo,
        log_raw_shading,
        raw_render_image,
        raw_albedo_image,
        raw_shading_images,
        sort=False,
    )

    if not is_bg_transparent:
        max_render_log = max(max_render_log, max_shading_log) * max_bg_constant
        max_shading_log = max_render_log
        max_albedo_log = 0.0

    preprocess_images(
        dataset_root,
        db_parts,
        num_views,
        is_bg_transparent=is_bg_transparent,
        is_strict_alpha=is_strict_alpha,
        render_eps=render_eps,
        albedo_eps=albedo_eps,
        shading_eps=shading_eps,
        save_numpy=save_numpy,
        save_png=save_pngs,
    )

    save_statistics(dataset_root=dataset_root + "_meta", eps=render_eps)


if __name__ == "__main__":
    main()

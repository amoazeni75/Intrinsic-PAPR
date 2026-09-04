import io
import json
import os
import sys
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from PIL import Image


def find_all_python_files_and_zip(src_dir, dst_path):
    # find all python files in src_dir
    python_files = []
    for root, _sub_dirs, filenames in os.walk(src_dir):
        if "experiment" in root:
            continue
        for filename in filenames:
            if filename.endswith(".py"):
                python_files.append(os.path.join(root, filename))

    # zip all python files
    with zipfile.ZipFile(dst_path, "w") as zip_file:
        for python_file in python_files:
            zip_file.write(python_file, os.path.relpath(python_file, src_dir))


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")
        ct = datetime.now()
        self.log.write("*" * 50 + "\n" + str(ct) + "\n" + "*" * 50 + "\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_colors(weights):
    num_points = weights.shape[0]
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    colors = np.full((num_points, 3), [1.0, 0.0, 0.0])
    colors[:, 0] *= weights[:num_points]
    colors[:, 2] = 1 - weights[:num_points]
    return colors


def write_metrics(scene_manager, log_dictionary, step):
    """Append one JSON line of scalar metrics to <log_dir>/metrics.jsonl."""
    record = {"step": step}
    for key, value in log_dictionary.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            record[key] = value
        elif isinstance(value, dict):
            record[key] = {
                k: v
                for k, v in value.items()
                if isinstance(v, (int, float, str, bool)) or v is None
            }
    path = os.path.join(scene_manager.scene_log_dir, "metrics.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def add_data_to_metrics(metrics_dict, data, name):
    # we will add min, max, mean, std of the data
    # data is an instance of torch.Tensor or numpy.ndarray
    if isinstance(data, torch.Tensor):
        metrics_dict.update(
            {
                f"{name}_min": data.min().item(),
                f"{name}_max": data.max().item(),
                f"{name}_mean": data.mean().item(),
                f"{name}_std": data.std().item(),
            }
        )
    elif isinstance(data, np.ndarray):
        metrics_dict.update(
            {
                f"{name}_min": data.min(),
                f"{name}_max": data.max(),
                f"{name}_mean": data.mean(),
                f"{name}_std": data.std(),
            }
        )


def get_training_main_plot(
    index,
    step,
    train_tgt_rgb,
    train_tgt_rgb_patch,
    train_pred_rgb_patch,
    test_tgt_rgb,
    test_pred_rgb,
    test_pred_foreground_rgb,
    points_np,
    pt_plot_scale,
    depth_np,
    train_tgt_rgb_raw_space=None,
    train_tgt_rgb_patch_raw_space=None,
    train_pred_rgb_patch_raw_space=None,
    test_tgt_rgb_raw_space=None,
    test_pred_rgb_raw_space=None,
    test_pred_foreground_rgb_raw_space=None,
    train_tgt_albedo=None,
    train_tgt_albedo_raw_space=None,
    train_tgt_albedo_patch=None,
    train_tgt_albedo_patch_raw_space=None,
    train_pred_albedo_patch=None,
    train_pred_albedo_patch_raw_space=None,
    test_tgt_albedo=None,
    test_tgt_albedo_raw_space=None,
    test_pred_albedo=None,
    test_pred_foreground_albedo=None,
    test_pred_albedo_raw_space=None,
    test_pred_foreground_albedo_raw_space=None,
    points_conf_scores_np=None,
    bg_attentions=None,
    bg_masks=None,
    metrics_dict=None,
):
    col_counts = 6
    row_counts = 3
    col_size = 30
    row_size = 30
    pad_size = 0

    if train_tgt_albedo is not None:
        row_counts = 2

    if train_tgt_rgb_raw_space is not None:
        row_counts += 1

    if train_tgt_albedo is not None:
        row_counts += 1
    if train_tgt_albedo_raw_space is not None:
        row_counts += 1

    row_size = (row_counts / 3.0) * row_size

    plot_index = 1

    fig = plt.figure(figsize=(col_size, row_size))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    # 1: train target rgb
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(train_tgt_rgb)
    ax.set_title(f"Iter: {step} tr tgt rgb", pad=pad_size)
    add_data_to_metrics(metrics_dict, train_tgt_rgb, "train_tgt_rgb")

    # 2: train target rgb patch
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(train_tgt_rgb_patch)
    ax.set_title(f"Iter: {step} tr tgt rgb patch", pad=pad_size)
    add_data_to_metrics(metrics_dict, train_tgt_rgb_patch, "train_tgt_rgb_patch")

    # 3: train pred rgb patch
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(train_pred_rgb_patch)
    ax.set_title(f"Iter: {step} tr pred rgb patch", pad=pad_size)
    add_data_to_metrics(metrics_dict, train_pred_rgb_patch, "train_pred_rgb_patch")

    # 4: test target rgb
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(test_tgt_rgb)
    ax.set_title(f"Iter: {step} eval tgt rgb", pad=pad_size)
    add_data_to_metrics(metrics_dict, test_tgt_rgb, "test_tgt_rgb")

    # 5: test pred rgb
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(test_pred_rgb)
    ax.set_title(f"Iter: {step} eval pred rgb", pad=pad_size)
    add_data_to_metrics(metrics_dict, test_pred_rgb, "test_pred_rgb")

    # 6: test pred foreground rgb
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(test_pred_foreground_rgb)
    ax.set_title(f"Iter: {step} eval pred foreground rgb", pad=pad_size)
    add_data_to_metrics(
        metrics_dict, test_pred_foreground_rgb, "test_pred_foreground_rgb"
    )

    # 7: train target rgb raw space
    if train_tgt_rgb_raw_space is not None:
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(train_tgt_rgb_raw_space)
        ax.set_title(f"Iter: {step} tr tgt rgb raw space", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, train_tgt_rgb_raw_space, "train_tgt_rgb_raw_space"
        )

        # 7: train target rgb patch raw space
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(train_tgt_rgb_patch_raw_space)
        ax.set_title(f"Iter: {step} tr tgt rgb patch raw space", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, train_tgt_rgb_patch_raw_space, "train_tgt_rgb_patch_raw_space"
        )

        # 8: train pred rgb patch raw space
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(train_pred_rgb_patch_raw_space)
        ax.set_title(f"Iter: {step} tr pred rgb patch raw space", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, train_pred_rgb_patch_raw_space, "train_pred_rgb_patch_raw_space"
        )

        # 9: test target rgb raw space
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(test_tgt_rgb_raw_space)
        ax.set_title(f"Iter: {step} eval tgt rgb raw space", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, test_tgt_rgb_raw_space, "test_tgt_rgb_raw_space"
        )

        # 10: test pred rgb raw space
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(test_pred_rgb_raw_space)
        ax.set_title(f"Iter: {step} eval pred rgb raw space", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, test_pred_rgb_raw_space, "test_pred_rgb_raw_space"
        )

        # 11: test pred foreground rgb raw space
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(test_pred_foreground_rgb_raw_space)
        ax.set_title(f"Iter: {step} eval pred foreground rgb raw space", pad=pad_size)
        add_data_to_metrics(
            metrics_dict,
            test_pred_foreground_rgb_raw_space,
            "test_pred_foreground_rgb_raw_space",
        )

    if train_tgt_albedo is not None:
        # 6: train target albedo
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(train_tgt_albedo)
        ax.set_title(f"Iter: {step} tr tgt albedo", pad=pad_size)
        add_data_to_metrics(metrics_dict, train_tgt_albedo, "train_tgt_albedo")

        # 7: train target albedo patch
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(train_tgt_albedo_patch)
        ax.set_title(f"Iter: {step} tr tgt albedo patch", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, train_tgt_albedo_patch, "train_tgt_albedo_patch"
        )

        # 8: train pred albedo patch
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(train_pred_albedo_patch)
        ax.set_title(f"Iter: {step} tr pred albedo patch", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, train_pred_albedo_patch, "train_pred_albedo_patch"
        )

        # 9: test target albedo
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(test_tgt_albedo)
        ax.set_title(f"Iter: {step} eval tgt albedo", pad=pad_size)
        add_data_to_metrics(metrics_dict, test_tgt_albedo, "test_tgt_albedo")

        # 10: test pred albedo
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(test_pred_albedo)
        ax.set_title(f"Iter: {step} eval pred albedo", pad=pad_size)
        add_data_to_metrics(metrics_dict, test_pred_albedo, "test_pred_albedo")

        # 11: test pred foreground albedo
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(test_pred_foreground_albedo)
        ax.set_title(f"Iter: {step} eval pred foreground albedo", pad=pad_size)
        add_data_to_metrics(
            metrics_dict, test_pred_foreground_albedo, "test_pred_foreground_albedo"
        )

        if train_tgt_albedo_raw_space is not None:
            # 11: train target albedo raw space
            plot_index += 1
            ax = fig.add_subplot(row_counts, col_counts, plot_index)
            ax.imshow(train_tgt_albedo_raw_space)
            ax.set_title(f"Iter: {step} tr tgt albedo raw space", pad=pad_size)
            add_data_to_metrics(
                metrics_dict, train_tgt_albedo_raw_space, "train_tgt_albedo_raw_space"
            )

            # 12: train target albedo patch raw space
            plot_index += 1
            ax = fig.add_subplot(row_counts, col_counts, plot_index)
            ax.imshow(train_tgt_albedo_patch_raw_space)
            ax.set_title(f"Iter: {step} tr tgt albedo patch raw space", pad=pad_size)
            add_data_to_metrics(
                metrics_dict,
                train_tgt_albedo_patch_raw_space,
                "train_tgt_albedo_patch_raw_space",
            )

            # 13: train pred albedo patch raw space
            plot_index += 1
            ax = fig.add_subplot(row_counts, col_counts, plot_index)
            ax.imshow(train_pred_albedo_patch_raw_space)
            ax.set_title(f"Iter: {step} tr pred albedo patch raw space", pad=pad_size)
            add_data_to_metrics(
                metrics_dict,
                train_pred_albedo_patch_raw_space,
                "train_pred_albedo_patch_raw_space",
            )

            # 14: test target albedo raw space
            plot_index += 1
            ax = fig.add_subplot(row_counts, col_counts, plot_index)
            ax.imshow(test_tgt_albedo_raw_space)
            ax.set_title(f"Iter: {step} eval tgt albedo raw space", pad=pad_size)
            add_data_to_metrics(
                metrics_dict, test_tgt_albedo_raw_space, "test_tgt_albedo_raw_space"
            )

            # 15: test pred albedo raw space
            plot_index += 1
            ax = fig.add_subplot(row_counts, col_counts, plot_index)
            ax.imshow(test_pred_albedo_raw_space)
            ax.set_title(f"Iter: {step} eval pred albedo raw space", pad=pad_size)
            add_data_to_metrics(
                metrics_dict, test_pred_albedo_raw_space, "test_pred_albedo_raw_space"
            )

            # 16: test pred foreground albedo raw space
            plot_index += 1
            ax = fig.add_subplot(row_counts, col_counts, plot_index)
            ax.imshow(test_pred_foreground_albedo_raw_space)
            ax.set_title(
                f"Iter: {step} eval pred foreground albedo raw space", pad=pad_size
            )
            add_data_to_metrics(
                metrics_dict,
                test_pred_foreground_albedo_raw_space,
                "test_pred_foreground_albedo_raw_space",
            )

    # depth map
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    cd = ax.imshow(depth_np)
    add_data_to_metrics(metrics_dict, depth_np, "depth_np")
    fig.colorbar(cd, ax=ax)
    ax.set_title("depth map", pad=pad_size)

    # point cloud
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index, projection="3d")
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    cur_color = "grey"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color)
    ax.set_title("Point Cloud", pad=pad_size)

    # attention
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    add_data_to_metrics(metrics_dict, bg_attentions, "bg_attentions")
    ax.imshow(bg_attentions)
    ax.set_title("bg attentions", pad=pad_size)

    # mask
    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    ax.imshow(bg_masks)
    add_data_to_metrics(metrics_dict, bg_masks, "bg_masks")
    ax.set_title("bg masks", pad=pad_size)

    fig.suptitle(
        "Main Plot\n%s\niter %d\nnum pts: %d" % (index, step, points_np.shape[0])
    )
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    img = Image.open(buffer)
    plt.close()

    return img, metrics_dict


def get_training_pcd_plot(
    index,
    step,
    ro,
    rd,
    points_np,
    coord_scale,
    pt_plot_scale,
    points_conf_scores_np=None,
):
    num_plots = 6 if points_conf_scores_np is not None else 4
    fig = plt.figure(figsize=(5 * num_plots, 6))

    H, W, _ = rd.shape

    ax = fig.add_subplot(1, num_plots, 1, projection="3d")
    ax.view_init(elev=0.0, azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=cur_color,
        s=0.8 * coord_scale,
    )
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(
        ro[0],
        ro[1],
        ro[2],
        rd[H // 2, W // 2, 0],
        rd[H // 2, W // 2, 1],
        rd[H // 2, W // 2, 2],
        length=2,
        alpha=1,
        color="blue",
    )
    ax.set_title("Point Cloud View 1")

    ax = fig.add_subplot(1, num_plots, 2, projection="3d")
    ax.view_init(elev=0.0, azim=180)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=cur_color,
        s=0.8 * coord_scale,
    )
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(
        ro[0],
        ro[1],
        ro[2],
        rd[H // 2, W // 2, 0],
        rd[H // 2, W // 2, 1],
        rd[H // 2, W // 2, 2],
        length=2,
        alpha=1,
        color="blue",
    )
    ax.set_title("Point Cloud View 2")

    ax = fig.add_subplot(1, num_plots, 3, projection="3d")
    ax.view_init(elev=0.0, azim=270)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=cur_color,
        s=0.8 * coord_scale,
    )
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(
        ro[0],
        ro[1],
        ro[2],
        rd[H // 2, W // 2, 0],
        rd[H // 2, W // 2, 1],
        rd[H // 2, W // 2, 2],
        length=2,
        alpha=1,
        color="blue",
    )
    ax.set_title("Point Cloud View 3")

    ax = fig.add_subplot(1, num_plots, 4, projection="3d")
    ax.view_init(elev=89.9, azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=cur_color,
        s=0.8 * coord_scale,
    )
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(
        ro[0],
        ro[1],
        ro[2],
        rd[H // 2, W // 2, 0],
        rd[H // 2, W // 2, 1],
        rd[H // 2, W // 2, 2],
        length=2,
        alpha=1,
        color="blue",
    )
    ax.set_title("Point Cloud View 1 Up")

    if points_conf_scores_np is not None:
        ax = fig.add_subplot(1, num_plots, 5)
        ax.scatter(range(len(points_conf_scores_np)), points_conf_scores_np)
        ax.set_title("Confidence Scores scatter plot")

        ax = fig.add_subplot(1, num_plots, 6)
        bins = np.linspace(-1, 1, 100).tolist()
        ax.hist(points_conf_scores_np, bins=bins)
        ax.set_title("Confidence Scores histogram")

    fig.suptitle("Point Clouds\n%s\niter %d" % (index, step))

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    img = Image.open(buffer)
    plt.close()

    return img


def print_nested_dict(d, step, indent=0):
    """
    Recursively prints a nested dictionary with keys and values on the same line and formatted list values.

    Args:
    d (dict): The dictionary to print.
    indent (int): The current indentation level.
    """
    print("@" * 100)
    print("\033[91m" + f"Step: {step}" + "\033[0m")
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + str(key) + ":")
            print_nested_dict(value, step, indent + 4)
        elif isinstance(value, list) and value:
            # Format list items to 6 decimal places
            formatted_list = ", ".join([f"{v:.6f}" for v in value])
            print(" " * indent + f"{key}: {formatted_list}")
        elif isinstance(value, list) and not value:
            # Don't print empty lists
            print(
                " " * indent + f"{key}: []"
            )  # Optional: can remove this line if you don't want to print empty lists
        else:
            print(" " * indent + f"{key}: {value}")
    print("@" * 100)


def print_log_statistics(
    scene_manager,
    log_dictionary,
    eval_metrics=None,
):
    step = scene_manager.step
    scene_config = scene_manager.scene_config
    model = scene_manager.model
    force_to_print = True if scene_manager.args.debug else False
    # Per-parameter gradient/value statistics are expensive; they are collected only
    # when explicitly requested with --log_gradient_stats.
    if getattr(scene_manager.args, "log_gradient_stats", False):
        gradient_stats = {
            "points_values": {"min": [], "max": [], "mean": [], "std": []},
            "points_grads": {"min": [], "max": [], "mean": [], "std": []},
            "points_conf_scores_values": {"min": [], "max": [], "mean": [], "std": []},
            "points_conf_scores_grads": {"min": [], "max": [], "mean": [], "std": []},
            "pc_feats_values": {"min": [], "max": [], "mean": [], "std": []},
            "pc_feats_grads": {"min": [], "max": [], "mean": [], "std": []},
            "transformer": {
                "embed_k_values": {"min": [], "max": [], "mean": [], "std": []},
                "embed_k_grads": {"min": [], "max": [], "mean": [], "std": []},
                "embed_q_values": {"min": [], "max": [], "mean": [], "std": []},
                "embed_q_grads": {"min": [], "max": [], "mean": [], "std": []},
                "embed_v_1_values": {"min": [], "max": [], "mean": [], "std": []},
                "embed_v_1_grads": {"min": [], "max": [], "mean": [], "std": []},
                "embed_v_2_albedo_values": {
                    "min": [],
                    "max": [],
                    "mean": [],
                    "std": [],
                },
                "embed_v_2_albedo_grads": {"min": [], "max": [], "mean": [], "std": []},
                "blocks_weights_values": {"min": [], "max": [], "mean": [], "std": []},
                "blocks_weights_grads": {"min": [], "max": [], "mean": [], "std": []},
                "blocks_biases_values": {"min": [], "max": [], "mean": [], "std": []},
                "blocks_biases_grads": {"min": [], "max": [], "mean": [], "std": []},
            },
        }
        if scene_config.models.use_renderer:
            gradient_stats["renderer"] = {
                "conv_weights_values": {"min": [], "max": [], "mean": [], "std": []},
                "conv_weights_grads": {"min": [], "max": [], "mean": [], "std": []},
                "conv_biases_values": {"min": [], "max": [], "mean": [], "std": []},
                "conv_biases_grads": {"min": [], "max": [], "mean": [], "std": []},
                "up_weights_values": {"min": [], "max": [], "mean": [], "std": []},
                "up_weights_grads": {"min": [], "max": [], "mean": [], "std": []},
                "up_biases_values": {"min": [], "max": [], "mean": [], "std": []},
                "up_biases_grads": {"min": [], "max": [], "mean": [], "std": []},
            }
        if scene_config.models.use_albedo:
            gradient_stats["albedo_model"] = {
                "conv_weights_values": {"min": [], "max": [], "mean": [], "std": []},
                "conv_weights_grads": {"min": [], "max": [], "mean": [], "std": []},
                "conv_biases_values": {"min": [], "max": [], "mean": [], "std": []},
                "conv_biases_grads": {"min": [], "max": [], "mean": [], "std": []},
                "up_weights_values": {"min": [], "max": [], "mean": [], "std": []},
                "up_weights_grads": {"min": [], "max": [], "mean": [], "std": []},
                "up_biases_values": {"min": [], "max": [], "mean": [], "std": []},
                "up_biases_grads": {"min": [], "max": [], "mean": [], "std": []},
            }
        if model.use_supervision_scaler:
            gradient_stats["supervision_scaler_values"] = {
                "min": [],
                "max": [],
                "mean": [],
                "std": [],
            }
            gradient_stats["supervision_scaler_grads"] = {
                "min": [],
                "max": [],
                "mean": [],
                "std": [],
            }

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                value = param.data
                key = None
                if "renderer" in name:
                    if "weight" in name:
                        key = (
                            "renderer",
                            "conv_weights" if "conv" in name else "up_weights",
                        )
                    elif "bias" in name:
                        key = (
                            "renderer",
                            "conv_biases" if "conv" in name else "up_biases",
                        )
                elif "albedo_model" in name:
                    if "weight" in name:
                        key = (
                            "albedo_model",
                            "conv_weights" if "conv" in name else "up_weights",
                        )
                    elif "bias" in name:
                        key = (
                            "albedo_model",
                            "conv_biases" if "conv" in name else "up_biases",
                        )
                elif "transformer" in name:
                    if "blocks" in name:
                        key = (
                            "transformer",
                            "blocks_weights" if "weight" in name else "blocks_biases",
                        )
                    else:
                        part = name.split(".")[2]  # embed_k, embed_q, etc.
                        key = ("transformer", part)
                else:
                    key = (name,)

                if key:
                    if len(key) == 1:
                        category_values = gradient_stats[key[0] + "_values"]
                        category_grads = gradient_stats[key[0] + "_grads"]
                    else:
                        # we don't know how deep the key is nested, so we need to traverse it
                        category = gradient_stats
                        for subkey in key[:-1]:
                            category = category[subkey]
                        category_values = category[key[-1] + "_values"]
                        category_grads = category[key[-1] + "_grads"]

                    # check if the tensor is not empty
                    if grad.numel() > 0:
                        category_grads["min"].append(grad.min().item())
                        category_grads["max"].append(grad.max().item())
                        category_grads["mean"].append(grad.mean().item())
                        category_grads["std"].append(grad.std().item())

                        category_values["min"].append(value.min().item())
                        category_values["max"].append(value.max().item())
                        category_values["mean"].append(value.mean().item())
                        category_values["std"].append(value.std().item())

        # we need to traverse the dictionary to get the stats, as it can be nested
        flattened_stats = {}

        def flatten_gradient_stats(subdict, parent_key=""):
            for key, value in subdict.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                if isinstance(value, dict):
                    flatten_gradient_stats(value, new_key)
                else:
                    # Assuming the value is a list, aggregate as mean of the list
                    flattened_stats[new_key] = sum(value) / len(value) if value else 0

        flatten_gradient_stats(gradient_stats)

        log_dictionary["scale"] = scene_manager.model.scaler.get_scale()
        log_dictionary.update(flattened_stats)

    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - reserved_memory
    ram_memory = psutil.virtual_memory().used / (1024**3)  # Convert bytes to GB
    log_dictionary.update(
        {
            "GPU_total_memory": total_memory / (1024**3),
            "GPU_reserved_memory": reserved_memory / (1024**3),
            "GPU_allocated_memory": allocated_memory / (1024**3),
            "GPU_free_memory": free_memory / (1024**3),
            "RAM_memory": ram_memory,
        }
    )

    scene_manager.total_train_losses.append(
        scene_manager.avg_total_train_loss / (scene_manager.eval_step_cnt + 1)
    )
    log_dictionary["total_train_losses"] = scene_manager.avg_total_train_loss / (
        scene_manager.eval_step_cnt + 1
    )
    scene_manager.avg_total_train_loss = 0

    phases = ["train"]
    spaces = [
        "pred_space",
        "original_space",
        "pred_space_cIMLE",
        "original_space_cIMLE",
    ]
    image_types = ["render"]
    if scene_manager.scene_config.models.use_albedo:
        image_types.append("albedo")
    for phase in phases:
        for space in spaces:
            for image_type in image_types:
                if (
                    len(getattr(scene_manager, f"{image_type}_losses")[phase][space])
                    != 0
                ):
                    # divide the average loss by the number of steps and then add it to the log dictionary; finally set it to zero
                    log_dictionary[f"{phase}_{image_type}_loss_{space}"] = getattr(
                        scene_manager, f"avg_{image_type}_loss_{space}"
                    ) / (scene_manager.eval_step_cnt + 1)
                    setattr(scene_manager, f"avg_{image_type}_loss_{space}", 0)

    scene_manager.eval_step_cnt = 0
    scene_manager.pt_lrs.append(scene_manager.model.pts_lr)
    log_dictionary["pt_lr"] = scene_manager.model.pts_lr
    scene_manager.tx_lrs.append(scene_manager.model.tx_lr)
    log_dictionary["tx_lr"] = scene_manager.model.tx_lr
    if scene_manager.scene_config.models.use_albedo:
        scene_manager.albedo_lrs.append(scene_manager.model.albedo_lr)
        log_dictionary["albedo_lr"] = scene_manager.model.albedo_lr

    log_dictionary["train_step"] = step

    if eval_metrics is not None:
        log_dictionary.update(eval_metrics)

    # Every scalar metric is appended, one JSON object per line, to metrics.jsonl in
    # the run directory. Non-serialisable entries (image arrays) are skipped.
    write_metrics(scene_manager, log_dictionary, step)

    if force_to_print:
        print_nested_dict(log_dictionary, step=step)

    print("step: {}: logged statistics".format(step))

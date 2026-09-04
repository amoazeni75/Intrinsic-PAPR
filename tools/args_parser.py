import argparse

import yaml
from ruamel.yaml.comments import CommentedSeq


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value


def add_arguments_from_config(parser, config, prefix=""):
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            add_arguments_from_config(parser, value, prefix=full_key)
        else:
            arg_type = type(value)
            if arg_type == type(None):
                arg_type = str
            is_list = arg_type is list
            if is_list:
                arg_type = type(value[0]) if value else str
            if is_list:
                parser.add_argument(
                    f"--{full_key}", type=arg_type, nargs="+", default=None
                )
            else:
                parser.add_argument(f"--{full_key}", type=arg_type, default=None)


def get_config_from_args(config_path):
    # we will open the config file (.yml) inside ./configs/config_name.yml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_train_test_options(args, train_options, debug=False):
    if isinstance(args, argparse.Namespace):
        args = vars(args)  # Convert Namespace to dictionary

    for key, value in args.items():
        _original_key = key
        keys = key.split(".")
        reference = train_options
        for sub_key in keys[:-1]:
            if sub_key in reference:
                reference = reference[sub_key]
            else:
                # print the error message with red color.
                print(
                    "\033[91m{}\033[00m".format("The key {} is not found.".format(sub_key))
                )
        if keys[-1] in reference:
            if value is not None:
                if isinstance(value, list):
                    reference[keys[-1]] = CommentedSeq(value)
                    reference[keys[-1]].fa.set_flow_style()
                else:
                    reference[keys[-1]] = value
                # print with green color.
                print(
                    "\033[92m{}\033[00m".format(
                        "The key {} is updated to {}.".format(_original_key, value)
                    )
                )
            elif debug:
                # print with bold orange color.
                print(
                    "\033[96m{}\033[00m".format(
                        "The key {} is not updated.".format(keys[-1])
                    )
                )
        else:
            # print the error message with red color.
            print(
                "\033[91m{}\033[00m".format("The key {} is not found.".format(keys[-1]))
            )

    return train_options


def parse_args(config):
    parser = argparse.ArgumentParser(description="Override configuration parameters")
    parser.add_argument(
        "--opt", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra information for debugging",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="The id of the gpu to use"
    )
    parser.add_argument(
        "--stage",
        choices=["train", "test"],
        default=None,
        help="The stage to run",
    )

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default=None,
        help=(
            "Override the scene directory used at test time. Needed because "
            "scene_N.test.datasets is a list and cannot be reached by the "
            "generated --scene_N.test.datasets.* flags."
        ),
    )
    parser.add_argument(
        "--log_gradient_stats",
        action="store_true",
        help="Log per-parameter gradient and value statistics at every eval step (slow)",
    )

    # test script
    parser.add_argument(
        "--save_point_cloud",
        action="store_true",
        help="save point cloud",
    )
    parser.add_argument(
        "--test_action",
        choices=[
            "transfer_albedo",
            "transfer_shading",
            "freefrom_transfer_albedo",
            "freefrom_transfer_shading",
            "render",
            "PCA",
            "change_brightness",
            "interpolate_albedo",
            "TSNE",
            "calculate_albedo_consistency",
            "2D_color_interpolation_with_UNet",
            "render_depth_pcd_for_comparison",
        ],
        help="The action to perform",
        default=None,
    )
    parser.add_argument(
        "--use_points_features",
        action="store_true",
        help="Using the points features for transfer",
    )
    parser.set_defaults(use_points_features=True)

    parser.add_argument(
        "--use_pixels_features",
        action="store_true",
        help="Using the pixels features for transfer",
    )
    parser.add_argument(
        "--source_scene_index",
        type=int,
        default=0,
        help="The index of the source scene",
    )
    parser.add_argument(
        "--target_scene_index",
        type=int,
        default=0,
        help="The index of the target scene",
    )
    parser.add_argument(
        "--source_area_indices",
        type=int,
        nargs="+",
        default=None,
        help="The indices of the source area",
    )
    parser.add_argument(
        "--source_area_path",
        type=str,
        default=None,
        help="The indices of the source area",
    )
    parser.add_argument(
        "--target_area_indices",
        type=int,
        nargs="+",
        default=None,
        help="The indices of the target area",
    )
    parser.add_argument(
        "--target_area_path",
        type=str,
        default=None,
        help="The indices of the target area",
    )
    parser.add_argument(
        "--source_target_area_selection_method",
        choices=["points_cloud_areas_boxes", "points_cloud_areas_file", "freeform_pixels"],
        default="freeform_pixels",
        help="The method to select the source and target areas",
    )

    parser.add_argument(
        "--how_many_samples",
        type=int,
        default=1,
        help="The number of samples",
    )
    parser.add_argument(
        "--source_point_index",
        type=int,
        default=None,
        help="The index of the source point",
    )
    parser.add_argument(
        "--use_source_point_index",
        action="store_true",
        help="Use source point index",
    )
    parser.add_argument(
        "--how_many_source_area_points",
        type=int,
        default=-1,
        help="The number of source area points",
    )
    parser.add_argument(
        "--include_time_in_name",
        action="store_true",
        help="Include time in the name",
    )

    # freeform editing
    parser.add_argument(
        "--freeform_source_key_frame_index",
        type=int,
        default=None,
        help="The index of the key frame",
    )
    parser.add_argument(
        "--freeform_source_point_method",
        choices=["all", "highest_attention"],
        default="all",
        help="The method to select the points",
    )
    parser.add_argument(
        "--freeform_target_key_frame_index",
        type=int,
        default=None,
        help="The index of the key frame",
    )
    parser.add_argument(
        "--freeform_target_point_method",
        choices=["all", "highest_attention"],
        default="all",
        help="The method to select the points",
    )

    # 2D color interpolation with UNet
    parser.add_argument(
        "--color_1_feature",
        type=str,
        default=None,
        help="The path to the UNet model",
    )
    parser.add_argument(
        "--color_2_feature",
        type=str,
        default=None,
        help="The path to the UNet model",
    )

    # render specs
    parser.add_argument(
        "--render_frame_type", choices=["onfly", "custom", "all", "range"], default=None
    )
    parser.add_argument(
        "--render_frame_start_index", default=None, help="The frame index", type=int
    )
    parser.add_argument(
        "--render_frame_end_index", default=None, help="The frame index", type=int
    )
    parser.add_argument(
        "--custom_frames",
        type=str,
        default="",
    )
    parser.add_argument(
        "--media_type",
        choices=["image", "video"],
        default=None,
        help="The media type",
    )
    parser.add_argument(
        "--force_using_train_views_for_test",
        action="store_true",
        help="Force using train views for test",
    )
    parser.add_argument(
        "--save_albedo_images",
        action="store_true",
        help="Save the albedo images",
    )
    parser.add_argument(
        "--save_shading_images",
        action="store_true",
        help="Save the shading images",
    )
    parser.add_argument(
        "--write_summary_on_image",
        action="store_true",
    )
    parser.add_argument(
        "--rotate_rendered_images",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--interpolate_colors_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--interpolate_colors_indices",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_pca_for_interpolation",
        action="store_true",
    )
    parser.add_argument(
        "--color_intensity",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--shading_intensity",
        type=float,
        default=1.0,
    )
    parser.add_argument("--intensity_start_range", type=float, default=0.5)
    parser.add_argument("--intensity_end_range", type=float, default=1.5)
    parser.add_argument("--intensity_num_steps", type=int, default=20)
    parser.add_argument(
        "--interpolate_colors_percentage",
        type=float,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--TSEN_refrence",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--TSEN_frames",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--include_metrics_in_name",
        action="store_true",
    )
    parser.add_argument(
        "--calculate_transfer_losses",
        action="store_true",
    )
    parser.add_argument(
        "--save_image_with_numpy",
        action="store_true",
    )
    parser.add_argument(
        "--render_bg_black",
        action="store_true",
    )
    parser.add_argument(
        "--decrease_shading_constant",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--albedo_consisntency_points_id",
        type=str,
        default=None,
    )
    add_arguments_from_config(parser, config)
    return parser.parse_args()


def get_args():
    initial_args = argparse.ArgumentParser(description="Intrinsic PAPR")
    initial_args.add_argument(
        "--opt", type=str, default="config.yaml", help="Path to the config file"
    )

    initial_args, _ = initial_args.parse_known_args()

    config = get_config_from_args(initial_args.opt)
    args = parse_args(config)

    config = update_train_test_options(args=args, train_options=config)
    return config, args

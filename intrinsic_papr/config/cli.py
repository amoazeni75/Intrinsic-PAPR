"""The command line, whose --key flags are generated from the merged configuration."""

import argparse

from .loader import get_config_from_args, update_train_test_options


def _bool_arg(text):
    """Parse a boolean override.

    argparse's `type=bool` would make every non-empty string True, so
    `--flag false` would silently mean True. Parse the word instead.
    """
    lowered = text.strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        "expected a boolean (true/false), got {!r}".format(text)
    )


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
            if arg_type is bool:
                arg_type = _bool_arg
            if is_list:
                parser.add_argument(
                    f"--{full_key}", type=arg_type, nargs="+", default=None
                )
            else:
                parser.add_argument(f"--{full_key}", type=arg_type, default=None)



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
            "freeform_transfer_albedo",
            "freeform_transfer_shading",
            "render",
            "change_brightness",
            "calculate_albedo_consistency",
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
        choices=["points_cloud_areas_boxes", "freeform_pixels"],
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
        "--save_albedo_images",
        action="store_true",
        help="Save the albedo images",
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
        "--color_intensity",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--shading_intensity",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--intensity_sweep",
        action="store_true",
        help="change_brightness: render a range of shading intensities instead of "
        "the single --shading_intensity, using the three flags below",
    )
    parser.add_argument("--intensity_start_range", type=float, default=0.5)
    parser.add_argument("--intensity_end_range", type=float, default=1.5)
    parser.add_argument("--intensity_num_steps", type=int, default=20)

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

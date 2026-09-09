"""Reading, merging and archiving the YAML configuration."""

import argparse
import os
import shutil

import yaml
from ruamel.yaml.comments import CommentedSeq


class DictAsMember(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            # must be AttributeError, or getattr(config, name, default) cannot fall back
            raise AttributeError(name) from None
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value


def copy_config_for_replay(config_path, log_dir):
    """
    Archive the config next to the run so it can be replayed with --opt <log_dir>/<name>.yml.

    Templates declare `base: base.yml`, so the base travels with them or the copy cannot load.
    """
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    with open(config_path, "r") as f:
        base_name = yaml.safe_load(f).get("base")
    if base_name is not None:
        base_path = os.path.join(os.path.dirname(config_path), base_name)
        shutil.copyfile(base_path, os.path.join(log_dir, os.path.basename(base_path)))

def merge_config(base, overlay):
    """
    Merge overlay into base, in place, and return base.

    Mappings merge key by key. Lists of mappings merge element by element, so a
    template can override one field of an entry in `test.datasets` without
    repeating the rest of it. Every other value in the overlay replaces the one
    in base.
    """
    for key, value in overlay.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merge_config(current, value)
        elif (
            isinstance(current, list)
            and isinstance(value, list)
            and current
            and value
            and all(isinstance(item, dict) for item in current)
            and all(isinstance(item, dict) for item in value)
        ):
            for i, item in enumerate(value):
                if i < len(current):
                    merge_config(current[i], item)
                else:
                    current.append(item)
        else:
            base[key] = value
    return base


def get_config_from_args(config_path):
    # we will open the config file (.yml) inside ./configs/config_name.yml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_name = config.pop("base", None)
    if base_name is None:
        return config

    base_path = os.path.join(os.path.dirname(config_path), base_name)
    with open(base_path, "r") as f:
        base_config = yaml.safe_load(f)
    return merge_config(base_config, config)


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
                    "{}".format("The key {} is not found.".format(sub_key))
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
                    "{}".format(
                        "The key {} is updated to {}.".format(_original_key, value)
                    )
                )
            elif debug:
                # print with bold orange color.
                print(
                    "{}".format(
                        "The key {} is not updated.".format(keys[-1])
                    )
                )
        else:
            # print the error message with red color.
            print(
                "{}".format("The key {} is not found.".format(keys[-1]))
            )

    return train_options

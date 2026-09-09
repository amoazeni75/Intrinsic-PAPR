"""Writing training metrics to metrics.jsonl and printing the periodic summary."""

import json
import os
import sys
import zipfile
from datetime import datetime


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
        self.terminal.flush()
        self.log.flush()

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

def print_nested_dict(d, step, indent=0):
    """
    Recursively prints a nested dictionary with keys and values on the same line and formatted list values.

    Args:
    d (dict): The dictionary to print.
    indent (int): The current indentation level.
    """
    print("@" * 100)
    print(f"Step: {step}")
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
    force_to_print = True if scene_manager.args.debug else False
    # eval_step_cnt is incremented after this function runs and reset to 0 below, so at
    # this point it already holds the number of steps accumulated since the last log.
    # It is 0 only on the very first log, and on the first log after a resume.
    steps_since_log = max(scene_manager.eval_step_cnt, 1)
    scene_manager.total_train_losses.append(
        scene_manager.avg_total_train_loss / (steps_since_log)
    )
    log_dictionary["total_train_losses"] = scene_manager.avg_total_train_loss / (
        steps_since_log
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
                    ) / (steps_since_log)
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

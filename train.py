"""Train one scene. See the README for the command line."""

import copy
import os
import shutil
import sys

from intrinsic_papr.config import copy_config_for_replay, get_args
from intrinsic_papr.seeding import setup_seed
from intrinsic_papr.training.metrics import (
    Logger,
    find_all_python_files_and_zip,
)
from intrinsic_papr.training.scene import SceneManager
from intrinsic_papr.training.trainer import train_and_eval

# imageio needs an ffmpeg binary to write videos; take whatever is on PATH
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", shutil.which("ffmpeg") or "ffmpeg")

if __name__ == "__main__":
    config, args = get_args()

    print("GPU ID:", args.gpu_id)
    # Restricting visibility to the chosen GPU renumbers it: whatever --gpu_id
    # names, it is the only visible device and torch sees it as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.stage = "train"

    log_dir = os.path.join(config["save_dir"], config["index"])
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, "train_error.log"), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    copy_config_for_replay(args.opt, log_dir)

    find_all_python_files_and_zip(".", os.path.join(log_dir, "code.zip"))

    setup_seed(config["seed"])


    # the scene to train, named scene_1 in every shipped config
    config_keys = list(config.keys())
    scene_keys = [
        key for key in config_keys if key.startswith("scene_") and len(key) == 7
    ]
    scene_keys.sort()
    assert len(scene_keys) == 1, (
        "Training runs one scene at a time; the config declares: {}".format(scene_keys)
    )
    scene_key = scene_keys[0]

    eval_config = copy.deepcopy(config)
    eval_config[scene_key]["dataset"].update(eval_config[scene_key]["eval"]["dataset"])
    eval_config = eval_config[scene_key]
    scene_idx = int(scene_key.split("_")[1])

    scene_manager = SceneManager(
        args=args,
        all_configs=config,
        scene_config=config[scene_key],
        eval_config=eval_config,
        scene_key=scene_key,
        scene_idx=scene_idx - 1,
        cuda_idx=0,
    )

    scene_manager.model.init_optimizers(total_steps=0)
    scene_manager.step = scene_manager.load_model(args.resume)
    scene_manager.model = scene_manager.model.to(scene_manager.device)
    scene_manager.eval_step_cnt = 0

    train_and_eval(scene_manager)

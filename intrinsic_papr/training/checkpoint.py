"""Reading and writing training checkpoints.

A checkpoint is a single ``checkpoints-<step>.pth`` holding the model weights,
the optimiser and scheduler state, the AMP scaler, the loss history and the
eval PSNRs. `load_legacy` reads an older multi-file layout that this code no
longer writes; it is kept so checkpoints from before that change still load.
"""

import os

import torch
from torch import nn

from ..seeding import setup_seed

# Every loss series carried in a checkpoint, keyed the way the manager stores it.
_LOSS_PHASES = ("train", "eval")
_LOSS_SPACES = (
    "pred_space",
    "original_space",
    "pred_space_cIMLE",
    "original_space_cIMLE",
)


def _loss_series(scene_manager):
    """Yield ``(key, phase, image_type, space)`` for each loss series to persist."""
    image_types = ["render"]
    if scene_manager.scene_config.models.use_albedo:
        image_types.append("albedo")
    for phase in _LOSS_PHASES:
        for space in _LOSS_SPACES:
            for image_type in image_types:
                key = "{}_{}_{}".format(phase, image_type, space)
                yield key, phase, image_type, space


def _restore_losses(scene_manager, read_series):
    """Put each stored loss series back on the manager.

    ``read_series(key)`` returns the tensor for one series, so the same
    restore works whether the series come from one file or from many.
    """
    for key, phase, image_type, space in _loss_series(scene_manager):
        loss = read_series(key)
        getattr(scene_manager, "{}_losses".format(image_type))[phase][space] = list(
            loss.detach().numpy()
        )


def _latest_step(checkpoint_dir, prefix):
    """The highest step among ``<prefix><step>.pth`` files, or None if there are none."""
    try:
        files = [
            name
            for name in os.listdir(checkpoint_dir)
            if name.startswith(prefix) and name.endswith(".pth")
        ]
        files = sorted(files, key=lambda name: int(name.split("-")[1].split(".")[0]))
        return int(files[-1].split("-")[1].split(".")[0])
    except (IndexError, ValueError):
        print(" Can't resume because no checkpoint found in {}".format(checkpoint_dir))
        return None


def _state_dicts(named_modules, key, state_dicts):
    """Restore optimisers or schedulers, asserting the checkpoint agrees on which exist."""
    for name, module in named_modules.items():
        if module is not None:
            module.load_state_dict(state_dicts[name])
        else:
            assert state_dicts[name] is None, key


def save(model):
    """Write the full training state to ``checkpoints-<step>.pth``."""
    scene_manager = model.scene_manager
    save_dict = {
        "step": scene_manager.step,
        # the next step re-seeds with this, so a resume continues the same stream
        "seed": scene_manager.step + 1,
        "model_state_dict": model.state_dict(),
        "optimizers_state_dict": {},
        "schedulers_state_dict": {},
        "scaler_state_dict": model.scaler.state_dict(),
        "losses": {},
    }
    for group, modules in (
        ("optimizers_state_dict", model.optimizers),
        ("schedulers_state_dict", model.schedulers),
    ):
        for name, module in modules.items():
            if module is None:
                save_dict[group] = None
            else:
                save_dict[group][name] = module.state_dict()

    for key, phase, image_type, space in _loss_series(scene_manager):
        save_dict["losses"][key] = torch.tensor(
            getattr(scene_manager, "{}_losses".format(image_type))[phase][space]
        )
    save_dict["eval_psnrs"] = torch.tensor(scene_manager.eval_psnrs)

    torch.save(
        save_dict,
        os.path.join(
            scene_manager.checkpoints_dir,
            "checkpoints-{}.pth".format(scene_manager.step),
        ),
    )


def load(model, manager, checkpoint_dir, specific_checkpoint=None, stage="train"):
    """Restore a checkpoint and return the step it was saved at, or 0 if there is none."""
    # rfind returns -1 for a bare filename, and s[-1:] is its last character, so
    # without the +1 every unqualified name like "checkpoints-250000.pth" was
    # tested as "h" and sent to the legacy loader.
    basename = specific_checkpoint[specific_checkpoint.rfind("/") + 1 :] if (
        specific_checkpoint is not None
    ) else ""
    if specific_checkpoint is not None and "checkpoints-" not in basename:
        return load_legacy(
            model, manager, checkpoint_dir, specific_checkpoint, stage=stage
        )

    print("Loading model from: ", checkpoint_dir)
    if specific_checkpoint is not None:
        # "checkpoints-<step>.pth" -> <step>. The old code indexed split("-")[2],
        # which needs three dash-separated parts; the documented
        # "checkpoints-250000.pth" has two, so it raised IndexError.
        step_to_load = int(basename.rsplit("-", 1)[-1].split(".")[0])
        print("step to load: ", step_to_load)
    else:
        step_to_load = _latest_step(checkpoint_dir, "checkpoints-")
        if step_to_load is None:
            return 0
        print("step to load: ", step_to_load)

    checkpoint_dict = torch.load(
        os.path.join(checkpoint_dir, "checkpoints-{}.pth".format(step_to_load))
    )
    print(
        "The checkpoint's step was: {}, so the next step is {}".format(
            checkpoint_dict["step"], checkpoint_dict["step"] + 1
        ),
    )
    setup_seed(checkpoint_dict["seed"])
    copy_state_dict(model, checkpoint_dict["model_state_dict"])

    # optimisers and schedulers only exist while training
    if stage == "train":
        _state_dicts(
            model.optimizers, "optimizers", checkpoint_dict["optimizers_state_dict"]
        )
        _state_dicts(
            model.schedulers, "schedulers", checkpoint_dict["schedulers_state_dict"]
        )

    model.scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
    _restore_losses(manager, lambda key: checkpoint_dict["losses"][key])
    manager.eval_psnrs = list(checkpoint_dict["eval_psnrs"].detach().numpy())
    return step_to_load


def load_legacy(model, manager, checkpoint_dir, specific_checkpoint=None, stage="train"):
    """Restore the older layout, where each piece of state was its own file."""

    def path(name):
        return os.path.join(checkpoint_dir, name)

    if specific_checkpoint is not None:
        step_to_load = int(specific_checkpoint.split("-")[1].split(".")[0])
    else:
        step_to_load = _latest_step(checkpoint_dir, "model-")
        if step_to_load is None:
            return 0

    if stage == "train":
        _state_dicts(
            model.optimizers,
            "optimizers",
            torch.load(path("optimizers-{}.pth".format(step_to_load))),
        )
        _state_dicts(
            model.schedulers,
            "schedulers",
            torch.load(path("schedulers-{}.pth".format(step_to_load))),
        )
        if os.path.exists(path("scaler-{}.pth".format(step_to_load))):
            model.scaler.load_state_dict(
                torch.load(path("scaler-{}.pth".format(step_to_load)))
            )
        _restore_losses(
            manager, lambda key: torch.load(path("{}_losses.pth".format(key)))
        )
        if os.path.exists(path("eval_psnrs.pth")):
            manager.eval_psnrs = list(torch.load(path("eval_psnrs.pth")).detach().numpy())

    model_state_dict = torch.load(path("model-{}.pth".format(step_to_load)))
    for step, state_dict in model_state_dict.items():
        copy_state_dict(model, state_dict)
        print(
            " Model loaded successfully at step {} from {}".format(step, checkpoint_dir)
        )
        if os.path.exists(path("seed-{}.pth".format(step_to_load))):
            setup_seed(torch.load(path("seed-{}.pth".format(step_to_load))))
        return int(step)


def copy_state_dict(model, state_dict):
    """Copy weights in, tolerating renamed modules and a changed point count.

    The point tensors are re-created rather than copied into, because the
    number of points changes as training prunes and grows the cloud.
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.startswith("renderer."):
            name = name.replace("renderer.", "renderer_UNet.")
        if name in ("points", "points_conf_scores", "pc_feats"):
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except (KeyError, RuntimeError) as err:
            print("Can't load", name, "-", err)

    model.points = nn.Parameter(
        state_dict["points"].data, requires_grad=model.points.requires_grad
    )
    if "points_last_grad" in state_dict:
        for name in (
            "points_last_grad",
            "points_acc_grad",
            "points_acc_grad_norm",
            "points_grad_cnt",
        ):
            setattr(
                model, name, nn.Parameter(state_dict[name].data, requires_grad=False)
            )
    if model.points_conf_scores is not None:
        model.points_conf_scores = nn.Parameter(
            state_dict["points_conf_scores"].data,
            requires_grad=model.points_conf_scores.requires_grad,
        )
    if (
        "learnable" in model.scene_manager.scene_config.geoms.point_feats.type
        and model.use_pc_feats
    ):
        model.pc_feats = nn.Parameter(
            state_dict["pc_feats"].data, requires_grad=model.pc_feats.requires_grad
        )
        print(
            "load pc_feats",
            model.pc_feats.shape,
            model.pc_feats.min(),
            model.pc_feats.max(),
        )

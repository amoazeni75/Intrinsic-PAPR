"""The training objective: reconstruction plus the space carving loss."""

import torch

from ..data.pipeline import run_image_pipeline


class LossResult:
    """The individual loss terms of one step, plus the total that is optimised.

    A term is ``None`` when its branch did not run, which is how the caller
    tells "this loss was zero" apart from "this loss was not computed".
    """

    __slots__ = (
        "total",
        "render_pred_space",
        "render_original_space",
        "albedo_pred_space",
        "albedo_original_space",
        "albedo_pred_space_cIMLE",
        "albedo_original_space_cIMLE",
    )

    def __init__(
        self,
        total,
        render_pred_space=None,
        render_original_space=None,
        albedo_pred_space=None,
        albedo_original_space=None,
        albedo_pred_space_cIMLE=None,
        albedo_original_space_cIMLE=None,
    ):
        self.total = total
        self.render_pred_space = render_pred_space
        self.render_original_space = render_original_space
        self.albedo_pred_space = albedo_pred_space
        self.albedo_original_space = albedo_original_space
        self.albedo_pred_space_cIMLE = albedo_pred_space_cIMLE
        self.albedo_original_space_cIMLE = albedo_original_space_cIMLE


def is_in_space_carving_loss_iters(current_iter, iters_list):
    """Whether ``current_iter`` falls inside any of the [start, end] windows."""
    for start, end in iters_list:
        if start <= current_iter <= end:
            return True
    return False


def get_space_carving_loss(pred, GTs):
    """Mean over pixels of the distance to the nearest of the cIMLE albedo samples.

    ``pred`` is [B, H, W, C] and ``GTs`` is [B, N_samples, H, W, C].

    Note this is the *unsquared* L2 norm, while Eq. 5 of the paper writes the
    square. The code is what produced the published numbers, so it is kept as
    is deliberately - do not "correct" it to a squared distance.
    """
    B, H, W, C = pred.shape

    # [B, 1, H, W, C] against [B, N_samples, H, W, C]
    diff = pred.unsqueeze(1) - GTs
    distances = torch.norm(diff, p=2, dim=-1)  # [B, N_samples, H, W]
    min_distances, _ = torch.min(distances, dim=1)  # [B, H, W]
    return min_distances.sum() / (B * H * W)


def _to_original_space(scene_manager, img, img_type):
    """Undo the prediction-space encoding so a loss can be taken in RGB space."""
    dataset_config = scene_manager.scene_config.dataset
    return run_image_pipeline(
        img=img,
        pipeline=scene_manager.scene_config.training.loss_func_preprocessing,
        eps=scene_manager.scene_config.models.predict_in_log_space_eps,
        min_val=getattr(dataset_config, "min_{}_log".format(img_type), None),
        max_val=getattr(dataset_config, "max_{}_log".format(img_type), None),
        white_bg_value=getattr(
            scene_manager.scene_config.geoms.background,
            "{}_init_scale".format(img_type),
            None,
        ),
    )


def _reconstruction_losses(
    scene_manager,
    pred_in_pred_space,
    gt_in_pred_space,
    img_type,
    loss_fn,
    also_in_original_space,
    loss_weight,
    clip,
):
    """The reconstruction loss in prediction space, and optionally in RGB space.

    Returns ``(pred_space_loss, original_space_loss)``. The second is ``None``
    unless ``also_in_original_space``.
    """

    def maybe_clip(img):
        return torch.clamp(img, 0, 1) if clip else img

    pred_space_loss = (
        loss_fn(maybe_clip(pred_in_pred_space), maybe_clip(gt_in_pred_space))
        * loss_weight
    )
    if not also_in_original_space:
        return pred_space_loss, None

    pred_in_original_space = _to_original_space(
        scene_manager, pred_in_pred_space, img_type
    )
    gt_in_original_space = _to_original_space(scene_manager, gt_in_pred_space, img_type)
    original_space_loss = (
        loss_fn(maybe_clip(pred_in_original_space), maybe_clip(gt_in_original_space))
        * loss_weight
    )
    return pred_space_loss, original_space_loss


def calculate_training_loss(
    scene_manager,
    render_pred_patch_pred_space,
    render_gt_patch_pred_space,
    albedo_pred_patch_pred_space=None,
    albedo_gt_patch_pred_space=None,
    clip=False,
):
    """Compute every active loss term for one step and sum them into a total.

    ``clip`` squashes both sides of a reconstruction loss into [0, 1] first;
    training passes ``False`` and evaluation passes ``True``.
    """
    models_config = scene_manager.scene_config.models
    result = LossResult(total=0)

    if models_config.include_rgb_loss:
        (
            result.render_pred_space,
            result.render_original_space,
        ) = _reconstruction_losses(
            scene_manager,
            render_pred_patch_pred_space,
            render_gt_patch_pred_space,
            "render",
            scene_manager.render_loss_fn,
            models_config.include_loss_in_original_space,
            models_config.rgb_loss_weight,
            clip,
        )

    if models_config.include_albedo_loss:
        space_carving = (
            scene_manager.scene_config.training.albedo_space_carving_loss.use
            and is_in_space_carving_loss_iters(
                scene_manager.step,
                scene_manager.scene_config.training.albedo_space_carving_loss.iters,
            )
        )
        if space_carving:
            if models_config.include_loss_in_pred_space:
                result.albedo_pred_space_cIMLE = (
                    get_space_carving_loss(
                        pred=albedo_pred_patch_pred_space,
                        GTs=albedo_gt_patch_pred_space,
                    )
                    * models_config.albedo_loss_weight
                )
            if models_config.include_loss_in_original_space:
                # This branch cannot run. The RGB-space albedo tensors are only
                # produced by the MSE branch below, which is the arm this one
                # excludes, so they are always None here. Kept as it was rather
                # than repaired, because no published run took this path.
                albedo_pred_patch_rgb_space = None
                albedo_gt_patch_rgb_space = None
                result.albedo_original_space_cIMLE = (
                    get_space_carving_loss(
                        albedo_pred_patch_rgb_space,
                        albedo_gt_patch_rgb_space,
                    )
                    * models_config.albedo_loss_weight
                )
        else:
            (
                result.albedo_pred_space,
                result.albedo_original_space,
            ) = _reconstruction_losses(
                scene_manager,
                albedo_pred_patch_pred_space,
                # any one of the cIMLE samples will do as the GT for an MSE loss
                albedo_gt_patch_pred_space[:, 0, :, :],
                "albedo",
                scene_manager.albedo_loss_fn,
                models_config.include_loss_in_original_space,
                models_config.albedo_loss_weight,
                clip,
            )

    weighted_terms = (
        (result.render_pred_space, models_config.weight_loss_in_pred_space),
        (result.render_original_space, models_config.weight_loss_render_in_original_space),
        (result.albedo_pred_space, models_config.weight_loss_in_pred_space),
        (result.albedo_original_space, models_config.weight_loss_albedo_in_original_space),
        (result.albedo_pred_space_cIMLE, models_config.weight_loss_in_pred_space),
        (
            result.albedo_original_space_cIMLE,
            models_config.weight_loss_albedo_in_original_space,
        ),
    )
    for loss, weight in weighted_terms:
        if loss is not None:
            result.total = result.total + loss * float(weight)

    return result

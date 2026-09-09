import json
import os
import shutil
import sys
import time

import imageio
import matplotlib
import numpy as np
import torch

# plots.py imports pyplot, so the backend must be headless before it is loaded
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from PIL import Image

from intrinsic_papr.data.rays import find_proj_coord, get_rays
from intrinsic_papr.data.pipeline import run_image_pipeline, write_a_text_on_image
from intrinsic_papr.data.cameras import get_render_poses
from intrinsic_papr.models.features import extract_features_from_feature_map
from intrinsic_papr.seeding import setup_seed
from intrinsic_papr.training.scene import SceneManager
from intrinsic_papr.config import copy_config_for_replay, get_args
from intrinsic_papr.training.metrics import Logger

# imageio needs an ffmpeg binary to write videos; take whatever is on PATH
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", shutil.which("ffmpeg") or "ffmpeg")


try:
    from skimage.measure import compare_ssim
except ImportError:
    from skimage.metrics import structural_similarity

    def compare_ssim(gt, img, win_size, channel_axis=2):
        # The images compared here are in [0, 1]. Without data_range, scikit-image
        # infers 2.0 from the float dtype, which makes the SSIM stabilising
        # constants four times too large and reports a slightly inflated score.
        return structural_similarity(
            gt, img, win_size=win_size, channel_axis=channel_axis, data_range=1.0
        )


def decode_render_and_albedo(
    model,
    feature_map,
    attn,
    bkg_split,
    rayd,
    albedo_feat_size,
    scene_manager,
):
    """Decode the render and albedo images from the attention feature map.

    Everything returned is in prediction space; the postprocessing pipelines
    convert it for display.
    """
    print(
        "getting rgb and albedo from feature map"
    )
    print("albedo UNet feat size: ", albedo_feat_size)

    rgb = None
    albedo_pred_test = None
    foreground_rgb = None
    foreground_albedo = None

    N, H, W, _ = rayd.shape
    with torch.no_grad():
        # albedo
        background_mask = (
            (attn[..., bkg_split:, :] * model.bkg_feats.expand(N, H, W, -1, -1))
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        attention_mask = attn[..., bkg_split:, :].squeeze().detach().cpu().numpy()
        if scene_manager.scene_config.models.use_albedo:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                albedo_input_features = extract_features_from_feature_map(
                    features_map=feature_map,
                    features_dim=albedo_feat_size,
                )

                foreground_albedo = (
                    model.albedo_model(
                        albedo_input_features.squeeze(-2).permute(0, 3, 1, 2)
                    )
                    .permute(0, 2, 3, 1)
                    .unsqueeze(-2)
                )
                if model.bkg_feats is not None:
                    bkg_attn = attn[..., bkg_split:, :]
                    bkg_feats = model.bkg_feats.expand(N, H, W, -1, -1)
                    if scene_manager.args.render_bg_black:
                        # render the background as black: invert the clamped bkg feats
                        bkg_feats = 1 - torch.clamp(bkg_feats, 0, 1)
                    if scene_manager.scene_config.models.normalize_topk_attn:
                        albedo_pred_test = (
                            foreground_albedo * (1 - bkg_attn) + bkg_feats * bkg_attn
                        )
                    else:
                        albedo_pred_test = foreground_albedo + bkg_feats * bkg_attn
                    albedo_pred_test = albedo_pred_test.squeeze(-2)
                else:
                    albedo_pred_test = foreground_albedo.squeeze(-2)

        # rgb
        if scene_manager.scene_config.models.use_renderer:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                foreground_rgb = (
                    model.renderer_UNet(feature_map.squeeze(-2).permute(0, 3, 1, 2))
                    .permute(0, 2, 3, 1)
                    .unsqueeze(-2)
                )  # (N, H, W, 1, 3)
                if model.bkg_feats is not None:
                    bkg_attn = attn[..., bkg_split:, :]
                    bkg_feats = model.bkg_feats.expand(N, H, W, -1, -1)
                    if scene_manager.args.render_bg_black:
                        # render the background as black: invert the clamped bkg feats
                        bkg_feats = 1 - torch.clamp(bkg_feats, 0, 1)
                    if scene_manager.scene_config.models.normalize_topk_attn:
                        rgb = foreground_rgb * (1 - bkg_attn) + bkg_feats * bkg_attn
                    else:
                        rgb = foreground_rgb + bkg_feats * bkg_attn
                    rgb = rgb.squeeze(-2)
                else:
                    rgb = foreground_rgb.squeeze(-2)
                foreground_rgb = foreground_rgb.squeeze()

    background_mask = np.clip(background_mask, 0, 1)
    attention_mask = np.clip(attention_mask, 0, 1)

    return (
        rgb,
        albedo_pred_test,
        foreground_rgb,
        foreground_albedo,
        background_mask,
        attention_mask,
    )


def get_name_to_save(
    frame_index, image_type, scene_manager, sample_idx, selected_source_points_index
):
    base_name = "view-{:04d}-test-{}-step-{}K".format(
        frame_index,
        image_type,
        (int(scene_manager.step / 1000) if scene_manager.step != -1 else "final"),
    )

    action_suffix = f"-{scene_manager.args.test_action}"
    extra_info = ""
    view_mode_suffix = f"-view-mode-{scene_manager.args.render_frame_type}"
    media_type_suffix = f"-{scene_manager.args.media_type}"

    if (
        scene_manager.args.test_action == "transfer_albedo"
        or scene_manager.args.test_action == "transfer_shading" or args.test_action == "freeform_transfer_shading"
        or scene_manager.args.test_action == "freeform_transfer_albedo"
        or scene_manager.args.test_action == "freeform_transfer_shading"
    ):
        extra_info = "sscene-{}-tscene-{}-sarea-{}-tarea-{}-p_id-{}-{}-s_pts-intensity-{}-".format(
            scene_manager.args.source_scene_index,
            scene_manager.args.target_scene_index,
            scene_manager.args.source_area_indices,
            "freeform",
            sample_idx,
            (
                "all"
                if scene_manager.args.how_many_source_area_points == -1
                else scene_manager.args.how_many_source_area_points
            ),
            (
                scene_manager.args.shading_intensity
                if scene_manager.args.test_action == "transfer_shading" or args.test_action == "freeform_transfer_shading"
                else scene_manager.args.color_intensity
            ),
        )
    if scene_manager.args.test_action == "change_brightness":
        extra_info += "shd-intensity-{:.4f}".format(
            scene_manager.args.shading_intensity
        )

    if scene_manager.args.test_action == "change_brightness":
        if selected_source_points_index is not None:
            extra_info += "-using-pt-{}".format(selected_source_points_index)
        else:
            extra_info += "-whole-image"

    if (
        not scene_manager.args.test_action == "change_brightness"
        and selected_source_points_index is not None
    ):
        extra_info += "-pt-{}".format(selected_source_points_index)

    return base_name + action_suffix + view_mode_suffix + extra_info + media_type_suffix


def _calculate_loss_and_save_image(
    pred,
    gt,
    img_type,
    frame_index,
    sample_idx,
    space,
    loss_log_dictionary,
    selected_source_points_index,
    scene_manager,
    save_image=True,
):

    if gt is None:
        # nothing to compare against: the image is still written, but no metric is
        # recorded, so the averages in loss_averages.txt stay over real comparisons only
        test_psnr = test_ssim = test_lpips_alex = test_lpips_vgg = loss = None
    else:
        test_psnr = -10.0 * np.log(((pred - gt) ** 2).mean().item()) / np.log(10.0)
        test_ssim = compare_ssim(
            pred.squeeze().detach().cpu().numpy(),
            gt.squeeze().detach().cpu().numpy(),
            11,
            channel_axis=2,
        )
        test_lpips_alex = (
            scene_manager.lpips_loss_fn_alex(
                pred.permute(0, 3, 1, 2),
                gt.permute(0, 3, 1, 2),
            )
            .squeeze()
            .item()
        )
        test_lpips_vgg = (
            scene_manager.lpips_loss_fn_vgg(
                pred.permute(0, 3, 1, 2),
                gt.permute(0, 3, 1, 2),
            )
            .squeeze()
            .item()
        )
        loss_fn = getattr(scene_manager, f"{img_type}_loss_fn")
        loss = loss_fn(pred, gt)
    if gt is None:
        print(f"({space}) Frame: {frame_index}, {img_type}, no ground truth, metrics skipped")
    else:
        print(
            f"({space}) Frame: {frame_index}, {img_type}, loss: {loss:.4f}, "
            f"psnr: {test_psnr:.4f}, ssim: {test_ssim:.4f}, "
            f"lpips_alex: {test_lpips_alex:.4f}, lpips_vgg: {test_lpips_vgg:.4f}"
        )

    if scene_manager.args.include_metrics_in_name and gt is not None:
        test_metrics = "-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}-{}".format(
            test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg, space
        )
    else:
        test_metrics = ""

    base_output_name = get_name_to_save(
        frame_index=frame_index,
        image_type=img_type,
        scene_manager=scene_manager,
        sample_idx=sample_idx,
        selected_source_points_index=selected_source_points_index,
    )

    final_name = base_output_name + test_metrics + ".png"

    if save_image:
        if scene_manager.args.rotate_rendered_images is not None:
            pred = torch.rot90(pred, dims=(1, 2))
        imageio.imwrite(
            os.path.join(
                scene_manager.test_log_dir,
                final_name,
            ),
            write_a_text_on_image(
                (pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
                text=None,
            ),
        )
        print(
            "Saved image: ",
            os.path.join(
                scene_manager.test_log_dir,
                final_name,
            ),
        )
        if scene_manager.args.save_image_with_numpy:
            npy_path = os.path.join(
                scene_manager.test_log_dir,
                final_name.replace(".png", ".npy"),
            )
            np.save(
                npy_path,
                pred.squeeze().detach().cpu().numpy(),
            )
            print("Saved numpy array: ", npy_path)

    if gt is not None:
        loss_log_dictionary[img_type]["loss"][space].append(loss.item())
        loss_log_dictionary[img_type]["psnr"][space].append(test_psnr)
        loss_log_dictionary[img_type]["ssim"][space].append(test_ssim)
        loss_log_dictionary[img_type]["lpips_alex"][space].append(test_lpips_alex)
        loss_log_dictionary[img_type]["lpips_vgg"][space].append(test_lpips_vgg)


def _to_display_space(scene_manager, image, pipeline_type, stats_type):
    """Run a test postprocessing pipeline over one image.

    ``pipeline_type`` picks the pipeline; ``stats_type`` picks the log-space
    min and max used to undo the normalisation. The two differ for predictions,
    which are denormalised with the *render* statistics whatever the image type
    is. That is asymmetric against the GT path, but it is what the published
    albedo numbers used, so it is kept deliberately.
    """
    models_config = scene_manager.scene_config.models
    in_log_space = (
        models_config.predict_rgb_in_log_space or models_config.predict_raw_in_log_space
    )
    dataset_config = scene_manager.scene_config.dataset
    return run_image_pipeline(
        img=image,
        pipeline=scene_manager.scene_config.test.datasets[0][
            "{}_postprocessing".format(pipeline_type)
        ],
        eps=models_config.predict_in_log_space_eps,
        min_val=(
            getattr(dataset_config, "min_{}_log".format(stats_type), None)
            if in_log_space
            else 0
        ),
        max_val=(
            getattr(dataset_config, "max_{}_log".format(stats_type), None)
            if in_log_space
            else 1
        ),
        white_bg_value=getattr(
            scene_manager.scene_config.geoms.background, "render_init_scale", None
        ),
        supervision_scaler=None,
    )


def calculate_loss_and_save_image(
    raw_pred,
    raw_gt,
    img_type,
    frame_index,
    sample_idx,
    loss_log_dictionary,
    selected_source_points_index,
    scene_manager,
):

    srgb_pred = _to_display_space(
        scene_manager, raw_pred, "{}_pred".format(img_type), "render"
    )
    srgb_gt = None
    if raw_gt is not None:
        srgb_gt = _to_display_space(
            scene_manager, raw_gt, "{}_GT".format(img_type), img_type
        )

    _calculate_loss_and_save_image(
        pred=raw_pred,
        gt=raw_gt,
        img_type=img_type,
        frame_index=frame_index,
        sample_idx=sample_idx,
        space="pred_space",
        loss_log_dictionary=loss_log_dictionary,
        save_image=False,
        selected_source_points_index=selected_source_points_index,
        scene_manager=scene_manager,
    )
    _calculate_loss_and_save_image(
        pred=srgb_pred,
        gt=srgb_gt,
        img_type=img_type,
        frame_index=frame_index,
        sample_idx=sample_idx,
        space="srgb_space",
        loss_log_dictionary=loss_log_dictionary,
        save_image=True,
        selected_source_points_index=selected_source_points_index,
        scene_manager=scene_manager,
    )

    return_srgb_pred = (srgb_pred.squeeze().detach().cpu().numpy() * 255).astype(
        np.uint8
    )
    return_srgb_gt = (
        (srgb_gt.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
        if srgb_gt is not None
        else None
    )
    return_raw_pred = (raw_pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    return_raw_gt = (
        (raw_gt.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
        if raw_gt is not None
        else None
    )

    return (return_srgb_pred, return_srgb_gt, return_raw_pred, return_raw_gt)



class FrameOutputs:
    """What rendering one frame produced, so callers name what they take."""

    __slots__ = (
        "render_srgb_pred",
        "render_srgb_gt",
        "render_raw_pred",
        "render_raw_gt",
        "albedo_srgb_pred",
        "albedo_srgb_gt",
        "albedo_raw_pred",
        "albedo_raw_gt",
        "selected_points",
        "selected_points_index",
        "selected_points_att",
        "depth_map",
        "camera_pose_np",
    )

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields[name])


class FrameEvaluation:
    """One frame pushed through the model: the feature map and what attended to what."""

    __slots__ = (
        "feature_map",
        "attn",
        "bkg_split",
        "selected_points",
        "selected_points_index",
        "selected_points_att",
        "rayo",
        "rayd",
        "c2w",
        "shape",
        "render_gt",
        "albedo_gt",
    )

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields[name])


def evaluate_frame(frame_idx, camera_poses, scene_manager):
    """Run the model over one whole frame, a tile at a time.

    ``camera_poses`` renders a pose that is not in the dataset, so there is no
    ground truth for it; passing None reads the frame from the eval dataset.
    """
    if camera_poses is not None:
        camera_pose = camera_poses[frame_idx].unsqueeze(0)  # (1, 4, 4)
        idx = torch.tensor([frame_idx])
        test_render_GT = None
        test_albedo_GT = None
        rayo, rayd = get_rays(
            scene_manager.eval_dataset.H,
            scene_manager.eval_dataset.W,
            scene_manager.eval_dataset.focal_x,
            scene_manager.eval_dataset.focal_y,
            camera_pose,
            coord=scene_manager.eval_dataset.dataset_args.rays.cam_world,
        )
        c2w = camera_pose.squeeze(0)  # (4, 4)
    else:
        _idx, _, _test_render_GT, _test_albedo_GT, _rayd, _rayo, _ = (
            scene_manager.eval_dataset[frame_idx]
        )
        idx = torch.tensor([_idx])
        if _test_render_GT is not None:
            test_render_GT = _test_render_GT.unsqueeze(0)
        else:
            test_render_GT = None
        if _test_albedo_GT is not None:
            test_albedo_GT = _test_albedo_GT[0].unsqueeze(0)
        else:
            test_albedo_GT = None
        rayd = _rayd.unsqueeze(0)
        rayo = _rayo.unsqueeze(0)

        c2w = scene_manager.eval_dataset.get_c2w(idx.squeeze())

    N, H, W, _ = rayd.shape
    num_pts, _ = scene_manager.model.points.shape

    rayo = rayo.to(scene_manager.device)
    rayd = rayd.to(scene_manager.device)
    c2w = c2w.to(scene_manager.device)

    if test_render_GT is not None:
        test_render_GT = test_render_GT.to(scene_manager.device)
    if test_albedo_GT is not None:
        test_albedo_GT = test_albedo_GT.to(scene_manager.device)

    topk = min([num_pts, scene_manager.model.select_k])
    pt_idxs = [topk * i // 5 for i in range(5)]

    # Unbounded scenes carry one extra attention slot per ray, the
    # background-sphere intersection. It belongs to the point sequence, so the
    # split between point attention and the learned background sits after it.
    # Bounded scenes add no slot and every size below is unchanged.
    num_slots = topk + (1 if scene_manager.model.append_bkg_points else 0)

    selected_points = torch.zeros(1, H, W, num_slots, 3)
    selected_points_att = torch.zeros(1, H, W, num_slots, 1)
    selected_points_index = torch.zeros(1, H, W, num_slots)

    bkg_seq_len_attn = 0
    tx_opt = scene_manager.scene_config.models.transformer
    feat_dim = (
        tx_opt.embed.d_ff_out
        if tx_opt.embed.share_embed
        else tx_opt.embed.value.d_ff_out
    )
    if scene_manager.model.bkg_feats is not None and scene_manager.model.bkg_type == 1:
        bkg_seq_len_attn = scene_manager.model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(scene_manager.device)
    attn = torch.zeros(N, H, W, num_slots + bkg_seq_len_attn, 1).to(
        scene_manager.device
    )

    with torch.no_grad():
        for height_start in range(0, H, scene_manager.scene_config.eval.max_height):
            for width_start in range(0, W, scene_manager.scene_config.eval.max_width):
                height_end = min(
                    height_start + scene_manager.scene_config.eval.max_height, H
                )
                width_end = min(
                    width_start + scene_manager.scene_config.eval.max_width, W
                )
                (
                    _,
                    feature_map[
                        :, height_start:height_end, width_start:width_end, :, :
                    ],
                    attn[:, height_start:height_end, width_start:width_end, :, :],
                    _,
                    _,
                    _,
                    _,
                ) = scene_manager.model.evaluate(
                    rayo,
                    rayd[:, height_start:height_end, width_start:width_end],
                    pt_idxs=pt_idxs,
                    step=scene_manager.step - 1,
                )

                selected_points[
                    :, height_start:height_end, width_start:width_end, :, :
                ] = scene_manager.model.selected_points

                selected_points_index[
                    :, height_start:height_end, width_start:width_end, :
                ] = scene_manager.model.select_k_ind

                selected_points_att[
                    :, height_start:height_end, width_start:width_end, :, :
                ] = scene_manager.model.top_k_attn
    return FrameEvaluation(
        feature_map=feature_map,
        attn=attn,
        bkg_split=num_slots,
        selected_points=selected_points,
        selected_points_index=selected_points_index,
        selected_points_att=selected_points_att,
        rayo=rayo,
        rayd=rayd,
        c2w=c2w,
        shape=(N, H, W),
        render_gt=test_render_GT,
        albedo_gt=test_albedo_GT,
    )


def collect_points_under_strokes(
    scene_manager, frame_idx, stroke_pixels, selection_method
):
    """The indices of the points that the stroke pixels attend to, in one frame.

    ``selection_method`` is "all" to take every point a pixel attends to, or
    "highest_attention" to take only the one it attends to most.
    """
    frame = evaluate_frame(frame_idx, None, scene_manager)
    _, H, W = frame.shape
    if selection_method not in ("all", "highest_attention"):
        raise ValueError(
            "Invalid freeform selection method: {}".format(selection_method)
        )

    points = set()
    # On an unbounded scene every ray carries one extra slot for the
    # background-sphere intersection, indexed one past the last real point so it
    # can address its own feature row. It is not a point of the scene and cannot
    # receive transferred features, so drop it here; passing it on would index
    # out of bounds. Bounded scenes have no such slot and nothing is dropped.
    real_point_count = scene_manager.model.points.shape[0]
    for x, y in stroke_pixels:
        if not (0 <= x < W and 0 <= y < H):
            continue
        pixel_points = [
            int(index) for index in frame.selected_points_index[0, y, x].cpu().numpy()
        ]
        if selection_method == "highest_attention":
            pixel_attn = frame.selected_points_att[0, y, x].cpu().numpy().reshape(-1)
            candidates = [
                slot
                for slot, index in enumerate(pixel_points)
                if index < real_point_count
            ]
            if not candidates:
                continue
            # max returns the first maximal element, as np.argmax did.
            pixel_points = [pixel_points[max(candidates, key=lambda s: pixel_attn[s])]]
        else:
            pixel_points = [
                index for index in pixel_points if index < real_point_count
            ]
        points.update(pixel_points)
    return points


def render_single_frame(
    frame_idx,
    sample_idx,
    loss_dictionary,
    selected_source_points_index,
    camera_poses,
    scene_manager,
):
    """Render one frame, save its images, and record its metrics."""
    frame = evaluate_frame(frame_idx, camera_poses, scene_manager)
    rayo, rayd = frame.rayo, frame.rayd
    c2w = frame.c2w
    test_render_GT, test_albedo_GT = frame.render_gt, frame.albedo_gt
    selected_points = frame.selected_points
    selected_points_index = frame.selected_points_index
    selected_points_att = frame.selected_points_att

    with torch.no_grad():
        (
            test_render_prediction_raw,
            test_albedo_prediction_raw,
            test_render_prediction_foreground_raw,
            test_albedo_prediction_foreground_raw,
            background_mask,
            attention_mask,
        ) = decode_render_and_albedo(
            model=scene_manager.model,
            feature_map=frame.feature_map,
            attn=frame.attn,
            bkg_split=frame.bkg_split,
            rayd=rayd,
            albedo_feat_size=scene_manager.model.albedo_UNet_inp_size,
            scene_manager=scene_manager,
        )
    # caluclate the loss and save the images
    (render_srgb_pred, render_srgb_gt, render_raw_pred, render_raw_gt) = (
        calculate_loss_and_save_image(
            raw_pred=test_render_prediction_raw,
            raw_gt=test_render_GT,
            img_type="render",
            frame_index=frame_idx,
            sample_idx=sample_idx,
            loss_log_dictionary=loss_dictionary,
            selected_source_points_index=selected_source_points_index,
            scene_manager=scene_manager,
        )
    )
    if scene_manager.args.save_albedo_images:
        (albedo_srgb_pred, albedo_srgb_gt, albedo_raw_pred, albedo_raw_gt) = (
            calculate_loss_and_save_image(
                raw_pred=test_albedo_prediction_raw,
                raw_gt=test_albedo_GT,
                img_type="albedo",
                frame_index=frame_idx,
                sample_idx=sample_idx,
                loss_log_dictionary=loss_dictionary,
                selected_source_points_index=selected_source_points_index,
                scene_manager=scene_manager,
            )
        )
    else:
        (
            albedo_srgb_pred,
            albedo_srgb_gt,
            albedo_raw_pred,
            albedo_raw_gt,
        ) = (
            -1,
            -1,
            -1,
            -1,
        )
    depth_map = compute_depth_map_from_attention(
        selected_points=selected_points,
        attn=frame.attn,
        rayo=rayo,
        scene_manager=scene_manager,
    )
    camera_pose_np = c2w.detach().cpu().numpy()

    return FrameOutputs(
        render_srgb_pred=render_srgb_pred,
        render_srgb_gt=render_srgb_gt,
        render_raw_pred=render_raw_pred,
        render_raw_gt=render_raw_gt,
        albedo_srgb_pred=albedo_srgb_pred,
        albedo_srgb_gt=albedo_srgb_gt,
        albedo_raw_pred=albedo_raw_pred,
        albedo_raw_gt=albedo_raw_gt,
        selected_points=selected_points,
        selected_points_index=selected_points_index,
        selected_points_att=selected_points_att,
        depth_map=depth_map,
        camera_pose_np=camera_pose_np,
    )


def calculate_albedo_consistency(
    args,
    scene_manager,
):
    """MACE: how much a point's albedo changes across the views that see it."""
    if not args.save_albedo_images:
        raise ValueError(
            "calculate_albedo_consistency reads the rendered albedo maps, "
            "so it needs --save_albedo_images"
        )
    evaluation_points_id = scene_manager.source_area_indices[
        args.source_area_indices[0]
    ]

    # Initialize dictionary using dictionary comprehension
    point_img_type_pixel_values_dict = {
        p_id: {
            "albedo_srgb_pred": [],
            "albedo_srgb_gt": [],
            "albedo_raw_pred": [],
            "albedo_raw_gt": [],
        }
        for p_id in evaluation_points_id
    }

    # Call the test function once and precompute necessary values
    return_dict = render_frames(
        scene_manager=scene_manager, sample_idx=0, keep_results=True
    )

    # Precompute model points
    model_points = (
        scene_manager.model.points[evaluation_points_id].detach().cpu().numpy()
    )

    # Loop through frames once, but vectorize pixel data extraction
    for i, frame in enumerate(return_dict["frames"]):
        # Project points to pixel coordinates
        c2w = scene_manager.eval_dataset.get_c2w(frame)
        c2w[-1, -1] = 1.0
        points_pixels = find_proj_coord(
            pc=model_points,
            c2w=c2w,
            H=scene_manager.eval_dataset.H,
            W=scene_manager.eval_dataset.W,
            focal_x=scene_manager.eval_dataset.focal_x,
            focal_y=scene_manager.eval_dataset.focal_y,
        ).astype(int)

        # Sample each projected point out of this frame's albedo maps.
        # NOTE: the sample is image[px, py], i.e. x-major, which is transposed
        # against the rest of this file. It is what the published MACE numbers
        # used, so it is kept deliberately - do not swap the two.
        height, width = scene_manager.eval_dataset.H, scene_manager.eval_dataset.W
        for p_idx, p_id in enumerate(evaluation_points_id):
            px, py = points_pixels[p_idx]
            if not (0 <= px < height and 0 <= py < width):
                # this point is not visible from this view
                continue
            for key in [
                "albedo_srgb_pred",
                "albedo_srgb_gt",
                "albedo_raw_pred",
                "albedo_raw_gt",
            ]:
                point_img_type_pixel_values_dict[p_id][key].append(
                    return_dict[key][i][px, py]
                )

    # Convert consistency lists to arrays for vectorized calculations
    pred_rgb_consistency = []
    gt_rgb_consistency = []
    pred_raw_consistency = []
    gt_raw_consistency = []

    for p_id in evaluation_points_id:
        # a point seen from fewer than two views has no consistency to measure
        if len(point_img_type_pixel_values_dict[p_id]["albedo_srgb_pred"]) < 2:
            continue
        # Stack all frames for each key as an array to vectorize calculations
        albedo_srgb_pred = np.array(
            point_img_type_pixel_values_dict[p_id]["albedo_srgb_pred"]
        )
        albedo_srgb_gt = np.array(
            point_img_type_pixel_values_dict[p_id]["albedo_srgb_gt"]
        )
        albedo_raw_pred = np.array(
            point_img_type_pixel_values_dict[p_id]["albedo_raw_pred"]
        )
        albedo_raw_gt = np.array(
            point_img_type_pixel_values_dict[p_id]["albedo_raw_gt"]
        )

        # Vectorize L2 loss calculation across frames
        pred_rgb_consistency.append(
            np.sqrt(np.sum((albedo_srgb_pred[1:] - albedo_srgb_pred[:-1]) ** 2, axis=1))
            / 2
        )
        gt_rgb_consistency.append(
            np.sqrt(np.sum((albedo_srgb_gt[1:] - albedo_srgb_gt[:-1]) ** 2, axis=1)) / 2
        )
        pred_raw_consistency.append(
            np.sqrt(np.sum((albedo_raw_pred[1:] - albedo_raw_pred[:-1]) ** 2, axis=1))
            / 2
        )
        gt_raw_consistency.append(
            np.sqrt(np.sum((albedo_raw_gt[1:] - albedo_raw_gt[:-1]) ** 2, axis=1)) / 2
        )

    # Convert results to arrays for more efficient further processing if needed
    if not pred_rgb_consistency:
        raise ValueError(
            "No evaluation point was visible from two or more of the rendered "
            "views, so albedo consistency is undefined. Render more views, or "
            "pick a --source_area_indices region the camera actually sees."
        )
    def print_mean_std(name, data, res_dict):
        # data is a list with one entry per point, holding that point's
        # frame-to-frame differences. A point that leaves the frame in some views
        # contributes a shorter entry than one visible throughout, so the lists
        # are ragged and cannot be stacked into a rectangular array. Taking the
        # per-point statistic first and averaging over points is what the
        # rectangular version computed, so scenes whose points are all visible
        # everywhere give exactly the same numbers as before.
        mean = np.mean([np.mean(point) for point in data]) / 255.0
        std = np.mean([np.std(point) for point in data]) / 255.0
        print(f"{name} mean: {mean:.4f}, std: {std:.4f}")
        res_dict[name] = {"mean": mean, "std": std}

    res_dict = {}
    print_mean_std("pred_rgb_consistency", pred_rgb_consistency, res_dict)
    print_mean_std("gt_rgb_consistency", gt_rgb_consistency, res_dict)
    print_mean_std("pred_raw_consistency", pred_raw_consistency, res_dict)
    print_mean_std("gt_raw_consistency", gt_raw_consistency, res_dict)

    # Save results to a json file
    with open(
        os.path.join(scene_manager.test_log_dir, "albedo_consistency.json"), "w"
    ) as f:
        json.dump(res_dict, f, indent=4)


def change_brightness_shading(
    args,
    scene_manager,
):
    """Scale the dominant shading direction of every point, then render.

    This is the scene-level illumination edit: PCA the shading half of the
    point features, scale the first component, and invert the transform.
    """
    shading_feat_size = (
        scene_manager.model.transformer.embed.dim_point_feat_MLP_1_shading
    )
    original_point_features = scene_manager.model.pc_feats.clone()

    original_shading_features = original_point_features[:, :shading_feat_size]

    # we will do PCA on the shading features
    pca = PCA(n_components=original_shading_features.shape[-1])
    mean_original_shading_features = np.mean(
        original_shading_features.detach().cpu().numpy(), axis=0
    )
    original_shading_features = (
        original_shading_features.detach().cpu().numpy()
        - mean_original_shading_features
    )
    pca.fit(original_shading_features)

    # print the eigen values to let the user choose the number of principal components we will keep
    print("PCA eigen values:", pca.explained_variance_ratio_)
    print("PCA eigen values sum:", np.sum(pca.explained_variance_ratio_))

    top_k_pca = 1  # we found the first eigen value is the most important

    if args.intensity_sweep:
        start = args.intensity_start_range
        stop = args.intensity_end_range
        step = (stop - start) / args.intensity_num_steps
        factors = [start + i * step for i in range(int((stop - start) / step))]
    else:
        factors = [args.shading_intensity]
    projected_shading_features = pca.transform(original_shading_features)
    for i in range(top_k_pca):
        for factor in factors:
            print("PCA component:", i, "factor:", factor)
            new_projected_shading_features = projected_shading_features.copy()
            new_projected_shading_features[:, i] *= factor

            new_shading_features = pca.inverse_transform(new_projected_shading_features)
            new_shading_features += mean_original_shading_features
            new_points_features = original_point_features.clone()
            args.shading_intensity = factor
            new_points_features[:, :shading_feat_size] = torch.from_numpy(
                new_shading_features
            ).to(scene_manager.device)
            scene_manager.model.pc_feats = torch.nn.Parameter(new_points_features)
            render_frames(scene_manager=scene_manager, sample_idx=0)


def get_frames_and_camera_poses(scene_manager):
    frames = []
    camera_poses = []
    if scene_manager.args.render_frame_type == "onfly":
        camera_poses = get_render_poses(scene=scene_manager.scene_config.index)
        frames = range(camera_poses.shape[0])
    elif scene_manager.args.render_frame_type == "all":
        frames = range(len(scene_manager.eval_dataset))
        camera_poses = None
    elif scene_manager.args.render_frame_type == "custom":
        # "x,y,z" -> [int(x), int(y), int(z)]
        frames = [int(frame) for frame in scene_manager.args.custom_frames.split(",")]
        camera_poses = None
    elif scene_manager.args.render_frame_type == "range":
        if (
            scene_manager.args.render_frame_start_index is not None
            and scene_manager.args.render_frame_end_index is not None
        ):
            if (
                scene_manager.args.render_frame_start_index
                < scene_manager.args.render_frame_end_index
            ):
                frames = range(
                    scene_manager.args.render_frame_start_index,
                    scene_manager.args.render_frame_end_index + 1,
                )
            else:
                part_1 = range(
                    scene_manager.args.render_frame_start_index,
                    len(scene_manager.eval_dataset),
                )
                part_2 = range(0, scene_manager.args.render_frame_end_index)
                union_set = set(part_1) | set(part_2)
                frames = sorted(union_set)

            # print with green color the start and end
            print(
                f"Start frame: {scene_manager.args.render_frame_start_index}, End frame: {scene_manager.args.render_frame_end_index}"
            )
        elif (
            scene_manager.args.render_frame_start_index is not None
            and scene_manager.args.render_frame_end_index is None
        ):
            frames.append(int(scene_manager.args.render_frame_start_index))
        else:
            frames = range(len(scene_manager.eval_dataset))
        camera_poses = None
    else:
        raise ValueError("Invalid render_frame_type")
    return frames, camera_poses


def calculate_average_and_format(dictionary):
    for key, value in dictionary.items():
        for metric, spaces in value.items():
            for space_key, values_list in spaces.items():
                if values_list:  # Check if the list is not empty
                    dictionary[key][metric][space_key] = np.mean(values_list)
                else:
                    dictionary[key][metric][
                        space_key
                    ] = None  # Set None if the list is empty
    return dictionary


def initialize_loss_dictionary():
    return {
        "render": {
            "loss": {"pred_space": [], "srgb_space": []},
            "psnr": {"pred_space": [], "srgb_space": []},
            "ssim": {"pred_space": [], "srgb_space": []},
            "lpips_alex": {"pred_space": [], "srgb_space": []},
            "lpips_vgg": {"pred_space": [], "srgb_space": []},
        },
        "albedo": {
            "loss": {"pred_space": [], "srgb_space": []},
            "psnr": {"pred_space": [], "srgb_space": []},
            "ssim": {"pred_space": [], "srgb_space": []},
            "lpips_alex": {"pred_space": [], "srgb_space": []},
            "lpips_vgg": {"pred_space": [], "srgb_space": []},
        },
        "shading": {
            "loss": {"pred_space": [], "srgb_space": []},
            "psnr": {"pred_space": [], "srgb_space": []},
            "ssim": {"pred_space": [], "srgb_space": []},
            "lpips_alex": {"pred_space": [], "srgb_space": []},
            "lpips_vgg": {"pred_space": [], "srgb_space": []},
        },
    }


def render_frames(
    scene_manager,
    sample_idx=1,
    keep_results=False,
):

    loss_dictionary = initialize_loss_dictionary()
    return_dict = {
        "render_srgb_pred": [],
        "render_srgb_gt": [],
        "render_raw_pred": [],
        "render_raw_gt": [],
        "albedo_srgb_pred": [],
        "albedo_srgb_gt": [],
        "albedo_raw_pred": [],
        "albedo_raw_gt": [],
        "selected_points": [],
        "selected_points_index": [],
        "selected_points_att": [],
        "depth_maps": [],
        "camera_poses": [],
        "frames": [],
    }

    frames, camera_poses = get_frames_and_camera_poses(scene_manager)

    for frame_idx in frames:
        t1 = time.time()
        frame = render_single_frame(
            frame_idx=frame_idx,
            sample_idx=sample_idx,
            loss_dictionary=loss_dictionary,
            selected_source_points_index=None,
            camera_poses=camera_poses,
            scene_manager=scene_manager,
        )
        if keep_results:
            # the two names that differ between the container and the collected dict
            renamed = {"depth_map": "depth_maps", "camera_pose_np": "camera_poses"}
            for field in FrameOutputs.__slots__:
                return_dict[renamed.get(field, field)].append(getattr(frame, field))
            return_dict["frames"].append(frame_idx)

        t2 = time.time()
        print(f"Frame {frame_idx} took {t2 - t1:.4f} seconds")

    updated_loss_dict = calculate_average_and_format(loss_dictionary)

    with open(
        os.path.join(scene_manager.test_log_dir, "loss_averages.txt"), "w"
    ) as file:
        file.write(
            str(updated_loss_dict)
            .replace(",", ",\n")
            .replace("{", "{\n")
            .replace("}", "\n}")
            .replace(" ", "    ")
        )
    return return_dict


def compute_depth_map_from_attention(selected_points, attn, rayo, scene_manager):
    if selected_points is None or attn is None or rayo is None:
        return None
    # Distance of every selected point to the image plane through the camera origin.
    plane_normal = -rayo
    plane_offset = torch.sum(plane_normal * rayo)
    normal_norm = torch.norm(plane_normal) + 1e-8
    point_distances = (
        torch.abs(
            torch.sum(selected_points.to(plane_normal.device) * plane_normal, dim=-1)
            - plane_offset
        )
        / normal_norm
    )
    if (
        scene_manager.model.bkg_feats is not None
        and scene_manager.model.bkg_type == 1
        and scene_manager.step <= scene_manager.scene_config.training.bkg_step
    ):
        num_bkg_feats = scene_manager.model.bkg_feats.shape[0]
        bkg_distance_padding = torch.zeros(
            point_distances.shape[0],
            point_distances.shape[1],
            point_distances.shape[2],
            num_bkg_feats,
            device=point_distances.device,
        )
        point_distances = torch.cat([point_distances, bkg_distance_padding], dim=-1)
    depth = torch.sum(
        attn.squeeze(-1).to(point_distances.device) * point_distances, dim=-1
    )
    return depth.squeeze().detach().cpu().numpy().astype(np.float32)


def transfer_points_features(scene_manager, original_pc_feats, args):
    print("transfering points features")
    print("scene: ", scene_manager.scene_config.index)

    new_points_features = original_pc_feats.clone()

    shading_feat_size = scene_manager.model.transformer.embed.dim_point_feat_MLP_1_shading
    albedo_feat_size = scene_manager.model.transformer.embed.dim_point_feat_MLP_2_albedo

    print("shading_feat_size: ", shading_feat_size)
    print("albedo_feat_size: ", albedo_feat_size)
    print("transfer type: ", args.test_action)

    # source_points_indices is a dictionary with the key as the area index and the value as the points indices
    target_area_point_indices = scene_manager.target_area_indices
    source_area_point_indices = scene_manager.source_area_indices

    for source_area_idx in source_area_point_indices.keys():
        if args.use_source_point_index:
            source_points_indices = [args.source_point_index]
        else:
            if int(args.how_many_source_area_points) == -1:
                source_points_indices = source_area_point_indices[source_area_idx]
                print("Using user provided source points: ", source_points_indices)
            else:
                source_points_indices = np.random.choice(
                    source_area_point_indices[source_area_idx],
                    args.how_many_source_area_points,
                    replace=False,
                )
                args.source_point_index = source_points_indices[0]
                print("Method for transfering features is mean")

        if args.test_action == "transfer_shading" or args.test_action == "freeform_transfer_shading":
            source_points_features_shading = scene_manager.model.pc_feats[source_points_indices, :shading_feat_size]
        if args.test_action == "transfer_albedo" or args.test_action == "freeform_transfer_albedo":
            source_points_features_albedo = scene_manager.model.pc_feats[source_points_indices, shading_feat_size:]

    if args.test_action == "transfer_shading" or args.test_action == "freeform_transfer_shading":
        mean_points_features_shading_source = args.shading_intensity * torch.mean(
            source_points_features_shading, dim=0, keepdim=True
        )
    if args.test_action == "transfer_albedo" or args.test_action == "freeform_transfer_albedo":
        mean_points_features_albedo_source = args.color_intensity * torch.mean(
            source_points_features_albedo, dim=0, keepdim=True
        )

    for target_area_idx in target_area_point_indices.keys():
        target_points_indices = target_area_point_indices[target_area_idx]

        # every target point takes the source mean, in the half of the feature
        # vector the action names
        if args.test_action in ("transfer_shading", "freeform_transfer_shading"):
            new_points_features[target_points_indices, :shading_feat_size] = (
                mean_points_features_shading_source.repeat(
                    len(target_points_indices), 1
                ).to(scene_manager.device)
            )
        if args.test_action in ("transfer_albedo", "freeform_transfer_albedo"):
            new_points_features[target_points_indices, shading_feat_size:] = (
                mean_points_features_albedo_source.repeat(
                    len(target_points_indices), 1
                ).to(scene_manager.device)
            )

    # model.pc_feats is nn.Parameter
    scene_manager.model.pc_feats = torch.nn.Parameter(
        new_points_features
    )

    return source_points_indices, target_points_indices


def do_action_transfer_albedo_shading(args, scene_manager):

    original_pc_feats = scene_manager.original_pc_feats.clone()
    n_transfers = args.how_many_samples
    if args.source_point_index is not None:
        n_transfers = 1

    for i in range(n_transfers):
        selected_source_points_index, affected_points_index = transfer_points_features(
            scene_manager=scene_manager,
            original_pc_feats=original_pc_feats,
            args=args,
        )

        render_frames(
            scene_manager=scene_manager,
            sample_idx=selected_source_points_index[0],
            keep_results=False,
        )

        # save the transfer mask
        if i == (args.how_many_samples - 1):
            frames, _ = get_frames_and_camera_poses(scene_manager=scene_manager)
            for f_index in frames:
                c2w = scene_manager.eval_dataset.c2w[f_index]
                c2w[-1, -1] = 1.0
                points_pixels_target = find_proj_coord(
                    pc=scene_manager.model.points[affected_points_index]
                    .detach()
                    .cpu()
                    .numpy(),
                    c2w=c2w,
                    H=scene_manager.eval_dataset.H,
                    W=scene_manager.eval_dataset.W,
                    focal_x=scene_manager.eval_dataset.focal_x,
                    focal_y=scene_manager.eval_dataset.focal_y,
                )
                points_pixels_source = find_proj_coord(
                    pc=scene_manager.model.points[selected_source_points_index]
                    .detach()
                    .cpu()
                    .numpy(),
                    c2w=c2w,
                    H=scene_manager.eval_dataset.H,
                    W=scene_manager.eval_dataset.W,
                    focal_x=scene_manager.eval_dataset.focal_x,
                    focal_y=scene_manager.eval_dataset.focal_y,
                )
                # create an image of size WxH . all pixels are 0 and the points pixels are 1
                points_pixels_image = np.zeros(
                    (scene_manager.eval_dataset.H, scene_manager.eval_dataset.W)
                )
                # a point can project outside the frame from some views
                for pixels, value in (
                    (points_pixels_target, 255),
                    (points_pixels_source, 125),
                ):
                    for point_pixel in pixels:
                        row, col = int(point_pixel[1]), int(point_pixel[0])
                        if (
                            0 <= row < scene_manager.eval_dataset.H
                            and 0 <= col < scene_manager.eval_dataset.W
                        ):
                            points_pixels_image[row, col] = value
                # save the image using PIL
                points_pixels_image = Image.fromarray(points_pixels_image).convert(
                    "RGB"
                )
                name_to_save = get_name_to_save(
                    frame_index=f_index,
                    image_type="transfer-mask",
                    scene_manager=scene_manager,
                    sample_idx=str(i + 1),
                    selected_source_points_index=str(selected_source_points_index[0]),
                )
                name_to_save += ".png"
                transfer_mask_path = os.path.join(scene_manager.test_log_dir, name_to_save)
                points_pixels_image.save(transfer_mask_path)
                print("Saved image: ", transfer_mask_path)


def do_action_rendering(scene_manager):
    render_frames(
        scene_manager=scene_manager,
        sample_idx=0,
    )


def read_pixel_coordinates(file_path):
    """Read a stroke file: one "x,y" integer pixel per line, blank lines ignored."""
    coordinates = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(",")
            if len(values) != 2:
                raise ValueError(
                    "Invalid coordinate format in line: {}. Expected 'x,y'".format(line)
                )
            coordinates.append([int(value) for value in values])
    return coordinates


def do_action_change_brightness(args, scene_manager):
    change_brightness_shading(args=args, scene_manager=scene_manager)


def do_action_albedo_consistency(args, scene_manager):
    calculate_albedo_consistency(args=args, scene_manager=scene_manager)


# Every --test_action, and the handler that runs it. All handlers take
# (args, scene_manager), so adding an action means adding one row here.
TEST_ACTIONS = {
    "render": lambda args, manager: do_action_rendering(scene_manager=manager),
    "transfer_albedo": do_action_transfer_albedo_shading,
    "transfer_shading": do_action_transfer_albedo_shading,
    "freeform_transfer_albedo": do_action_transfer_albedo_shading,
    "freeform_transfer_shading": do_action_transfer_albedo_shading,
    "change_brightness": do_action_change_brightness,
    "calculate_albedo_consistency": do_action_albedo_consistency,
}

# The actions that can take their source and target regions from stroke files.
FREEFORM_ACTIONS = (
    "transfer_albedo",
    "transfer_shading",
    "freeform_transfer_albedo",
    "freeform_transfer_shading",
    "change_brightness",
    "calculate_albedo_consistency",
)


if __name__ == "__main__":
    config, args = get_args()
    args.stage = "test"

    log_dir = os.path.join(config["save_dir"], config["index"])
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, "test.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, "test_error.log"), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    if args.opt != os.path.join(log_dir, os.path.basename(args.opt)):
        copy_config_for_replay(args.opt, log_dir)

    setup_seed(config["seed"])

    if args.source_point_index is not None or args.test_action == "render":
        args.how_many_samples = 1

    # the scene to test, named scene_1 in every shipped config
    scene_keys = sorted(
        key for key in config if key.startswith("scene_") and len(key) == 7
    )
    assert len(scene_keys) == 1, (
        "Testing runs one scene at a time; the config declares: {}".format(scene_keys)
    )
    scene_key = scene_keys[0]

    if args.test_dataset_path is not None:
        for dataset in config[scene_key]["test"]["datasets"]:
            dataset["path"] = args.test_dataset_path
    for dataset in config[scene_key]["test"]["datasets"]:
        # A key left blank in a test block means "not specified here", so it must
        # not overwrite what the scene dataset block or the command line already
        # set. mipnerf360.yml leaves `factor:` blank, and the generated
        # --scene_N.test.datasets.* flags cannot reach into this list, so without
        # this filter a Mip-NeRF 360 scene can never be tested.
        config[scene_key]["dataset"].update(
            {key: value for key, value in dataset.items() if value is not None}
        )
    scene_idx = int(scene_key.split("_")[1])

    scene_manager = SceneManager(
        args=args,
        all_configs=config,
        scene_config=config[scene_key],
        eval_config=config[scene_key],
        scene_key=scene_key,
        scene_idx=scene_idx - 1,
        cuda_idx=args.gpu_id,
    )
    scene_manager.setup_test_steps()

    if (
        args.test_action in FREEFORM_ACTIONS
        and args.source_target_area_selection_method == "freeform_pixels"
    ):
        source_points = sorted(
            collect_points_under_strokes(
                scene_manager=scene_manager,
                frame_idx=args.freeform_source_key_frame_index,
                stroke_pixels=read_pixel_coordinates(args.source_area_path),
                selection_method=args.freeform_source_point_method,
            )
        )
        target_points = sorted(
            collect_points_under_strokes(
                scene_manager=scene_manager,
                frame_idx=args.freeform_target_key_frame_index,
                stroke_pixels=read_pixel_coordinates(args.target_area_path),
                selection_method=args.freeform_target_point_method,
            )
        )
        print("Freeform # of source points: ", len(source_points))
        print("Freeform # of target points: ", len(target_points))
        scene_manager.source_area_indices["freeform"] = source_points
        scene_manager.target_area_indices["freeform"] = target_points

    handler = TEST_ACTIONS.get(args.test_action)
    if handler is None:
        raise NotImplementedError(
            "Unknown --test_action {}. Choose one of: {}".format(
                args.test_action, ", ".join(sorted(TEST_ACTIONS))
            )
        )
    handler(args, scene_manager)

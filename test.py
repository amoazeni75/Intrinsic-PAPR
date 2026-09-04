import glob
import json
import os
import shutil
import sys
import time

import imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset.utils import *
from models.utils import *
from scene_manager import SceneManager
from tools.args_parser import *
from tools.logger import *

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"


try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity

    def compare_ssim(gt, img, win_size, channel_axis=2):
        return structural_similarity(
            gt, img, win_size=win_size, channel_axis=channel_axis
        )


def get_srgb_render_albedo_shading_from_feature_map(
    model,
    feature_map,
    attn,
    topk,
    rayd,
    albedo_feat_size,
    albedo_feat_side,
    scene_manager,
):
    print(
        "\033[91m"
        + "********** getting rgb and albedo from feature map **********"
        + "\033[0m"
    )
    print("albedo UNet feat size: ", albedo_feat_size)
    print("albedo_feat_side: ", albedo_feat_side)

    rgb = None
    albedo_pred_test = None
    shading_pred_test = None
    foreground_rgb = None
    foreground_albedo = None
    foreground_shading = None

    N, H, W, _ = rayd.shape
    with torch.no_grad():
        # albedo
        background_mask = (
            (attn[..., topk:, :] * model.bkg_feats.expand(N, H, W, -1, -1))
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        attention_mask = attn[..., topk:, :].squeeze().detach().cpu().numpy()
        if scene_manager.scene_config.models.use_albedo:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                albedo_input_features = extract_features_from_feature_map(
                    features_map=feature_map,
                    features_dim=albedo_feat_size,
                    side=albedo_feat_side,
                )

                foreground_albedo = (
                    model.albedo_model(
                        albedo_input_features.squeeze(-2).permute(0, 3, 1, 2)
                    )
                    .permute(0, 2, 3, 1)
                    .unsqueeze(-2)
                )
                if model.bkg_feats is not None:
                    bkg_attn = attn[..., topk:, :]
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
                    bkg_attn = attn[..., topk:, :]
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
                foreground_rgb_unsqueezed = foreground_rgb
                foreground_rgb = foreground_rgb.squeeze()
        elif scene_manager.scene_config.models.use_implicit_renderer:
            # albedo and shading are in log space and unbounded -> inv_trans -> log_pred_raw -> add together
            rgb = cacluate_rgb_from_albedo_and_shading(
                albedo=albedo_pred_test,
                shading=shading_pred_test,
                scene_config=model.scene_config,
            )

    background_mask = np.clip(background_mask, 0, 1)
    attention_mask = np.clip(attention_mask, 0, 1)

    if shading_pred_test is None:
        # we will calculate the shading image on fly using rgb / albedo
        # for shading we make it gray scale
        foreground_shading = foreground_rgb_unsqueezed / (
            foreground_albedo
            + float(scene_manager.scene_config.models.predict_in_log_space_eps)
        )
        bkg_feats = model.bkg_feats.expand(N, H, W, -1, -1)
        if scene_manager.args.render_bg_black:
            # render the background as black: invert the clamped bkg feats
            bkg_feats = 1 - torch.clamp(bkg_feats, 0, 1)
        foreground_shading -= args.decrease_shading_constant
        shading_pred_test = foreground_shading * (1 - bkg_attn) + bkg_feats * bkg_attn
        shading_pred_test = torch.mean(
            shading_pred_test.squeeze(-2), dim=-1, keepdim=True
        )

    return (
        rgb,
        albedo_pred_test,
        shading_pred_test,
        foreground_rgb,
        foreground_albedo,
        foreground_shading,
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
        or scene_manager.args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading"
        or scene_manager.args.test_action == "freefrom_transfer_albedo"
        or scene_manager.args.test_action == "freefrom_transfer_shading"
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
                if scene_manager.args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading"
                else scene_manager.args.color_intensity
            ),
        )
    if scene_manager.args.test_action == "change_brightness":
        extra_info += "{}-intensity-{:.4f}".format(
            (
                "col"
                if scene_manager.args.test_action == "interpolate_albedo"
                else "shd"
            ),
            (
                scene_manager.args.color_intensity
                if scene_manager.args.test_action == "interpolate_albedo"
                else scene_manager.args.shading_intensity
            ),
        )

    if scene_manager.args.test_action == "change_brightness":
        if selected_source_points_index is not None:
            extra_info += "-using-pt-{}".format(selected_source_points_index)
        else:
            extra_info += "-whole-image"

    if scene_manager.args.test_action == "interpolate_albedo":
        color_names = scene_manager.args.interpolate_colors_name.split(",")
        color_indices = [
            int(color_index)
            for color_index in scene_manager.args.interpolate_colors_indices.split(",")
        ]
        color_percentages = [
            float(percentage)
            for percentage in scene_manager.args.interpolate_colors_percentage
        ]
        for i, color_name in enumerate(color_names):
            extra_info += "-{}-{}-{}".format(
                color_name, color_indices[i], color_percentages[i]
            )
        extra_info += "-use-pca" if scene_manager.args.use_pca_for_interpolation else ""
        extra_info += "-col-intensity-{}-shd-intensity-{}".format(
            scene_manager.args.color_intensity, scene_manager.args.shading_intensity
        )
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
        test_psnr = 0.0
        test_ssim = 1.0
        test_lpips_alex = 0.0
        test_lpips_vgg = 0.0
        loss = torch.tensor(0.0)
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
    print_message = f"({space}) Frame: {frame_index}, {img_type}, loss: {loss:.4f},  test_psnr: {test_psnr:.4f}, test_ssim: {test_ssim:.4f}, test_lpips_alex: {test_lpips_alex:.4f}, test_lpips_vgg: {test_lpips_vgg:.4f}"
    print(print_message)

    if scene_manager.args.include_metrics_in_name:
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

    # add losses to the loss log dictionary
    loss_log_dictionary[img_type]["loss"][space].append(loss.item())
    loss_log_dictionary[img_type]["psnr"][space].append(test_psnr)
    loss_log_dictionary[img_type]["ssim"][space].append(test_ssim)
    loss_log_dictionary[img_type]["lpips_alex"][space].append(test_lpips_alex)
    loss_log_dictionary[img_type]["lpips_vgg"][space].append(test_lpips_vgg)


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

    srgb_pred = preprocess_postproces_images_pipeline(
        img=raw_pred,
        pipline=scene_manager.scene_config.test.datasets[0][
            f"{img_type}_pred_postprocessing"
        ],
        eps=scene_manager.scene_config.models.predict_in_log_space_eps,
        min_val=(
            getattr(
                scene_manager.scene_config.dataset, "min_{}_log".format("render"), None
            )
            if scene_manager.scene_config.models.predict_rgb_in_log_space
            or scene_manager.scene_config.models.predict_raw_in_log_space
            else 0
        ),
        max_val=(
            getattr(
                scene_manager.scene_config.dataset, "max_{}_log".format("render"), None
            )
            if scene_manager.scene_config.models.predict_rgb_in_log_space
            or scene_manager.scene_config.models.predict_raw_in_log_space
            else 1
        ),
        white_bg_value=getattr(
            scene_manager.scene_config.geoms.background, "render_init_scale", None
        ),
        supervision_scaler=None,
    )
    if raw_gt is not None:
        srgb_gt = preprocess_postproces_images_pipeline(
            img=raw_gt,
            pipline=scene_manager.scene_config.test.datasets[0][
                f"{img_type}_GT_postprocessing"
            ],
            eps=scene_manager.scene_config.models.predict_in_log_space_eps,
            min_val=(
                getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format(img_type),
                    None,
                )
                if scene_manager.scene_config.models.predict_rgb_in_log_space
                or scene_manager.scene_config.models.predict_raw_in_log_space
                else 0
            ),
            max_val=(
                getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format(img_type),
                    None,
                )
                if scene_manager.scene_config.models.predict_rgb_in_log_space
                or scene_manager.scene_config.models.predict_raw_in_log_space
                else 1
            ),
            white_bg_value=getattr(
                scene_manager.scene_config.geoms.background, "render_init_scale", None
            ),
            supervision_scaler=None,
        )
    else:
        srgb_gt = None

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


def render_single_frame(
    frame_idx,
    sample_idx,
    loss_dictionary,
    selected_source_points_index,
    camera_poses,
    scene_manager,
    target_pixels=None,
    get_target_points_only=False,
    target_pixels_method=None,
):
    scene_manager.model.move_shared_components_to_device()
    if camera_poses is not None:
        camera_pose = camera_poses[frame_idx].unsqueeze(0)  # (1, 4, 4)
        idx = torch.tensor([frame_idx])
        test_render_GT = None
        test_albedo_GT = None
        test_shading_GT = None
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
        # shading is derived from the render and the albedo, it has no ground truth
        test_shading_GT = None
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
    if test_shading_GT is not None:
        test_shading_GT = test_shading_GT.to(scene_manager.device)

    topk = min([num_pts, scene_manager.model.select_k])
    pt_idxs = [topk * i // 5 for i in range(5)]

    selected_points = torch.zeros(1, H, W, topk, 3)
    selected_points_att = torch.zeros(1, H, W, topk, 1)
    selected_points_index = torch.zeros(1, H, W, topk)

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
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(scene_manager.device)

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
                    c2w,
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
                ] = scene_manager.model.top_k_att_TSNE

        # Get target points and attention vectors if target_pixels is provided
        all_freefrom_target_points = set()
        per_pixel_attentions = {}
        per_point_opacity = {}
        if target_pixels is not None:
            for x, y, alpha in target_pixels:
                if 0 <= x < W and 0 <= y < H:
                    # Get all points and their attention values for this pixel
                    if target_pixels_method == "all":
                        pixel_points = selected_points_index[0, y, x].cpu().numpy()
                        pixel_attn = selected_points_att[0, y, x].cpu().numpy()
                        # convert items in pixel_points to int
                        pixel_points = [int(item) for item in pixel_points]
                    elif target_pixels_method == "highest_attention":
                        pixel_points = selected_points_index[0, y, x].cpu().numpy()
                        pixel_attn = selected_points_att[0, y, x].cpu().numpy()
                        # convert items in pixel_points to int
                        pixel_points = [int(item) for item in pixel_points]
                        # get the index of the highest attention value
                        highest_attn_index = np.argmax(pixel_attn)
                        pixel_points = [pixel_points[highest_attn_index]]
                        pixel_attn = [pixel_attn[highest_attn_index]]
                    else:
                        raise ValueError(f"Invalid freeform selection method: {scene_manager.args.freeform_select_all_points}")

                    all_freefrom_target_points.update(pixel_points)
                    for point_idx in pixel_points:
                        per_point_opacity[point_idx] = alpha
                    per_pixel_attentions[f"{x},{y}"] = pixel_attn

        if get_target_points_only:
            return all_freefrom_target_points, per_pixel_attentions, per_point_opacity

        # Here we either get the prediction without transfering features or with transfering features depending on the args.transfer_feature
        (
            test_render_prediction_raw,
            test_albedo_prediction_raw,
            test_shading_prediction_raw,
            test_render_prediction_foreground_raw,
            test_albedo_prediction_foreground_raw,
            test_shading_prediction_foreground_raw,
            background_mask,
            attention_mask,
        ) = get_srgb_render_albedo_shading_from_feature_map(
            model=scene_manager.model,
            feature_map=feature_map,
            attn=attn,
            topk=topk,
            rayd=rayd,
            albedo_feat_size=scene_manager.model.albedo_UNet_inp_size,
            albedo_feat_side=scene_manager.model.albedo_feat_side,
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
    if scene_manager.args.save_shading_images:
        calculate_loss_and_save_image(
            raw_pred=test_shading_prediction_raw,
            raw_gt=test_shading_GT,
            img_type="shading",
            frame_index=frame_idx,
            sample_idx=sample_idx,
            loss_log_dictionary=loss_dictionary,
            selected_source_points_index=selected_source_points_index,
            scene_manager=scene_manager,
        )

    depth_map = compute_depth_map_from_attention(
        selected_points=selected_points,
        attn=attn,
        rayo=rayo,
        scene_manager=scene_manager,
    )
    camera_pose_np = c2w.detach().cpu().numpy()

    return (
        render_srgb_pred,
        render_srgb_gt,
        render_raw_pred,
        render_raw_gt,
        albedo_srgb_pred,
        albedo_srgb_gt,
        albedo_raw_pred,
        albedo_raw_gt,
        selected_points,
        selected_points_index,
        selected_points_att,
        depth_map,
        camera_pose_np,
    )


def load_selected_area_pixels_idx(area, load_path):
    # we will open the txt file and load the pixels idx
    # pixels x,y coordinates each line
    with open(os.path.join(load_path, "test", f"{area}_pixels.txt"), "r") as f:
        lines = f.readlines()
        pixels_in_area = []
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            line = line.split(",")
            pixels_in_area.append([int(float(line[0])), int(float(line[1]))])

    return pixels_in_area


def PCA_on_features(features):
    pca = PCA(n_components=features.shape[-1])
    mean_features = np.mean(features.detach().cpu().numpy(), axis=0)
    centered_original_features = features.detach().cpu().numpy() - mean_features
    pca.fit(centered_original_features)

    print("PCA eigen values:", pca.explained_variance_ratio_)
    print("PCA eigen values sum:", np.sum(pca.explained_variance_ratio_))

    projected_features = pca.transform(centered_original_features)

    return pca, projected_features, mean_features


def interpolate_albedo(args, scene_manager):

    color_indices = [
        int(color_index) for color_index in args.interpolate_colors_indices.split(",")
    ]
    if args.interpolate_colors_percentage is not None:
        color_percentage = [
            float(percentage) / 100.0
            for percentage in args.interpolate_colors_percentage
        ]
    else:
        color_percentage = None
    if len(color_indices) == 1:
        factors = [color_percentage]
    elif len(color_indices) == 2:
        if color_percentage is not None:
            factors = [color_percentage]
        else:
            factors = [0, 0.25, 0.5, 0.75, 1]
    elif len(color_indices) >= 3:
        factors = [color_percentage]

    target_area_points_idx = []
    for area_idx in managers[args.target_scene_index].target_area_indices:
        target_area_points_idx.extend(
            managers[args.target_scene_index].target_area_indices[area_idx]
        )
    # NOTE: these sizes come from the transformer embedding, i.e. they describe the
    # transformer-facing feature split, not the raw stored point features.
    albedo_feat_size = scene_manager.model.transformer.embed.dim_point_feat_MLP_2_albedo
    shading_feat_size = (
        scene_manager.model.transformer.embed.dim_point_feat_MLP_1_shading
    )  # not always the albedo_feat_size and shading_feat_size are the same

    original_point_features = scene_manager.model.pc_feats.clone()
    if scene_manager.model.albedo_feat_side == "right":
        original_albedo_features = original_point_features[:, shading_feat_size:]
    elif scene_manager.model.albedo_feat_side == "left":
        original_albedo_features = original_point_features[:, :albedo_feat_size]
    else:
        raise ValueError("The albedo_feat_side is not defined")

    # we need to do PCA on the albedo features
    if manager.args.use_pca_for_interpolation:
        albedo_pca, projected_albedo_features, mean_original_albedo_features = (
            PCA_on_features(original_albedo_features)
        )
    else:
        projected_albedo_features = original_albedo_features
    for factor in factors:
        if manager.args.use_pca_for_interpolation:
            new_projected_albedo_features = projected_albedo_features.copy()
        else:
            new_projected_albedo_features = (
                original_albedo_features.detach().cpu().numpy()
            )
            projected_albedo_features = original_albedo_features.detach().cpu().numpy()
        if isinstance(factor, list):
            new_color_features = (
                factor[0] * projected_albedo_features[color_indices[0], :]
            )
            for i in range(1, len(factor)):
                new_color_features += (
                    factor[i] * projected_albedo_features[color_indices[i], :]
                )
        else:
            new_color_features = (
                factor * projected_albedo_features[color_indices[0], :]
                + (1 - factor) * projected_albedo_features[color_indices[1], :]
            )
        new_color_features *= args.color_intensity
        new_projected_albedo_features[target_area_points_idx, :] = new_color_features

        if manager.args.use_pca_for_interpolation:
            new_albedo_features = albedo_pca.inverse_transform(
                new_projected_albedo_features
            )
            new_albedo_features += mean_original_albedo_features
        else:
            new_albedo_features = new_projected_albedo_features

        new_points_features = original_point_features.clone()
        if scene_manager.model.albedo_feat_side == "right":
            new_points_features[:, shading_feat_size:] = torch.from_numpy(
                new_albedo_features
            ).to(scene_manager.device)
        elif scene_manager.model.albedo_feat_side == "left":
            new_points_features[:, :albedo_feat_size] = torch.from_numpy(
                new_albedo_features
            ).to(scene_manager.device)
        else:
            raise ValueError("The albedo_feat_side is not defined")

        # shading featuer adjustment
        if scene_manager.model.albedo_feat_side == "right":
            new_points_features[target_area_points_idx, :shading_feat_size] = (
                manager.args.shading_intensity
                * original_point_features[target_area_points_idx, :shading_feat_size]
            )
        elif scene_manager.model.albedo_feat_side == "left":
            new_points_features[target_area_points_idx, albedo_feat_size:] = (
                manager.args.shading_intensity
                * original_point_features[target_area_points_idx, albedo_feat_size:]
            )

        # render results
        scene_manager.model.pc_feats = torch.nn.Parameter(new_points_features)
        render_frames(
            scene_manager=manager,
            sample_idx=args.interpolate_colors_indices,
            keep_results=False,
        )


def calculate_albedo_consistency(
    args,
    scene_manager,
):
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
            W=scene_manager.eval_dataset.W,
            focal=scene_manager.eval_dataset.focal_x,
        ).astype(int)

        # Collect pixel values in one go using array indexing
        for p_idx, p_id in enumerate(evaluation_points_id):
            px, py = points_pixels[p_idx]
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
    pred_rgb_consistency = np.array(pred_rgb_consistency)
    gt_rgb_consistency = np.array(gt_rgb_consistency)
    pred_raw_consistency = np.array(pred_raw_consistency)
    gt_raw_consistency = np.array(gt_raw_consistency)

    def print_mean_std(name, data, res_dict):
        # data is nxf array where n is number of points and f is number of frames
        # we take the mean across frames for each point and then the mean across points
        mean = np.mean(np.mean(data, axis=1)) / 255.0
        # we take the std across frames for each point and then the mean across points
        std = np.mean(np.std(data, axis=1)) / 255.0
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

    # transfer shading from source to the target
    if (
        args.source_point_index is not None
        and args.transfer_feature
        and args.use_points_features
    ):
        target_area_points_idx = managers[args.target_scene_index].target_area_indices
        source_area_points_idx = managers[args.source_scene_index].source_area_indices
        transfer_points_features(
            scene_managers=managers,
            source_scene_index=args.source_scene_index,
            target_scene_index=args.target_scene_index,
            source_points_indices=source_area_points_idx,
            targets_points_indices=target_area_points_idx,
            transfer_albedo=args.transfer_albedo,
            transfer_shading=args.transfer_shading,
        )

    shading_feat_size = (
        scene_manager.model.transformer.embed.dim_point_feat_MLP_1_shading
    )
    albedo_feat_size = scene_manager.model.transformer.embed.dim_point_feat_MLP_2_albedo
    original_point_features = scene_manager.model.pc_feats.clone()

    if scene_manager.model.albedo_feat_side == "right":
        original_shading_features = original_point_features[:, :shading_feat_size]
    elif scene_manager.model.albedo_feat_side == "left":
        original_shading_features = original_point_features[:, albedo_feat_size:]
    else:
        raise ValueError("The albedo_feat_side is not defined")

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

    if args.generate_gif_from_frames:
        # from 0.25 to 4 + 50 values in between
        start = args.intensity_start_range
        stop = args.intensity_end_range
        step = (stop - start) / args.intensity_num_steps
        factors = [start + i * step for i in range(int((stop - start) / step))]
    else:
        factors = [args.shading_intensity]
    projected_shading_features = pca.transform(original_shading_features)
    for i in range(top_k_pca):
        for factor in factors:
            print("********* PCA component:", i, "factor:", factor)
            new_projected_shading_features = projected_shading_features.copy()
            if args.transfer_feature:
                new_projected_shading_features[
                    target_area_points_idx[args.target_area_indices[0]], i
                ] *= factor
            else:
                new_projected_shading_features[:, i] *= factor

            new_shading_features = pca.inverse_transform(new_projected_shading_features)
            new_shading_features += mean_original_shading_features
            new_points_features = original_point_features.clone()
            args.shading_intensity = factor
            if scene_manager.model.albedo_feat_side == "right":
                new_points_features[:, :shading_feat_size] = torch.from_numpy(
                    new_shading_features
                ).to(scene_manager.device)
            elif scene_manager.model.albedo_feat_side == "left":
                new_points_features[:, albedo_feat_size:] = torch.from_numpy(
                    new_shading_features
                ).to(scene_manager.device)
            else:
                raise ValueError("The albedo_feat_side is not defined")
            scene_manager.model.pc_feats = torch.nn.Parameter(new_points_features)
            render_frames(scene_manager=scene_manager, sample_idx=0)


def collect_unique_point_ids_and_colors(point_index_map, reference_colors_image):
    """
    Collect, for every point id that is the top-attention point of at least one pixel,
    the reference colour of the first pixel it appears in. Point ids that hit any
    pure-white ([255, 255, 255]) reference pixel are skipped.

    Args:
    - point_index_map (torch.Tensor): point indices with shape (N, H, W, num_pts).
    - reference_colors_image (numpy.ndarray): reference colours with shape (H, W, 3).

    Returns:
    - selected_points_ids (list): List of selected points ids.
    - selected_points_colors (numpy.ndarray): Array of selected points colors.
    """
    # Initialize lists to store selected points ids and colors
    point_index_map = point_index_map.detach().cpu().numpy().astype(np.int32)
    selected_points_ids = []
    selected_points_colors = []

    # Extract dimensions
    N, H, W, num_pts = point_index_map.shape

    # Reshape the point indices to facilitate vectorized operations
    flat_point_index_map = point_index_map.reshape(N * H * W, num_pts)

    # Flatten the reference colours to facilitate vectorized operations
    flat_reference_colors = reference_colors_image.reshape(-1, 3)

    # Get unique ids in the point index map
    unique_ids = np.unique(flat_point_index_map[:, 0])

    # Find indices where the reference colour is [255, 255, 255]
    white_indices = np.where(
        (flat_reference_colors == np.array([255, 255, 255])).all(axis=1)
    )[0]

    for point_id in unique_ids:
        if point_id not in selected_points_ids:
            # Get the pixel indices where this point id is the top-attention point
            pixel_indices = np.where(flat_point_index_map[:, 0] == point_id)[0]
            # Skip the id if any of those pixels is white in the reference image
            if (
                len(pixel_indices) > 0
                and len(np.intersect1d(white_indices, pixel_indices)) == 0
            ):
                selected_points_ids.append(point_id)
                # Get the color for the first occurrence of the id
                selected_points_colors.append(flat_reference_colors[pixel_indices[0]])

    # Convert selected_points_colors to NumPy array
    selected_points_colors = np.array(selected_points_colors)

    return selected_points_ids, selected_points_colors


def generate_TSNE_plot(args, scene_manager):

    albedo_feat_size = scene_manager.model.transformer.embed.dim_point_feat_MLP_2_albedo
    shading_feat_size = (
        scene_manager.model.transformer.embed.dim_point_feat_MLP_1_shading
    )  # not always the albedo_feat_size and shading_feat_size are the same

    original_point_features = scene_manager.model.pc_feats.clone()
    if scene_manager.model.albedo_feat_side == "right":
        original_albedo_features = original_point_features[:, shading_feat_size:]
    elif scene_manager.model.albedo_feat_side == "left":
        original_albedo_features = original_point_features[:, :albedo_feat_size]
    else:
        raise ValueError("The albedo_feat_side is not defined")

    selected_points_ids = []
    selected_points_colors = []

    frames = args.TSEN_frames.split(",")
    for f in frames:
        args.render_frame_start_index = int(f)
        frame_results = render_frames(
            scene_manager=scene_manager, sample_idx=0, keep_results=True
        )
        render_srgb_pred = frame_results["render_srgb_pred"][0]
        albedo_srgb_pred = frame_results["albedo_srgb_pred"][0]
        selected_points_index = frame_results["selected_points_index"][0]
        selected_points_att = frame_results["selected_points_att"][0]

        # we keep the id of the point with highest atten in each selected_points -> (N, H, W, 1)
        highest_attention_indices = selected_points_att.argmax(
            dim=-2, keepdim=True
        ).squeeze(-1)
        # selected_points_index becomes the id of the point with the highest attention
        selected_points_index = torch.gather(
            selected_points_index, dim=-1, index=highest_attention_indices
        )
        reference_colors = (
            albedo_srgb_pred if args.TSEN_refrence == "albedo" else render_srgb_pred
        )
        this_frame_selected_points_ids, this_frame_selected_points_colors = (
            collect_unique_point_ids_and_colors(
                point_index_map=selected_points_index,
                reference_colors_image=reference_colors,
            )
        )
        # we will add the selected points ids and colors to the global lists if they are not already there
        for point_id, color in zip(
            this_frame_selected_points_ids, this_frame_selected_points_colors
        ):
            if point_id not in selected_points_ids:
                selected_points_ids.append(point_id)
                selected_points_colors.append(color)
        print("Number of selected points:", len(selected_points_ids))
        print("Frame:", f)

    albedo_features = original_albedo_features[selected_points_ids, :]
    print("Number of selected points:", len(selected_points_ids))

    # TSNE plot
    albedo_features = albedo_features.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=0)
    albedo_features_tsne = tsne.fit_transform(albedo_features)

    plt.figure(figsize=(10, 10))
    # for each point we have its color
    for i in range(len(selected_points_colors)):
        plt.scatter(
            albedo_features_tsne[i, 0],
            albedo_features_tsne[i, 1],
            color=(selected_points_colors[i] / 255).tolist(),
        )
    # save
    tsne_path = os.path.join(
        scene_manager.log_dir,
        f"TSNE_plot_views_{args.TSEN_frames.replace(',', '_')}_refrence_{args.TSEN_refrence}_points_{len(selected_points_colors)}.png",
    )
    plt.savefig(tsne_path)
    print("Saved image: ", tsne_path)


def generate_gif_from_frames(scene_manager):
    frames = []
    files = glob.glob(
        os.path.join(
            scene_manager.log_dir,
            "test-*.png",
        )
    )
    # print with red the number of frames
    print(
        "\033[91m"
        + "We will generate gif from frames for scene {}. Number of frames: ".format(
            scene_manager.scene_idx
        )
        + str(len(files))
        + "\033[0m"
    )
    if len(files) == 0:
        print("\033[91m" + "No frames found." + "\033[0m")
        return
    # sort by frame number, test-0000-.....png
    files = sorted(files, key=lambda x: int(x.split("-")[1]))
    for file in files:
        # verified: raw images do not need special handling here
        frames.append(imageio.imread(file))
    mp4_path = os.path.join(
        scene_manager.log_dir,
        "360_views.mp4",
    )
    imageio.mimsave(
        mp4_path,
        frames,
        fps=30,
        format="mp4",
    )
    print("Saved video: ", mp4_path)
    print("GIF generated successfully.")


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
                    len(scene_manager.dataset),
                )
                part_2 = range(0, scene_manager.args.render_frame_end_index)
                union_set = set(part_1) | set(part_2)
                frames = sorted(union_set)

            # print with green color the start and end
            print(
                f"\033[92mStart frame: {scene_manager.args.render_frame_start_index}, End frame: {scene_manager.args.render_frame_end_index}\033[0m"
            )
        elif (
            scene_manager.args.render_frame_start_index is not None
            and scene_manager.args.render_frame_end_index is None
        ):
            frames.append(int(scene_manager.args.render_frame_start_index))
        else:
            frames = range(len(scene_manager.dataset))
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
        (
            render_srgb_pred,
            render_srgb_gt,
            render_raw_pred,
            render_raw_gt,
            albedo_srgb_pred,
            albedo_srgb_gt,
            albedo_raw_pred,
            albedo_raw_gt,
            selected_points,
            selected_points_index,
            selected_points_att,
            depth_map,
            camera_pose_np,
        ) = render_single_frame(
            frame_idx=frame_idx,
            sample_idx=sample_idx,
            loss_dictionary=loss_dictionary,
            selected_source_points_index=None,
            camera_poses=camera_poses,
            scene_manager=scene_manager,
        )
        if keep_results:
            return_dict["render_srgb_pred"].append(render_srgb_pred)
            return_dict["render_srgb_gt"].append(render_srgb_gt)
            return_dict["render_raw_pred"].append(render_raw_pred)
            return_dict["render_raw_gt"].append(render_raw_gt)
            return_dict["albedo_srgb_pred"].append(albedo_srgb_pred)
            return_dict["albedo_srgb_gt"].append(albedo_srgb_gt)
            return_dict["albedo_raw_pred"].append(albedo_raw_pred)
            return_dict["albedo_raw_gt"].append(albedo_raw_gt)
            return_dict["selected_points"].append(selected_points)
            return_dict["selected_points_index"].append(selected_points_index)
            return_dict["selected_points_att"].append(selected_points_att)
            return_dict["depth_maps"].append(depth_map)
            return_dict["camera_poses"].append(camera_pose_np)
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


def save_depth_outputs(depth_map, output_dir):
    if depth_map is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    depth_map = np.asarray(depth_map, dtype=np.float32)
    np.save(os.path.join(output_dir, "depth.npy"), depth_map)

    valid_mask = np.isfinite(depth_map) & (depth_map > 1e-8)
    depth_norm = np.zeros_like(depth_map, dtype=np.float32)
    if np.any(valid_mask):
        valid_values = depth_map[valid_mask]
        depth_min = valid_values.min()
        depth_range = max(valid_values.max() - depth_min, 1e-8)
        depth_norm[valid_mask] = (valid_values - depth_min) / depth_range
    depth_display = depth_norm.copy()
    depth_display[~valid_mask] = np.nan

    cmap = plt.get_cmap("viridis")
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color="white")

    if np.any(valid_mask):
        depth_rgba = cmap(depth_display)
        depth_img = (depth_rgba[..., :3] * 255).astype(np.uint8)
    else:
        depth_img = np.full((*depth_map.shape, 3), 255, dtype=np.uint8)

    depth_png_path = os.path.join(output_dir, "depth.png")
    print("Saving depth image: ", depth_png_path)
    imageio.imwrite(depth_png_path, depth_img)


def save_point_cloud_view(points_np, camera_pose_np, output_path):
    if points_np is None or camera_pose_np is None:
        return
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        s=1.0,
        c=points_np[:, 2],
        cmap="viridis",
        alpha=0.5,
    )
    zoom_ratio = 1.0
    pts_min = points_np.min(axis=0)
    pts_max = points_np.max(axis=0)
    center = (pts_min + pts_max) * 0.5
    span = np.maximum(pts_max - pts_min, 1e-6)
    half_span = 0.5 * span * zoom_ratio
    ax.set_xlim(center[0] - half_span[0], center[0] + half_span[0])
    ax.set_ylim(center[1] - half_span[1], center[1] + half_span[1])
    ax.set_zlim(center[2] - half_span[2], center[2] + half_span[2])

    camera_origin = camera_pose_np[:3, 3]
    ax.scatter(
        camera_origin[0],
        camera_origin[1],
        camera_origin[2],
        c="red",
        s=30,
        label="camera",
    )
    forward = camera_pose_np[:3, 2]
    forward_norm = forward / (np.linalg.norm(forward) + 1e-8)
    elev = np.degrees(
        np.arctan2(forward_norm[2], np.linalg.norm(forward_norm[:2]) + 1e-8)
    )
    azim = np.degrees(np.arctan2(forward_norm[1], forward_norm[0] + 1e-8))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    print("Saving point cloud view: ", output_path)
    fig.savefig(output_path)
    plt.close(fig)


def transfer_points_features(scene_managers, original_pc_feats, args, per_point_opacity=None):
    print("\033[91m" + "********** transfering points features **********" + "\033[0m")
    print("source scene: ", scene_managers[args.source_scene_index].scene_config.index)
    print("target scene: ", scene_managers[args.target_scene_index].scene_config.index)

    new_points_features = original_pc_feats.clone()

    shading_feat_size = scene_managers[
        args.target_scene_index
    ].model.transformer.embed.dim_point_feat_MLP_1_shading
    albedo_feat_size = scene_managers[
        args.target_scene_index
    ].model.transformer.embed.dim_point_feat_MLP_2_albedo

    print("shading_feat_size: ", shading_feat_size)
    print("albedo_feat_size: ", albedo_feat_size)
    print(
        "albedo side: ", scene_managers[args.target_scene_index].model.albedo_feat_side
    )
    print("transfer type: ", args.test_action)

    # source_points_indices is a dictionary with the key as the area index and the value as the points indices
    target_area_point_indices = scene_managers[
        args.target_scene_index
    ].target_area_indices
    source_area_point_indices = scene_managers[
        args.source_scene_index
    ].source_area_indices

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

        if scene_managers[args.source_scene_index].model.albedo_feat_side == "right":
            if args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading":
                source_points_features_shading = scene_managers[
                    args.source_scene_index
                ].model.pc_feats[source_points_indices, :shading_feat_size]
            if args.test_action == "transfer_albedo" or args.test_action == "freefrom_transfer_albedo":
                source_points_features_albedo = scene_managers[
                    args.source_scene_index
                ].model.pc_feats[source_points_indices, shading_feat_size:]
        elif scene_managers[args.source_scene_index].model.albedo_feat_side == "left":
            if args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading":
                source_points_features_shading = scene_managers[
                    args.source_scene_index
                ].model.pc_feats[source_points_indices, albedo_feat_size:]
            if args.test_action == "transfer_albedo" or args.test_action == "freefrom_transfer_albedo":
                source_points_features_albedo = scene_managers[
                    args.source_scene_index
                ].model.pc_feats[source_points_indices, :albedo_feat_size]
        else:
            raise ValueError("The albedo_feat_side is not defined")

    if args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading":
        mean_points_features_shading_source = args.shading_intensity * torch.mean(
            source_points_features_shading, dim=0, keepdim=True
        )
    if args.test_action == "transfer_albedo" or args.test_action == "freefrom_transfer_albedo":
        mean_points_features_albedo_source = args.color_intensity * torch.mean(
            source_points_features_albedo, dim=0, keepdim=True
        )

    for target_area_idx in target_area_point_indices.keys():
        target_points_indices = target_area_point_indices[target_area_idx]

        # Create opacity tensor for target points
        if per_point_opacity is not None:
            # Create a tensor of ones with same length as target_points_indices
            opacity_tensor = torch.ones(len(target_points_indices), 1).to(scene_managers[args.target_scene_index].device)
            # Update opacity values for points that have them
            for i, point_idx in enumerate(target_points_indices):
                if point_idx in per_point_opacity:
                    opacity_tensor[i, 0] = per_point_opacity[point_idx]
        else:
            opacity_tensor = torch.ones(len(target_points_indices), 1).to(scene_managers[args.target_scene_index].device)

        if scene_managers[args.target_scene_index].model.albedo_feat_side == "right":
            if args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading":
                # Apply opacity to shading features
                new_points_features[target_points_indices, :shading_feat_size] = (
                    mean_points_features_shading_source.repeat(
                        len(target_points_indices), 1
                    ).to(scene_managers[args.target_scene_index].device) * opacity_tensor
                )
            if args.test_action == "transfer_albedo" or args.test_action == "freefrom_transfer_albedo":
                # Apply opacity to albedo features
                new_points_features[target_points_indices, shading_feat_size:] = (
                    mean_points_features_albedo_source.repeat(
                        len(target_points_indices), 1
                    ).to(scene_managers[args.target_scene_index].device) * opacity_tensor
                )
        elif scene_managers[args.target_scene_index].model.albedo_feat_side == "left":
            if args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_shading":
                # Apply opacity to shading features
                new_points_features[target_points_indices, albedo_feat_size:] = (
                    mean_points_features_shading_source.repeat(
                        len(target_points_indices), 1
                    ).to(scene_managers[args.target_scene_index].device) * opacity_tensor
                )
            if args.test_action == "transfer_albedo" or args.test_action == "freefrom_transfer_albedo":
                # Apply opacity to albedo features
                new_points_features[target_points_indices, :albedo_feat_size] = (
                    mean_points_features_albedo_source.repeat(
                        len(target_points_indices), 1
                    ).to(scene_managers[args.target_scene_index].device) * opacity_tensor
                )
        else:
            raise ValueError("The albedo_feat_side is not defined")

    # model.pc_feats is nn.Parameter
    scene_managers[args.target_scene_index].model.pc_feats = torch.nn.Parameter(
        new_points_features
    )

    return source_points_indices, target_points_indices


def do_action_transfer_albedo_shading(args, scene_managers, per_point_opacity=None):

    original_pc_feats = scene_managers[
        args.target_scene_index
    ].original_pc_feats.clone()
    n_transfers = args.how_many_samples
    if args.source_point_index is not None:
        n_transfers = 1

    for i in range(n_transfers):
        selected_source_points_index, affected_points_index = transfer_points_features(
            scene_managers=scene_managers,
            original_pc_feats=original_pc_feats,
            args=args,
            per_point_opacity=per_point_opacity,
        )

        manager = scene_managers[args.target_scene_index]

        render_frames(
            scene_manager=manager,
            sample_idx=selected_source_points_index[0],
            keep_results=False,
        )

        # save the transfer mask
        if i == (args.how_many_samples - 1):
            frames, _ = get_frames_and_camera_poses(scene_manager=manager)
            for f_index in frames:
                c2w = scene_managers[args.target_scene_index].eval_dataset.c2w[f_index]
                c2w[-1, -1] = 1.0
                points_pixels_target = find_proj_coord(
                    pc=manager.model.points[affected_points_index]
                    .detach()
                    .cpu()
                    .numpy(),
                    c2w=c2w,
                    W=manager.eval_dataset.W,
                    focal=manager.eval_dataset.focal_x,
                )
                points_pixels_source = find_proj_coord(
                    pc=manager.model.points[selected_source_points_index]
                    .detach()
                    .cpu()
                    .numpy(),
                    c2w=c2w,
                    W=manager.eval_dataset.W,
                    focal=manager.eval_dataset.focal_x,
                )
                # create an image of size WxH . all pixels are 0 and the points pixels are 1
                points_pixels_image = np.zeros(
                    (manager.eval_dataset.H, manager.eval_dataset.W)
                )
                for point_pixel in points_pixels_target:
                    points_pixels_image[int(point_pixel[1]), int(point_pixel[0])] = 255
                for point_pixel in points_pixels_source:
                    points_pixels_image[int(point_pixel[1]), int(point_pixel[0])] = 125
                # save the image using PIL
                points_pixels_image = Image.fromarray(points_pixels_image).convert(
                    "RGB"
                )
                name_to_save = get_name_to_save(
                    frame_index=f_index,
                    image_type="transfer-mask",
                    scene_manager=manager,
                    sample_idx=str(i + 1),
                    selected_source_points_index=str(selected_source_points_index[0]),
                )
                name_to_save += ".png"
                transfer_mask_path = os.path.join(manager.test_log_dir, name_to_save)
                points_pixels_image.save(transfer_mask_path)
                print("Saved image: ", transfer_mask_path)


def do_action_rendering(scene_manager):
    render_frames(
        scene_manager=scene_manager,
        sample_idx=0,
    )


def do_action_render_depth_pcd(scene_manager):
    frames, camera_poses = get_frames_and_camera_poses(scene_manager)
    frames = list(frames)
    if not frames:
        print("\033[91mNo frames provided for render_depth_pcd_for_comparison.\033[0m")
        return
    comparison_root = os.path.join(
        scene_manager.test_log_dir, "render_depth_pcd_for_comparison"
    )
    os.makedirs(comparison_root, exist_ok=True)
    loss_dictionary = initialize_loss_dictionary()
    points_np = scene_manager.model.points.detach().cpu().numpy()
    for frame_idx in frames:
        view_dir = os.path.join(comparison_root, f"view-{frame_idx:04d}")
        os.makedirs(view_dir, exist_ok=True)
        (
            render_srgb_pred,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            depth_map,
            camera_pose_np,
        ) = render_single_frame(
            frame_idx=frame_idx,
            sample_idx=0,
            loss_dictionary=loss_dictionary,
            selected_source_points_index=None,
            camera_poses=camera_poses,
            scene_manager=scene_manager,
        )
        rgb_to_save = (
            render_srgb_pred
            if isinstance(render_srgb_pred, np.ndarray)
            else np.zeros((1, 1, 3), dtype=np.uint8)
        )
        imageio.imwrite(
            os.path.join(view_dir, "render_rgb.png"),
            rgb_to_save,
        )
        save_depth_outputs(depth_map, view_dir)
        save_point_cloud_view(
            points_np,
            camera_pose_np,
            os.path.join(view_dir, "point_cloud.png"),
        )


def read_pixel_coordinates(file_path):
    """
    Read pixel coordinates from a text file.
    Each line should be in format: x,y or x,y,alpha
    If alpha is not provided, it defaults to 1.0
    Returns a list of [x,y,alpha] coordinates.
    """
    coordinates = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                values = line.split(",")
                if len(values) == 2:
                    x, y = map(int, values)
                    alpha = 1.0
                elif len(values) == 3:
                    x, y, alpha = map(float, values)
                    x, y = int(x), int(y)
                else:
                    raise ValueError(f"Invalid coordinate format in line: {line}. Expected 'x,y' or 'x,y,alpha'")
                coordinates.append([x, y, alpha])
    return coordinates


def get_freeform_points_indices(
    args, scene_managers, file_path, frame_idx, target_pixels_method
):
    """
    Handle free-form editing by reading target pixels from a file and transferring features.
    Uses attention-weighted feature transfer based on the attention vectors of target pixels.
    """
    # Read target pixels from file
    target_pixels = read_pixel_coordinates(file_path)

    # Get target points and per-point opacity by rendering a single frame
    target_points, _, per_point_opacity = render_single_frame(
        frame_idx=frame_idx,
        sample_idx=0,
        loss_dictionary={},  # Not needed for point collection
        selected_source_points_index=None,
        camera_poses=None,
        scene_manager=scene_managers[args.target_scene_index],
        target_pixels=target_pixels,
        get_target_points_only=True,
        target_pixels_method=target_pixels_method,
    )
    target_points = list(target_points)
    return target_points, per_point_opacity


if __name__ == "__main__":
    config, args = get_args()
    args.stage = "test"

    log_dir = os.path.join(config["save_dir"], config["index"])
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, "test.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, "test_error.log"), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    if args.opt != os.path.join(log_dir, os.path.basename(args.opt)):
        shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    setup_seed(config["seed"])

    if args.source_point_index is not None or args.test_action == "render":
        args.how_many_samples = 1

    config_keys = list(config.keys())
    scene_keys = [
        key for key in config_keys if key.startswith("scene_") and len(key) == 7
    ]
    scene_keys.sort()
    managers = []
    if len(scene_keys) == 0:
        print("\033[91m" + "No scene found." + "\033[0m")
        exit()
    for scene_key in scene_keys:
        if args.test_dataset_path is not None:
            for dataset in config[scene_key]["test"]["datasets"]:
                dataset["path"] = args.test_dataset_path
        for i, dataset in enumerate(config[scene_key]["test"]["datasets"]):
            config[scene_key]["dataset"].update(dataset)
        scene_idx = int(scene_key.split("_")[1])
        if args.source_scene_index == scene_idx - 1:
            s_area = args.source_area_indices
        else:
            s_area = None
        if args.target_scene_index == scene_idx - 1:
            t_area = args.target_area_indices
        else:
            t_area = None
        managers.append(
            SceneManager(
                args=args,
                all_configs=config,
                scene_config=config[scene_key],
                eval_config=config[scene_key],
                scene_key=scene_key,
                scene_idx=scene_idx - 1,
                cuda_idx=args.gpu_id if len(scene_keys) == 1 else scene_idx - 1,
                phase="test",
            )
        )

    # intialize the optimizers
    for manager in managers:
        manager.setup_test_steps()

    # freeform editing loading
    if args.test_action in ["transfer_shading", "transfer_albedo", "freefrom_transfer_albedo", "freefrom_transfer_shading", "change_brightness", "interpolate_albedo", "calculate_albedo_consistency"] and args.source_target_area_selection_method == "freeform_pixels":
        source_points, _ = get_freeform_points_indices(args=args, scene_managers=managers, file_path=args.source_area_path, frame_idx=args.freeform_source_key_frame_index, target_pixels_method=args.freeform_source_point_method)
        target_points, per_point_opacity = get_freeform_points_indices(args=args, scene_managers=managers, file_path=args.target_area_path, frame_idx=args.freeform_target_key_frame_index, target_pixels_method=args.freeform_target_point_method)
        print("Freeform # of source points: ", len(source_points))
        print("Freeform # of target points: ", len(target_points))
        managers[args.target_scene_index].target_area_indices["freeform"] = target_points
        managers[args.source_scene_index].source_area_indices["freeform"] = source_points
        managers[args.target_scene_index].target_points_opacity = per_point_opacity

    # rendering only
    if args.test_action == "render":
        for manager in managers:
            do_action_rendering(scene_manager=manager)
        exit()

    if args.test_action == "render_depth_pcd_for_comparison":
        for manager in managers:
            do_action_render_depth_pcd(scene_manager=manager)
        exit()

    # transfer albedo or shading
    if args.test_action == "transfer_albedo" or args.test_action == "transfer_shading" or args.test_action == "freefrom_transfer_albedo" or args.test_action == "freefrom_transfer_shading":
        do_action_transfer_albedo_shading(args=args, scene_managers=managers)
        exit()


    # brightness interpolation
    if args.test_action == "change_brightness":
        change_brightness_shading(
            args=args, scene_manager=managers[args.target_scene_index]
        )
        exit()

    # albedo interpolation
    if args.test_action == "interpolate_albedo":
        interpolate_albedo(args=args, scene_manager=managers[args.target_scene_index])
        exit()

    # TSNE plot
    if args.test_action == "TSNE":
        generate_TSNE_plot(args=args, scene_manager=managers[args.target_scene_index])
        exit()

    # Albedo consistency
    if args.test_action == "calculate_albedo_consistency":
        calculate_albedo_consistency(args=args, scene_manager=managers[0])
        exit()

    if args.test_action == "2D_color_interpolation_with_UNet":
        resolution = 10
        # np load the color featuer 1 and 2
        color_1 = np.load(args.color_1_feature)
        color_1 = torch.tensor(color_1).to(managers[0].device).reshape(1, 1, 1, -1)

        color_2 = np.load(args.color_2_feature)
        color_2 = torch.tensor(color_2).to(managers[0].device).reshape(1, 1, 1, -1)

        alpha = np.linspace(0, 1, resolution)
        beta = np.linspace(0, 1, resolution)
        alpha_grid, beta_grid = np.meshgrid(alpha, beta)

        interpolated_colors = np.zeros((resolution, resolution, 3))
        for i in range(resolution):
            for j in range(resolution):
                # generate a zero tensor with the shape of [B, 10, 10, 3]
                feat = (
                    torch.zeros((1, 10, 10, color_1.shape[-1]))
                    .float()
                    .to(managers[0].device)
                )

                final_input_feat = (
                    alpha_grid[i, j] * color_1 + beta_grid[i, j] * color_2
                ).reshape(1, 1, 1, -1)

                final_input_feat = final_input_feat * 1.5
                # replace all featuers in feat by color 1
                feat[:, :, :, :] = torch.tensor(final_input_feat).to(managers[0].device)

                # call the renderer
                raw_pred = (
                    managers[0]
                    .model.renderer_UNet(feat.squeeze(-2).permute(0, 3, 1, 2))
                    .permute(0, 2, 3, 1)
                    .unsqueeze(-2)
                )  # (N, H, W, 1, 3)

                # skip the first row, the first column, the last row, and the last column
                raw_pred = raw_pred[:, 1:-1, 1:-1, :, :]
                # get the average of output and set it for all pixels
                raw_pred = raw_pred.mean(dim=(1, 2), keepdim=True).repeat(
                    1, 10, 10, 1, 1
                )

                # save the image
                srgb_pred = preprocess_postproces_images_pipeline(
                    img=raw_pred,
                    pipline=managers[0].scene_config.test.datasets[0][
                        f"render_pred_postprocessing"
                    ],
                    eps=managers[0].scene_config.models.predict_in_log_space_eps,
                    min_val=(
                        getattr(
                            managers[0].scene_config.dataset,
                            "min_{}_log".format("render"),
                            None,
                        )
                        if managers[0].scene_config.models.predict_rgb_in_log_space
                        or managers[0].scene_config.models.predict_raw_in_log_space
                        else 0
                    ),
                    max_val=(
                        getattr(
                            managers[0].scene_config.dataset,
                            "max_{}_log".format("render"),
                            None,
                        )
                        if managers[0].scene_config.models.predict_rgb_in_log_space
                        or managers[0].scene_config.models.predict_raw_in_log_space
                        else 1
                    ),
                    white_bg_value=getattr(
                        managers[0].scene_config.geoms.background,
                        "render_init_scale",
                        None,
                    ),
                    supervision_scaler=None,
                )

                return_srgb_pred = (
                    srgb_pred.squeeze().detach().cpu().numpy() * 255
                ).astype(np.uint8)

                interpolated_colors[i, j, :] = return_srgb_pred[5, 5, :]


        plt.imshow(interpolated_colors.astype(int))
        interpolated_colors_path = "interpolated_colors_UNET.png"
        plt.savefig(interpolated_colors_path)
        print("Saved image: ", interpolated_colors_path)

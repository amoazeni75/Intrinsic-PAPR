import bisect
import copy
import os
import shutil
import sys
import time

import numpy as np
import torch

from models.utils import *
from scene_manager import SceneManager
from tools.args_parser import *
from tools.logger import *

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
torch.autograd.set_detect_anomaly(True)


softplus_activation = SoftplusActivation()


def eval_step(
    scene_manager,
    batch,
    train_rgb_out,
    train_pred_albedo_patch=None,
    log_dictionary=None,
    add_to_log_dictionary=False,
):
    (
        train_img_idx,
        _,
        train_patch,
        train_tgt_albedo_patch,
        _,
        _,
        _,
    ) = batch

    train_tgt_albedo_patch = train_tgt_albedo_patch[0, 0].unsqueeze(0)

    (
        train_img,
        train_tgt_albedo,
        train_rayd,
        train_rayo,
        _,
    ) = scene_manager.train_dataset.get_full_img(train_img_idx[0])
    train_tgt_albedo = train_tgt_albedo[0, 0].unsqueeze(0)
    img, test_tgt_albedo, rayd, rayo, _ = (
        scene_manager.eval_dataset.get_full_img(scene_manager.scene_config.eval.img_idx)
    )
    test_tgt_albedo = test_tgt_albedo[0, 0].unsqueeze(0)
    c2w = scene_manager.train_dataset.get_c2w(scene_manager.scene_config.eval.img_idx)

    N, H, W, _ = rayd.shape
    num_pts, _ = scene_manager.model.points.shape

    rayo = rayo.to(scene_manager.device)
    rayd = rayd.to(scene_manager.device)
    img = img.to(scene_manager.device)
    if scene_manager.scene_config.models.use_albedo:
        test_tgt_albedo = test_tgt_albedo.to(scene_manager.device)
    c2w = c2w.to(scene_manager.device)

    topk = min([num_pts, scene_manager.model.select_k])
    pt_idxs = [topk * i // 5 for i in range(5)]

    selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    transformer_opt = scene_manager.scene_config.models.transformer
    feat_dim = (
        transformer_opt.embed.d_ff_out
        if transformer_opt.embed.share_embed
        else transformer_opt.embed.value.d_ff_out
    )
    if (
        scene_manager.model.bkg_feats is not None
        and scene_manager.model.bkg_type == 1
        and scene_manager.step <= scene_manager.scene_config.training.bkg_step
    ):
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
                    step=scene_manager.step,
                )

                selected_points[
                    :, height_start:height_end, width_start:width_end, :, :
                ] = scene_manager.model.selected_points

        bg_attention = np.clip(
            attn[..., topk:, :].squeeze().detach().cpu().numpy(), 0, 1
        )
        bg_mask = (
            (
                attn[..., topk:, :]
                * scene_manager.model.bkg_feats.expand(N, H, W, -1, -1)
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        bg_mask = np.clip(bg_mask, 0, 1)
        # albedo output
        if scene_manager.scene_config.models.use_albedo:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                albedo_input_features = extract_features_from_feature_map(
                    features_map=feature_map,
                    features_dim=scene_manager.model.albedo_UNet_inp_size,
                    side=scene_manager.model.albedo_feat_side,
                )
                foreground_albedo = (
                    scene_manager.model.albedo_model(
                        albedo_input_features.squeeze(-2).permute(0, 3, 1, 2)
                    )
                    .permute(0, 2, 3, 1)
                    .unsqueeze(-2)
                )
                if (
                    scene_manager.model.bkg_feats is not None
                    and scene_manager.step
                    <= scene_manager.scene_config.training.bkg_step
                ):
                    bkg_attn = attn[..., topk:, :]
                    if scene_manager.scene_config.models.normalize_topk_attn:
                        albedo = (
                            foreground_albedo * (1 - bkg_attn)
                            + scene_manager.model.bkg_feats.expand(N, H, W, -1, -1)
                            * bkg_attn
                        )
                    else:
                        albedo = (
                            foreground_albedo
                            + scene_manager.model.bkg_feats.expand(N, H, W, -1, -1)
                            * bkg_attn
                        )
                    test_pred_albedo = albedo.squeeze(-2)
                else:
                    test_pred_albedo = foreground_albedo.squeeze(-2)

                foreground_albedo = foreground_albedo.squeeze(-2)

        else:
            test_pred_albedo = None

        test_pred_shading = None

        # renderer output
        if scene_manager.scene_config.models.use_renderer:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                foreground_rgb = (
                    scene_manager.model.renderer_UNet(
                        feature_map.squeeze(-2).permute(0, 3, 1, 2)
                    )
                    .permute(0, 2, 3, 1)
                    .unsqueeze(-2)
                )  # (N, H, W, 1, 3)
                if (
                    scene_manager.model.bkg_feats is not None
                    and scene_manager.step
                    <= scene_manager.scene_config.training.bkg_step
                ):
                    bkg_attn = attn[..., topk:, :]
                    if scene_manager.scene_config.models.normalize_topk_attn:
                        rgb = (
                            foreground_rgb * (1 - bkg_attn)
                            + scene_manager.model.bkg_feats.expand(N, H, W, -1, -1)
                            * bkg_attn
                        )
                    else:
                        rgb = (
                            foreground_rgb
                            + scene_manager.model.bkg_feats.expand(N, H, W, -1, -1)
                            * bkg_attn
                        )
                    rgb = rgb.squeeze(-2)
                else:
                    rgb = foreground_rgb.squeeze(-2)
                foreground_rgb = foreground_rgb.squeeze(-2)
        elif scene_manager.scene_config.models.use_implicit_renderer:
            rgb = cacluate_rgb_from_albedo_and_shading(
                albedo=test_pred_albedo,
                shading=test_pred_shading,
                scene_config=scene_manager.model.args,
            )  # it's normalized and in log space
            foreground_rgb = rgb
        else:
            rgb = None
        scene_manager.model.clear_grad()

    # apply the last activation function on Render, Albedo, and Shading,
    rgb = scene_manager.model.last_act(rgb)  # eval prediction rgb
    if scene_manager.scene_config.models.use_albedo:
        test_pred_albedo = scene_manager.model.last_act(test_pred_albedo)

    # preprocessing on prediction images
    if len(scene_manager.scene_config.dataset.render_pred_preprocessing) != 0:
        rgb = preprocess_postproces_images_pipeline(
            img=rgb,
            pipline=scene_manager.scene_config.dataset.render_pred_preprocessing,
            eps=scene_manager.scene_config.models.predict_in_log_space_eps,
            min_val=getattr(
                scene_manager.scene_config.dataset, "min_{}_log".format("render"), None
            ),
            max_val=getattr(
                scene_manager.scene_config.dataset, "max_{}_log".format("render"), None
            ),
            white_bg_value=getattr(
                scene_manager.scene_config.geoms.background, "render_init_scale", None
            ),
        )
    if (
        scene_manager.scene_config.models.use_albedo
        and len(scene_manager.scene_config.dataset.albedo_pred_preprocessing) != 0
    ):
        test_pred_albedo = preprocess_postproces_images_pipeline(
            img=test_pred_albedo,
            pipline=scene_manager.scene_config.dataset.albedo_pred_preprocessing,
            eps=scene_manager.scene_config.models.predict_in_log_space_eps,
            min_val=getattr(
                scene_manager.scene_config.dataset, "min_{}_log".format("albedo"), None
            ),
            max_val=getattr(
                scene_manager.scene_config.dataset, "max_{}_log".format("albedo"), None
            ),
            white_bg_value=getattr(
                scene_manager.scene_config.geoms.background, "albedo_init_scale", None
            ),
        )

    # prepare the raw images for visualization, these are exactly the things model sees during training
    train_tgt_rgb_raw_space = None
    train_tgt_rgb_patch_raw_space = None
    train_pred_rgb_patch_raw_space = None
    test_tgt_rgb_raw_space = None
    test_pred_rgb_raw_space = None
    test_pred_foreground_rgb_raw_space = None
    train_tgt_albedo_raw_space = None
    train_tgt_albedo_patch_raw_space = None
    train_pred_albedo_patch_raw_space = None
    test_tgt_albedo_raw_space = None
    test_pred_albedo_raw_space = None
    test_pred_foreground_albedo_raw_space = None

    if (
        len(scene_manager.scene_config.dataset.render_pred_postprocessing) != 0
        or len(scene_manager.scene_config.dataset.render_GT_postprocessing) != 0
    ):
        train_tgt_rgb_raw_space = train_img.squeeze().cpu().numpy().astype(np.float32)
        train_tgt_rgb_patch_raw_space = train_patch[0].cpu().numpy().astype(np.float32)
        train_pred_rgb_patch_raw_space = (
            train_rgb_out[0].detach().cpu().numpy().astype(np.float32)
        )
        test_tgt_rgb_raw_space = img.squeeze().cpu().numpy().astype(np.float32)
        test_pred_rgb_raw_space = rgb.squeeze().cpu().numpy().astype(np.float32)
        test_pred_foreground_rgb_raw_space = (
            foreground_rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        )

    # albedo
    if scene_manager.scene_config.models.use_albedo and (
        len(scene_manager.scene_config.dataset.albedo_GT_postprocessing) != 0
        or len(scene_manager.scene_config.training.albedo_pred_postprocessing) != 0
    ):
        train_tgt_albedo_raw_space = (
            train_tgt_albedo.squeeze().cpu().numpy().astype(np.float32)
        )
        train_tgt_albedo_patch_raw_space = (
            train_tgt_albedo_patch[0].detach().cpu().numpy().astype(np.float32)
        )
        train_pred_albedo_patch_raw_space = (
            train_pred_albedo_patch[0].detach().cpu().numpy().astype(np.float32)
        )
        test_tgt_albedo_raw_space = (
            test_tgt_albedo.squeeze().cpu().numpy().astype(np.float32)
        )
        test_pred_albedo_raw_space = (
            test_pred_albedo.squeeze().cpu().numpy().astype(np.float32)
        )
        test_pred_foreground_albedo_raw_space = (
            foreground_albedo.squeeze().detach().cpu().numpy().astype(np.float32)
        )

    # calculate the loss function on model prediction
    (
        eval_total_loss,
        eval_render_loss_pred_space,
        eval_render_loss_original_space,
        eval_albedo_loss_pred_space,
        eval_albedo_loss_original_space,
        eval_shading_loss_pred_space,
        eval_shading_loss_original_space,
        eval_albedo_loss_original_space_cIMLE,
        eval_albedo_loss_pred_space_cIMLE,
        log_dictionary,
    ) = calculate_training_loss(
        scene_manager=scene_manager,
        render_pred_patch_pred_space=rgb,
        render_gt_patch_pred_space=img,
        albedo_pred_patch_pred_space=test_pred_albedo,
        albedo_gt_patch_pred_space=test_tgt_albedo,
        clip=True,
        log_dictioanry=log_dictionary,
        add_to_log_dictioanry=add_to_log_dictionary,
        phase="eval",
    )

    ################################## Adding Losses to the log dictionary ##################################
    scene_manager.total_eval_losses.append(eval_total_loss.item())
    phases = ["eval"]
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
                # add the corresponding loss.item() to the list of getattr(self.scene_manager, f"{image_type}_losses")[phase][space] if it's not None
                if locals().get(f"{image_type}_loss_{space}") is not None:
                    getattr(scene_manager, f"{image_type}_losses")[phase][space].append(
                        locals().get(f"{image_type}_loss_{space}").item()
                    )
                    # average the loss
                    setattr(
                        scene_manager,
                        f"avg_{image_type}_loss_{space}",
                        getattr(scene_manager, f"avg_{image_type}_loss_{space}")
                        + locals().get(f"{image_type}_loss_{space}").item(),
                    )
                    if scene_manager.args.debug:
                        print(
                            "\033[91m",
                            end="",
                        )
                        print(
                            f"step: {scene_manager.step}, {phase}_step loss for",
                            image_type,
                            space,
                            ": "
                            + str(locals().get(f"{image_type}_loss_{space}").item()),
                        )
                        print("\033[0m", end="")

    ################################## Adding Losses to the log dictionary ##################################
    coord_scale = scene_manager.scene_config.dataset.coord_scale
    pt_plot_scale = 1.0 * coord_scale
    if "Barn" in scene_manager.scene_config.dataset.path:
        pt_plot_scale *= 1.8
    if "Family" in scene_manager.scene_config.dataset.path:
        pt_plot_scale *= 0.5

    # calculate depth, weighted sum the distances from top K points to image plane
    image_plane_normal = -rayo
    image_plane_offset = torch.sum(image_plane_normal * rayo)
    point_to_plane_dists = torch.abs(
        torch.sum(
            selected_points.to(image_plane_normal.device) * image_plane_normal, -1
        )
        - image_plane_offset
    ) / torch.norm(image_plane_normal)
    if (
        scene_manager.model.bkg_feats is not None
        and scene_manager.model.bkg_type == 1
        and scene_manager.step <= scene_manager.scene_config.training.bkg_step
    ):
        point_to_plane_dists = torch.cat(
            [
                point_to_plane_dists,
                torch.ones(N, H, W, scene_manager.model.bkg_feats.shape[0]).to(
                    point_to_plane_dists.device
                )
                * 0,
            ],
            dim=-1,
        )
    current_depth = (
        (
            torch.sum(
                attn.squeeze(-1).to(image_plane_normal.device) * point_to_plane_dists,
                dim=-1,
            )
        )
        .detach()
        .cpu()
    )

    print(
        "Eval step:",
        scene_manager.step,
    )

    if (
        len(scene_manager.scene_config.dataset.render_GT_postprocessing) != 0
        or len(scene_manager.scene_config.training.render_pred_postprocessing) != 0
    ):
        train_tgt_rgb = (
            preprocess_postproces_images_pipeline(
                img=train_img,
                pipline=scene_manager.scene_config.dataset.render_GT_postprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        train_tgt_rgb_patch = (
            preprocess_postproces_images_pipeline(
                img=train_patch[0].unsqueeze(0),
                pipline=scene_manager.scene_config.dataset.render_GT_postprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        test_tgt_rgb = (
            preprocess_postproces_images_pipeline(
                img=img,
                pipline=scene_manager.scene_config.dataset.render_GT_postprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        test_pred_rgb = (
            preprocess_postproces_images_pipeline(
                img=rgb,
                pipline=scene_manager.scene_config.dataset.render_pred_postprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        train_pred_rgb_patch = (
            preprocess_postproces_images_pipeline(
                img=train_rgb_out[0].unsqueeze(0),
                pipline=scene_manager.scene_config.dataset.render_pred_postprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        test_pred_foreground_rgb = (
            preprocess_postproces_images_pipeline(
                img=foreground_rgb,
                pipline=scene_manager.scene_config.dataset.render_pred_postprocessing,
                eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                min_val=getattr(
                    scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        train_tgt_rgb = train_img.squeeze().cpu().numpy().astype(np.float32)
        train_tgt_rgb_patch = train_patch[0].cpu().numpy().astype(np.float32)
        test_tgt_rgb = img.squeeze().cpu().numpy().astype(np.float32)
        test_pred_rgb = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        train_pred_rgb_patch = (
            train_rgb_out[0].detach().cpu().numpy().astype(np.float32)
        )
        test_pred_foreground_rgb = (
            foreground_rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        )

    if scene_manager.scene_config.models.use_albedo:
        if (
            len(scene_manager.scene_config.dataset.albedo_GT_postprocessing) != 0
            or len(scene_manager.scene_config.training.albedo_pred_postprocessing) != 0
        ):
            train_tgt_albedo = (
                preprocess_postproces_images_pipeline(
                    img=train_tgt_albedo,
                    pipline=scene_manager.scene_config.dataset.albedo_GT_postprocessing,
                    eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                    min_val=getattr(
                        scene_manager.scene_config.dataset,
                        "min_{}_log".format("albedo"),
                        None,
                    ),
                    max_val=getattr(
                        scene_manager.scene_config.dataset,
                        "max_{}_log".format("albedo"),
                        None,
                    ),
                    white_bg_value=getattr(
                        scene_manager.scene_config.geoms.background,
                        "albedo_init_scale",
                        None,
                    ),
                )
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            train_tgt_albedo_patch = (
                preprocess_postproces_images_pipeline(
                    img=train_tgt_albedo_patch[0].unsqueeze(0),
                    pipline=scene_manager.scene_config.dataset.albedo_GT_postprocessing,
                    eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                    min_val=getattr(
                        scene_manager.scene_config.dataset,
                        "min_{}_log".format("albedo"),
                        None,
                    ),
                    max_val=getattr(
                        scene_manager.scene_config.dataset,
                        "max_{}_log".format("albedo"),
                        None,
                    ),
                    white_bg_value=getattr(
                        scene_manager.scene_config.geoms.background,
                        "albedo_init_scale",
                        None,
                    ),
                )
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            train_pred_albedo_patch = (
                preprocess_postproces_images_pipeline(
                    img=train_pred_albedo_patch[0].unsqueeze(0),
                    pipline=scene_manager.scene_config.dataset.albedo_pred_postprocessing,
                    eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                    min_val=getattr(
                        scene_manager.scene_config.dataset,
                        "min_{}_log".format("albedo"),
                        None,
                    ),
                    max_val=getattr(
                        scene_manager.scene_config.dataset,
                        "max_{}_log".format("albedo"),
                        None,
                    ),
                    white_bg_value=getattr(
                        scene_manager.scene_config.geoms.background,
                        "albedo_init_scale",
                        None,
                    ),
                )
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            test_tgt_albedo = (
                preprocess_postproces_images_pipeline(
                    img=test_tgt_albedo,
                    pipline=scene_manager.scene_config.dataset.albedo_GT_postprocessing,
                    eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                    min_val=getattr(
                        scene_manager.scene_config.dataset,
                        "min_{}_log".format("albedo"),
                        None,
                    ),
                    max_val=getattr(
                        scene_manager.scene_config.dataset,
                        "max_{}_log".format("albedo"),
                        None,
                    ),
                    white_bg_value=getattr(
                        scene_manager.scene_config.geoms.background,
                        "albedo_init_scale",
                        None,
                    ),
                )
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            test_pred_albedo = (
                preprocess_postproces_images_pipeline(
                    img=test_pred_albedo,
                    pipline=scene_manager.scene_config.dataset.albedo_pred_postprocessing,
                    eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                    min_val=getattr(
                        scene_manager.scene_config.dataset,
                        "min_{}_log".format("albedo"),
                        None,
                    ),
                    max_val=getattr(
                        scene_manager.scene_config.dataset,
                        "max_{}_log".format("albedo"),
                        None,
                    ),
                    white_bg_value=getattr(
                        scene_manager.scene_config.geoms.background,
                        "albedo_init_scale",
                        None,
                    ),
                )
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            test_pred_foreground_albedo = (
                preprocess_postproces_images_pipeline(
                    img=foreground_albedo,
                    pipline=scene_manager.scene_config.dataset.albedo_pred_postprocessing,
                    eps=scene_manager.scene_config.models.predict_in_log_space_eps,
                    min_val=getattr(
                        scene_manager.scene_config.dataset,
                        "min_{}_log".format("albedo"),
                        None,
                    ),
                    max_val=getattr(
                        scene_manager.scene_config.dataset,
                        "max_{}_log".format("albedo"),
                        None,
                    ),
                    white_bg_value=getattr(
                        scene_manager.scene_config.geoms.background,
                        "albedo_init_scale",
                        None,
                    ),
                )
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            train_tgt_albedo = (
                train_tgt_albedo.squeeze().cpu().numpy().astype(np.float32)
            )
            train_tgt_albedo_patch = (
                train_tgt_albedo_patch[0].cpu().numpy().astype(np.float32)
            )
            train_pred_albedo_patch = (
                train_pred_albedo_patch[0].detach().cpu().numpy().astype(np.float32)
            )
            test_tgt_albedo = test_tgt_albedo.squeeze().cpu().numpy().astype(np.float32)
            test_pred_albedo = (
                test_pred_albedo.squeeze().detach().cpu().numpy().astype(np.float32)
            )
            test_pred_foreground_albedo = (
                foreground_albedo.squeeze().detach().cpu().numpy().astype(np.float32)
            )
    else:
        train_tgt_albedo = None
        train_tgt_albedo_patch = None
        train_pred_albedo_patch = None
        test_tgt_albedo = None
        test_pred_albedo = None
        test_pred_foreground_albedo = None

    points_np = scene_manager.model.points.detach().cpu().numpy()
    depth = current_depth.squeeze().numpy().astype(np.float32)
    points_conf_scores_np = None
    if scene_manager.model.points_conf_scores is not None:
        points_conf_scores_np = (
            scene_manager.model.points_conf_scores.squeeze().detach().cpu().numpy()
        )

    eval_psnr = (
        -10.0
        * np.log(((test_pred_rgb - test_tgt_rgb) ** 2).mean().item())
        / np.log(10.0)
    )

    scene_manager.eval_psnrs.append(eval_psnr.item())

    scene_metrics_dict = {
        "attention_max": attn.max().item(),
        "attention_min": attn.min().item(),
        "attention_mean": attn.mean().item(),
        "attention_std": attn.std().item(),
        "bkg_feats_max": scene_manager.model.bkg_feats.max().item(),
        "bkg_feats_min": scene_manager.model.bkg_feats.min().item(),
        "bkg_feats_mean": scene_manager.model.bkg_feats.mean().item(),
        "bkg_feats_std": scene_manager.model.bkg_feats.std().item(),
        "eval_psnr": eval_psnr.item(),
    }
    # update all keys and attach scene index before the keys
    metrics_dict = {
        f"scene_{scene_manager.scene_config.scene_idx}_{key}": value
        for key, value in scene_metrics_dict.items()
    }

    # main plot
    main_plot, metrics_dict = get_training_main_plot(
        index=scene_manager.scene_config.index,
        step=scene_manager.step,
        train_tgt_rgb=train_tgt_rgb,
        train_tgt_rgb_patch=train_tgt_rgb_patch,
        train_pred_rgb_patch=train_pred_rgb_patch,
        test_tgt_rgb=test_tgt_rgb,
        test_pred_rgb=test_pred_rgb,
        test_pred_foreground_rgb=test_pred_foreground_rgb,
        points_np=points_np,
        pt_plot_scale=pt_plot_scale,
        depth_np=depth,
        train_tgt_albedo=train_tgt_albedo,
        train_tgt_albedo_patch=train_tgt_albedo_patch,
        train_pred_albedo_patch=train_pred_albedo_patch,
        test_tgt_albedo=test_tgt_albedo,
        test_pred_albedo=test_pred_albedo,
        test_pred_foreground_albedo=test_pred_foreground_albedo,
        points_conf_scores_np=points_conf_scores_np,
        train_tgt_rgb_raw_space=train_tgt_rgb_raw_space,
        train_tgt_rgb_patch_raw_space=train_tgt_rgb_patch_raw_space,
        train_pred_rgb_patch_raw_space=train_pred_rgb_patch_raw_space,
        test_tgt_rgb_raw_space=test_tgt_rgb_raw_space,
        test_pred_rgb_raw_space=test_pred_rgb_raw_space,
        test_pred_foreground_rgb_raw_space=test_pred_foreground_rgb_raw_space,
        train_tgt_albedo_raw_space=train_tgt_albedo_raw_space,
        train_tgt_albedo_patch_raw_space=train_tgt_albedo_patch_raw_space,
        train_pred_albedo_patch_raw_space=train_pred_albedo_patch_raw_space,
        test_tgt_albedo_raw_space=test_tgt_albedo_raw_space,
        test_pred_albedo_raw_space=test_pred_albedo_raw_space,
        test_pred_foreground_albedo_raw_space=test_pred_foreground_albedo_raw_space,
        bg_attentions=bg_attention,
        bg_masks=bg_mask,
        metrics_dict=metrics_dict,
    )
    main_plot_path = os.path.join(
        scene_manager.train_main_plots_dir,
        "%s_iter_%d.png" % (scene_manager.scene_config.index, scene_manager.step),
    )
    main_plot.save(main_plot_path)

    # point cloud plot
    train_rayo_np = train_rayo.squeeze().detach().cpu().numpy()
    train_rayd_np = train_rayd.squeeze().detach().cpu().numpy()

    pcd_plot = get_training_pcd_plot(
        scene_manager.scene_config.index,
        scene_manager.step,
        train_rayo_np,
        train_rayd_np,
        points_np,
        scene_manager.scene_config.dataset.coord_scale,
        pt_plot_scale,
        points_conf_scores_np,
    )
    pcd_plot_path = os.path.join(
        scene_manager.train_pcd_plots_dir,
        "%s_iter_%d.png" % (scene_manager.scene_config.index, scene_manager.step),
    )
    pcd_plot.save(pcd_plot_path)

    del rayo, rayd, c2w
    del eval_total_loss, eval_psnr
    del selected_points, attn

    return metrics_dict, log_dictionary


def check_applying_scaler_to_supervision(scene_manager):
    def is_within_active_interval(start, end, how_long, current_iteration):
        # Check if current iteration is within the range of start and end
        if current_iteration < start or current_iteration > end:
            return False

        if how_long == -1:
            return True

        # Find how far we are from the start
        relative_position = current_iteration - start

        # Determine if the current iteration falls within one of the intervals
        # We divide the relative position by the cycle length (how_long * 2) to see which segment we are in
        interval_length = 2 * how_long
        within_active_period = relative_position % interval_length < how_long

        return within_active_period

    if scene_manager.scene_config.models.supervision_scaler.use:
        if (
            "scale_supervision_fg"
            in scene_manager.scene_config.training.GT_albedo_preprocessing
        ):
            return is_within_active_interval(
                start=scene_manager.scene_config.models.supervision_scaler.apply_interval_start,
                end=scene_manager.scene_config.models.supervision_scaler.apply_interval_end,
                how_long=scene_manager.scene_config.models.supervision_scaler.apply_interval_length,
                current_iteration=scene_manager.step,
            )
        else:
            return False
    else:
        return False


def train_step(
    batch,
    scene_manager,
    log_dictionary=None,
    add_to_log_dictionary=False,
):
    img_idx, _, tgt, tgt_albedo, rayd, rayo, alpha_channel = batch
    if scene_manager.args.debug:
        # print the step and image index in red
        print("\033[91m", end="")
        print("step: {}, image index: {}".format(scene_manager.step, img_idx))
        print("\033[0m", end="")
    c2w = scene_manager.train_dataset.get_c2w(img_idx[0])

    rayo = rayo.to(scene_manager.device)
    rayd = rayd.to(scene_manager.device)
    tgt = tgt.to(scene_manager.device)
    c2w = c2w.to(scene_manager.device)
    alpha_channel = alpha_channel.to(scene_manager.device)
    if scene_manager.scene_config.models.use_albedo:
        tgt_albedo = tgt_albedo.to(scene_manager.device)

    # we do the scaling and the rest of preprocessing on the GT albedo here
    if (
        scene_manager.scene_config.models.supervision_scaler.use
        and len(scene_manager.scene_config.training.GT_albedo_preprocessing) == 0
    ):
        raise ValueError(
            "We are set to use the supervision scaler, but the GT albedo preprocessing is not set"
        )

    if len(scene_manager.scene_config.training.GT_albedo_preprocessing) != 0:
        if check_applying_scaler_to_supervision(scene_manager):
            albedo_supervision_scaler = scene_manager.model.supervision_scaler[img_idx]
            if scene_manager.args.debug:
                print(
                    "\033[91m"
                    + "DEBUG: we are using the supervision scaler for the GT albedo"
                    + "\033[0m"
                )
        else:
            albedo_supervision_scaler = (
                scene_manager.model.supervision_scaler[img_idx].clone().detach()
            )
            if scene_manager.args.debug:
                print(
                    "\033[91m"
                    + "DEBUG: we detached the supervision scaler from the graph"
                    + "\033[0m"
                )

        # we need to change the view of alpha_channel to be the same as tgt_albedo (B, N_samples, H, W, 1)
        expanded_alpha_channel = alpha_channel.unsqueeze(1)
        expanded_alpha_channel = expanded_alpha_channel.expand(
            -1, tgt_albedo.shape[1], -1, -1, -1
        )
        albedo_supervision_scaler = (
            albedo_supervision_scaler.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, tgt_albedo.shape[1], 1, 1, 1)
        )

        tgt_albedo = preprocess_postproces_images_pipeline(
            img=tgt_albedo,
            pipline=scene_manager.scene_config.training.GT_albedo_preprocessing,
            eps=scene_manager.scene_config.models.predict_in_log_space_eps,
            min_val=getattr(
                scene_manager.scene_config.dataset, "min_{}_log".format("albedo"), None
            ),
            max_val=getattr(
                scene_manager.scene_config.dataset, "max_{}_log".format("albedo"), None
            ),
            white_bg_value=getattr(
                scene_manager.scene_config.geoms.background, "albedo_init_scale"
            ),
            supervision_scaler=softplus_activation(albedo_supervision_scaler),
            clamp_min=getattr(
                scene_manager.scene_config.dataset, "min_{}_log".format("albedo"), None
            ),
            clamp_max=getattr(
                scene_manager.scene_config.dataset, "max_{}_log".format("albedo"), None
            ),
            alpha_channel=expanded_alpha_channel,
        )

    scene_manager.model.clear_grad()
    scene_manager.model.move_shared_components_to_device()
    rgb_out, albedo_out, _ = scene_manager.model(rayo, rayd, c2w, scene_manager.step)
    rgb_out = scene_manager.model.last_act(rgb_out)
    if scene_manager.scene_config.models.use_albedo:
        albedo_out = scene_manager.model.last_act(albedo_out)

    (
        total_loss,
        render_loss_pred_space,
        render_loss_original_space,
        albedo_loss_pred_space,
        albedo_loss_original_space,
        shading_loss_pred_space,
        shading_loss_original_space,
        albedo_loss_pred_space_cIMLE,
        albedo_loss_rgb_space_cIMLE,
        log_dictionary,
    ) = calculate_training_loss(
        scene_manager=scene_manager,
        render_pred_patch_pred_space=rgb_out,
        render_gt_patch_pred_space=tgt,
        albedo_pred_patch_pred_space=albedo_out,
        albedo_gt_patch_pred_space=tgt_albedo,
        clip=False,
        log_dictioanry=log_dictionary,
        add_to_log_dictioanry=add_to_log_dictionary,
        phase="train",
    )
    scene_manager.model.scaler.scale(total_loss).backward()
    scene_manager.model.step(scene_manager.step)
    if (
        scene_manager.scene_config.scaler_min_scale > 0
        and scene_manager.model.scaler.get_scale()
        < scene_manager.scene_config.scaler_min_scale
    ):
        scene_manager.model.scaler.update(scene_manager.scene_config.scaler_min_scale)
    else:
        scene_manager.model.scaler.update()

    # debug: when we don't use the scaler to scale the albedo GT, the gradients of the parameters should be zero
    if scene_manager.args.debug and check_applying_scaler_to_supervision(scene_manager):
        # print the gradient of the model.supervision_scaler with a red message
        print("\033[91m", end="")
        print("Step: ", scene_manager.step)
        print(
            f"step: {scene_manager.step}, Max grad albedo gt scaler: ",
            scene_manager.model.supervision_scaler.grad.max().item(),
        )
        print(
            f"step: {scene_manager.step}, Min grad albedo gt scaler: ",
            scene_manager.model.supervision_scaler.grad.min().item(),
        )
        print(
            f"step: {scene_manager.step}, Mean grad albedo gt scaler: ",
            scene_manager.model.supervision_scaler.grad.mean().item(),
        )
        print(
            f"step: {scene_manager.step}, Std grad albedo gt scaler: ",
            scene_manager.model.supervision_scaler.grad.std().item(),
        )
        # values
        print(
            f"step: {scene_manager.step}, Max albedo gt scaler: ",
            scene_manager.model.supervision_scaler.max().item(),
        )
        print(
            f"step: {scene_manager.step}, Min albedo gt scaler: ",
            scene_manager.model.supervision_scaler.min().item(),
        )
        print(
            f"step: {scene_manager.step}, Mean albedo gt scaler: ",
            scene_manager.model.supervision_scaler.mean().item(),
        )
        print(
            f"step: {scene_manager.step}, Std albedo gt scaler: ",
            scene_manager.model.supervision_scaler.std().item(),
        )

        print("\033[0m", end="")

    ################################## Adding Losses to the log dictionary ##################################
    scene_manager.avg_total_train_loss += total_loss.item()
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
                # add the corresponding loss.item() to the list of getattr(self.scene_manager, f"{image_type}_losses")[phase][space] if it's not None
                if locals().get(f"{image_type}_loss_{space}") is not None:
                    getattr(scene_manager, f"{image_type}_losses")[phase][space].append(
                        locals().get(f"{image_type}_loss_{space}").item()
                    )
                    # average the loss
                    setattr(
                        scene_manager,
                        f"avg_{image_type}_loss_{space}",
                        getattr(scene_manager, f"avg_{image_type}_loss_{space}")
                        + locals().get(f"{image_type}_loss_{space}").item(),
                    )
                    if scene_manager.args.debug:
                        print(
                            "\033[91m",
                            end="",
                        )
                        print(
                            f"step: {scene_manager.step}, train_step loss for",
                            image_type,
                            space,
                            ": "
                            + str(locals().get(f"{image_type}_loss_{space}").item()),
                        )
                        print("\033[0m", end="")

    ################################## Adding Losses to the log dictionary ##################################

    return (
        rgb_out,
        albedo_out,
        log_dictionary,
    )


def train_and_eval(managers):
    start_step = managers[0].step
    eval_metrics = None

    print(start_step, managers[0].scene_config.training.steps)
    start_time = time.time()
    counter = start_step
    while True:
        if len(managers) == 1:
            manager_index = 0
        else:
            manager_index = (
                counter // managers[0].all_configs.interchange_step % len(managers)
            )
        scene_manager = managers[manager_index]
        counter += 1
        _, batch = next(enumerate(scene_manager.train_dataloader))
        log_dictionary = {}
        if (
            (scene_manager.scene_config.training.prune_steps > 0)
            and (scene_manager.step < scene_manager.scene_config.training.prune_stop)
            and (scene_manager.step >= scene_manager.scene_config.training.prune_start)
        ):
            if (
                len(scene_manager.scene_config.training.prune_steps_list) > 0
                and scene_manager.step % scene_manager.scene_config.training.prune_steps
                == 0
            ):
                current_prune_thresh = (
                    scene_manager.scene_config.training.prune_thresh_list[
                        bisect.bisect_left(
                            scene_manager.scene_config.training.prune_steps_list,
                            scene_manager.step,
                        )
                    ]
                )
                scene_manager.model.clean_optimizer()
                scene_manager.model.clean_scheduler()
                num_pruned = scene_manager.model.prune_points(current_prune_thresh)
                scene_manager.model.init_optimizers(scene_manager.step)
                scene_manager.pruned = True
                print(
                    "Scene: %d, Step %d: Pruned %d points, prune threshold %f"
                    % (
                        scene_manager.scene_idx + 1,
                        scene_manager.step,
                        num_pruned,
                        current_prune_thresh,
                    )
                )

            elif (
                scene_manager.step % scene_manager.scene_config.training.prune_steps
                == 0
            ):
                scene_manager.model.clean_optimizer()
                scene_manager.model.clean_scheduler()
                num_pruned = scene_manager.model.prune_points(
                    scene_manager.scene_config.training.prune_thresh
                )
                scene_manager.model.init_optimizers(scene_manager.step)
                scene_manager.pruned = True
                print(
                    "Scene: %d, Step %d: Pruned %d points"
                    % (scene_manager.scene_idx + 1, scene_manager.step, num_pruned)
                )

        if (
            scene_manager.pruned
            and len(scene_manager.scene_config.training.add_steps_list) > 0
        ):
            if scene_manager.step in scene_manager.scene_config.training.add_steps_list:
                current_add_num = scene_manager.scene_config.training.add_num_list[
                    scene_manager.scene_config.training.add_steps_list.index(
                        scene_manager.step
                    )
                ]
                scene_manager.model.clean_optimizer()
                scene_manager.model.clean_scheduler()
                num_added = scene_manager.model.add_points(current_add_num)
                scene_manager.model.init_optimizers(scene_manager.step)
                scene_manager.model.added_points = True

                if scene_manager.scene_config.training.select_k_factor > 0:
                    scene_manager.model.select_k = torch.tensor(
                        int(
                            scene_manager.scene_config.training.select_k_factor
                            * scene_manager.model.select_k
                        )
                    ).to(scene_manager.device)

                if scene_manager.scene_config.training.sample_k_factor > 0:
                    scene_manager.model.sample_k = torch.tensor(
                        int(
                            scene_manager.scene_config.training.sample_k_factor
                            * scene_manager.model.sample_k
                        )
                    ).to(scene_manager.device)

                print(
                    "Scene: %d, Step %d: Added %d points"
                    % (scene_manager.scene_idx, scene_manager.step, num_added)
                )

        elif (
            scene_manager.pruned
            and (scene_manager.scene_config.training.add_steps > 0)
            and (
                scene_manager.step % scene_manager.scene_config.training.add_steps == 0
            )
            and (scene_manager.step < scene_manager.scene_config.training.add_stop)
            and (scene_manager.step >= scene_manager.scene_config.training.add_start)
        ):
            scene_manager.model.clean_optimizer()
            scene_manager.model.clean_scheduler()
            num_added = scene_manager.model.add_points(
                scene_manager.scene_config.training.add_num
            )
            scene_manager.model.init_optimizers(scene_manager.step)
            scene_manager.model.added_points = True

            if scene_manager.scene_config.training.select_k_factor > 0:
                scene_manager.model.select_k = torch.tensor(
                    int(
                        scene_manager.scene_config.training.select_k_factor
                        * scene_manager.model.select_k
                    )
                ).to(scene_manager.device)

            if scene_manager.scene_config.training.sample_k_factor > 0:
                scene_manager.model.sample_k = torch.tensor(
                    int(
                        scene_manager.scene_config.training.sample_k_factor
                        * scene_manager.model.sample_k
                    )
                ).to(scene_manager.device)

            print(
                "Scene: %d, Step %d: Added %d points, topk = %d, sample k = %d"
                % (
                    scene_manager.scene_idx,
                    scene_manager.step,
                    num_added,
                    scene_manager.model.select_k,
                    scene_manager.model.sample_k,
                )
            )
        ###################################### train step ######################################
        (
            rgb_pred,
            albedo_pred,
            log_dictionary,
        ) = train_step(
            batch,
            scene_manager=scene_manager,
            log_dictionary=log_dictionary,
            add_to_log_dictionary=(
                True
                if scene_manager.step % scene_manager.all_configs.print_step == 0
                else False
            ),
        )
        ###################################### eval step ######################################
        if (
            scene_manager.step % scene_manager.scene_config.eval.step == 0
            or scene_manager.step >= scene_manager.scene_config.training.steps
        ):
            eval_start_time = time.time()
            eval_metrics, log_dictionary = eval_step(
                scene_manager=scene_manager,
                batch=batch,
                train_rgb_out=rgb_pred,
                train_pred_albedo_patch=(
                    albedo_pred
                    if scene_manager.scene_config.models.use_albedo
                    else None
                ),
                log_dictionary=log_dictionary,
                add_to_log_dictionary=(
                    True
                    if scene_manager.step % scene_manager.all_configs.print_step == 0
                    else False
                ),
            )
            print(f"Eval time: {time.time() - eval_start_time} sec")
        ###################################### Log statistics step ######################################
        if (
            scene_manager.step % scene_manager.all_configs.print_step == 0
            or scene_manager.step >= scene_manager.scene_config.training.steps
        ):
            log_start_time = time.time()
            print_log_statistics(
                scene_manager=scene_manager,
                log_dictionary=log_dictionary,
                eval_metrics=eval_metrics,
            )
            elapsed_time = time.time() - start_time
            elapsed_units = "seconds"
            if elapsed_time > 60 and elapsed_time < 3600:
                elapsed_time = elapsed_time / 60
                elapsed_units = "minutes"
            elif elapsed_time > 3600:
                elapsed_time = elapsed_time / 3600
                elapsed_units = "hours"
            print(
                "step: %d, time elapsed: %.2f %s, current time: %s"
                % (
                    scene_manager.step,
                    elapsed_time,
                    elapsed_units,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            eval_metrics = None
            print(f"Logging statistics: {time.time() - log_start_time} sec")
        ###################################### save model step ######################################
        if (
            scene_manager.step % scene_manager.all_configs.save_checkpoint_step == 0
            or scene_manager.step >= scene_manager.scene_config.training.steps
        ):
            scene_manager.model.save()
            setup_seed(scene_manager.step + 1)


        scene_manager.step += 1
        scene_manager.eval_step_cnt += 1

        if scene_manager.step >= scene_manager.scene_config.training.steps:
            print("Training finished")
            # VolumetricBank.save() takes no arguments; it reads the step, seed and
            # output directory from the scene manager it already holds.
            scene_manager.model.save()
            break

if __name__ == "__main__":
    config, args = get_args()

    print("GPU ID:", args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.stage = "train"

    log_dir = os.path.join(config["save_dir"], config["index"])
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, "train_error.log"), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    find_all_python_files_and_zip(".", os.path.join(log_dir, "code.zip"))

    setup_seed(config["seed"])


    # create scene managers, we need to see how many scene_i do we have in the config
    config_keys = list(config.keys())
    scene_keys = [
        key for key in config_keys if key.startswith("scene_") and len(key) == 7
    ]
    scene_keys.sort()
    managers = []
    for scene_key in scene_keys:
        eval_config = copy.deepcopy(config)
        eval_config[scene_key]["dataset"].update(
            eval_config[scene_key]["eval"]["dataset"]
        )
        eval_config = eval_config[scene_key]
        scene_config = config[scene_key]
        scene_idx = int(scene_key.split("_")[1])
        managers.append(
            SceneManager(
                args=args,
                all_configs=config,
                scene_config=scene_config,
                eval_config=eval_config,
                scene_key=scene_key,
                scene_idx=scene_idx - 1,
                cuda_idx=args.gpu_id if len(scene_keys) == 1 else scene_idx - 1,
            )
        )
    print(
        "\033[91m"
        + "******************Share components between scenes******************"
        + "\033[0m"
    )
    print("*" * 70)

    # intialize the optimizers
    for manager in managers:
        manager.model.init_optimizers(total_steps=0)
        manager.step = manager.load_model(args.resume)
        manager.model = manager.model.to(manager.device)
        manager.eval_step_cnt = manager.step

    # we need to first create the model, then load the wieghts; finally, sharing the components
    # TODO: for now, don't support the shared components

    train_and_eval(managers)

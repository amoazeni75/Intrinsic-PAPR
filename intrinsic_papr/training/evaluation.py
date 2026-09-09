"""The periodic evaluation during training: render a held-out view and plot it."""

import os

import numpy as np
import torch

from ..data.pipeline import apply_image_pipeline
from ..models.features import extract_features_from_feature_map
from .losses import calculate_training_loss
from .plots import get_training_main_plot, get_training_pcd_plot


_PLOT_SCALE_BY_SCENE = {"Barn": 1.8, "Family": 0.5}


def point_cloud_plot_scale(dataset_config):
    """Axis extent for the point-cloud plots, in world units."""
    scale = dataset_config.coord_scale
    for name, factor in _PLOT_SCALE_BY_SCENE.items():
        if name in dataset_config.path:
            scale *= factor
    return scale


_PANEL_SLOTS = (
    ("train_tgt", False, False),
    ("train_tgt_patch", True, False),
    ("train_pred_patch", True, True),
    ("test_tgt", False, False),
    ("test_pred", False, True),
    ("test_pred_foreground", False, True),
)


def panel_arrays(scene_config, sources, img_type, gt_pipeline, pred_pipeline):
    """Turn the six source tensors of one image type into float32 numpy panels.

    Pass empty pipelines to get the arrays in the space the model trains in,
    which is what the raw-space row of the plot shows.
    """
    apply_pipelines = len(gt_pipeline) != 0 or len(pred_pipeline) != 0
    panels = {}
    for name, is_patch, is_pred in _PANEL_SLOTS:
        tensor = sources[name]
        if apply_pipelines:
            tensor = apply_image_pipeline(
                scene_config,
                tensor[0].unsqueeze(0) if is_patch else tensor,
                pred_pipeline if is_pred else gt_pipeline,
                img_type,
            ).squeeze()
        elif is_patch:
            tensor = tensor[0]
        else:
            tensor = tensor.squeeze()
        panels[name] = tensor.detach().cpu().numpy().astype(np.float32)
    return panels


def decode_head(scene_manager, decoder, decoder_input, attn, bkg_split, shape):
    """Run one decoder over the feature map and composite the background behind it.

    Returns ``(composited, foreground)``, both [N, H, W, C].
    """
    N, H, W = shape
    model = scene_manager.model
    foreground = decoder(decoder_input).permute(0, 2, 3, 1).unsqueeze(-2)
    if (
        model.bkg_feats is not None
        and scene_manager.step <= scene_manager.scene_config.training.bkg_step
    ):
        bkg_attn = attn[..., bkg_split:, :]
        background = model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
        if scene_manager.scene_config.models.normalize_topk_attn:
            composited = foreground * (1 - bkg_attn) + background
        else:
            composited = foreground + background
        composited = composited.squeeze(-2)
    else:
        composited = foreground.squeeze(-2)
    return composited, foreground.squeeze(-2)


def depth_from_attention(scene_manager, selected_points, attn, rayo, shape):
    """Depth as the attention-weighted distance from the top-k points to the image plane."""
    N, H, W = shape
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
        # the background points sit at distance zero
        point_to_plane_dists = torch.cat(
            [
                point_to_plane_dists,
                torch.zeros(
                    N, H, W, scene_manager.model.bkg_feats.shape[0]
                ).to(point_to_plane_dists.device),
            ],
            dim=-1,
        )
    return (
        torch.sum(
            attn.squeeze(-1).to(image_plane_normal.device) * point_to_plane_dists,
            dim=-1,
        )
        .detach()
        .cpu()
    )


def eval_step(
    scene_manager,
    batch,
    train_rgb_out,
    train_pred_albedo_patch=None,
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

    use_albedo = scene_manager.scene_config.models.use_albedo
    # Without an albedo head the dataset yields placeholders instead of albedo
    # tensors, so drop them here and let the guards below skip the albedo work.
    train_tgt_albedo_patch = (
        train_tgt_albedo_patch[0, 0].unsqueeze(0) if use_albedo else None
    )

    (
        train_img,
        train_tgt_albedo,
        train_rayd,
        train_rayo,
        _,
    ) = scene_manager.train_dataset.get_full_img(train_img_idx[0])
    train_tgt_albedo = train_tgt_albedo[0, 0].unsqueeze(0) if use_albedo else None
    img, test_tgt_albedo, rayd, rayo, _ = (
        scene_manager.eval_dataset.get_full_img(scene_manager.scene_config.eval.img_idx)
    )
    test_tgt_albedo = test_tgt_albedo[0, 0].unsqueeze(0) if use_albedo else None
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

    # Unbounded scenes carry one extra attention slot per ray, the
    # background-sphere intersection. It belongs to the point sequence, so the
    # split between point attention and the learned background sits after it.
    # Bounded scenes add no slot and every size below is unchanged.
    num_slots = topk + (1 if scene_manager.model.append_bkg_points else 0)

    selected_points = torch.zeros(1, H, W, num_slots, 3)

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
                    step=scene_manager.step,
                )

                selected_points[
                    :, height_start:height_end, width_start:width_end, :, :
                ] = scene_manager.model.selected_points

        bg_attention = np.clip(
            attn[..., num_slots:, :].squeeze().detach().cpu().numpy(), 0, 1
        )
        bg_mask = (
            (
                attn[..., num_slots:, :]
                * scene_manager.model.bkg_feats.expand(N, H, W, -1, -1)
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        bg_mask = np.clip(bg_mask, 0, 1)
        # `out_fuse_type` other than 1 leaves the four names below unbound, which
        # is what the original code did; the later reads then raise NameError
        # rather than quietly rendering something wrong.
        image_shape = (N, H, W)
        if scene_manager.scene_config.models.use_albedo:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                albedo_input_features = extract_features_from_feature_map(
                    features_map=feature_map,
                    features_dim=scene_manager.model.albedo_UNet_inp_size,
                )
                test_pred_albedo, foreground_albedo = decode_head(
                    scene_manager,
                    scene_manager.model.albedo_model,
                    albedo_input_features.squeeze(-2).permute(0, 3, 1, 2),
                    attn,
                    num_slots,
                    image_shape,
                )
        else:
            test_pred_albedo = None

        if scene_manager.scene_config.models.use_renderer:
            if scene_manager.scene_config.models.out_fuse_type in [1]:
                rgb, foreground_rgb = decode_head(
                    scene_manager,
                    scene_manager.model.renderer_UNet,
                    feature_map.squeeze(-2).permute(0, 3, 1, 2),
                    attn,
                    num_slots,
                    image_shape,
                )
        else:
            rgb = None
        scene_manager.model.clear_grad()

    # apply the last activation function on Render, Albedo, and Shading,
    rgb = scene_manager.model.last_act(rgb)  # eval prediction rgb
    if scene_manager.scene_config.models.use_albedo:
        test_pred_albedo = scene_manager.model.last_act(test_pred_albedo)

    # preprocessing on prediction images
    if len(scene_manager.scene_config.dataset.render_pred_preprocessing) != 0:
        rgb = apply_image_pipeline(
            scene_manager.scene_config,
            rgb,
            scene_manager.scene_config.dataset.render_pred_preprocessing,
            "render",
        )
    if (
        scene_manager.scene_config.models.use_albedo
        and len(scene_manager.scene_config.dataset.albedo_pred_preprocessing) != 0
    ):
        test_pred_albedo = apply_image_pipeline(
            scene_manager.scene_config,
            test_pred_albedo,
            scene_manager.scene_config.dataset.albedo_pred_preprocessing,
            "albedo",
        )

    # The tensors each of the six plot panels is built from. The display rows
    # run them through the postprocessing pipelines; the raw-space rows show
    # them as they are, which is what the model actually trains on.
    render_sources = {
        "train_tgt": train_img,
        "train_tgt_patch": train_patch,
        "train_pred_patch": train_rgb_out,
        "test_tgt": img,
        "test_pred": rgb,
        "test_pred_foreground": foreground_rgb,
    }
    albedo_sources = (
        {
            "train_tgt": train_tgt_albedo,
            "train_tgt_patch": train_tgt_albedo_patch,
            "train_pred_patch": train_pred_albedo_patch,
            "test_tgt": test_tgt_albedo,
            "test_pred": test_pred_albedo,
            "test_pred_foreground": foreground_albedo,
        }
        if scene_manager.scene_config.models.use_albedo
        else None
    )

    dataset_config = scene_manager.scene_config.dataset
    render_raw = None
    if (
        len(dataset_config.render_pred_postprocessing) != 0
        or len(dataset_config.render_GT_postprocessing) != 0
    ):
        render_raw = panel_arrays(
            scene_manager.scene_config, render_sources, "render", [], []
        )
    albedo_raw = None
    if scene_manager.scene_config.models.use_albedo and (
        len(dataset_config.albedo_GT_postprocessing) != 0
        or len(dataset_config.albedo_pred_postprocessing) != 0
    ):
        albedo_raw = panel_arrays(
            scene_manager.scene_config, albedo_sources, "albedo", [], []
        )

    # calculate the loss function on model prediction
    # Only the total is used; the individual eval terms were bound and never read.
    eval_total_loss = calculate_training_loss(
        scene_manager=scene_manager,
        render_pred_patch_pred_space=rgb,
        render_gt_patch_pred_space=img,
        albedo_pred_patch_pred_space=test_pred_albedo,
        albedo_gt_patch_pred_space=test_tgt_albedo,
        clip=True,
    ).total

    ################################## Adding Losses to the log dictionary ##################################
    scene_manager.total_eval_losses.append(eval_total_loss.item())

    ################################## Adding Losses to the log dictionary ##################################
    pt_plot_scale = point_cloud_plot_scale(scene_manager.scene_config.dataset)

    current_depth = depth_from_attention(
        scene_manager, selected_points, attn, rayo, image_shape
    )

    print(
        "Eval step:",
        scene_manager.step,
    )

    render_panels = panel_arrays(
        scene_manager.scene_config,
        render_sources,
        "render",
        dataset_config.render_GT_postprocessing,
        dataset_config.render_pred_postprocessing,
    )
    albedo_panels = None
    if scene_manager.scene_config.models.use_albedo:
        albedo_panels = panel_arrays(
            scene_manager.scene_config,
            albedo_sources,
            "albedo",
            dataset_config.albedo_GT_postprocessing,
            dataset_config.albedo_pred_postprocessing,
        )

    points_np = scene_manager.model.points.detach().cpu().numpy()
    depth = current_depth.squeeze().numpy().astype(np.float32)
    points_conf_scores_np = None
    if scene_manager.model.points_conf_scores is not None:
        points_conf_scores_np = (
            scene_manager.model.points_conf_scores.squeeze().detach().cpu().numpy()
        )

    # PSNR is measured on the postprocessed render, which is what the paper reports
    eval_psnr = (
        -10.0
        * np.log(
            (
                (render_panels["test_pred"] - render_panels["test_tgt"]) ** 2
            ).mean().item()
        )
        / np.log(10.0)
    )

    scene_manager.eval_psnrs.append(eval_psnr.item())

    scene_metrics_dict = {"eval_psnr": eval_psnr.item()}
    # update all keys and attach scene index before the keys
    metrics_dict = {
        f"scene_{scene_manager.scene_config.scene_idx}_{key}": value
        for key, value in scene_metrics_dict.items()
    }

    # main plot
    main_plot = get_training_main_plot(
        index=scene_manager.scene_config.index,
        step=scene_manager.step,
        render_panels=render_panels,
        depth_np=depth,
        points_np=points_np,
        pt_plot_scale=pt_plot_scale,
        bg_attentions=bg_attention,
        bg_masks=bg_mask,
        render_raw_panels=render_raw,
        albedo_panels=albedo_panels,
        albedo_raw_panels=albedo_raw,
        points_conf_scores_np=points_conf_scores_np,
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

    return metrics_dict


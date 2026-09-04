import os

import numpy as np
import torch
import torch.nn as nn

from .mlp import get_mapping_mlp
from .renderer import get_generator
from .tx import get_transformer
from .utils import (
    activation_func,
    add_points_knn,
    cacluate_rgb_from_albedo_and_shading,
    create_learning_rate_fn,
    extract_features_from_feature_map,
    normalize_vector,
    preprocess_postproces_images_pipeline,
    setup_seed,
)


class VolumetricBank(nn.Module):

    def __init__(
        self,
        scene_manager,
        scene_idx=0,
        shared_components=None,
    ):
        super(VolumetricBank, self).__init__()
        self.scene_manager = scene_manager
        self.eps = scene_manager.scene_config.eps
        self.device = scene_manager.device
        self.scene_idx = scene_idx
        self.shared_components = shared_components
        self.use_albedo = self.scene_manager.scene_config.models.use_albedo
        self.use_amp = self.scene_manager.scene_config.use_amp
        point_opt = self.scene_manager.scene_config.geoms.points
        pc_feat_opt = self.scene_manager.scene_config.geoms.point_feats
        bkg_feat_opt = self.scene_manager.scene_config.geoms.background
        self.coord_scale = self.scene_manager.scene_config.dataset.coord_scale

        self.amp_dtype = (
            torch.float16
            if self.scene_manager.scene_config.amp_dtype == "float16"
            else torch.bfloat16
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.register_buffer(
            "select_k",
            torch.tensor(point_opt.select_k, device=self.device, dtype=torch.int32),
        )
        self.register_buffer(
            "sample_k",
            torch.tensor(point_opt.sample_k, device=self.device, dtype=torch.int32),
        )

        if point_opt.load_path:
            points = torch.load(point_opt.load_path)
        else:
            # Init point positions
            pt_init_center = [i * self.coord_scale for i in point_opt.init_center]
            pt_init_scale = [i * self.coord_scale for i in point_opt.init_scale]
            if point_opt.init_type == "norm-cube":
                points = self._cube_normal_pc(
                    pt_init_center, point_opt.num, pt_init_scale
                )
            else:
                raise NotImplementedError(
                    "Point init type [{:s}] is not found".format(point_opt.init_type)
                )
        self.points = torch.nn.Parameter(points, requires_grad=True)
        self.points_last_grad = torch.nn.Parameter(
            torch.zeros(points.shape[0], 3, device=self.device), requires_grad=False
        )
        self.points_acc_grad = torch.nn.Parameter(
            torch.zeros(points.shape[0], 3, device=self.device), requires_grad=False
        )
        self.points_acc_grad_norm = torch.nn.Parameter(
            torch.zeros(points.shape[0], device=self.device), requires_grad=False
        )
        self.points_grad_cnt = torch.nn.Parameter(
            torch.zeros(points.shape[0], device=self.device), requires_grad=False
        )

        # Init point confidence scores
        self.points_conf_scores = None
        if point_opt.conf_type == 1:
            self.points_conf_scores = torch.nn.Parameter(
                torch.ones(point_opt.num, 1, device=self.device)
                * point_opt.conf_init_val,
                requires_grad=True,
            )

        # Init mapping MLP, only if fine-tuning with IMLE for the exposure control
        self.mapping_mlp = None
        if self.scene_manager.scene_config.models.mapping_mlp.use:
            self.mapping_mlp = get_mapping_mlp(
                self.scene_manager.scene_config.models,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
            )

        # Setup dims
        tx_opt = self.scene_manager.scene_config.models.transformer
        self.renderer_UNet_inp_size = (
            tx_opt.embed.d_ff_out
            if tx_opt.embed.share_embed
            else tx_opt.embed.value.d_ff_out
        )
        # renderer receives the whole feature map from the transformer
        self.albedo_UNet_inp_size = int(
            self.renderer_UNet_inp_size
            * self.scene_manager.scene_config.models.albedo.features.UNet_ratio
        )
        self.albedo_feat_side = (
            self.scene_manager.scene_config.models.albedo.features.side
        )
        print("#" * 80)
        print(
            "\033[92m"
            + "Renderer input size: {}, Albedo UNet input size: {}, Albedo ratio: {}, Albedo features side: {}".format(
                self.renderer_UNet_inp_size,
                self.albedo_UNet_inp_size,
                self.scene_manager.scene_config.models.albedo.features.UNet_ratio,
                self.albedo_feat_side,
            )
            + "\033[00m"
        )
        print("#" * 80)

        # Init UNet
        if self.scene_manager.scene_config.models.use_renderer:
            if (
                self.shared_components is not None
                and "Renderer_UNet" in self.shared_components
            ):
                self.renderer_UNet = self.shared_components["Renderer_UNet"]
            else:
                self.renderer_UNet = get_generator(
                    self.scene_manager.scene_config.models.renderer.generator,
                    in_c=self.renderer_UNet_inp_size,
                    out_c=3,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                )
        elif not self.scene_manager.scene_config.models.use_implicit_renderer:
            assert (
                self.scene_manager.scene_config.models.transformer.embed.share_embed
                and self.scene_manager.scene_config.models.transformer.embed.d_ff_out
                == 3
            ) or (
                not self.scene_manager.scene_config.models.transformer.embed.share_embed
                and self.scene_manager.scene_config.models.transformer.embed.value.d_ff_out
                == 3
            ), "Value embedding MLP should have output dim 3 if not using renderer"

        # Init albedo branch
        if self.use_albedo:
            if (
                self.shared_components is not None
                and "Albedo_UNet" in self.shared_components
            ):
                self.albedo_model = self.shared_components["Albedo_UNet"]
            else:
                self.albedo_model = get_generator(
                    self.scene_manager.scene_config.models.albedo.generator,
                    in_c=self.albedo_UNet_inp_size,
                    out_c=3,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                )

        v_extra_dim = 0
        k_extra_dim = 0
        q_extra_dim = 0

        self.bkg_feats = None
        self.bkg_score = None
        self.bkg_type = bkg_feat_opt.type
        if bkg_feat_opt.use_bkg_feat:
            if bkg_feat_opt.init_type == "ones":
                bkg_feat_init_func = torch.ones
            else:
                raise NotImplementedError(
                    "Background init type [{:s}] is not found".format(
                        bkg_feat_opt.init_type
                    )
                )
            self.bkg_score = torch.tensor(
                bkg_feat_opt.constant, device=self.device, dtype=torch.float32
            ).reshape(1)

            if bkg_feat_opt.type == 1:  # Use in attn
                feat_dim = 3
                self.bkg_feats = nn.Parameter(
                    bkg_feat_init_func(
                        bkg_feat_opt.seq_len, feat_dim, device=self.device
                    )
                    * bkg_feat_opt.render_init_scale,
                    requires_grad=bkg_feat_opt.learnable,
                )
            else:
                raise NotImplementedError(
                    "Background feature type [{:d}] is not found".format(
                        bkg_feat_opt.type
                    )
                )

        # ------------------------------------------------------------------
        # Unbounded scenes (Mip-NeRF 360): append one extra "point" per ray at
        # the forward intersection of that ray with a large background sphere.
        # It gives the attention something to attend to where no foreground
        # point exists, instead of leaving far-field rays unsupported.
        # ------------------------------------------------------------------
        self.append_bkg_points = bool(bkg_feat_opt.get("append_bkg_points", False))
        if self.append_bkg_points:
            self.bkg_sphere_radius = float(bkg_feat_opt.get("sphere_radius", 150.0))
            self.bkg_sphere_center = [
                c * self.coord_scale
                for c in bkg_feat_opt.get("sphere_center", [0.0, 0.0, 0.0])
            ]

        self.use_pc_feats = (
            pc_feat_opt.use_ink or pc_feat_opt.use_inq or pc_feat_opt.use_inv
        )
        if self.use_pc_feats:
            if pc_feat_opt.type == "learnable":
                self.pc_feats = nn.Parameter(
                    torch.randn(point_opt.num, pc_feat_opt.dim) * pc_feat_opt.factor,
                    requires_grad=True,
                )
            print(
                "Point features: ",
                self.pc_feats.shape,
                self.pc_feats.min(),
                self.pc_feats.max(),
                self.pc_feats.mean(),
                self.pc_feats.std(),
            )

        # Dedicated feature and confidence rows for the appended background point.
        # They live outside self.pc_feats / self.points_conf_scores so that point
        # pruning and growing never touch them.
        self.bkg_point_feats = None
        self.bkg_point_conf = None
        if self.append_bkg_points:
            if self.use_pc_feats:
                self.bkg_point_feats = nn.Parameter(
                    torch.randn(1, pc_feat_opt.dim) * pc_feat_opt.factor,
                    requires_grad=True,
                )
            if self.points_conf_scores is not None:
                self.bkg_point_conf = nn.Parameter(
                    torch.full(
                        (1, self.points_conf_scores.shape[-1]),
                        float(point_opt.conf_init_val),
                    ),
                    requires_grad=True,
                )

        if pc_feat_opt.use_inv:
            v_extra_dim = self.pc_feats.shape[-1]
            print("Using v_extra_dim: ", v_extra_dim)
        if pc_feat_opt.use_ink:
            k_extra_dim = self.pc_feats.shape[-1]
            print("Using k_extra_dim: ", k_extra_dim)
        if pc_feat_opt.use_inq:
            q_extra_dim = self.pc_feats.shape[-1]
            print("Using q_extra_dim: ", q_extra_dim)

        self.last_act = activation_func(self.scene_manager.scene_config.models.last_act)

        transformer = get_transformer(
            args=self.scene_manager.scene_config.models.transformer,
            seq_len=point_opt.num,
            v_extra_dim=v_extra_dim,
            k_extra_dim=k_extra_dim,
            q_extra_dim=q_extra_dim,
            eps=self.eps,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            albedo_value_MLP_input_portion=self.scene_manager.scene_config.models.albedo.features.Emb_value_MLP_inp_ratio,
            albedo_value_MLP_output_portion=self.scene_manager.scene_config.models.albedo.features.Emb_value_MLP_out_ratio,
            albedo_value_MLP_feature_side=self.albedo_feat_side,
            shared_components=self.shared_components,
        )
        self.transformer = transformer

        self.seql_k = self.select_k
        self.seql_v = self.select_k
        self.seql_q = self.select_k
        if self.scene_manager.scene_config.models.transformer.q_type in [1]:
            self.seql_q = 1

        # Optimizers are deliberately not built here. The main script calls
        # init_optimizers() once the shared components have been set up.
        if self.scene_manager.scene_config.models.supervision_scaler.use:
            self.supervision_scaler = nn.Parameter(
                torch.ones(
                    self.scene_manager.scene_config.models.supervision_scaler.size,
                    device=self.device,
                )
                * self.scene_manager.scene_config.models.supervision_scaler.intial_value,
                requires_grad=True,
            )
            self.use_supervision_scaler = True
        else:
            self.supervision_scaler = None
            self.use_supervision_scaler = False

    def init_optimizers(self, total_steps):
        lr_opt = self.scene_manager.scene_config.training.lr
        print("LR factor: ", lr_opt.lr_factor)
        optimizer_points = torch.optim.Adam(
            [self.points], lr=lr_opt.points.base_lr * lr_opt.lr_factor
        )
        optimizer_tx = torch.optim.Adam(
            self.transformer.parameters(),
            lr=lr_opt.transformer.base_lr * lr_opt.lr_factor,
            weight_decay=lr_opt.transformer.weight_decay,
        )

        lr_scheduler_points = create_learning_rate_fn(
            optimizer_points,
            self.scene_manager.scene_config.training.steps,
            lr_opt.points,
            debug=self.scene_manager.args.debug,
        )
        lr_scheduler_tx = create_learning_rate_fn(
            optimizer_tx,
            self.scene_manager.scene_config.training.steps,
            lr_opt.transformer,
            debug=self.scene_manager.args.debug,
        )

        self.optimizers = {
            "points": optimizer_points,
            "transformer": optimizer_tx,
        }

        self.schedulers = {
            "points": lr_scheduler_points,
            "transformer": lr_scheduler_tx,
        }

        if self.use_pc_feats:
            if "learnable" in self.scene_manager.scene_config.geoms.point_feats.type:
                pc_feat_params = [self.pc_feats]
                if self.bkg_point_feats is not None:
                    pc_feat_params.append(self.bkg_point_feats)
                optimizer_pc_feats = torch.optim.Adam(
                    pc_feat_params,
                    lr=lr_opt.feats.base_lr * lr_opt.lr_factor,
                    weight_decay=lr_opt.feats.weight_decay,
                )
            lr_scheduler_pc_feats = create_learning_rate_fn(
                optimizer_pc_feats,
                self.scene_manager.scene_config.training.steps,
                lr_opt.feats,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["pc_feats"] = optimizer_pc_feats
            self.schedulers["pc_feats"] = lr_scheduler_pc_feats

        if self.mapping_mlp is not None:
            optimizer_mapping_mlp = torch.optim.Adam(
                self.mapping_mlp.parameters(),
                lr=lr_opt.mapping_mlp.base_lr * lr_opt.lr_factor,
                weight_decay=lr_opt.mapping_mlp.weight_decay,
            )
            lr_scheduler_mapping_mlp = create_learning_rate_fn(
                optimizer_mapping_mlp,
                self.scene_manager.scene_config.training.steps,
                lr_opt.mapping_mlp,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["mapping_mlp"] = optimizer_mapping_mlp
            self.schedulers["mapping_mlp"] = lr_scheduler_mapping_mlp

        if self.scene_manager.scene_config.models.use_renderer:
            optimizer_renderer = torch.optim.Adam(
                self.renderer_UNet.parameters(),
                lr=lr_opt.generator.base_lr * lr_opt.lr_factor,
                weight_decay=lr_opt.generator.weight_decay,
            )
            lr_scheduler_renderer = create_learning_rate_fn(
                optimizer_renderer,
                self.scene_manager.scene_config.training.steps,
                lr_opt.generator,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["renderer"] = optimizer_renderer
            self.schedulers["renderer"] = lr_scheduler_renderer

        if self.use_albedo:
            optimizer_albedo = torch.optim.Adam(
                self.albedo_model.parameters(),
                lr=lr_opt.albedo.base_lr * lr_opt.lr_factor,
                weight_decay=lr_opt.albedo.weight_decay,
            )
            lr_scheduler_albedo = create_learning_rate_fn(
                optimizer_albedo,
                self.scene_manager.scene_config.training.steps,
                lr_opt.albedo,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["albedo"] = optimizer_albedo
            self.schedulers["albedo"] = lr_scheduler_albedo

        if (
            self.bkg_feats is not None
            and self.scene_manager.scene_config.geoms.background.learnable
        ):
            optimizer_bkg_feats = torch.optim.Adam(
                [self.bkg_feats],
                lr=lr_opt.bkg_feats.base_lr * lr_opt.lr_factor,
                weight_decay=lr_opt.bkg_feats.weight_decay,
            )
            lr_scheduler_bkg_feats = create_learning_rate_fn(
                optimizer_bkg_feats,
                self.scene_manager.scene_config.training.steps,
                lr_opt.bkg_feats,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["bkg_feats"] = optimizer_bkg_feats
            self.schedulers["bkg_feats"] = lr_scheduler_bkg_feats

        if self.points_conf_scores is not None:
            conf_params = [self.points_conf_scores]
            if self.bkg_point_conf is not None:
                conf_params.append(self.bkg_point_conf)
            optimizer_points_conf_scores = torch.optim.Adam(
                conf_params,
                lr=lr_opt.points_conf_scores.base_lr * lr_opt.lr_factor,
                weight_decay=lr_opt.points_conf_scores.weight_decay,
            )
            lr_scheduler_points_conf_scores = create_learning_rate_fn(
                optimizer_points_conf_scores,
                self.scene_manager.scene_config.training.steps,
                lr_opt.points_conf_scores,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["points_conf_scores"] = optimizer_points_conf_scores
            self.schedulers["points_conf_scores"] = lr_scheduler_points_conf_scores

        if self.use_supervision_scaler:
            optimizer_supervision_scaler = torch.optim.Adam(
                [self.supervision_scaler],
                lr=lr_opt.supervision_scaler.base_lr * lr_opt.lr_factor,
                weight_decay=lr_opt.supervision_scaler.weight_decay,
            )
            lr_scheduler_supervision_scaler = create_learning_rate_fn(
                optimizer_supervision_scaler,
                self.scene_manager.scene_config.training.steps,
                lr_opt.supervision_scaler,
                debug=self.scene_manager.args.debug,
            )

            self.optimizers["supervision_scaler"] = optimizer_supervision_scaler
            self.schedulers["supervision_scaler"] = lr_scheduler_supervision_scaler

        for name in self.scene_manager.scene_config.training.fix_keys:
            if name in self.optimizers:
                print("Fixing {}".format(name))
                self.optimizers.pop(name)
                self.schedulers.pop(name)

        if total_steps > 0:
            for name, scheduler in self.schedulers.items():
                if scheduler is not None:
                    for i in range(total_steps):
                        scheduler.step()

    def clean_optimizer(self):
        self.optimizers.clear()
        del self.optimizers

    def clean_scheduler(self):
        self.schedulers.clear()
        del self.schedulers

    def clear_grad(self):
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                optimizer.zero_grad()

    def _cube_pc(self, center, num_pts, scale):
        xs = np.random.uniform(-scale[0], scale[0], num_pts) + center[0]
        ys = np.random.uniform(-scale[1], scale[1], num_pts) + center[1]
        zs = np.random.uniform(-scale[2], scale[2], num_pts) + center[2]
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()

    def _cube_normal_pc(self, center, num_pts, scale):
        axis_num_pts = int(num_pts ** (1.0 / 3.0))
        xs = np.linspace(-scale[0], scale[0], axis_num_pts) + center[0]
        ys = np.linspace(-scale[1], scale[1], axis_num_pts) + center[1]
        zs = np.linspace(-scale[2], scale[2], axis_num_pts) + center[2]
        points = np.array([[i, j, k] for i in xs for j in ys for k in zs])
        rest_num_pts = num_pts - points.shape[0]
        if rest_num_pts > 0:
            rest_points = self._cube_pc(center, rest_num_pts, scale)
            points = np.concatenate([points, rest_points], axis=0)
        return torch.from_numpy(points).float()

    def _calculate_global_distances(self, rays_o, rays_d, points):
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape

        rays_d = rays_d.unsqueeze(-2)  # (N, H, W, 1, 3)
        rays_o = rays_o.reshape(N, 1, 1, 1, 3)
        points = points.reshape(1, 1, 1, num_pts, 3)

        if self.scene_manager.scene_config.geoms.points.select_k_type == "d2r":
            origin_to_points = points - rays_o  # (N, 1, 1, num_pts, 3)
            parallel_component = rays_d * (
                torch.sum(origin_to_points * rays_d, dim=-1)
                / (torch.sum(rays_d * rays_d, dim=-1) + self.eps)
            ).unsqueeze(-1)
            perp_component = (
                origin_to_points - parallel_component
            )  # (N, H, W, num_pts, 3)
            dists_to_rays = torch.norm(perp_component, dim=-1)
        else:
            raise ValueError("Invalid select_k type")

        _, sampled_k_ind = dists_to_rays.topk(
            self.sample_k,
            dim=-1,
            largest=False,
            sorted=self.scene_manager.scene_config.geoms.points.select_k_sorted,
        )  # (N, H, W, sample_k)
        if self.sample_k == self.select_k:
            select_k_ind = sampled_k_ind
        elif self.scene_manager.scene_config.geoms.points.sample_k_type == "uniform":
            select_inds = np.random.choice(
                self.sample_k.item(), self.select_k.item(), replace=False
            )  # (N, H, W, select_k)
            select_k_ind = sampled_k_ind[..., select_inds]

        return select_k_ind

    def _calculate_distances(self, rays_o, rays_d, points):
        N, H, W, _ = rays_d.shape

        unit_rays = normalize_vector(rays_d, eps=self.eps).unsqueeze(
            -2
        )  # (N, H, W, 1, 3)
        origin_to_points = points - rays_o.reshape(
            N, 1, 1, 1, 3
        )  # (N, 1, 1, num_pts, 3)
        parallel_component = unit_rays * (
            torch.sum(origin_to_points * unit_rays, dim=-1)
            / (torch.sum(unit_rays * unit_rays, dim=-1) + self.eps)
        ).unsqueeze(-1)
        perp_component = (
            origin_to_points - parallel_component
        )  # (N, H, W, num_pts, 3)

        dists_to_rays = torch.norm(perp_component, dim=-1).unsqueeze(-1)
        proj_dists = torch.norm(parallel_component, dim=-1).unsqueeze(-1)

        return proj_dists, dists_to_rays, parallel_component, perp_component

    def get_bkg_sphere_intersection(self, rays_o, rays_d):
        """Forward intersection of each ray with the background sphere.

        Returns one point per ray, shaped (N, H, W, 3). Rays that miss the
        sphere are clamped to the tangent point, so every ray always yields a
        usable background position.
        """
        N, H, W, _ = rays_d.shape
        center = torch.tensor(
            self.bkg_sphere_center, dtype=rays_d.dtype, device=rays_d.device
        )
        unit_rays = normalize_vector(rays_d, eps=self.eps)
        center_to_origin = rays_o.view(N, 1, 1, 3) - center.view(1, 1, 1, 3)
        half_linear_term = torch.sum(center_to_origin * unit_rays, dim=-1)
        constant_term = (
            torch.sum(center_to_origin * center_to_origin, dim=-1)
            - self.bkg_sphere_radius**2
        )
        discriminant = torch.clamp(
            half_linear_term * half_linear_term - constant_term, min=0.0
        )
        ray_t = torch.clamp(-half_linear_term + torch.sqrt(discriminant), min=0.0)
        return rays_o.view(N, 1, 1, 3) + unit_rays * ray_t.unsqueeze(-1)

    def _gather_with_bkg(self, per_point_tensor, bkg_row, select_k_ind):
        """Index a per-point tensor by select_k_ind, honouring the background slot.

        select_k_ind carries one extra index per ray, equal to the number of real
        points, which addresses bkg_row.
        """
        if bkg_row is None:
            return per_point_tensor[select_k_ind]
        table = torch.cat([per_point_tensor, bkg_row.to(per_point_tensor.dtype)], dim=0)
        return table[select_k_ind]

    def _get_points(self, rays_o, rays_d):
        points = self.points
        if self.select_k >= points.shape[0] or self.select_k < 0:
            return points, None

        select_k_ind = self._calculate_global_distances(
            rays_o, rays_d, points
        )  # (N, H, W, num_pts)
        selected_points = points[select_k_ind, :]  # (N, H, W, select_k, 3)

        if self.append_bkg_points:
            # One extra slot per ray: the ray-sphere intersection, indexed by
            # points.shape[0] so that every per-point gather can find its
            # dedicated background row.
            intersection = self.get_bkg_sphere_intersection(rays_o, rays_d)
            selected_points = torch.cat(
                [selected_points, intersection.unsqueeze(-2)], dim=-2
            )
            bkg_index = torch.full(
                select_k_ind.shape[:-1] + (1,),
                points.shape[0],
                dtype=select_k_ind.dtype,
                device=select_k_ind.device,
            )
            select_k_ind = torch.cat([select_k_ind, bkg_index], dim=-1)

        self.selected_points = selected_points
        self.select_k_ind = select_k_ind

        return selected_points, select_k_ind

    def prune_points(self, thresh):
        if self.points_conf_scores is not None:
            if self.scene_manager.scene_config.training.prune_type == "<":
                mask = self.points_conf_scores[:, 0] > thresh
            elif self.scene_manager.scene_config.training.prune_type == ">":
                mask = self.points_conf_scores[:, 0] < thresh
            print("@@@@@@@@@  pruned {}/{}".format(torch.sum(mask == 0), mask.shape[0]))

            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(
                self.points[mask, :], requires_grad=cur_requires_grad
            )
            print("@@@@@@@@@ New points: ", self.points.shape)

            cur_requires_grad = self.points_conf_scores.requires_grad
            self.points_conf_scores = nn.Parameter(
                self.points_conf_scores[mask, :], requires_grad=cur_requires_grad
            )
            print("@@@@@@@@@ New points_conf_scores: ", self.points_conf_scores.shape)
            self.points_last_grad = nn.Parameter(
                self.points_last_grad[mask, :], requires_grad=False
            )
            self.points_acc_grad = nn.Parameter(
                self.points_acc_grad[mask, :], requires_grad=False
            )
            self.points_acc_grad_norm = nn.Parameter(
                self.points_acc_grad_norm[mask], requires_grad=False
            )
            self.points_grad_cnt = nn.Parameter(
                self.points_grad_cnt[mask], requires_grad=False
            )

            if (
                self.use_pc_feats
                and "learnable"
                in self.scene_manager.scene_config.geoms.point_feats.type
            ):
                cur_requires_grad = self.pc_feats.requires_grad
                self.pc_feats = nn.Parameter(
                    self.pc_feats[mask, :], requires_grad=cur_requires_grad
                )
                print("@@@@@@@@@ New pc_feats: ", self.pc_feats.shape)

            return torch.sum(mask == 0)
        return 0

    def add_points(self, add_num):
        points = self.points.detach().cpu()
        point_features = None
        cur_num_points = points.shape[0]

        if (
            "max_points" in self.scene_manager.scene_config
            and self.scene_manager.scene_config.max_points > 0
            and (cur_num_points + add_num) >= self.scene_manager.scene_config.max_points
        ):
            add_num = self.scene_manager.scene_config.max_points - cur_num_points
            if add_num <= 0:
                return 0

        if (
            self.use_pc_feats
            and "learnable" in self.scene_manager.scene_config.geoms.point_feats.type
        ):
            point_features = self.pc_feats.detach().cpu()

        (
            new_points,
            num_new_points,
            new_conf_scores,
            new_point_features,
        ) = add_points_knn(
            coords=points,
            influ_scores=self.points_conf_scores.detach().cpu(),
            add_num=add_num,
            k=self.scene_manager.scene_config.geoms.points.add_k,
            comb_type=self.scene_manager.scene_config.geoms.points.add_type,
            sample_k=self.scene_manager.scene_config.geoms.points.add_sample_k,
            sample_type=self.scene_manager.scene_config.geoms.points.add_sample_type,
            point_features=point_features,
            last_coord_grad=self.points_last_grad.detach().cpu(),
            acc_coord_grad=self.points_acc_grad.detach().cpu(),
            acc_coord_grad_norm=self.points_acc_grad_norm.detach().cpu(),
            grad_cnt=self.points_grad_cnt.detach().cpu(),
            move_scale=1.0,
        )
        print("@@@@@@@@@  added {} points".format(num_new_points))
        if new_points is not None:
            print(points.dtype, new_points.dtype)

        if num_new_points > 0:
            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(
                torch.cat([points, new_points], dim=0).to(self.points.device),
                requires_grad=cur_requires_grad,
            )
            print("@@@@@@@@@ New points: ", self.points.shape)

            if self.points_conf_scores is not None:
                cur_requires_grad = self.points_conf_scores.requires_grad
                self.points_conf_scores = nn.Parameter(
                    torch.cat(
                        [
                            self.points_conf_scores,
                            new_conf_scores.to(self.points_conf_scores.device),
                        ],
                        dim=0,
                    ),
                    requires_grad=cur_requires_grad,
                )
                print(
                    "@@@@@@@@@ New points_conf_scores: ", self.points_conf_scores.shape
                )

            self.points_last_grad = nn.Parameter(
                torch.zeros(self.points.shape[0], 3, device=self.points.device),
                requires_grad=False,
            )
            self.points_acc_grad = nn.Parameter(
                torch.zeros(self.points.shape[0], 3, device=self.points.device),
                requires_grad=False,
            )
            self.points_acc_grad_norm = nn.Parameter(
                torch.zeros(self.points.shape[0], device=self.points.device),
                requires_grad=False,
            )
            self.points_grad_cnt = nn.Parameter(
                torch.zeros(self.points.shape[0], device=self.points.device),
                requires_grad=False,
            )

            if (
                self.use_pc_feats
                and "learnable"
                in self.scene_manager.scene_config.geoms.point_feats.type
            ):
                cur_requires_grad = self.pc_feats.requires_grad
                self.pc_feats = nn.Parameter(
                    torch.cat(
                        [self.pc_feats, new_point_features.to(self.pc_feats.device)],
                        dim=0,
                    ),
                    requires_grad=cur_requires_grad,
                )
                print("@@@@@@@@@ New pc_feats: ", self.pc_feats.shape)

        return num_new_points

    def _get_kqv(self, rays_o, rays_d, points, select_k_ind, pd_factor=1.0):
        proj_dists, dists_to_rays, vec_p2o, vec_p2r = self._calculate_distances(
            rays_o, rays_d, points
        )

        proj_dists = proj_dists / pd_factor
        vec_p2o = vec_p2o / pd_factor

        N, H, W, _ = rays_d.shape
        num_pts = points.shape[-2]

        if points.dim() == 2:
            points = points.expand(N, H, W, -1, 3)

        k_type = self.scene_manager.scene_config.models.transformer.k_type
        k_L = self.scene_manager.scene_config.models.transformer.embed.k_L
        if k_type == 1:
            key = [points.detach(), vec_p2o, vec_p2r]
        else:
            raise ValueError("Invalid key type")
        assert len(key) == (len(k_L))

        q_type = self.scene_manager.scene_config.models.transformer.q_type
        q_L = self.scene_manager.scene_config.models.transformer.embed.q_L
        if q_type == 1:
            query = [rays_d.unsqueeze(-2)]
        else:
            raise ValueError("Invalid query type")
        assert len(query) == (len(q_L))

        v_type = self.scene_manager.scene_config.models.transformer.v_type
        v_L = self.scene_manager.scene_config.models.transformer.embed.v_L
        if v_type == 1:
            value = [vec_p2o, vec_p2r]
        else:
            raise ValueError("Invalid value type")
        assert len(value) == (len(v_L))

        k_extra = None
        q_extra = None
        v_extra = None
        if self.scene_manager.scene_config.geoms.point_feats.use_ink:
            if self.select_k >= self.points.shape[0]:
                k_extra = [self.pc_feats.expand(N, H, W, num_pts, -1)]
            else:
                k_extra = [
                    self._gather_with_bkg(
                        self.pc_feats, self.bkg_point_feats, select_k_ind
                    )
                ]
        if self.scene_manager.scene_config.geoms.point_feats.use_inq:
            if self.select_k >= self.points.shape[0]:
                q_extra = [self.pc_feats.expand(N, H, W, num_pts, -1)]
            else:
                q_extra = [
                    self._gather_with_bkg(
                        self.pc_feats, self.bkg_point_feats, select_k_ind
                    )
                ]
        if self.scene_manager.scene_config.geoms.point_feats.use_inv:
            if self.select_k >= self.points.shape[0]:
                v_extra = [self.pc_feats.expand(N, H, W, num_pts, -1)]
            else:
                v_extra = [
                    self._gather_with_bkg(
                        self.pc_feats, self.bkg_point_feats, select_k_ind
                    )
                ]

        return key, query, value, k_extra, q_extra, v_extra

    def step(self, step=-1):
        self.points_last_grad.data = self.points.grad
        self.points_acc_grad.data += self.points.grad
        self.points_acc_grad_norm.data += torch.norm(self.points.grad, dim=-1)
        self.points_grad_cnt.data += (self.points.grad.sum(-1) != 0).float()

        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                self.scaler.step(optimizer)

        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.step()

        self.tx_lr = 0
        if "transformer" in self.optimizers:
            if self.schedulers["transformer"] is not None:
                self.tx_lr = self.schedulers["transformer"].get_last_lr()[0]
            else:
                self.tx_lr = self.optimizers["transformer"].param_groups[0]["lr"]

        self.pts_lr = 0
        if "points" in self.optimizers:
            if self.schedulers["points"] is not None:
                self.pts_lr = self.schedulers["points"].get_last_lr()[0]
            else:
                self.pts_lr = self.optimizers["points"].param_groups[0]["lr"]

        self.albedo_lr = 0
        if "albedo" in self.optimizers:
            if self.schedulers["albedo"] is not None:
                self.albedo_lr = self.schedulers["albedo"].get_last_lr()[0]
            else:
                self.albedo_lr = self.optimizers["albedo"].param_groups[0]["lr"]

        self.supervision_scaler_lr = 0
        if "supervision_scaler" in self.optimizers:
            if self.schedulers["supervision_scaler"] is not None:
                self.supervision_scaler_lr = self.schedulers[
                    "supervision_scaler"
                ].get_last_lr()[0]
            else:
                self.supervision_scaler_lr = self.optimizers[
                    "supervision_scaler"
                ].param_groups[0]["lr"]

    def evaluate(self, rays_o, rays_d, c2w, pt_idxs, step=-1, pd_factor=1.0):
        points, select_k_ind = self._get_points(rays_o, rays_d)
        self.select_k_ind = select_k_ind
        key, query, value, k_extra, q_extra, v_extra = self._get_kqv(
            rays_o, rays_d, points, select_k_ind, pd_factor
        )
        N, H, W, _ = rays_d.shape
        num_pts = points.shape[-2]

        cur_points_conf_score = (
            self._gather_with_bkg(
                self.points_conf_scores, self.bkg_point_conf, select_k_ind
            )
            if self.points_conf_scores is not None
            else None
        )

        embedk, embedq, embedv, encode, scores = self.transformer(
            key, query, value, k_extra, q_extra, v_extra, step=step
        )

        if self.scene_manager.scene_config.models.out_fuse_type == 1:
            embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
            scores = scores.reshape(N, H, W, -1, 1)

            if (
                cur_points_conf_score is not None
                and step <= self.scene_manager.scene_config.training.score_step
            ):
                scores = scores * cur_points_conf_score
            if (
                self.bkg_feats is not None
                and step <= self.scene_manager.scene_config.training.bkg_step
            ):
                if self.bkg_type == 1:
                    bkg_seq_len = self.bkg_feats.shape[0]
                    scores = torch.cat(
                        [scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)],
                        dim=-2,
                    )
                softmax = nn.Softmax(dim=3)
                attn = softmax(
                    scores * self.scene_manager.scene_config.models.sftmax_temp
                )
                topk_attn = attn[..., :num_pts, :]
                if self.scene_manager.scene_config.models.normalize_topk_attn:
                    topk_attn = topk_attn / torch.sum(topk_attn, dim=3, keepdim=True)
                self.top_k_att_TSNE = topk_attn
                fused_features = torch.sum(
                    embedv * topk_attn, dim=3, keepdim=True
                )  # (N, H, W, 1, C)
            else:
                softmax = nn.Softmax(dim=3)
                attn = softmax(
                    scores * self.scene_manager.scene_config.models.sftmax_temp
                )
                if self.scene_manager.scene_config.models.normalize_topk_attn:
                    attn = attn / torch.sum(attn, dim=3, keepdim=True)
                fused_features = torch.sum(
                    embedv * attn, dim=3, keepdim=True
                )  # (N, H, W, 1, C)

            out = torch.zeros(N, H, W, 3, device=attn.device)

            encode = encode.reshape(N, H, W, self.seql_v, -1)
            embedk = embedk.reshape(N, H, W, self.seql_k, -1)
            embedq = embedq.reshape(N, H, W, self.seql_q, -1)

        return (
            encode[..., pt_idxs, :],
            fused_features,
            attn,
            out,
            embedk[..., pt_idxs, :],
            embedq,
            embedv[..., pt_idxs, :],
        )

    def forward(self, rays_o, rays_d, c2w, step=-1, shading_code=None):
        gamma, beta = None, None
        if shading_code is not None and self.mapping_mlp is not None:
            affine = self.mapping_mlp(shading_code)
            affine_dim = affine.shape[-1]
            gamma, beta = affine[: affine_dim // 2], affine[affine_dim // 2 :]

        points, select_k_ind = self._get_points(rays_o, rays_d)
        key, query, value, k_extra, q_extra, v_extra = self._get_kqv(
            rays_o, rays_d, points, select_k_ind
        )
        N, H, W, _ = rays_d.shape
        num_pts = points.shape[-2]

        cur_points_conf_score = (
            self._gather_with_bkg(
                self.points_conf_scores, self.bkg_point_conf, select_k_ind
            )
            if self.points_conf_scores is not None
            else None
        )
        _, _, embedv, encode, scores = self.transformer(
            key, query, value, k_extra, q_extra, v_extra, step=step
        )

        if self.scene_manager.scene_config.models.out_fuse_type == 1:
            assert (
                self.scene_manager.scene_config.models.use_renderer
                or self.scene_manager.scene_config.models.use_implicit_renderer
            )
            embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
            scores = scores.reshape(N, H, W, -1, 1)

            if (
                cur_points_conf_score is not None
                and step <= self.scene_manager.scene_config.training.score_step
            ):
                scores = scores * cur_points_conf_score

            if (
                self.bkg_feats is not None
                and step <= self.scene_manager.scene_config.training.bkg_step
            ):
                if self.bkg_type == 1:
                    bkg_seq_len = self.bkg_feats.shape[0]
                    scores = torch.cat(
                        [scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)],
                        dim=-2,
                    )
                softmax = nn.Softmax(dim=3)
                attn = softmax(
                    scores * self.scene_manager.scene_config.models.sftmax_temp
                )
                topk_attn = attn[..., :num_pts, :]
                bkg_attn = attn[..., num_pts:, :]
                if self.scene_manager.scene_config.models.normalize_topk_attn:
                    topk_attn = topk_attn / torch.sum(topk_attn, dim=3, keepdim=True)
                fused_features = torch.sum(embedv * topk_attn, dim=3)  # (N, H, W, C)

                # albedo
                if self.use_albedo:
                    inp_feat_albedo = extract_features_from_feature_map(
                        features_map=fused_features,
                        features_dim=self.albedo_UNet_inp_size,
                        side=self.albedo_feat_side,
                    )
                    foreground_albedo = (
                        self.albedo_model(
                            inp_feat_albedo.permute(0, 3, 1, 2), gamma=gamma, beta=beta
                        )
                        .permute(0, 2, 3, 1)
                        .unsqueeze(-2)
                    )
                    if self.scene_manager.scene_config.models.normalize_topk_attn:
                        albedo = (
                            foreground_albedo * (1 - bkg_attn)
                            + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                        )
                    else:
                        albedo = (
                            foreground_albedo
                            + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                        )
                    albedo_output = albedo.squeeze(-2)
                else:
                    albedo_output = None

                # Shading is not predicted; it is derived from the render and the
                # albedo downstream (see calculate_shading_from_albedo_and_rendered_image).
                shading_output = None

                # renderer
                if self.scene_manager.scene_config.models.use_renderer:
                    foreground = (
                        self.renderer_UNet(
                            fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta
                        )
                        .permute(0, 2, 3, 1)
                        .unsqueeze(-2)
                    )  # (N, H, W, 1, 3)
                    if self.scene_manager.scene_config.models.normalize_topk_attn:
                        rgb = (
                            foreground * (1 - bkg_attn)
                            + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                        )
                    else:
                        rgb = (
                            foreground
                            + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                        )
                    rgb = rgb.squeeze(-2)
                elif self.scene_manager.scene_config.models.use_implicit_renderer:
                    # here we calculate the rgb as the sum of the albedo and shading in normalised log space
                    rgb = cacluate_rgb_from_albedo_and_shading(
                        albedo=albedo_output,
                        shading=shading_output,
                        scene_config=self.args,
                    )
                else:
                    rgb = fused_features

            else:
                softmax = nn.Softmax(dim=3)
                attn = softmax(
                    scores * self.scene_manager.scene_config.models.sftmax_temp
                )
                fused_features = torch.sum(embedv * attn, dim=3)  # (N, H, W, C)
                # albedo
                if self.use_albedo:
                    inp_feat_albedo = extract_features_from_feature_map(
                        features_map=fused_features,
                        features_dim=self.albedo_UNet_inp_size,
                        side=self.albedo_feat_side,
                    )
                    albedo_output = (
                        self.albedo_model(
                            inp_feat_albedo.permute(0, 3, 1, 2), gamma=gamma, beta=beta
                        )
                        .permute(0, 2, 3, 1)
                        .float()
                    )
                else:
                    albedo_output = None

                # Shading is not predicted; it is derived from the render and the
                # albedo downstream (see calculate_shading_from_albedo_and_rendered_image).
                shading_output = None

                if self.scene_manager.scene_config.models.use_renderer:
                    rgb = self.renderer_UNet(
                        fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta
                    ).permute(
                        0, 2, 3, 1
                    )  # (N, H, W, 3)
                elif self.scene_manager.scene_config.models.use_implicit_renderer:
                    rgb = cacluate_rgb_from_albedo_and_shading(
                        albedo=albedo_output,
                        shading=shading_output,
                        args=self.args,
                    )
                else:
                    rgb = fused_features

        if len(self.scene_manager.scene_config.dataset.render_pred_preprocessing) != 0:
            rgb = preprocess_postproces_images_pipeline(
                img=rgb,
                pipline=self.scene_manager.scene_config.dataset.render_pred_preprocessing,
                eps=self.scene_manager.scene_config.training.pred_preprocessing_eps,
                min_val=getattr(
                    self.scene_manager.scene_config.dataset,
                    "min_{}_log".format("render"),
                    None,
                ),
                max_val=getattr(
                    self.scene_manager.scene_config.dataset,
                    "max_{}_log".format("render"),
                    None,
                ),
                white_bg_value=getattr(
                    self.scene_manager.scene_config.geoms.background,
                    "render_init_scale",
                    None,
                ),
            )
        if (
            albedo_output is not None
            and len(self.scene_manager.scene_config.dataset.albedo_pred_preprocessing)
            != 0
        ):
            albedo_output = preprocess_postproces_images_pipeline(
                img=albedo_output,
                pipline=self.scene_manager.scene_config.dataset.albedo_pred_preprocessing,
                eps=self.scene_manager.scene_config.training.pred_preprocessing_eps,
                min_val=getattr(
                    self.scene_manager.scene_config.dataset,
                    "min_{}_log".format("albedo"),
                    None,
                ),
                max_val=getattr(
                    self.scene_manager.scene_config.dataset,
                    "max_{}_log".format("albedo"),
                    None,
                ),
                white_bg_value=getattr(
                    self.scene_manager.scene_config.geoms.background,
                    "albedo_init_scale",
                    None,
                ),
            )

        return (
            rgb,
            albedo_output,
            shading_output,
        )

    def save(self):
        save_dict = {
            "step": self.scene_manager.step,
            "seed": self.scene_manager.step + 1,
            "model_state_dict": self.state_dict(),
            "optimizers_state_dict": {},
            "schedulers_state_dict": {},
            "scaler_state_dict": self.scaler.state_dict(),
            "losses": {},
        }
        # Optimizers
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                save_dict["optimizers_state_dict"][name] = optimizer.state_dict()
            else:
                save_dict["optimizers_state_dict"] = None
        # Schedulers
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                save_dict["schedulers_state_dict"][name] = scheduler.state_dict()
            else:
                save_dict["schedulers_state_dict"] = None
        # Losses
        phases = ["train", "eval"]
        spaces = [
            "pred_space",
            "original_space",
            "pred_space_cIMLE",
            "original_space_cIMLE",
        ]
        image_types = ["render"]
        if self.scene_manager.scene_config.models.use_albedo:
            image_types.append("albedo")
        for phase in phases:
            for space in spaces:
                for image_type in image_types:
                    save_dict["losses"][f"{phase}_{image_type}_{space}"] = torch.tensor(
                        getattr(self.scene_manager, f"{image_type}_losses")[phase][
                            space
                        ]
                    )
        # Eval PSNRs
        save_dict["eval_psnrs"] = torch.tensor(self.scene_manager.eval_psnrs)
        torch.save(
            save_dict,
            os.path.join(
                self.scene_manager.checkpoints_dir,
                f"checkpoints-{self.scene_manager.step}.pth",
            ),
        )

    def load(self, manager, checkpoint_dir, specific_checkpoint=None, stage="train"):
        # check if "/checkpoints-{step}.pth" is the format, we use the new version otherwise we use the old version
        if (
            specific_checkpoint is not None
            and "checkpoints-"
            not in specific_checkpoint[specific_checkpoint.rfind("/") :]
        ):
            return self.load_old_version(
                manager, checkpoint_dir, specific_checkpoint, stage=stage
            )

        print("#" * 100)
        print("Loading model from: ", checkpoint_dir)
        if specific_checkpoint is not None:
            step_to_load = int(specific_checkpoint.split("-")[2].split(".")[0])
            print("step to load: ", step_to_load)
        else:
            try:
                # list all files with the model-{step}.pth
                # and load the latest one
                files = os.listdir(checkpoint_dir)
                files = [
                    file
                    for file in files
                    if file.startswith("checkpoints-") and file.endswith(".pth")
                ]
                files = sorted(files, key=lambda x: int(x.split("-")[1].split(".")[0]))
                step_to_load = int(files[-1].split("-")[1].split(".")[0])
                print("step to load: ", step_to_load)
            except:
                # print with red color
                print(
                    "\033[91m Can't resume because no checkpoint found in {}\033[00m".format(
                        checkpoint_dir
                    )
                )
                return 0

        # open the checkpoint file
        checkpoint_dict = torch.load(
            os.path.join(checkpoint_dir, f"checkpoints-{step_to_load}.pth")
        )
        print(
            f"The checkpoint's step was: {checkpoint_dict['step']}, so the next step is {checkpoint_dict['step'] + 1}",
        )
        # 1: seed
        setup_seed(checkpoint_dict["seed"])
        print(
            f"\033[92m seed loaded successfully and set to {checkpoint_dict['seed']}\033[00m"
        )

        # 2: load model state dict
        self.load_my_state_dict(checkpoint_dict["model_state_dict"])
        print("\033[92m model loaded successfully \033[00m")

        # 3: load optimizers
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                optimizer.load_state_dict(
                    checkpoint_dict["optimizers_state_dict"][name]
                )
                print("\033[92m {} optimizer loaded successfully \033[00m".format(name))
            else:
                assert checkpoint_dict["optimizers_state_dict"][name] is None

        # 4: load schedulers
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.load_state_dict(
                    checkpoint_dict["schedulers_state_dict"][name]
                )
                print("\033[92m {} scheduler loaded successfully \033[00m".format(name))
            else:
                assert checkpoint_dict["schedulers_state_dict"][name] is None

        # 5: load scaler
        self.scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
        print("\033[92m scaler loaded successfully \033[00m")

        # 6: load losses
        phases = ["train", "eval"]
        spaces = [
            "pred_space",
            "original_space",
            "pred_space_cIMLE",
            "original_space_cIMLE",
        ]
        image_types = ["render"]
        if manager.scene_config.models.use_albedo:
            image_types.append("albedo")
        for phase in phases:
            for space in spaces:
                for image_type in image_types:
                    loss = checkpoint_dict["losses"][f"{phase}_{image_type}_{space}"]
                    getattr(manager, f"{image_type}_losses")[phase][space] = list(
                        loss.detach().numpy()
                    )
                    print(
                        "\033[92m {} {} {} losses loaded successfully \033[00m".format(
                            phase, image_type, space
                        )
                    )

        # 7: load eval_psnrs
        manager.eval_psnrs = list(checkpoint_dict["eval_psnrs"].detach().numpy())

        return step_to_load

    def load_old_version(
        self,
        manager,
        checkpoint_dir,
        specific_checkpoint=None,
        stage="train",
    ):
        print("*" * 100)
        if specific_checkpoint is not None:
            step_to_load = int(specific_checkpoint.split("-")[1].split(".")[0])
        else:
            try:
                # list all files with the model-{step}.pth
                # and load the latest one
                files = os.listdir(checkpoint_dir)
                files = [
                    file
                    for file in files
                    if file.startswith("model-") and file.endswith(".pth")
                ]
                files = sorted(files, key=lambda x: int(x.split("-")[1].split(".")[0]))
                step_to_load = int(files[-1].split("-")[1].split(".")[0])
            except:
                # print with red color
                print(
                    "\033[91m Can't resume because no checkpoint found in {}\033[00m".format(
                        checkpoint_dir
                    )
                )
                return 0

        if stage == "train":
            optimizers_state_dict = torch.load(
                os.path.join(checkpoint_dir, f"optimizers-{step_to_load}.pth")
            )
            for name, optimizer in self.optimizers.items():
                if optimizer is not None:
                    optimizer.load_state_dict(optimizers_state_dict[name])
                    print(
                        "\033[92m {} optimizer loaded successfully \033[00m".format(
                            name
                        )
                    )
                else:
                    assert optimizers_state_dict[name] is None

            schedulers_state_dict = torch.load(
                os.path.join(checkpoint_dir, f"schedulers-{step_to_load}.pth")
            )
            for name, scheduler in self.schedulers.items():
                if scheduler is not None:
                    scheduler.load_state_dict(schedulers_state_dict[name])
                    print(
                        "\033[92m {} scheduler loaded successfully \033[00m".format(
                            name
                        )
                    )
                else:
                    assert schedulers_state_dict[name] is None

            if os.path.exists(
                os.path.join(checkpoint_dir, f"scaler-{step_to_load}.pth")
            ):
                scaler_state_dict = torch.load(
                    os.path.join(checkpoint_dir, f"scaler-{step_to_load}.pth")
                )
                self.scaler.load_state_dict(scaler_state_dict)
                print("\033[92m scaler loaded successfully \033[00m")

            phases = ["train", "eval"]
            spaces = [
                "pred_space",
                "original_space",
                "pred_space_cIMLE",
                "original_space_cIMLE",
            ]
            image_types = ["render"]
            if manager.scene_config.models.use_albedo:
                image_types.append("albedo")

            for phase in phases:
                for space in spaces:
                    for image_type in image_types:
                        loss = torch.load(
                            os.path.join(
                                checkpoint_dir,
                                f"{phase}_{image_type}_{space}_losses.pth",
                            )
                        )
                        getattr(manager, f"{image_type}_losses")[phase][space] = list(
                            loss.detach().numpy()
                        )
                        print(
                            "\033[92m {} {} {} losses loaded successfully \033[00m".format(
                                phase, image_type, space
                            )
                        )

            if os.path.exists(os.path.join(checkpoint_dir, "eval_psnrs.pth")):
                eval_psnrs = torch.load(os.path.join(checkpoint_dir, "eval_psnrs.pth"))
                manager.eval_psnrs = list(eval_psnrs.detach().numpy())
                print("\033[92m eval_psnrs loaded successfully \033[00m")

        model_state_dict = torch.load(
            os.path.join(checkpoint_dir, f"model-{step_to_load}.pth")
        )
        for step, state_dict in model_state_dict.items():
            self.load_my_state_dict(state_dict)
            print("*" * 100)
            print(
                "\033[92m Model loaded successfully at step {} from {}\033[00m".format(
                    step, checkpoint_dir
                )
            )

            # load the seed and set the seed
            if os.path.exists(os.path.join(checkpoint_dir, f"seed-{step_to_load}.pth")):
                seed = torch.load(
                    os.path.join(checkpoint_dir, f"seed-{step_to_load}.pth")
                )
                setup_seed(seed)
            return int(step)

    def load_my_state_dict(self, state_dict, exclude_keys=[]):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name.startswith("renderer."):
                name = name.replace("renderer.", "renderer_UNet.")
            print(name, param.shape)
            for exclude_key in exclude_keys:
                if exclude_key in name:
                    print("exclude", name)
                    break
            else:
                if name not in ["points", "points_conf_scores", "pc_feats"]:
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except:
                        print("Can't load", name)

        # when we instantiate the model, we don't know the size of the pc_feats; because the size of the points could be different during training
        self.points = nn.Parameter(
            state_dict["points"].data, requires_grad=self.points.requires_grad
        )
        if "points_last_grad" in state_dict:
            self.points_last_grad = nn.Parameter(
                state_dict["points_last_grad"].data, requires_grad=False
            )
            self.points_acc_grad = nn.Parameter(
                state_dict["points_acc_grad"].data, requires_grad=False
            )
            self.points_acc_grad_norm = nn.Parameter(
                state_dict["points_acc_grad_norm"].data, requires_grad=False
            )
            self.points_grad_cnt = nn.Parameter(
                state_dict["points_grad_cnt"].data, requires_grad=False
            )
        if self.points_conf_scores is not None:
            self.points_conf_scores = nn.Parameter(
                state_dict["points_conf_scores"].data,
                requires_grad=self.points_conf_scores.requires_grad,
            )
        if (
            "learnable" in self.scene_manager.scene_config.geoms.point_feats.type
            and self.use_pc_feats
        ):
            self.pc_feats = nn.Parameter(
                state_dict["pc_feats"].data, requires_grad=self.pc_feats.requires_grad
            )
            print(
                "load pc_feats",
                self.pc_feats.shape,
                self.pc_feats.min(),
                self.pc_feats.max(),
            )

    def move_shared_components_to_device(self):
        if self.shared_components is None or len(self.shared_components) == 0:
            return
        for item in self.shared_components.values():
            item.to(self.device)


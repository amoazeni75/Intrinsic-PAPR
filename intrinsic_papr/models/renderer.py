
import numpy as np
import torch
import torch.nn as nn

from ..data.pipeline import run_image_pipeline
from ..training import checkpoint
from ..training.schedules import create_learning_rate_fn
from .activations import activation_func
from .attention import get_transformer
from .features import extract_features_from_feature_map
from .generator import get_generator
from .points import add_points_knn, normalize_vector


class VolumetricBank(nn.Module):

    def __init__(
        self,
        scene_manager,
        scene_idx=0,
    ):
        super(VolumetricBank, self).__init__()
        self.scene_manager = scene_manager
        self.eps = scene_manager.scene_config.eps
        self.device = scene_manager.device
        self.scene_idx = scene_idx
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

        self._init_points(point_opt)
        self._init_decoders()
        extra_dims = self._init_background(point_opt, pc_feat_opt, bkg_feat_opt)
        self._init_transformer(point_opt, extra_dims)
        # Optimizers are deliberately not built here. The main script calls
        # init_optimizers() once the shared components have been set up.
        if self.scene_manager.scene_config.models.supervision_scaler.use:
            self.supervision_scaler = nn.Parameter(
                torch.ones(
                    self.scene_manager.scene_config.models.supervision_scaler.size,
                    device=self.device,
                )
                * self.scene_manager.scene_config.models.supervision_scaler.initial_value,
                requires_grad=True,
            )
            self.use_supervision_scaler = True
        else:
            self.supervision_scaler = None
            self.use_supervision_scaler = False


    def _init_points(self, point_opt):
        """The point cloud itself, plus the gradient statistics pruning reads."""
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


    def _init_decoders(self):
        """The render and albedo UNets, and the feature widths they take."""
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
        print(
            "Renderer input size: {}, Albedo UNet input size: {}, Albedo ratio: {}".format(
                self.renderer_UNet_inp_size,
                self.albedo_UNet_inp_size,
                self.scene_manager.scene_config.models.albedo.features.UNet_ratio,
            )
        )

        # Init UNet
        if self.scene_manager.scene_config.models.use_renderer:
            self.renderer_UNet = get_generator(
                self.scene_manager.scene_config.models.renderer.generator,
                in_c=self.renderer_UNet_inp_size,
                out_c=3,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
            )
        else:
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
            self.albedo_model = get_generator(
                self.scene_manager.scene_config.models.albedo.generator,
                in_c=self.albedo_UNet_inp_size,
                out_c=3,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
            )


    def _init_background(self, point_opt, pc_feat_opt, bkg_feat_opt):
        """Point features, the background feature rows, and the unbounded-scene sphere.

        Returns the extra key/query/value widths the transformer needs, which
        depend on whether the point features are fed into each of them.
        """
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
                "Point features: {}, range [{:.3f}, {:.3f}]".format(
                    tuple(self.pc_feats.shape),
                    self.pc_feats.min().item(),
                    self.pc_feats.max().item(),
                )
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
        if pc_feat_opt.use_ink:
            k_extra_dim = self.pc_feats.shape[-1]
        if pc_feat_opt.use_inq:
            q_extra_dim = self.pc_feats.shape[-1]

        self.last_act = activation_func(self.scene_manager.scene_config.models.last_act)

        return v_extra_dim, k_extra_dim, q_extra_dim

    def _init_transformer(self, point_opt, extra_dims):
        """The attention stack, sized by the extra dims the point features add."""
        v_extra_dim, k_extra_dim, q_extra_dim = extra_dims
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
        )
        self.transformer = transformer

        self.seql_k = self.select_k
        self.seql_v = self.select_k
        self.seql_q = self.select_k
        if self.scene_manager.scene_config.models.transformer.q_type in [1]:
            self.seql_q = 1


    def _optimizer_groups(self):
        """Every trainable group: its name, whether it is active, its parameters
        and the learning-rate config it reads.

        The parameters are thunks because the attribute behind a group only
        exists when that group is active. The order is the order the optimisers
        are stored in, which is also the order `step`, `save` and `load` walk.
        """
        config = self.scene_manager.scene_config

        def pc_feat_params():
            params = [self.pc_feats]
            if self.bkg_point_feats is not None:
                params.append(self.bkg_point_feats)
            return params

        def conf_params():
            params = [self.points_conf_scores]
            if self.bkg_point_conf is not None:
                params.append(self.bkg_point_conf)
            return params

        return (
            ("points", True, lambda: [self.points], "points"),
            ("transformer", True, self.transformer.parameters, "transformer"),
            ("pc_feats", self.use_pc_feats, pc_feat_params, "feats"),
            (
                "renderer",
                config.models.use_renderer,
                lambda: self.renderer_UNet.parameters(),
                "generator",
            ),
            (
                "albedo",
                self.use_albedo,
                lambda: self.albedo_model.parameters(),
                "albedo",
            ),
            (
                "bkg_feats",
                self.bkg_feats is not None and config.geoms.background.learnable,
                lambda: [self.bkg_feats],
                "bkg_feats",
            ),
            (
                "points_conf_scores",
                self.points_conf_scores is not None,
                conf_params,
                "points_conf_scores",
            ),
            (
                "supervision_scaler",
                self.use_supervision_scaler,
                lambda: [self.supervision_scaler],
                "supervision_scaler",
            ),
        )

    def init_optimizers(self, total_steps):
        """Build one Adam and one schedule per active group.

        ``total_steps`` fast-forwards the schedules, for resuming a run whose
        optimiser state was not restored.
        """
        lr_opt = self.scene_manager.scene_config.training.lr
        print("LR factor: ", lr_opt.lr_factor)

        self.optimizers = {}
        self.schedulers = {}
        for name, active, parameters, lr_key in self._optimizer_groups():
            if not active:
                continue
            # getattr, not [], so nested dicts come back wrapped
            lr_group = getattr(lr_opt, lr_key)
            optimizer = torch.optim.Adam(
                list(parameters()),
                lr=lr_group.base_lr * lr_opt.lr_factor,
                weight_decay=lr_group.weight_decay,
            )
            self.optimizers[name] = optimizer
            self.schedulers[name] = create_learning_rate_fn(
                optimizer,
                self.scene_manager.scene_config.training.steps,
                lr_group,
                debug=self.scene_manager.args.debug,
            )

        for name in self.scene_manager.scene_config.training.fix_keys:
            if name in self.optimizers:
                print("Fixing {}".format(name))
                self.optimizers.pop(name)
                self.schedulers.pop(name)

        if total_steps > 0:
            for scheduler in self.schedulers.values():
                if scheduler is not None:
                    for _ in range(total_steps):
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
            mask = self.points_conf_scores[:, 0] > thresh
            print("pruned {}/{}".format(torch.sum(mask == 0), mask.shape[0]))

            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(
                self.points[mask, :], requires_grad=cur_requires_grad
            )
            print("New points: ", self.points.shape)

            cur_requires_grad = self.points_conf_scores.requires_grad
            self.points_conf_scores = nn.Parameter(
                self.points_conf_scores[mask, :], requires_grad=cur_requires_grad
            )
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
            sample_k=self.scene_manager.scene_config.geoms.points.add_sample_k,
            sample_type=self.scene_manager.scene_config.geoms.points.add_sample_type,
            point_features=point_features,
            acc_coord_grad_norm=self.points_acc_grad_norm.detach().cpu(),
        )
        print("added {} points".format(num_new_points))
        if num_new_points > 0:
            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(
                torch.cat([points, new_points], dim=0).to(self.points.device),
                requires_grad=cur_requires_grad,
            )
            print("New points: ", self.points.shape)

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

        # the same point features feed whichever of the three branches asks for them;
        # only touch pc_feats when at least one does, as the original three blocks did
        point_feat_opt = self.scene_manager.scene_config.geoms.point_feats
        extra = None
        if point_feat_opt.use_ink or point_feat_opt.use_inq or point_feat_opt.use_inv:
            if self.select_k >= self.points.shape[0]:
                extra = [self.pc_feats.expand(N, H, W, num_pts, -1)]
            else:
                extra = [
                    self._gather_with_bkg(
                        self.pc_feats, self.bkg_point_feats, select_k_ind
                    )
                ]
        k_extra = extra if point_feat_opt.use_ink else None
        q_extra = extra if point_feat_opt.use_inq else None
        v_extra = extra if point_feat_opt.use_inv else None

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

        # supervision_scaler had a fourth readback here; nothing ever read it
        for attr, name in (
            ("tx_lr", "transformer"),
            ("pts_lr", "points"),
            ("albedo_lr", "albedo"),
        ):
            setattr(self, attr, self._current_lr(name))

    def _current_lr(self, name):
        """The learning rate in force for one optimizer group, 0 if it has none."""
        if name not in self.optimizers:
            return 0
        scheduler = self.schedulers[name]
        if scheduler is not None:
            return scheduler.get_last_lr()[0]
        return self.optimizers[name].param_groups[0]["lr"]

    def evaluate(self, rays_o, rays_d, pt_idxs, step=-1, pd_factor=1.0):
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
            key, query, value, k_extra, q_extra, v_extra
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

    def _preprocess_prediction(self, image, img_type):
        """Run the prediction-side pipeline over one model output.

        Returns the image untouched when no pipeline is configured, which is the
        case for every shipped config. Note this reads
        `training.pred_preprocessing_eps`, which no config defines and which is a
        different epsilon from `models.predict_in_log_space_eps` used everywhere
        else - so a config that turns a pipeline on here has to add it too.
        """
        pipeline = getattr(
            self.scene_manager.scene_config.dataset,
            "{}_pred_preprocessing".format(img_type),
        )
        if image is None or len(pipeline) == 0:
            return image
        return run_image_pipeline(
            img=image,
            pipeline=pipeline,
            eps=self.scene_manager.scene_config.training.pred_preprocessing_eps,
            min_val=getattr(
                self.scene_manager.scene_config.dataset,
                "min_{}_log".format(img_type),
                None,
            ),
            max_val=getattr(
                self.scene_manager.scene_config.dataset,
                "max_{}_log".format(img_type),
                None,
            ),
            white_bg_value=getattr(
                self.scene_manager.scene_config.geoms.background,
                "{}_init_scale".format(img_type),
                None,
            ),
        )

    def forward(self, rays_o, rays_d, step=-1):
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
            key, query, value, k_extra, q_extra, v_extra
        )

        if self.scene_manager.scene_config.models.out_fuse_type == 1:
            assert self.scene_manager.scene_config.models.use_renderer
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
                    )
                    foreground_albedo = (
                        self.albedo_model(
                            inp_feat_albedo.permute(0, 3, 1, 2)
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
                            fused_features.permute(0, 3, 1, 2)
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
                    )
                    albedo_output = (
                        self.albedo_model(
                            inp_feat_albedo.permute(0, 3, 1, 2)
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
                        fused_features.permute(0, 3, 1, 2)
                    ).permute(
                        0, 2, 3, 1
                    )  # (N, H, W, 3)
                else:
                    rgb = fused_features

        rgb = self._preprocess_prediction(rgb, "render")
        albedo_output = self._preprocess_prediction(albedo_output, "albedo")

        return (
            rgb,
            albedo_output,
            shading_output,
        )

    def save(self):
        checkpoint.save(self)

    def load(self, manager, checkpoint_dir, specific_checkpoint=None, stage="train"):
        return checkpoint.load(
            self, manager, checkpoint_dir, specific_checkpoint, stage=stage
        )

    def load_my_state_dict(self, state_dict):
        checkpoint.copy_state_dict(self, state_dict)


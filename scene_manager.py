import datetime
import json
import os

import lpips
import numpy as np
import torch

from dataset import get_dataset, get_loader
from models import get_loss, get_model
from tools.args_parser import *


class SceneManager:
    def __init__(
        self,
        args,
        all_configs,
        scene_config,
        eval_config,
        scene_key,
        scene_idx,
        cuda_idx=0,
        phase="train",
    ):
        print(
            "\033[91m"
            "For scene {}, CUDA index: {}".format(scene_idx, cuda_idx) + "\033[0m"
        )
        self.args = args
        self.all_configs = all_configs
        self.scene_config = scene_config
        self.scene_key = scene_key
        self.scene_idx = scene_idx
        self.eval_config = eval_config
        self.pruned = False
        self.pt_lrs = []
        self.tx_lrs = []
        self.albedo_lrs = []
        self.target_points_opacity = None

        self.device = torch.device(
            "cuda:%d" % cuda_idx if torch.cuda.is_available() else "cpu"
        )

        self.scene_config["scene_idx"] = scene_idx

        # update the print_step, and eval step
        if self.args.debug:
            self.all_configs["print_step"] = min(10, self.all_configs["print_step"])
            self.all_configs["save_checkpoint_step"] = min(
                10, self.all_configs["save_checkpoint_step"]
            )
            self.scene_config["eval"]["step"] = min(
                20, self.all_configs["print_step"] * 2
            )
            print(
                "\033[91m"
                "Debug mode is on. The print_step is changed to: {}, eval step is changed to: {}, save_checkpoint is changed to : {}".format(
                    self.all_configs["print_step"],
                    self.scene_config["eval"]["step"],
                    self.all_configs["save_checkpoint_step"],
                )
                + "\033[0m"
            )
        else:
            self.scene_config["eval"]["step"] = 2 * self.all_configs["print_step"]
            print(
                "\033[91m"
                "Eval step changed to: {}".format(self.scene_config["eval"]["step"])
                + "\033[0m"
            )

        if self.args.stage == "test":
            self.scene_config["dataset"]["batch_size"] = 1
            self.scene_config["dataset"]["read_offline"] = False
            # print with red color
            print(
                "\033[91m"
                + "********** We are in the test stage and we changed the batch size to 1 and  read_offline to False **********"
                + "\033[0m"
            )

        # Load the dataset
        self.setup_datasets()

        # setup the directories
        self.setup_directories()

        # setup the model
        self.setup_model()

        # setup the loss function
        self.setup_loss_fn()

        coord_scale = self.scene_config.dataset.coord_scale
        self.pt_plot_scale = 0.8 * coord_scale
        if "Barn" in self.scene_config.dataset.path:
            self.pt_plot_scale *= 1.5
        if "Family" in self.scene_config.dataset.path:
            self.pt_plot_scale *= 0.5

    def setup_datasets(self):
        relative_dataset_path = self.scene_config["dataset"]["path"]
        self.scene_config["dataset"]["path"] = os.path.join(
            self.all_configs["dataset_root"], self.scene_config["dataset"]["path"]
        )
        self.eval_config["dataset"]["path"] = os.path.join(
            self.all_configs["dataset_root"], self.eval_config["dataset"]["path"]
        )
        if (
            self.scene_config["models"]["predict_raw_in_log_space"]
            or self.scene_config["models"]["predict_rgb_in_log_space"]
        ):
            dataset_stats = load_dataset_statistics(
                self.scene_config,
                dataset_path=os.path.join(
                    self.all_configs["dataset_root"], relative_dataset_path
                ),
            )
            for key, value in dataset_stats.items():
                self.scene_config["dataset"][key] = value
                self.eval_config["dataset"][key] = value
                print(
                    "\033[92m{}\033[00m".format(
                        "The key {} is updated to {}.".format(key, value)
                    )
                )

        self.scene_config = DictAsMember(self.scene_config)
        self.eval_config = DictAsMember(self.eval_config)
        self.all_configs = DictAsMember(self.all_configs)

        self.train_dataset = get_dataset(
            dataset_args=self.scene_config.dataset,
            scene_config=self.scene_config,
            mode="train",
            use_albedo=self.scene_config.models.use_albedo,
            debug=self.args.debug,
        )
        self.train_dataloader = get_loader(
            self.train_dataset, self.scene_config.dataset, mode="train"
        )
        if self.args.force_using_train_views_for_test:
            self.eval_dataset = self.train_dataset
            print("We are using the train dataset for evaluation.")
        else:
            self.eval_dataset = get_dataset(
                dataset_args=self.eval_config.dataset,
                scene_config=self.scene_config,
                mode="test",
                use_albedo=self.scene_config.models.use_albedo,
                debug=self.args.debug,
            )
        self.scene_config["models"]["supervision_scaler"]["size"] = len(
            self.train_dataset
        )

    def setup_model(self):
        self.model = get_model(
            scene_manager=self,
            scene_idx=self.scene_idx,
            shared_components=self.all_configs.shared_components,
        )
        self.model = self.model.to(self.device)
        print("\033[91m" "Model's learnable parameters with their details:" + "\033[0m")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        print("#" * 100)

    def setup_loss_fn(self):
        self.render_loss_fn = get_loss(self.scene_config.training.render_losses).to(
            self.device
        )
        if self.scene_config.models.use_albedo:
            # loss function for albedo
            self.albedo_loss_fn = get_loss(self.scene_config.training.albedo_losses).to(
                self.device
            )
        else:
            self.albedo_loss_fn = None

        self.render_losses = {
            "train": {
                "pred_space": [],
                "original_space": [],
                "original_space_cIMLE": [],
                "pred_space_cIMLE": [],
            },
            "eval": {
                "pred_space": [],
                "original_space": [],
                "original_space_cIMLE": [],
                "pred_space_cIMLE": [],
            },
        }
        self.albedo_losses = {
            "train": {
                "pred_space": [],
                "original_space": [],
                "original_space_cIMLE": [],
                "pred_space_cIMLE": [],
            },
            "eval": {
                "pred_space": [],
                "original_space": [],
                "original_space_cIMLE": [],
                "pred_space_cIMLE": [],
            },
        }
        self.total_train_losses = []
        self.total_eval_losses = []
        self.eval_psnrs = []

        self.avg_total_train_loss = 0.0
        self.avg_render_loss_pred_space = 0.0
        self.avg_render_loss_original_space = 0.0
        self.avg_albedo_loss_pred_space = 0.0
        self.avg_albedo_loss_original_space = 0.0
        self.avg_albedo_loss_pred_space_cIMLE = 0.0
        self.avg_albedo_loss_original_space_cIMLE = 0.0

    def setup_directories(self):

        # the whole structure of the directories is something like this:
        # save_dir/all_configs.index = exp_dir
        # ---- scene_i_name (train_log_dir)
        # -------- checkpoints
        # -------- train_main_plots
        # -------- train_pcd_plots
        # -------- test
        # ------------ action_camera_media_step

        self.exp_dir = os.path.join(
            self.all_configs["save_dir"], self.all_configs["index"]
        )
        os.makedirs(self.exp_dir, exist_ok=True)

        # per-scene run directory; training metrics are appended here as metrics.jsonl
        self.scene_log_dir = os.path.join(self.exp_dir, self.scene_config["index"])
        os.makedirs(self.scene_log_dir, exist_ok=True)

        # checkpoint directory is the args.opt
        if self.args.stage == "train":
            self.checkpoints_dir = os.path.join(
                self.exp_dir, self.scene_config["index"], "checkpoints"
            )
        else:
            checkpoint_exp_dir = self.args.opt[: self.args.opt.rfind("/")]
            self.checkpoints_dir = os.path.join(
                checkpoint_exp_dir, self.scene_config["index"], "checkpoints"
            )

        self.train_main_plots_dir = os.path.join(
            self.exp_dir, self.scene_config["index"], "train_main_plots"
        )
        self.train_pcd_plots_dir = os.path.join(
            self.exp_dir, self.scene_config["index"], "train_pcd_plots"
        )
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        print("Checkpoints dir: ", self.checkpoints_dir)
        os.makedirs(self.train_main_plots_dir, exist_ok=True)
        print("Train main plots dir: ", self.train_main_plots_dir)
        os.makedirs(self.train_pcd_plots_dir, exist_ok=True)
        print("Train pcd plots dir: ", self.train_pcd_plots_dir)
        if self.args.stage == "test":
            assert (
                self.args.test_action is not None
            ), "Test action must be chosen from: (transfer_albedo, transfer_shading, render, PCA, change_brightness, interpolate_albedo, TSNE, calculate_albedo_consistency)"

            assert (
                self.args.render_frame_type is not None
            ), "Render frame type must be chosen from: (onfly, custom, all)"
            assert (
                self.args.media_type is not None
            ), "Media type must be chosen from: (image, video)"
            test_root = "test"
            if self.args.force_using_train_views_for_test:
                os.makedirs(
                    os.path.join(
                        self.exp_dir,
                        self.scene_config.index,
                        "test_using_train_views",
                    ),
                    exist_ok=True,
                )
                test_root = "test_using_train_views"
            else:
                os.makedirs(
                    os.path.join(self.exp_dir, self.scene_config.index, "test"),
                    exist_ok=True,
                )
            if self.scene_config.load_path is None:
                resume_step = "latest"
            else:
                resume_step = self.scene_config.load_path.split("-")[1].split(".")[0]
            time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            action_name = self.args.test_action
            if (
                self.args.test_action == "transfer_albedo"
                or self.args.test_action == "transfer_shading"
            ):
                action_name = action_name + "_{}_to_{}".format(
                    self.args.source_area_indices, self.args.target_area_indices
                )
            elif self.args.test_action == "freefrom_transfer_albedo" or self.args.test_action == "freefrom_transfer_shading":
                action_name = action_name + "_source_type_{}_target_type_{}".format(self.args.freeform_source_point_method, self.args.freeform_target_point_method)
            self.test_log_dir = os.path.join(
                self.exp_dir,
                self.scene_config.index,
                test_root,
                f"{action_name}_{self.args.render_frame_type}_{self.args.media_type}_step_{resume_step}_{time_now if self.args.include_time_in_name else ''}",
            )
            os.makedirs(self.test_log_dir, exist_ok=True)
            print("Test log dir: ", self.test_log_dir)

    def save_point_cloud(self):
        import open3d as o3d

        print(
            "\033[91m"
            + "********** saving point cloud to {}. **********".format(
                os.path.join(
                    self.exp_dir,
                    "points_{}_{}.ply".format(self.scene_idx, self.step),
                )
            )
            + "\033[0m"
        )
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(
            self.model.points.detach().cpu().numpy()
        )
        # we need to save the point cloud on the parent folder of log_dir
        o3d.io.write_point_cloud(
            os.path.join(
                self.exp_dir,
                self.scene_config.index,
                "points_{}_{}.ply".format(self.scene_idx, self.step - 1),
            ),
            point_cloud,
        )

    def get_points_in_box(self, box, points):
        corners = np.array(box)
        box_min_corner = np.min(corners, axis=0)
        box_max_corner = np.max(corners, axis=0)

        points_np = points.detach().cpu().numpy()
        inside_box = np.logical_and(
            points_np >= box_min_corner, points_np <= box_max_corner
        ).all(axis=1)
        return np.where(inside_box)[0]

    def load_points_areas_for_feature_transfer_from_boxes(
        self, source_area_indices, target_area_indices
    ):
        if source_area_indices is None:
            source_area_indices = []
        if target_area_indices is None:
            target_area_indices = []

        self.source_area_indices = {}
        self.target_area_indices = {}
        # print with red color
        if len(source_area_indices) == 0 and len(target_area_indices) == 0:
            return
        print(
            "\033[91m"
            + "********** Loading points areas from boxes ***********"
            + "\033[0m"
        )
        # we need to load the json file located in the root of the dataset
        boxes_file = os.path.join(
            self.scene_config.dataset.path,
            "point_cloud_areas.json",
        )
        with open(boxes_file, "r") as f:
            boxes = json.load(f)
        for area_idx in source_area_indices:
            self.source_area_indices[area_idx] = self.get_points_in_box(
                box=boxes[str(area_idx)], points=self.model.points
            )
            print(
                "\033[91m" + "area_idx:",
                area_idx,
                "number of points: ",
                len(self.source_area_indices[area_idx]),
                "\033[0m",
            )

        for area_idx in target_area_indices:
            self.target_area_indices[area_idx] = self.get_points_in_box(
                box=boxes[str(area_idx)], points=self.model.points
            )
            print(
                "\033[91m" + "area_idx:",
                area_idx,
                "number of points: ",
                len(self.target_area_indices[area_idx]),
                "\033[0m",
            )

    def load_points_areas_for_feature_transfer_from_file(
        self, source_area_indices, target_area_indices
    ):
        # print with red color
        print(
            "\033[91m"
            + "********** Loading points areas from file: {} ***********".format(
                os.path.join(
                    self.all_scenes_config.save_dir,
                    self.all_scenes_config.index,
                    self.scene_config.index,
                    "test",
                    "points_{}_{}.txt".format(self.scene_idx, self.resume_step),
                )
            )
            + "\033[0m"
        )
        tmp_log_dir = self.log_dir[: self.log_dir.rfind("/")]
        with open(
            os.path.join(
                tmp_log_dir,
                "points_{}_{}.txt".format(self.scene_idx, self.resume_step),
            ),
            "r",
        ) as f:
            lines = f.readlines()
            points = []
            labels = []
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                line = line.split(" ")
                points.append([float(line[0]), float(line[1]), float(line[2])])
                labels.append(int(float(line[3])))
            points = np.array(points)
            labels = np.array(labels)
            print("Labels are: ", np.unique(labels))
            if source_area_indices is not None:
                for area_idx in source_area_indices:
                    self.source_area_indices[area_idx] = np.where(labels == area_idx)[0]
                    if len(self.source_area_indices[area_idx]) == 0:
                        raise ValueError(
                            "The area_idx {} is not in the point cloud".format(area_idx)
                        )
                    print(
                        "\033[91m" + "area_idx:",
                        area_idx,
                        "number of points: ",
                        len(self.source_area_indices[area_idx]),
                        "\033[0m",
                    )
            if target_area_indices is not None:
                for area_idx in target_area_indices:
                    self.target_area_indices[area_idx] = np.where(labels == area_idx)[0]
                    if len(self.target_area_indices[area_idx]) == 0:
                        raise ValueError(
                            "The area_idx {} is not in the point cloud".format(area_idx)
                        )
                    print(
                        "\033[91m" + "area_idx:",
                        area_idx,
                        "number of points: ",
                        len(self.target_area_indices[area_idx]),
                        "\033[0m",
                    )

    def load_points_areas_for_feature_transfer(
        self, source_area_indices, target_area_indices
    ):
        print(
            "\033[91m"
            + "********** Loading points for the specified regions ***********"
            + "\033[0m"
        )
        if self.args.source_target_area_selection_method == "points_cloud_areas_boxes":
            self.load_points_areas_for_feature_transfer_from_boxes(
                source_area_indices, target_area_indices
            )
        elif self.args.source_target_area_selection_method == "points_cloud_areas_file":
            self.load_points_areas_for_feature_transfer_from_file(
                source_area_indices, target_area_indices
            )

    def load_pixels_areas_for_feature_transfer(
        self, source_area_indices, target_area_indices
    ):
        # print with red color
        print(
            "\033[91m"
            + "********** We will show you the frame you want to select areas for transfering features. Please select two areas, one is dark area, the other is bright area. **********"
            + "\033[0m"
        )
        print(
            "\033[91m"
            + "********** Loading frame id: **********"
            + str(self.args.frame_idx)
            + "\033[0m"
        )
        if source_area_indices is not None:
            for area_idx in source_area_indices:
                self.source_area_indices[area_idx] = self.load_selected_area_pixels_idx(
                    area=area_idx,
                    load_path=os.path.join(
                        self.all_scenes_config.save_dir,
                        self.all_scenes_config.index,
                        self.scene_config.index,
                    ),
                )
                print(
                    "\033[91m" + "area_idx:",
                    area_idx,
                    "number of points: ",
                    len(self.source_area_indices[area_idx]),
                    "\033[0m",
                )
        if target_area_indices is not None:
            for area_idx in target_area_indices:
                self.target_area_indices[area_idx] = self.load_selected_area_pixels_idx(
                    area=area_idx,
                    load_path=os.path.join(
                        self.all_scenes_config.save_dir,
                        self.all_scenes_config.index,
                        self.scene_config.index,
                    ),
                )
                print(
                    "\033[91m" + "area_idx:",
                    area_idx,
                    "number of points: ",
                    len(self.target_area_indices[area_idx]),
                    "\033[0m",
                )

    def load_model(self, resume):
        start_step = 0
        if resume > 0 or self.scene_config.load_path is not None:
            start_step = self.model.load(
                manager=self,
                checkpoint_dir=self.checkpoints_dir,
                specific_checkpoint=self.scene_config.load_path,
                stage=self.args.stage,
            )
            print("!!!!! Resume training from step %s" % start_step)
        else:
            if self.args.stage == "test":
                raise ValueError("We can not test without loading the model.")
        return start_step if start_step == 0 else start_step + 1

    def setup_test_steps(self):
        self.step = self.load_model(self.args.resume)
        self.model = self.model.to(self.device)
        self.eval_step_cnt = self.step

        if self.args.save_point_cloud:
            self.save_point_cloud()

        # loading the source and target areas for feature transfer
        if (
            self.args.test_action
            in ["transfer_albedo", "transfer_shading", "interpolate_albedo"]
            and self.args.use_points_features
        ) or self.args.test_action == "calculate_albedo_consistency":
            self.load_points_areas_for_feature_transfer(
                self.args.source_area_indices, self.args.target_area_indices
            )

        if (
            self.args.test_action
            in ["transfer_albedo", "transfer_shading", "interpolate_albedo"]
        ) and self.args.use_pixels_features:
            self.load_pixels_areas_for_feature_transfer(
                self.args.source_area_indices, self.args.target_area_indices
            )

        if self.args.source_target_area_selection_method == "freeform_pixels":
            self.source_area_indices = {
                "freeform": []
            }
            self.target_area_indices = {
                "freeform": []
            }

        self.original_pc_feats = self.model.pc_feats.clone()

        self.lpips_loss_fn_alex = lpips.LPIPS(net="alex", version="0.1")
        self.lpips_loss_fn_alex = self.lpips_loss_fn_alex.to(self.device)

        self.lpips_loss_fn_vgg = lpips.LPIPS(net="vgg", version="0.1")
        self.lpips_loss_fn_vgg = self.lpips_loss_fn_vgg.to(self.device)

        if self.scene_config.models.supervision_scaler.use:
            # save the scalers in npy file
            np.save(
                os.path.join(
                    self.exp_dir,
                    self.scene_config.index,
                    "albedo_supervision_scalers_{}.npy".format(self.scene_idx),
                ),
                self.model.supervision_scaler.detach().cpu().numpy(),
            )


def load_dataset_statistics(scene_config, dataset_path):
    eps = float(scene_config["models"]["predict_in_log_space_eps"])
    eps = "{:.0e}".format(eps)
    image_space = "rgb" if scene_config["models"]["predict_rgb_in_log_space"] else "raw"
    extra_prefix = ""
    if scene_config["dataset"]["convert_image_to_raw_space"]:
        if scene_config["dataset"]["force_convert_image_to_raw_space_white_bg"]:
            extra_prefix = "reconstructed_white_bg_"
        else:
            extra_prefix = "reconstructed_transparent_bg_"
        # print with red color we are using reconstructed images
        print("*" * 100)
        print(
            "\033[91m"
            "The reconstructed RAW image statistics are used for the dataset statistics."
            "\033[0m"
        )
        print("*" * 100)
    # Mip-NeRF 360 scenes keep their frames in a resolution subdirectory, so the
    # statistics sit next to it as <scene>/images_<factor>_meta rather than
    # <scene>_meta like the object-centric and Tanks & Temples layouts.
    if scene_config["dataset"].get("type") == "mip360":
        factor = int(scene_config["dataset"].get("factor", 1) or 1)
        images_dir = f"images_{factor}" if factor > 1 else "images"
        meta_dir = os.path.join(dataset_path, images_dir + "_meta")
    else:
        meta_dir = dataset_path + "_meta"

    stat_file = os.path.join(
        meta_dir,
        f"{extra_prefix}{image_space}_statistics_eps_{eps}{scene_config['dataset']['train_albedo_extraction_method']}.json",
    )

    if not os.path.exists(stat_file):
        raise FileNotFoundError(
            f"Dataset statistics not found at {stat_file}.\n"
            "Run the albedo/shading extraction step for this scene first; it writes "
            "raw_statistics_eps_<eps>.json into the scene's _meta directory."
        )

    # return the statistics
    with open(stat_file, "r") as f:
        stats = json.load(f)
    return stats

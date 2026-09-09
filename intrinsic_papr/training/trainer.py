"""The training loop: one step, the prune/grow schedule, and the loop that drives them."""

import bisect
import time
from datetime import datetime

import torch

from ..data.pipeline import run_image_pipeline
from ..models.activations import SoftplusActivation
from ..seeding import setup_seed
from .evaluation import eval_step
from .losses import calculate_training_loss
from .metrics import print_log_statistics


softplus_activation = SoftplusActivation()


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


def train_step(batch, scene_manager):
    img_idx, _, tgt, tgt_albedo, rayd, rayo, alpha_channel = batch
    if scene_manager.args.debug:
        # print the step and image index in red
        print("step: {}, image index: {}".format(scene_manager.step, img_idx))
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
        scene_manager.scene_config.models.use_albedo
        and len(scene_manager.scene_config.training.GT_albedo_preprocessing) != 0
    ):
        if check_applying_scaler_to_supervision(scene_manager):
            albedo_supervision_scaler = scene_manager.model.supervision_scaler[img_idx]
            if scene_manager.args.debug:
                print(
                    "DEBUG: we are using the supervision scaler for the GT albedo"
                )
        else:
            albedo_supervision_scaler = (
                scene_manager.model.supervision_scaler[img_idx].clone().detach()
            )
            if scene_manager.args.debug:
                print(
                    "DEBUG: we detached the supervision scaler from the graph"
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

        tgt_albedo = run_image_pipeline(
            img=tgt_albedo,
            pipeline=scene_manager.scene_config.training.GT_albedo_preprocessing,
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
            # The tensor reaching here is already normalised to [0, 1], so the
            # clamp guards that range: the learnable scaler can push the target
            # above 1. Clamping with the log-space bounds instead, as this used
            # to, collapsed the target towards a constant.
            clamp_min=0.0,
            clamp_max=1.0,
            alpha_channel=expanded_alpha_channel,
        )

    scene_manager.model.clear_grad()
    rgb_out, albedo_out, _ = scene_manager.model(rayo, rayd, scene_manager.step)
    rgb_out = scene_manager.model.last_act(rgb_out)
    if scene_manager.scene_config.models.use_albedo:
        albedo_out = scene_manager.model.last_act(albedo_out)

    losses = calculate_training_loss(
        scene_manager=scene_manager,
        render_pred_patch_pred_space=rgb_out,
        render_gt_patch_pred_space=tgt,
        albedo_pred_patch_pred_space=albedo_out,
        albedo_gt_patch_pred_space=tgt_albedo,
        clip=False,
    )
    scene_manager.model.scaler.scale(losses.total).backward()
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


    ################################## Adding Losses to the log dictionary ##################################
    scene_manager.avg_total_train_loss += losses.total.item()
    logged_losses = {
        ("render", "pred_space"): losses.render_pred_space,
        ("render", "original_space"): losses.render_original_space,
        ("albedo", "pred_space"): losses.albedo_pred_space,
        ("albedo", "original_space"): losses.albedo_original_space,
        ("albedo", "pred_space_cIMLE"): losses.albedo_pred_space_cIMLE,
        ("albedo", "original_space_cIMLE"): losses.albedo_original_space_cIMLE,
    }
    for (image_type, space), loss in logged_losses.items():
        if loss is None:
            continue
        if image_type == "albedo" and not scene_manager.scene_config.models.use_albedo:
            continue
        value = loss.item()
        getattr(scene_manager, f"{image_type}_losses")["train"][space].append(value)
        attr = f"avg_{image_type}_loss_{space}"
        setattr(scene_manager, attr, getattr(scene_manager, attr) + value)
        if scene_manager.args.debug:
            print(
                f"step: {scene_manager.step}, train_step loss for",
                image_type,
                space,
                ": " + str(value),
            )

    ################################## Adding Losses to the log dictionary ##################################

    return rgb_out, albedo_out


def train_and_eval(scene_manager):
    eval_metrics = None

    start_time = time.time()
    while True:
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
        rgb_pred, albedo_pred = train_step(batch, scene_manager=scene_manager)
        ###################################### eval step ######################################
        if (
            scene_manager.step % scene_manager.scene_config.eval.step == 0
            or scene_manager.step >= scene_manager.scene_config.training.steps
        ):
            eval_metrics = eval_step(
                scene_manager=scene_manager,
                batch=batch,
                train_rgb_out=rgb_pred,
                train_pred_albedo_patch=(
                    albedo_pred
                    if scene_manager.scene_config.models.use_albedo
                    else None
                ),
            )
                ###################################### Log statistics step ######################################
        if (
            scene_manager.step % scene_manager.all_configs.print_step == 0
            or scene_manager.step >= scene_manager.scene_config.training.steps
        ):
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


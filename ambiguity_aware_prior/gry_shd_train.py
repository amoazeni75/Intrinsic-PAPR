#  100 Images ~ 8GB RAM
#  16 Images  ~ 12GB VRAM, each step ~ 0.2 sec
# 800 images -> generating z codes ~ 6 min

import gc
import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from argparse import Namespace
from time import time

import torch
from altered_midas.midas_net import MidasNet
from chrislib.general import uninvert
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.custom_datasets import (
    CustomMultiDataset,
    HypersimColorfulDataset,
    MINoAugmentationDataset,
    multiple_dataset_sampler,
)
from models.midas_net_small import MidasNet_small
from models.tools import *

set_seed(42)

ignore_list = [
    "ai_001_002/images/scene_cam_01_final_hdf5/frame.0047",
    "ai_035_010/images/scene_cam_00_final_hdf5/frame.0007",
    "ai_016_001/images/scene_cam_01_final_hdf5/frame.0095",
    "ai_048_007/images/scene_cam_01_final_hdf5/frame.0097",
    "ai_051_002/images/scene_cam_04_final_hdf5/frame.0095",
    "ai_044_005/images/scene_cam_01_final_hdf5/frame.0055",
    "ai_048_009/images/scene_cam_00_final_hdf5/frame.0051",
    "ai_047_002/images/scene_cam_00_final_hdf5/frame.0034",
    "ai_029_001/images/scene_cam_01_final_hdf5/frame.0006",
    "ai_051_002/images/scene_cam_06_final_hdf5/frame.0022",
    "ai_016_001/images/scene_cam_01_final_hdf5/frame.0088",
    "ai_005_007/images/scene_cam_00_final_hdf5/frame.0075",
    "ai_051_005/images/scene_cam_02_final_hdf5/frame.0087",
    "ai_036_003/images/scene_cam_01_final_hdf5/frame.0029",
    "ai_055_002/images/scene_cam_01_final_hdf5/frame.0025",
    "ai_008_009/images/scene_cam_00_final_hdf5/frame.0062",
    "ai_031_009/images/scene_cam_00_final_hdf5/frame.0014",
    "ai_003_005/images/scene_cam_01_final_hdf5/frame.0023",
    "ai_048_003/images/scene_cam_01_final_hdf5/frame.0056",
    "ai_001_002/images/scene_cam_01_final_hdf5/frame.0027",
    "ai_019_007/images/scene_cam_00_final_hdf5/frame.0092",
    "ai_035_003/images/scene_cam_00_final_hdf5/frame.0006",
    "ai_052_010/images/scene_cam_00_final_hdf5/frame.0082",
    "ai_004_005/images/scene_cam_00_final_hdf5/frame.0095",
    "ai_055_004/images/scene_cam_00_final_hdf5/frame.0003",
    "ai_039_010/images/scene_cam_00_final_hdf5/frame.0085",
    "ai_039_010/images/scene_cam_00_final_hdf5/frame.0085",
]


# intialize the multi-scale gradient loss
msg_loss = MSGLoss(
    scales=4,
    taps=[2, 1, 1, 1],
    k_size=[3, 5, 7, 9],
)

# stage 1 is grayscale shaidng estimation (ordinal shading as input)
STAGE = 1


def get_time_str(seconds):
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def log_training_stats(log_dict, model, step, results_path):
    for i in range(4):
        meanshift = getattr(model, f"style_mod{i}_meanshift")
        varshift = getattr(model, f"style_mod{i}_varshift")

        log_dict[f"style_mod{i}_meanshift_min"] = meanshift.min().item()
        log_dict[f"style_mod{i}_meanshift_max"] = meanshift.max().item()
        log_dict[f"style_mod{i}_meanshift_mean"] = meanshift.mean().item()
        log_dict[f"style_mod{i}_meanshift_var"] = meanshift.var().item()

        log_dict[f"style_mod{i}_varshift_min"] = varshift.min().item()
        log_dict[f"style_mod{i}_varshift_max"] = varshift.max().item()
        log_dict[f"style_mod{i}_varshift_mean"] = varshift.mean().item()
        log_dict[f"style_mod{i}_varshift_var"] = varshift.var().item()
    log_dict["step"] = step
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({k: v for k, v in log_dict.items()
                            if isinstance(v, (int, float, str, bool)) or v is None}) + "\n")


def calculate_loss_values(pred_shd, gt_shd, gt_alb, in_img, msk, keepdim=False):
    # compute the implied albedo by dividing the shading from the image
    alb = in_img / uninvert(pred_shd).clip(1e-4)

    # compute l2 and multi-scale gradient loss for both albedo and shading
    shd_lp_loss = lp_loss(pred_shd, gt_shd, msk, keepdim=keepdim)
    shd_msg_loss = msg_loss(pred_shd, gt_shd, msk, keepdim=keepdim)

    alb_lp_loss = lp_loss(alb, gt_alb, msk, keepdim=keepdim)
    alb_msg_loss = msg_loss(alb, gt_alb, msk, keepdim=keepdim)

    # everything is combined with unit scale, but this can be adjusted
    shd_loss = shd_lp_loss + (shd_msg_loss * 1.0)
    alb_loss = alb_lp_loss + (alb_msg_loss * 1.0)

    loss = shd_loss + (1.0 * alb_loss)

    return loss, shd_loss, alb_loss


if __name__ == "__main__":

    # bunch of flags to make things faster
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    TRN_DATASET_PROBS = [0.5, 0.5]

    TRN_DATASET_PROBS = [x / sum(TRN_DATASET_PROBS) for x in TRN_DATASET_PROBS]

    args = get_args()

    DEBUG = args.debug

    # cIMLE flags
    D_LATENT = args.d_latent
    NUM_SAMPLE = args.IMLE_num_samples
    MAX_EPOCH = args.epochs

    BASE_LR = args.base_lr
    MLP_LR = args.mlp_lr
    MLP_LR2 = args.mlp_lr2
    PRETRAIN_MLP = True if args.pretrain_mlp else False
    PRETRAIN_EPOCHS = args.pretrain_steps
    USE_SCHEDULER = args.use_scheduler

    cfg = Namespace()

    exp_name = args.exp_name

    cfg.learning_rate = args.base_lr
    cfg.batchsize = args.batchsize
    cfg.num_workers = args.workers
    cfg.epochs = args.epochs
    cfg.clip = args.clip
    cfg.train_iters = args.train_iters
    cfg.checkpoint = args.checkpoint
    cfg.cache_probability = args.cache_prob
    cfg.cache_length = args.cache_length

    # save results path
    LOG_DIR = os.path.join(args.weights_path, exp_name)
    results_path = os.path.join(args.weights_path, exp_name, "results")
    checkpoint_path = os.path.join(args.weights_path, exp_name, "checkpoints")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # since each worker thread will have it's own cache (for each dataset)
    # I think I want to divide the cache length by num workers to not overload RAM
    cache_length = int(args.cache_length / (args.workers + 1))

    # training data --------------------
    train_datasets = [
        HypersimColorfulDataset(
            args.HYPERSIM_PATH,
            "splits/skimmed_train_list.p",
            stage=STAGE,
            clip=args.clip,
            cache_prb=0.0,
            cache_len=cache_length,
            ignore_list=ignore_list,
            is_subset=args.is_subset,
        ),
        MINoAugmentationDataset(
            args.MID_PATH,
            stage=STAGE,
            clip=args.clip,
            cache_prb=0.0,
            cache_len=cache_length,
            ignore_list=ignore_list,
        ),
    ]
    total_number_of_data = sum([len(dataset) for dataset in train_datasets])

    model = MidasNet_small(
        activation="sigmoid",
        exportable=False,
        input_channels=5,
        output_channels=1,
        use_cIMLE_pretrained=args.use_IMLE,
    )
    ord_model = MidasNet(
        activation="sigmoid",
        input_channels=3,
        output_channels=1,
    )

    model, ord_model = load_model(model, ord_model, args)

    model = model.to(args.device)
    model.train()
    ord_model = ord_model.cuda()
    ord_model.eval()
    msg_loss.to_device(args.device)


    ### For joint training
    optimizer = ModelOptimizer_AdaIn(model, BASE_LR, MLP_LR2, fixed_backbone=False)

    ### If we want to pretrain the MLP first
    if PRETRAIN_MLP:
        pretrain_optimizer = ModelOptimizer_AdaIn(
            model, BASE_LR, MLP_LR, fixed_backbone=True
        )
    ############################################################################################################

    if not DEBUG:
        if args.compile:
            model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    ### Minibatch to handle larger sample size
    mini_batch_size = args.IMLE_mini_batch_size
    num_sets = int(NUM_SAMPLE / mini_batch_size)
    true_num_samples = num_sets * mini_batch_size  # just take the floor

    start_training = time()
    time_is_up = False
    for epoch in range(args.epochs):
        if time_is_up:
            break

        ################### AdaIN init ###################
        if (
            epoch == 0 and args.num_data_for_AdaIN_init > 0
        ):  # we need to initailize the AdaIN
            start_init = time()
            # let's see if we have the values just load it
            if os.path.exists("./checkpoints/mean_var_adain.npy"):
                ### Set mean and variance
                mean0, var0, mean1, var1, mean2, var2, mean3, var3 = (
                    load_mean_var_adain("./checkpoints/mean_var_adain.npy", args.device)
                )
                model.set_mean_var_shifts(
                    mean0, var0, mean1, var1, mean2, var2, mean3, var3
                )

                print("AdaIn weights init done.")
                print("========================")
            else:
                model.eval()
                print("Initializing AdaIn layers")

                ### Make the mean=0 and variance=1
                ### Calculate the statistics of a subset of the data
                subset_sampler = multiple_dataset_sampler(
                    num_datasets=len(train_datasets),
                    total_number_of_data=args.num_data_for_AdaIN_init,
                    probs=TRN_DATASET_PROBS,
                )
                subset_dataset = CustomMultiDataset(
                    datasets=train_datasets,
                    datasets_indices=subset_sampler.get_dataset_indices(),
                    total_num_requested_data=args.num_data_for_AdaIN_init,
                    use_cashed_data=False,
                    save_cashed_data=False,
                )
                subset_dataloader = DataLoader(
                    subset_dataset,
                    batch_size=args.AdaIN_init_batch,
                    num_workers=0,
                    shuffle=False,
                    pin_memory=True,
                )
                ### Iterate through to get the dataset statistics
                z_np = np.empty(
                    (args.num_data_for_AdaIN_init, D_LATENT), dtype=np.float32
                )

                ### Hardcoded dimensions for resnext model  ---> Fix for other architectures?
                print("Initializing encoder.")
                all_ada0 = torch.zeros((args.num_data_for_AdaIN_init, 32))
                all_ada1 = torch.zeros((args.num_data_for_AdaIN_init, 48))
                all_ada2 = torch.zeros((args.num_data_for_AdaIN_init, 136))
                all_ada3 = torch.zeros((args.num_data_for_AdaIN_init, 384))

                with torch.no_grad():
                    ada_in_init_counter = 0
                    for _, (data, _) in enumerate(
                        tqdm(subset_dataloader, desc="AdaIn init")
                    ):
                        in_img = data["input"].to(args.device)
                        ord_base = data["ord_base"].to(args.device)
                        ord_full = data["ord_full"].to(args.device)
                        msk = data["mask"].to(args.device)

                        if msk.sum() == 0:
                            print("bad batch, skipping")
                            print(data["fname"])

                        # input is image, inverse base and full ordinal shading
                        inp = torch.cat(
                            [in_img, ord_base, ord_full], 1
                        )  # batch, 5, h, w

                        batch_size = inp.shape[0]
                        C = inp.shape[1]
                        H = inp.shape[2]
                        W = inp.shape[3]

                        num_images = inp.shape[0]
                        inp = inp.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                        inp = inp.view(-1, C, H, W)

                        ## Hard coded d_latent
                        z = torch.normal(
                            0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT)
                        )
                        z = z.view(-1, D_LATENT).cuda()

                        ### Get activations
                        adain0, adain1, adain2, adain3 = model.get_adain_init_act(
                            inp, z
                        )
                        adain0 = adain0.detach().cpu()
                        adain1 = adain1.detach().cpu()
                        adain2 = adain2.detach().cpu()
                        adain3 = adain3.detach().cpu()

                        ### Take the mean
                        adain0 = torch.mean(
                            adain0.view(adain0.shape[0], adain0.shape[1], -1), axis=-1
                        )
                        adain1 = torch.mean(
                            adain1.view(adain1.shape[0], adain1.shape[1], -1), axis=-1
                        )
                        adain2 = torch.mean(
                            adain2.view(adain2.shape[0], adain2.shape[1], -1), axis=-1
                        )
                        adain3 = torch.mean(
                            adain3.view(adain3.shape[0], adain3.shape[1], -1), axis=-1
                        )

                        adain0 = torch.mean(adain0, axis=0)
                        adain1 = torch.mean(adain1, axis=0)
                        adain2 = torch.mean(adain2, axis=0)
                        adain3 = torch.mean(adain3, axis=0)

                        all_ada0[ada_in_init_counter] = adain0
                        all_ada1[ada_in_init_counter] = adain1
                        all_ada2[ada_in_init_counter] = adain2
                        all_ada3[ada_in_init_counter] = adain3

                        ada_in_init_counter += 1
                        del (
                            inp,
                            z,
                            adain0,
                            adain1,
                            adain2,
                            adain3,
                            in_img,
                            ord_base,
                            ord_full,
                            msk,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

                    ### Calculate mean and variance
                    mean0 = torch.mean(all_ada0, axis=0)
                    mean1 = torch.mean(all_ada1, axis=0)
                    mean2 = torch.mean(all_ada2, axis=0)
                    mean3 = torch.mean(all_ada3, axis=0)

                    var0 = torch.var(all_ada0, axis=0)
                    var1 = torch.var(all_ada1, axis=0)
                    var2 = torch.var(all_ada2, axis=0)
                    var3 = torch.var(all_ada3, axis=0)

                    ### Save mean and variance to dictionary
                    mean0 = mean0.to("cpu").detach().numpy().squeeze()
                    mean1 = mean1.to("cpu").detach().numpy().squeeze()
                    mean2 = mean2.to("cpu").detach().numpy().squeeze()
                    mean3 = mean3.to("cpu").detach().numpy().squeeze()
                    var0 = var0.to("cpu").detach().numpy().squeeze()
                    var1 = var1.to("cpu").detach().numpy().squeeze()
                    var2 = var2.to("cpu").detach().numpy().squeeze()
                    var3 = var3.to("cpu").detach().numpy().squeeze()

                    output_dict = {
                        "mean0": mean0,
                        "mean1": mean1,
                        "mean2": mean2,
                        "mean3": mean3,
                        "var0": var0,
                        "var1": var1,
                        "var2": var2,
                        "var3": var3,
                    }

                    np.save(os.path.join(LOG_DIR, "mean_var_adain.npy"), output_dict)

                    #########################

                    ### Set mean and variance
                    mean0, var0, mean1, var1, mean2, var2, mean3, var3 = (
                        load_mean_var_adain(
                            os.path.join(LOG_DIR, "mean_var_adain.npy"), args.device
                        )
                    )
                    print("mean0", mean0)
                    print("var0", var0)
                    print("mean1", mean1)
                    print("var1", var1)
                    print("mean2", mean2)
                    print("var2", var2)
                    print("mean3", mean3)
                    print("var3", var3)

                    model.set_mean_var_shifts(
                        mean0, var0, mean1, var1, mean2, var2, mean3, var3
                    )

                    print("AdaIn weights init done.")
                    print("========================")

                    if args.only_output_adain_init:
                        exit()

                model.train()

            # print with red color
            print(
                "\033[91m"
                + "AdaIn weights init time: "
                + get_time_str(time() - start_init)
                + "\033[0m"
            )
        ################### AdaIN init ###################

        # we assume that we generate z codes at the beginning of each epoch; so, if you don't want to
        # refresh z codes frquently, you need to increase the number of train iters
        print("epoch: {}, refreshing z codes".format(epoch))
        start_z_generation = time()
        model.eval()

        ### Iterate over dataset
        number_of_z_codes = (
            args.train_iters * args.batchsize
        )  # the number of data we want to use for training
        # between each refresh of z codes
        selected_z_np = np.empty((number_of_z_codes, D_LATENT), dtype=np.float32)
        print("Total number of samples to find the nearest sample:", number_of_z_codes)

        subset_train_between_2_interval_sampler = multiple_dataset_sampler(
            num_datasets=len(train_datasets),
            total_number_of_data=number_of_z_codes,
            probs=TRN_DATASET_PROBS,
        )
        subset_train_between_2_interval_dataset = CustomMultiDataset(
            datasets=train_datasets,
            datasets_indices=subset_train_between_2_interval_sampler.get_dataset_indices(),
            total_num_requested_data=number_of_z_codes,
            use_cashed_data=False,
            save_cashed_data=True,
        )
        subset_train_between_2_interval_dataloader = DataLoader(
            subset_train_between_2_interval_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

        with torch.no_grad():
            for i, (data, _) in enumerate(
                tqdm(
                    subset_train_between_2_interval_dataloader,
                    desc="Generating Z codes",
                )
            ):

                ### Batch size
                in_img = data["input"].to(args.device, non_blocking=True)
                ord_base = data["ord_base"].to(args.device, non_blocking=True)
                ord_full = data["ord_full"].to(args.device, non_blocking=True)
                gt_shd = data["gt_shd"].to(args.device, non_blocking=True)
                gt_alb = data["gt_alb"].to(args.device, non_blocking=True)
                msk = data["mask"].to(args.device, non_blocking=True)
                inp = torch.cat([in_img, ord_base, ord_full], 1)  # batch, 5, h, w
                if msk.sum() == 0:
                    print("bad batch, skipping")
                    print(data["fname"])
                    continue
                batch_size = inp.shape[0]
                C = inp.shape[1]
                H = inp.shape[2]
                W = inp.shape[3]

                ### Loss values
                all_losses = torch.zeros((batch_size, true_num_samples)).cuda()
                all_z = torch.zeros((batch_size, true_num_samples, D_LATENT)).cuda()

                ### Repeat for the number of samples
                num_images = inp.shape[0]
                inp_img = in_img.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                inp_img = inp_img.view(-1, 3, H, W)

                ord_base = ord_base.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                ord_base = ord_base.view(-1, 1, H, W)

                ord_full = ord_full.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                ord_full = ord_full.view(-1, 1, H, W)

                gt_shd = gt_shd.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                gt_shd = gt_shd.view(-1, 1, H, W)

                gt_alb = gt_alb.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                gt_alb = gt_alb.view(-1, 3, H, W)

                msk = msk.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                msk = msk.view(-1, 1, H, W)

                inp = inp.unsqueeze(1).repeat(1, mini_batch_size, 1, 1, 1)
                inp = inp.view(-1, C, H, W)

                ### Iterate over the minibatch
                for k in range(num_sets):

                    z = torch.normal(
                        0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT)
                    )
                    z = z.view(-1, D_LATENT).cuda()

                    pred_shd = model(inp, z)
                    loss, shd_loss, alb_loss = calculate_loss_values(
                        pred_shd, gt_shd, gt_alb, in_img, msk, keepdim=True
                    )
                    total_raw = loss.view(batch_size, mini_batch_size)
                    z = z.view(batch_size, mini_batch_size, D_LATENT)

                    for s in range(mini_batch_size):
                        all_losses[:, k * mini_batch_size + s] = total_raw[:, s]
                        all_z[:, k * mini_batch_size + s, :] = z[:, s, :]

                all_z = all_z.view(batch_size, true_num_samples, D_LATENT)

                idx_to_take = torch.argmin(all_losses, axis=-1)

                for j in range(batch_size):
                    selected_z_np[i * batch_size + j, :] = (
                        all_z[j][idx_to_take[j]].cpu().data.numpy()
                    )

                torch.cuda.empty_cache()
                gc.collect()

        print(
            "\033[91m"
            + f"epoch: {epoch}, Z code generation time: "
            + get_time_str(time() - start_z_generation)
            + "\033[0m"
        )
        model.train()
        subset_train_between_2_interval_dataset.set_z_codes(selected_z_np)
        subset_train_between_2_interval_dataset.set_use_cashed_data()
        subset_train_between_2_interval_dataloader = DataLoader(
            subset_train_between_2_interval_dataset,
            batch_size=args.batchsize,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

        losses = {}
        losses["shd_loss"] = 0.0
        losses["alb_loss"] = 0.0
        losses["all_loss"] = 0.0

        ### Training loop for the current epoch
        print("Start training at epoch: ", epoch)
        start_train_steps = time()
        i_step = 0
        for _, (data, cur_batch_z) in enumerate(
            tqdm(
                subset_train_between_2_interval_dataloader,
                desc=f"Training data, step: {i_step + 1}",
            )
        ):
            model_time_start = time()

            in_img = data["input"].to(args.device, non_blocking=True)
            ord_base = data["ord_base"].to(args.device, non_blocking=True)
            ord_full = data["ord_full"].to(args.device, non_blocking=True)
            gt_shd = data["gt_shd"].to(args.device, non_blocking=True)
            gt_alb = data["gt_alb"].to(args.device, non_blocking=True)
            msk = data["mask"].to(args.device, non_blocking=True)
            inp = torch.cat([in_img, ord_base, ord_full], 1)  # batch, 5, h, w
            cur_batch_z = cur_batch_z.to(args.device, non_blocking=True)
            if msk.sum() == 0:
                print("bad batch, skipping")
                print(data["fname"])
            pred_shd = model(inp, cur_batch_z)
            loss, shd_loss, alb_loss = calculate_loss_values(
                pred_shd, gt_shd, gt_alb, in_img, msk
            )

            if PRETRAIN_MLP and (epoch * args.train_iters + i_step) < PRETRAIN_EPOCHS:
                pretrain_optimizer.optim(loss)
            else:
                optimizer.optim(loss)

            model_time = time() - model_time_start

            losses["alb_loss"] += alb_loss.item()
            losses["shd_loss"] += shd_loss.item()
            losses["all_loss"] += loss.item()

            if (epoch * args.train_iters + i_step) % args.i_print_step == 0:
                time_of_steps = get_time_str(time() - start_train_steps)
                start_train_steps = time()
                time_of_training = get_time_str(time() - start_training)
                print(
                    f"epoch: {epoch}, step: {i_step}, alb_loss: {losses['alb_loss'] / (i_step + 1)}, shd_loss: {losses['shd_loss'] / (i_step + 1)}, all_loss: {losses['all_loss'] / (i_step + 1)}, time_of_steps: {time_of_steps}, time_of_training: {time_of_training}"
                )
                metrics_log = {
                    "alb_loss": losses["alb_loss"] / (i_step + 1),
                    "shd_loss": losses["shd_loss"] / (i_step + 1),
                    "all_loss": losses["all_loss"] / (i_step + 1),
                    "time_of_steps": time_of_steps,
                    "time_of_training": time_of_training,
                    "step": epoch * args.train_iters + i_step,
                }
                qual_out = qual_eval(
                    ord_model, model, args, results_path, epoch, i_step
                )
                metrics_log.update(qual_out)
                log_training_stats(
                    metrics_log,
                    model,
                    (epoch * args.train_iters + i_step),
                    results_path,
                )

            if (epoch * args.train_iters + i_step) % args.i_eval_step == 0:

                best_weights_path = os.path.join(checkpoint_path, "best.pt")
                epoch_weights_path = os.path.join(
                    checkpoint_path, f"weights_e_{epoch}_s_{i_step}.pt"
                )
                torch.save(model.state_dict(), best_weights_path)
                torch.save(model.state_dict(), epoch_weights_path)

            i_step += 1

        torch.cuda.empty_cache()
        gc.collect()

    print("Total training time: ", get_time_str(time() - start_training))
    best_weights_path = os.path.join(checkpoint_path, "best.pt")
    torch.save(model.state_dict(), best_weights_path)

    print("Training done.")

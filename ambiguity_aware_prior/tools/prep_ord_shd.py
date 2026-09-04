import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import gc
import random
from pathlib import Path
from time import time

import numpy as np
import torch
from altered_midas.midas_net import MidasNet
from chrislib.general import get_brightness, invert, np_to_pil, round_32, to2np, uninvert
from chrislib.resolution_util import optimal_resize
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import HypersimColorfulDataset

STAGE = 0
# Top-level Hypersim directory. Override with the HYPERSIM_PATH environment variable.
HYPERSIM_PATH = os.environ.get("HYPERSIM_PATH", "./data/Hypersim")


def base_resize(img):
    h, w, _ = img.shape

    max_dim = max(h, w)
    scale = 384 / max_dim

    new_h, new_w = scale * h, scale * w
    new_h, new_w = round_32(new_h), round_32(new_w)

    net_input = resize(img, (new_h, new_w, 3), anti_aliasing=True)
    return net_input


def equalize_predictions(img, base, full, msk, p=0.5):

    if len(msk.shape) == 3:
        msk = msk[:, :, 0]  # just take the first channel

    h, w, _ = img.shape

    full_shd = uninvert(full)
    base_shd = uninvert(base)

    full_alb = get_brightness(img) / full_shd.clip(1e-5)
    base_alb = get_brightness(img) / base_shd.clip(1e-5)

    rand_msk = (np.random.randn(h, w) > p).astype(np.uint8) * msk

    flat_full_alb = full_alb[rand_msk == 1]
    flat_base_alb = base_alb[rand_msk == 1]

    scale, _, _, _ = np.linalg.lstsq(
        flat_full_alb.reshape(-1, 1), flat_base_alb, rcond=None
    )

    new_full_alb = scale * full_alb
    new_full_shd = get_brightness(img) / new_full_alb.clip(1e-5)
    new_full = invert(new_full_shd)

    return base, new_full



parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers", type=int, default=4, help="number of worker processes for loading data"
)
parser.add_argument(
    "--batch_sz", type=int, default=1, help="batch size for dataloading"
)
parser.add_argument(
    "--bit_depth",
    type=int,
    default=8,
    choices=[8, 16],
    help="bit-depth to save shading images",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="whether or not existing shading should be overwritten",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "hypersim",
        "fsvg",
        "structure3d",
        "interiorverse",
        "eden",
        "prid",
        "matrix_city",
        "lumos",
    ],
    help="name of the dataset to preprocess",
)
parser.add_argument("--weights_path", type=str, help="path to ordinal network weights")
args = parser.parse_args()

ignore_list = ["ai_001_002/images/scene_cam_01_final_hdf5/frame.0047"]

if args.dataset == "hypersim":
    dataset = HypersimColorfulDataset(
        HYPERSIM_PATH,
        "splits/skimmed_train_list.p",
        stage=STAGE,
        clip=15,
        use_pred_shd=False,
        random=False,
        augment=False,
        ignore_list=ignore_list,
    )

loader = DataLoader(
    dataset, batch_size=args.batch_sz, num_workers=args.workers, shuffle=True
)

ord_model = MidasNet(
    activation="sigmoid",
    input_channels=3,
    output_channels=1,
)

ord_model.load_state_dict(torch.load(args.weights_path, map_location="cuda"))

ord_model = ord_model.cuda()
ord_model.eval()


failed_ones = []
print("batches:", len(loader))
for batch in tqdm(loader):

    inp_batch = batch["input"]
    if inp_batch.shape == torch.Size([1]) and inp_batch == -1:
        print(f"skipping {batch['fname']}, already exists")
        continue
    gt_shd_batch = batch["gt_shd"]
    msk_batch = batch["mask"]
    fname_batch = batch["fname"]

    for inp, gt_shd, msk, fname in zip(inp_batch, gt_shd_batch, msk_batch, fname_batch):

        shd_fname = dataset.get_ord_fname(fname)

        if (
            os.path.exists(shd_fname) and not args.overwrite
        ) or shd_fname in ignore_list:
            print(f"skipping {shd_fname}, already exists")
            continue
        try:
            shd_dir = Path(shd_fname).parent
            if not os.path.exists(shd_dir):
                os.makedirs(shd_dir)

            start = time()

            inp = to2np(inp)
            gt_shd = to2np(gt_shd)
            msk = to2np(msk)

            o_h, o_w, _ = inp.shape

            # scale input to base size and R0 size
            base_inp = base_resize(inp)
            orig_inp = resize(inp, (round_32(o_h), round_32(o_w)))
            full_inp = optimal_resize(inp, 0.0)

            base_inp = (
                torch.from_numpy(base_inp).permute(2, 0, 1).unsqueeze(0).cuda().float()
            )
            orig_inp = (
                torch.from_numpy(orig_inp).permute(2, 0, 1).unsqueeze(0).cuda().float()
            )
            full_inp = (
                torch.from_numpy(full_inp).permute(2, 0, 1).unsqueeze(0).cuda().float()
            )

            with torch.no_grad():
                base_est = to2np(ord_model(base_inp).squeeze(0))
                orig_est = to2np(ord_model(orig_inp).squeeze(0))
                full_est = to2np(ord_model(full_inp).squeeze(0))

            # resize estimations to the original image size
            base_est = resize(base_est, (o_h, o_w))
            orig_est = resize(orig_est, (o_h, o_w))
            full_est = resize(full_est, (o_h, o_w))

            # we want to scale the full and orig size estimations to match the base estimation scale
            _, orig_est = equalize_predictions(inp, base_est, orig_est, msk)
            _, full_est = equalize_predictions(inp, base_est, full_est, msk)

            all_shd = np.concatenate([base_est, orig_est, full_est], -1)

            np_to_pil(all_shd, bits=args.bit_depth).save(shd_fname)
        except Exception as e:
            print(f"failed to process {fname}: {e}")
            failed_ones.append(fname)

        finally:
            # Move tensors to CPU and delete them
            del base_inp, orig_inp, full_inp, base_est, orig_est, full_est
            torch.cuda.empty_cache()
            gc.collect()

    # free up some memory
    del batch
    torch.cuda.empty_cache()
    gc.collect()


print(f"failed ones: {failed_ones}")

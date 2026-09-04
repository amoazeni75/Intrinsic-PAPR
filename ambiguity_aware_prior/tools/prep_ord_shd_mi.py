import os
from random import shuffle

# this is set so that opencv can load exr files
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
from glob import glob

import cv2
import numpy as np
import torch
from altered_midas.midas_net import MidasNet
from chrislib.data_util import np_to_pil
from chrislib.general import get_brightness, get_tonemap_scale, invert, round_32, to2np, uninvert
from chrislib.resolution_util import optimal_resize
from skimage.transform import resize
from tqdm import tqdm


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


def process_scene(root_dir, scene_name, model):

    images = []
    albedos = []
    shadings = []

    if len(glob(f"{root_dir}/{scene_name}/*_ord_shd.png")) > 0:
        print(f"skipping {scene_name} - already processed")
        return

    for img_idx in range(0, 25):
        print(f" \t processing image ({img_idx + 1} / 25)")
        img = cv2.imread(
            f"{root_dir}/{scene_name}/dir_{img_idx}_mip2.exr",
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )[:, :, ::-1]
        prb = cv2.imread(
            f"{root_dir}/{scene_name}/probes/dir_{img_idx}_gray256.exr",
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )[:, :, ::-1]

        wb_img = img

        tm_scale = get_tonemap_scale(wb_img)
        tm_img = (tm_scale * wb_img).clip(0, 1)

        images.append(tm_img)


        o_h, o_w, _ = tm_img.shape

        # scale input to base size and R0 size
        base_inp = base_resize(tm_img)
        orig_inp = resize(tm_img, (round_32(o_h), round_32(o_w)))
        full_inp = optimal_resize(tm_img, 0.0)

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
            base_est = to2np(model(base_inp).squeeze(0))
            orig_est = to2np(model(orig_inp).squeeze(0))
            full_est = to2np(model(full_inp).squeeze(0))

        # resize estimations to the original image size
        base_est = resize(base_est, (o_h, o_w))
        orig_est = resize(orig_est, (o_h, o_w))
        full_est = resize(full_est, (o_h, o_w))

        msk = (tm_img > 0.001).astype(np.uint8)

        # we want to scale the full and orig size estimations to match the base estimation scale
        _, orig_est = equalize_predictions(tm_img, base_est, orig_est, msk)
        _, full_est = equalize_predictions(tm_img, base_est, full_est, msk)

        all_shd = np.concatenate([base_est, orig_est, full_est], -1)

        shadings.append(all_shd)

    for idx, shd in enumerate(shadings):
        np_to_pil(shd).save(f"{root_dir}/{scene_name}/{idx}_ord_shd.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mid_path", type=str, help="path to the MIDIntrinsic (train or test)"
    )
    parser.add_argument(
        "--weights_path", type=str, help="path to ordinal network weights to use"
    )
    args = parser.parse_args()

    ord_model = MidasNet(
        activation="sigmoid",
        input_channels=3,
        output_channels=1,
    )

    ord_model.load_state_dict(torch.load(args.weights_path, map_location="cuda"))

    ord_model = ord_model.cuda()
    ord_model.eval()

    scenes = [
        x
        for x in os.listdir(args.mid_path)
        if os.path.isdir(os.path.join(args.mid_path, x))
    ]
    num_scenes = len(scenes)
    print(f"found {num_scenes} scenes")

    shuffle(scenes)

    for i, scene_name in enumerate(tqdm(scenes)):
        print(f"({i + 1} / {num_scenes}) - processing {scene_name}")
        process_scene(args.mid_path, scene_name, ord_model)

import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import *

import argparse

parser = argparse.ArgumentParser(
    description="Prepare Tanks & Temples scenes (NSVF layout) for albedo extraction"
)
parser.add_argument(
    "--root", type=str, required=True, help="Directory holding the T&T scenes"
)
parser.add_argument(
    "--scenes",
    type=str,
    nargs="+",
    default=["Barn", "Caterpillar", "Family", "Ignatius", "Truck"],
)
parser.add_argument("--resize_w", type=int, default=1088)
parser.add_argument("--resize_h", type=int, default=640)
_args = parser.parse_args()

resize_w = _args.resize_w
resize_h = _args.resize_h
root = _args.root
datasets = _args.scenes

# For each dataset, we copy of it for two instances:
# 1. with name pretrained_albedo_transparent_bg_{dataset}
# 2. with name pretrained_albedo_white_bg_{dataset}
for dataset in datasets:
    # Copy the dataset to the new location with the new name
    # remove the old directories if they exist
    if os.path.exists(
        os.path.join(root, f"pretrained_albedo_transparent_bg_{dataset}")
    ):
        shutil.rmtree(os.path.join(root, f"pretrained_albedo_transparent_bg_{dataset}"))
    if os.path.exists(os.path.join(root, f"pretrained_albedo_white_bg_{dataset}")):
        shutil.rmtree(os.path.join(root, f"pretrained_albedo_white_bg_{dataset}"))
    shutil.copytree(
        os.path.join(root, dataset),
        os.path.join(root, f"pretrained_albedo_transparent_bg_{dataset}"),
    )
    shutil.copytree(
        os.path.join(root, dataset),
        os.path.join(root, f"pretrained_albedo_white_bg_{dataset}"),
    )
    print(f"Copying dataset {dataset} done")

# For each dataset, we will read images inside both previous directories and resize them
for dataset in datasets:
    print(f"Resizing images for dataset {dataset}")
    # Read images from the first directory
    img_dir = os.path.join(root, f"pretrained_albedo_transparent_bg_{dataset}", "rgb")
    img_paths = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    ]
    for img_path in img_paths:
        # open the image with PIL and resize it and save it
        img = Image.open(img_path)
        img = img.resize((resize_w, resize_h))
        img.save(img_path)

    # Read images from the second directory
    img_dir = os.path.join(root, f"pretrained_albedo_white_bg_{dataset}", "rgb")
    img_paths = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    ]
    for img_path in img_paths:
        # open the image with PIL and resize it and save it
        img = Image.open(img_path)
        img = img.resize((resize_w, resize_h))
        img.save(img_path)
    print(f"Resizing images for dataset {dataset} done")


# for the transparent background, we need to make it transparent
datasets = ["Barn", "Caterpillar", "Family", "Truck"]
for dataset in datasets:
    print(f"Making background transparent for dataset {dataset}")
    img_dir = os.path.join(root, f"pretrained_albedo_transparent_bg_{dataset}", "rgb")
    img_paths = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    ]
    for img_path in img_paths:
        img, _, _, _ = load_resize_normal_image(
            image_path=img_path,
            scene_config=None,
            convert_image_to_raw_space=False,
            force_convert_image_to_raw_space_white_bg=False,
            img_type="render",
            pre_post_processing_steps=None,
        )
        img, alpha = make_img_bg_transparent(img, white_values=1.0)
        img = np.dstack((img, alpha))
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(img_path)


# for the white background, we need to make it white
datasets = ["Ignatius"]
for dataset in datasets:
    print(f"Making background white for dataset {dataset}")
    img_dir = os.path.join(root, f"pretrained_albedo_white_bg_{dataset}", "rgb")
    img_paths = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    ]
    for img_path in img_paths:
        img, _, _, _ = load_resize_normal_image(
            image_path=img_path,
            scene_config=None,
            convert_image_to_raw_space=False,
            force_convert_image_to_raw_space_white_bg=False,
            img_type="render",
            pre_post_processing_steps=["white_bg"],
        )
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        img = np.dstack((img, alpha))
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(img_path)

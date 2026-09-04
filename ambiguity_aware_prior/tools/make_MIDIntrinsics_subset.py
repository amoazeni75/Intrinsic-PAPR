import math
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm


def get_file_counts_ignore_probes(root_dir):
    file_counts = defaultdict(int)

    for scene in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene)
        if os.path.isdir(scene_path):
            for item in os.listdir(scene_path):
                item_path = os.path.join(scene_path, item)
                # Skip 'probes' directories
                if os.path.isdir(item_path) and item == "probes":
                    continue
                # Count files that match the required patterns
                if item.endswith("_ord_shd.png"):
                    base_name = item.split("_ord_shd.png")[0]
                    if os.path.exists(
                        os.path.join(scene_path, f"dir_{base_name}_mip2.exr")
                    ):
                        file_counts[scene] += 1

    return file_counts


def select_files_ignore_probes(file_counts, total_images_needed):
    selected_files = defaultdict(list)

    total_files_available = sum(count for count in file_counts.values())

    for scene, count in file_counts.items():
        scene_ratio = count / total_files_available
        num_images_to_select = math.floor(scene_ratio * total_images_needed)
        selected_files[scene] = num_images_to_select

    return selected_files


def copy_subset_ignore_probes(root_dir, dest_dir, selected_files):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    total_files_to_copy = sum(
        [num_images * 2 for num_images in selected_files.values()]
    )
    progress_bar = tqdm(total=total_files_to_copy, desc="Copying files", unit="file")

    for scene, num_images in selected_files.items():
        scene_path = os.path.join(root_dir, scene)
        files = sorted(
            [f for f in os.listdir(scene_path) if f.endswith("_ord_shd.png")]
        )
        selected_frames = random.sample(files, num_images)
        os.makedirs(scene_path, exist_ok=True)
        os.makedirs(os.path.join(dest_dir, scene), exist_ok=True)
        shutil.copy(
            os.path.join(scene_path, "albedo.exr"), os.path.join(dest_dir, scene)
        )

        for frame in selected_frames:
            f_id = frame.split("_ord_shd.png")[0]
            for ext in ["{}_ord_shd.png", "dir_{}_mip2.exr"]:
                src_file = os.path.join(scene_path, ext.format(f_id))
                dest_scene_path = os.path.join(dest_dir, scene)
                shutil.copy(src_file, dest_scene_path)
                progress_bar.update(1)

    progress_bar.close()


def create_fair_subset_ignore_probes(root_dir, dest_dir, total_images_needed):
    file_counts = get_file_counts_ignore_probes(root_dir)
    selected_files = select_files_ignore_probes(file_counts, total_images_needed)
    copy_subset_ignore_probes(root_dir, dest_dir, selected_files)


# Example usage:
import argparse

parser = argparse.ArgumentParser(
    description="Build a subset of the MIDIntrinsics dataset"
)
parser.add_argument(
    "--root_dir", type=str, required=True, help="MIDIntrinsics root directory"
)
parser.add_argument(
    "--dest_dir", type=str, required=True, help="Where to write the subset"
)
args = parser.parse_args()

root_dir = args.root_dir
dest_dir = args.dest_dir
total_images_needed = 5000  # Adjust this number as needed

create_fair_subset_ignore_probes(root_dir, dest_dir, total_images_needed)


os.system(f"tar -cvf {dest_dir}.tar {dest_dir}")

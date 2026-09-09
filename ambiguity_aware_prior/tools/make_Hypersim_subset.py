import math
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm


def get_file_counts(root_dir):
    file_counts = defaultdict(lambda: defaultdict(int))

    for scene in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene, "images")
        if os.path.isdir(scene_path):
            for camera in os.listdir(scene_path):
                camera_path = os.path.join(scene_path, camera)
                if os.path.isdir(camera_path):
                    for f in os.listdir(camera_path):
                        if f.endswith(".color.hdf5"):
                            base_name = f.split(".color.hdf5")[0]
                            if os.path.exists(
                                os.path.join(
                                    camera_path, f"{base_name}.diffuse_reflectance.hdf5"
                                )
                            ) and os.path.exists(
                                os.path.join(camera_path, f"{base_name}.ord_shd.png")
                            ):
                                file_counts[scene][camera] += 1
    return file_counts


def select_files(file_counts, total_images_needed):
    selected_files = defaultdict(lambda: defaultdict(list))

    # Calculate total available images (each image has 3 files)
    total_files_available = sum(
        sum(count for count in cameras.values()) for cameras in file_counts.values()
    )

    # Ensure fairness by determining how many images to pick per camera
    for scene, cameras in file_counts.items():
        for camera, count in cameras.items():
            camera_ratio = count / total_files_available
            num_images_to_select = math.floor(camera_ratio * total_images_needed)
            selected_files[scene][camera] = num_images_to_select

    return selected_files


def copy_subset(root_dir, dest_dir, selected_files):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Calculate total number of files to be copied
    total_files = (
        sum(
            num_images
            for cameras in selected_files.values()
            for num_images in cameras.values()
        )
        * 3
    )

    # Initialize progress bar
    with tqdm(total=total_files, desc="Copying files") as pbar:
        for scene, cameras in selected_files.items():
            for camera, num_images in cameras.items():
                scene_path = os.path.join(root_dir, scene, "images")
                camera_path = os.path.join(scene_path, camera)
                files = sorted(
                    [f for f in os.listdir(camera_path) if f.endswith(".color.hdf5")]
                )
                selected_frames = random.sample(files, num_images)

                for frame in selected_frames:
                    base_name = frame.split(".color.hdf5")[0]
                    for ext in [
                        "color.hdf5",
                        "diffuse_reflectance.hdf5",
                        "ord_shd.png",
                    ]:
                        src_file = os.path.join(camera_path, f"{base_name}.{ext}")
                        dest_scene_path = os.path.join(dest_dir, scene, "images")
                        dest_camera_path = os.path.join(dest_scene_path, camera)
                        if not os.path.exists(dest_camera_path):
                            os.makedirs(dest_camera_path)
                        shutil.copy(src_file, dest_camera_path)
                        pbar.update(1)


def create_fair_subset(root_dir, dest_dir, total_images_needed):
    file_counts = get_file_counts(root_dir)
    selected_files = select_files(file_counts, total_images_needed)
    copy_subset(root_dir, dest_dir, selected_files)


# Example usage:
import argparse

parser = argparse.ArgumentParser(description="Build a subset of the Hypersim dataset")
parser.add_argument(
    "--root_dir", type=str, required=True, help="Hypersim root directory"
)
parser.add_argument(
    "--dest_dir", type=str, required=True, help="Where to write the subset"
)
args = parser.parse_args()

root_dir = args.root_dir
dest_dir = args.dest_dir
total_images_needed = 5000

create_fair_subset(root_dir, dest_dir, total_images_needed)

# tar the destination directory without compression
os.system(f"tar -cvf {dest_dir}.tar {dest_dir}")

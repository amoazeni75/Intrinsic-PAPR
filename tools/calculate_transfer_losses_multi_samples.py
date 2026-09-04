import os
import sys

# Resolve sibling tools regardless of the working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

import cv2
import numpy as np
import PIL
from intrinsic_decomposition_using_prior_model import *
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Calculate transfer losses")
    parser.add_argument(
        "--transferred_images_path",
        type=str,
        default=None,
        help="Path to the transferred image",
    )
    parser.add_argument(
        "--original_render_image_path",
        type=str,
        default=None,
        help="Path to the original render image",
    )
    parser.add_argument(
        "--mask_img_path",
        type=str,
        default=None,
        help="Path to the mask image",
    )
    parser.add_argument(
        "--prior_model",
        choices=["yagiz_v1", "yagiz_v2", "diffusion", "GT", "cIMLE_yagiz_v1"],
        default="yagiz_v1",
    )
    parser.add_argument(
        "--transfer_type",
        type=str,
        default=None,
        help="Type of transfer to evaluate",
    )
    parser.add_argument(
        "--ref_rgb",
        type=str,
        default=None,
        help="Type of transfer to evaluate",
    )
    parser.add_argument(
        "--filter_keywords",
        type=str,
        default=None,
        help="Filter the transferred images",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default=None,
        help="Name of the report",
    )
    args = parser.parse_args()
    return args


def get_source_target_pixel_coordinates(mask_img_path):
    mask_img = Image.open(mask_img_path)
    mask_img = np.array(mask_img)
    # we need to double check if the mask is 3D, we select the first channel
    if len(mask_img.shape) == 3:
        mask_img = mask_img[:, :, 0]
    target_pixels_coordinate = np.argwhere(mask_img == 255)
    source_pixels_coordinate = np.argwhere(mask_img == 125)

    # print with red color
    print(
        "\033[91m" + "number of source pixels: ",
        len(source_pixels_coordinate),
        "\033[0m",
    )
    print(
        "\033[91m" + "number of target pixels: ",
        len(target_pixels_coordinate),
        "\033[0m",
    )
    return source_pixels_coordinate, target_pixels_coordinate


def mse_distance(a, b):
    mse_dist = np.mean((a - b) ** 2)
    return mse_dist


def calculate_transfer_accuracy(
    original_img,
    transferred_img,
    source_area_pixel_coordinates,
    target_area_pixel_coordinates,
    source_rgb_color=None,
):
    # This part is independent of the decomposition method and transfer type
    # The goal is that the color of source in the original image should be close to the color of the target in the transferred image
    # We can use the L2 distance between the source and target colors as a metric

    if source_rgb_color is not None:
        source_pixel_color = np.array(source_rgb_color).reshape(1, -1) / 255.0
    else:
        if source_area_pixel_coordinates.shape[0] == 1:
            source_pixel_color = original_img[
                source_area_pixel_coordinates[0][0], source_area_pixel_coordinates[0][1]
            ].reshape(1, -1)
        elif id(source_area_pixel_coordinates) == id(target_area_pixel_coordinates):
            source_pixel_color = original_img[
                source_area_pixel_coordinates[:, 0], source_area_pixel_coordinates[:, 1]
            ]
        else:
            source_pixel_color = np.mean(
                original_img[
                    source_area_pixel_coordinates[:, 0],
                    source_area_pixel_coordinates[:, 1],
                ],
                axis=0,
            )

    # find the l2 distance between the source color and the target pixels' colors
    target_pixel_colors = transferred_img[
        target_area_pixel_coordinates[:, 0], target_area_pixel_coordinates[:, 1]
    ]

    mse = mse_distance(target_pixel_colors, source_pixel_color)
    return mse


def calculate_transfer_losses(
    transferred_img,
    original_render_img,
    source_area_pixel_coordinates,
    target_area_pixel_coordinates,
    args,
    name,
):
    """
    transferred_img: np.array, shape (H, W, 3), dtype float32, range [0, 1]
    original_render_img: np.array, shape (H, W, 3), dtype float32, range [0, 1]
    """
    # step 1: we decompose the transferred_img and original_render_img into shading and albedo components

    (
        _,
        _,
        rgb_shading_transferred,
        rgb_albedo_transferred,
        _,
    ) = get_shading_albedo(img=transferred_img, prior_model=args.prior_model, args=args)
    rgb_shading_transferred = rgb_shading_transferred[0]
    rgb_albedo_transferred = rgb_albedo_transferred[0]

    (
        _,
        _,
        rgb_shading_original,
        rgb_albedo_original,
        _,
    ) = get_shading_albedo(
        img=original_render_img, prior_model=args.prior_model, args=args
    )
    rgb_shading_original = rgb_shading_original[0]
    rgb_albedo_original = rgb_albedo_original[0]

    # step 2: surface details preservation error
    surface_details_preser_error = calculate_surface_details_preservation_error(
        transferred_img,
        original_render_img,
    )

    # step 3: complementary transfer accuracy: we get the MSE
    # loss between the target area before and and after the transfer
    if args.transfer_type == "albedo":
        rgb_space_complementary_transfer_accuracy_loss = calculate_transfer_accuracy(
            transferred_img=rgb_shading_transferred,
            original_img=rgb_shading_original,
            source_area_pixel_coordinates=target_area_pixel_coordinates,
            target_area_pixel_coordinates=target_area_pixel_coordinates,
            source_rgb_color=None,
        )
    elif args.transfer_type == "shading":
        rgb_space_complementary_transfer_accuracy_loss = calculate_transfer_accuracy(
            transferred_img=rgb_albedo_transferred,
            original_img=rgb_albedo_original,
            source_area_pixel_coordinates=target_area_pixel_coordinates,
            target_area_pixel_coordinates=target_area_pixel_coordinates,
            source_rgb_color=None,
        )

    # step 4: calculate the transfer accuracy
    source_rgb_color = args.ref_rgb.split(",") if args.ref_rgb is not None else None
    source_rgb_color = (
        [float(color) for color in source_rgb_color]
        if source_rgb_color is not None
        else None
    )
    if args.transfer_type == "albedo":
        rgb_space_transfer_accuracy_loss = calculate_transfer_accuracy(
            transferred_img=rgb_albedo_transferred,
            original_img=rgb_albedo_original,
            source_area_pixel_coordinates=None,
            target_area_pixel_coordinates=target_area_pixel_coordinates,
            source_rgb_color=source_rgb_color,
        )
    elif args.transfer_type == "shading":
        rgb_space_transfer_accuracy_loss = calculate_transfer_accuracy(
            transferred_img=rgb_shading_transferred,
            original_img=rgb_shading_original,
            source_area_pixel_coordinates=None,
            target_area_pixel_coordinates=target_area_pixel_coordinates,
            source_rgb_color=source_rgb_color,
        )

    return (
        surface_details_preser_error,
        rgb_space_transfer_accuracy_loss,
        rgb_space_complementary_transfer_accuracy_loss,
    )


def calculate_surface_details_preservation_error(
    transferred_img,
    original_render_img,
):
    """
    transferred_img: np.array, shape (H, W, 3), dtype float32, range [0, 1]
    original_render_img: np.array, shape (H, W, 3), dtype float32, range [0, 1]
    """
    _transferred_img = (transferred_img.copy() * 255.0).astype(np.uint8)
    _original_render_img = (original_render_img.copy() * 255.0).astype(np.uint8)

    # Convert to graycsale
    img_trans_gray = cv2.cvtColor(_transferred_img, cv2.COLOR_BGR2GRAY)
    img_orig_gray = cv2.cvtColor(_original_render_img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_trans_blur = cv2.GaussianBlur(img_trans_gray, (3, 3), 0)
    img_orig_blur = cv2.GaussianBlur(img_orig_gray, (3, 3), 0)

    img_trans_sobelxy = cv2.Sobel(
        src=img_trans_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=7
    )
    # normalize the image
    img_trans_sobelxy_normalized = cv2.convertScaleAbs(img_trans_sobelxy)
    img_orig_sobelxy = cv2.Sobel(
        src=img_orig_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=7
    )
    img_orig_sobelxy_normalized = cv2.convertScaleAbs(img_orig_sobelxy)

    img_diff = mse_distance(img_trans_sobelxy_normalized, img_orig_sobelxy_normalized)
    return img_diff


def load_transfered_images(transferred_img_path_root, filter_keyword):
    # 1: list all files in the directory
    # 2: keep the .png files
    # 3: keep the ones that has the filter keywords
    # 4: drop mask images
    # 5: load the images

    # 1: list all files in the directory
    transferred_img_paths = os.listdir(transferred_img_path_root)
    transferred_img_paths = [
        os.path.join(transferred_img_path_root, img_path)
        for img_path in transferred_img_paths
    ]

    # 2: keep the .png files
    transferred_img_paths = [
        img_path for img_path in transferred_img_paths if ".png" in img_path
    ]

    # 3: keep the ones that has the filter keywords
    transferred_img_paths = [
        img_path for img_path in transferred_img_paths if filter_keyword in img_path
    ]
    # 4: drop mask images
    transferred_img_paths = [
        img_path for img_path in transferred_img_paths if "mask" not in img_path
    ]

    # 5: load the images
    transferred_imgs = []
    for img_path in transferred_img_paths:
        transferred_img = PIL.Image.open(img_path)
        transferred_img = np.array(transferred_img).astype(np.float32) / 255.0
        transferred_img = transferred_img[:, :, :3]
        transferred_imgs.append(transferred_img)

    return transferred_imgs


def main():
    args = get_args()
    setup_and_load_models(args.prior_model, args=args)

    transferred_img_paths = args.transferred_images_path.split(",")
    original_render_img_paths = args.original_render_image_path.split(",")
    mask_img_paths = args.mask_img_path.split(",")
    filter_keywords = (
        args.filter_keywords.split(",")
        if args.filter_keywords is not None
        else ["", "", ""]
    )

    # Open the report file once in write mode
    if args.report_name:
        report_path = os.path.join(os.getcwd(), "{}.txt".format(args.report_name))
        report_file = open(report_path, "w")
    else:
        report_file = None

    for i in range(len(transferred_img_paths)):
        transfered_images = load_transfered_images(
            transferred_img_paths[i], filter_keywords[i]
        )

        if ".npy" not in original_render_img_paths[i]:
            original_render_img = PIL.Image.open(original_render_img_paths[i])
            original_render_img = (
                np.array(original_render_img).astype(np.float32) / 255.0
            )
            original_render_img = original_render_img[:, :, :3]
        else:
            raise NotImplementedError("Please use the image format instead of npy")

        source_pixels_coordinate, target_pixels_coordinate = (
            get_source_target_pixel_coordinates(mask_img_paths[i])
        )
        if "GS-IR" in transferred_img_paths[i]:
            name = "GS-IR"
        elif "DPIR" in transferred_img_paths[i]:
            name = "DPIR"
        elif "seal-3d" in transferred_img_paths[i]:
            name = "SEAL-3D"
        else:
            name = "PAPR"
        print(name)

        preservation_errors = []
        transfer_acc_errors = []
        transfer_complementary_acc_errors = []
        counter = 0
        for transferred_img in tqdm(
            transfered_images, desc="Processing transferred images"
        ):
            (
                surface_details_preser_error,
                transfer_acc_rgb,
                transfer_complementary_acc_rgb,
            ) = calculate_transfer_losses(
                transferred_img=transferred_img,
                original_render_img=original_render_img,
                source_area_pixel_coordinates=source_pixels_coordinate,
                target_area_pixel_coordinates=target_pixels_coordinate,
                args=args,
                name=name,
            )
            preservation_errors.append(surface_details_preser_error)
            transfer_acc_errors.append(transfer_acc_rgb)
            transfer_complementary_acc_errors.append(transfer_complementary_acc_rgb)
            counter += 1

        report_lines = [
            "Method: {}".format(name),
            "Avg preservation error: {}".format(np.mean(preservation_errors)),
            "Std preservation error: {}".format(np.std(preservation_errors)),
            "Avg transfer accuracy: {}".format(np.mean(transfer_acc_errors)),
            "Std transfer accuracy: {}".format(np.std(transfer_acc_errors)),
            "Avg complementary transfer accuracy: {}".format(
                np.mean(transfer_complementary_acc_errors)
            ),
            "Std complementary transfer accuracy: {}".format(
                np.std(transfer_complementary_acc_errors)
            ),
            "*" * 50,
        ]
        report_text = "\n".join(report_lines)
        print(report_text)

        if report_file:
            report_file.write(report_text + "\n")

    if report_file:
        report_file.close()


if __name__ == "__main__":
    main()

"""
Image loading and the pre/post-processing pipeline.

These live with the data layer rather than the model layer: they are what turns a file on
disk into a tensor in the training space, and back again for display and metrics.
"""

import imageio
import Imath
import numpy as np
import OpenEXR
import torch
from PIL import Image, ImageDraw, ImageFont


def shift_scale_image(img, min_val, max_val):
    """
    Shift and scale the image to the range [min_val, max_val]
    image: [B, H, W, C]
    min_val: float
    max_val: float
    """
    img = (img - min_val) / (max_val - min_val)
    return img


def inv_shift_scale_image(img, min_val, max_val):
    """
    Inverse shift and scale the image to the range [0, 1]
    image: [B, H, W, C]
    min_val: float
    max_val: float
    """
    img = img * (max_val - min_val) + min_val
    return img


def apply_image_pipeline(scene_config, img, pipeline, img_type, **extra):
    """
    Run a pre/post-processing pipeline for one image type.

    The normalisation bounds and the background value are the ones recorded for that image
    type, so `img_type` picks the dataset statistics that belong with `pipeline`.

    scene_config: the scene's config
    img: the image to process
    pipeline: list of step names, e.g. dataset.albedo_GT_postprocessing
    img_type: "render", "albedo" or "shading"
    extra: forwarded to run_image_pipeline (supervision_scaler, alpha_channel, clamp bounds)
    """
    return run_image_pipeline(
        img=img,
        pipeline=pipeline,
        eps=scene_config.models.predict_in_log_space_eps,
        min_val=getattr(scene_config.dataset, "min_{}_log".format(img_type), None),
        max_val=getattr(scene_config.dataset, "max_{}_log".format(img_type), None),
        white_bg_value=getattr(
            scene_config.geoms.background, "{}_init_scale".format(img_type), None
        ),
        **extra,
    )


def run_image_pipeline(
    img,
    pipeline,
    eps,
    min_val,
    max_val,
    white_bg_value,
    supervision_scaler=None,
    alpha_channel=None,
    clamp_min=0.0,
    clamp_max=1.0,
):
    eps = float(eps)
    if img is None:
        return None
    if len(pipeline) == 0:
        return img
    result = img
    for process in pipeline:
        if process == "log+eps":
            if isinstance(result, np.ndarray):
                result = np.log(result + eps)
            else:
                result = torch.log(result + eps)
        elif process == "normalize":
            result = shift_scale_image(
                img=result,
                min_val=min_val,
                max_val=max_val,
            )
        elif process == "inv_normalize":
            result = inv_shift_scale_image(
                img=result,
                min_val=min_val,
                max_val=max_val,
            )
        elif process == "exp-eps":
            if isinstance(result, np.ndarray):
                result = np.exp(result)
            else:
                result = torch.exp(result)
            result = result - eps
        elif process == "tone_map":
            result = tone_map_image(result)
        elif process == "scale_supervision_fg":
            result = result * supervision_scaler * alpha_channel
        elif process == "clamp":
            if isinstance(result, np.ndarray):
                result = np.clip(result, clamp_min, clamp_max)
            else:
                result = torch.clamp(result, clamp_min, clamp_max)
        elif "white_bg" in process:
            if result.shape[-1] == 4 or alpha_channel is not None:
                if alpha_channel is not None:
                    if isinstance(result, np.ndarray):
                        _alpha_channel = alpha_channel[..., None]
                    else:
                        _alpha_channel = alpha_channel[..., :1]
                    result = (
                        result[..., :3] * _alpha_channel
                        + (1.0 - _alpha_channel) * white_bg_value
                    )
                else:
                    result = (
                        result[..., :3] * result[..., -1:]
                        + (1.0 - result[..., -1:]) * white_bg_value
                    )
            else:
                raise ValueError("The image should have 4 channels")
        else:
            raise ValueError(f"Invalid preprocessing step: {process}")
    return result


def calculate_shading_from_albedo_and_rendered_image(albedo, rendered_img, epsilon):
    copy_albedo = np.copy(albedo)
    mask_albedo = copy_albedo == 0
    copy_albedo[mask_albedo] = epsilon
    shading_channels = rendered_img / copy_albedo
    shading_channels[mask_albedo] = 0
    return shading_channels


def retrieve_raw_from_rgb(rgb_img):
    assert rgb_img.shape[-1] == 3, "The image should have 3 channels"
    # we do inverse gamma correction
    raw_img = invert_tone_map_image(rgb_img)
    return raw_img


def read_exr_with_alpha(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exr_file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the color channels and alpha channel as 32-bit floats
    channels = ["R", "G", "B", "A"]  # Include alpha channel
    channel_data = [exr_file.channel(c, pt) for c in channels]

    # Convert the strings to numpy arrays
    channel_arrays = [np.frombuffer(cd, dtype=np.float32) for cd in channel_data]
    for ca in channel_arrays:
        ca.shape = (size[1], size[0])  # Numpy arrays have (row, col) structure

    alpha_channel = channel_arrays.pop()  # Remove the alpha channel from the list
    return (
        channel_arrays,
        alpha_channel,
        size,
    )  # Returns list of numpy arrays for R, G, B, A and the size


def load_resize_normal_image(
    image_path,
    scene_config,
    img_type,
    convert_image_to_raw_space,
    force_convert_image_to_raw_space_white_bg,
    pre_post_processing_steps,
    resize_w=None,
    resize_h=None,
    factor=None,
    constant_bg=1.0,
    alpha_channel=None,
    debug=False,
    force_to_load_alpha_channel=False,
):
    """
    Load a png or raw image file stored as a numpy array
    returns the image as a numpy array, float32, [0,1]
    """
    if scene_config is not None and debug:
        print(f"Loading image: {image_path}")
        print(f"Image type: {img_type}")
        print(f"Convert image to raw space: {convert_image_to_raw_space}")
        print(
            f"Force convert image to raw space white bg: {force_convert_image_to_raw_space_white_bg}"
        )
        print(f"Background init_scale: {constant_bg}")
    if img_type == "shading":
        render_raw, _, _, _ = load_resize_normal_image(
            image_path.replace("_shading", ""),
            scene_config=None,
            img_type="render",
            convert_image_to_raw_space=(
                True if ".png" in image_path else convert_image_to_raw_space
            ),
            force_convert_image_to_raw_space_white_bg=force_convert_image_to_raw_space_white_bg,
            pre_post_processing_steps=pre_post_processing_steps,
            debug=debug,
        )
        albedo_raw, _, _, _ = load_resize_normal_image(
            image_path.replace("_shading", "_albedo"),
            scene_config=None,
            img_type="albedo",
            convert_image_to_raw_space=(
                True if ".png" in image_path else convert_image_to_raw_space
            ),
            force_convert_image_to_raw_space_white_bg=force_convert_image_to_raw_space_white_bg,
            pre_post_processing_steps=pre_post_processing_steps,
            debug=debug,
        )
        image = calculate_shading_from_albedo_and_rendered_image(
            albedo=albedo_raw,
            rendered_img=render_raw,
            epsilon=1e-6,
        )
    else:
        # if the image extension is .exr, load it as a numpy array
        if image_path.endswith(".npy"):
            image = np.load(image_path).astype(np.float32)
            if img_type == "render" or force_to_load_alpha_channel:
                if image.shape[-1] == 4:
                    alpha_channel = image[..., -1]
                else:
                    alpha_channel = np.ones_like(image[..., 0])
            image = image[..., :3]
        elif image_path.endswith(".exr"):
            image_channels, image_alpha, _ = read_exr_with_alpha(image_path)
            if img_type == "render" or force_to_load_alpha_channel:
                alpha_channel = image_alpha
            image = np.stack(image_channels, axis=-1)
        elif image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            # Mip-NeRF 360 ships .JPG frames; they decode the same way as PNG.
            image = imageio.imread(image_path)
            if resize_w is not None and resize_h is not None and factor is not None:
                new_w = resize_w // factor
                new_h = resize_h // factor
            elif resize_w is not None and resize_h is not None and factor is None:
                new_w = resize_w
                new_h = resize_h
            elif resize_w is None and resize_h is None and factor is not None:
                H, W = image.shape[:2]
                new_w = W // factor
                new_h = H // factor
            else:
                H, W = image.shape[:2]
                new_w = W
                new_h = H
            image = Image.fromarray(image).resize((new_w, new_h))
            image = (np.array(image) / 255.0).astype(np.float32)
            if img_type == "render":
                if image.shape[-1] == 4:
                    alpha_channel = image[..., -1]
                else:
                    alpha_channel = np.ones_like(image[..., 0])
            image = image[..., :3]
        else:
            raise ValueError(
                "Unsupported image format for {}. Supported: .npy, .exr, .png, "
                ".jpg/.jpeg".format(image_path)
            )

    if convert_image_to_raw_space:
        print("Converting the image to raw space")
        if force_convert_image_to_raw_space_white_bg:
            image = (
                image[..., :3] * alpha_channel[..., None]
                + (1.0 - alpha_channel[..., None]) * 1
            )
        image = retrieve_raw_from_rgb(image)

    original_H = image.shape[0]
    original_W = image.shape[1]

    # preprocessing steps
    if scene_config is not None and len(pre_post_processing_steps) != 0:
        image = run_image_pipeline(
            img=image,
            pipeline=pre_post_processing_steps,
            eps=scene_config.models.predict_in_log_space_eps,
            min_val=getattr(scene_config.dataset, "min_{}_log".format(img_type), None),
            max_val=getattr(scene_config.dataset, "max_{}_log".format(img_type), None),
            white_bg_value=constant_bg,
            alpha_channel=alpha_channel,
        )

    return image, alpha_channel, original_H, original_W


def tone_map_image(image, gamma=2.2):
    """
    Tone map the image using gamma correction
    image: [B, H, W, C]
    gamma: float
    """
    image[image < 0] = 0
    return image ** (1.0 / gamma)


def invert_tone_map_image(image, gamma=2.2):
    """
    Invert tone map the image using gamma correction
    image: [B, H, W, C]
    gamma: float
    """
    return image**gamma


def write_a_text_on_image(image, text, font_size=18):
    if text is None:
        return image
    # convert the image to PIL image if it is not
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    # pick a bold font
    font = ImageFont.truetype("FreeMono.ttf", font_size)
    draw.text((30, 10), text, font=font, fill=(0, 0, 0))
    return image


def make_img_bg_transparent(img, white_values=1.0, bg_value=0):
    # img is a numpy array, [H, W, C]
    alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    img = img[:, :, :3]
    mask_r = img[:, :, 0] == white_values
    mask_g = img[:, :, 1] == white_values
    mask_b = img[:, :, 2] == white_values
    mask = np.logical_and(mask_r, np.logical_and(mask_g, mask_b))
    alpha_channel[mask] = 0

    # update the bg value
    img[mask] = bg_value

    return img, alpha_channel

import argparse
import glob
import json
import math
import os
import sys

import imageio.v2 as imageio
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import *

yagiz_intrinsic_model = None
yagiz_pipeline = None
uninvert = None

cIMLE_yagiz_pipeline = None
cIMLE_yagiz_intrinsic_model = None



def get_args():
    parser = argparse.ArgumentParser(
        description="2D intrinsic decomposition using prior model"
    )
    parser.add_argument(
        "--save_raw_images",
        action="store_true",
    )
    parser.add_argument(
        "--prior_model",
        choices=["yagiz_v1", "yagiz_v2", "GT", "cIMLE_yagiz_v1"],
        default="yagiz_v1",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_type",
        choices=["tanks_temples", "nerf_synthetic", "custom"],
    )
    parser.add_argument(
        "--make_bg_white",
        action="store_true",
    )
    parser.add_argument(
        "--make_bg_transparent",
        action="store_true",
    )
    parser.add_argument(
        "--resize_w",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resize_h",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--format",
        choices=["png", "exr"],
        default="png",
    )
    parser.add_argument(
        "--dataset_albedo_postfix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dataset_render_postfix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--save_albedo",
        action="store_true",
    )
    parser.add_argument(
        "--save_shading",
        action="store_true",
    )
    parser.add_argument(
        "--save_render",
        action="store_true",
    )

    parser.add_argument(
        "--cIMLE_number_of_samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cIMLE_d_latent",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--custom_image_path",
        type=str,
        default=None,
    )
    return parser.parse_args()


def setup_and_load_models(prior_model, args=None):
    global yagiz_intrinsic_model
    global yagiz_pipeline
    global uninvert
    global cIMLE_yagiz_pipeline
    global cIMLE_yagiz_intrinsic_model

    if "yagiz" in prior_model and "cIMLE" not in prior_model:
        from chrislib.general import uninvert
        from intrinsic.model_util import load_models
        from intrinsic.pipeline import run_pipeline as yagiz_pipeline

        yagiz_intrinsic_model = load_models("paper_weights")
    elif "yagiz" in prior_model and "cIMLE" in prior_model:
        from chrislib.general import uninvert

        from ambiguity_aware_prior.models.tools import load_cIMLE_model
        from ambiguity_aware_prior.models.tools import (
            run_pipeline as cIMLE_yagiz_pipeline,
        )

        cIMLE_yagiz_intrinsic_model = load_cIMLE_model(
            cIMLE_iid_model_checkpoint=args.model_checkpoint
        )
    print(f"Loaded {prior_model} model")


def get_image_paths(
    dataset_root,
    dataset_type,
    postfix="",
    format="png",
    splits=["train", "test", "val"],
    custom_image_path=None,
):
    img_paths = {}
    if dataset_type == "tanks_temples":
        img_paths["rgb"] = glob.glob(os.path.join(dataset_root, "rgb", f"*.{format}"))
        print("number of images in rgb: ", len(img_paths["rgb"]))
    elif dataset_type == "nerf_synthetic":
        for s in splits:
            img_paths[s] = glob.glob(
                os.path.join(dataset_root, f"{s}{postfix}", f"*.{format}")
            )
            print(f"number of images in {s}: ", len(img_paths[s]))
    elif dataset_type == "custom":
        img_paths["custom"] = [custom_image_path]
        print("number of images in rgb: ", len(img_paths["custom"]))
    else:
        raise ValueError("Invalid dataset type")

    return img_paths


def setup_directories(root_output, img_paths, prior_model, args):
    for split_name in img_paths:
        if args.save_albedo and (
            "train" in split_name or "rgb" in split_name or "custom" in split_name
        ):
            os.makedirs(
                os.path.join(root_output, f"{split_name}_albedo_{prior_model}"),
                exist_ok=True,
            )
        if args.save_shading:
            os.makedirs(
                os.path.join(root_output, f"{split_name}_shading_{prior_model}"),
                exist_ok=True,
            )
        print(f"Created directories for {split_name} in {root_output}")
    os.makedirs(f"{root_output}_meta", exist_ok=True)


def get_shading_albedo_yagiz(img, is_inp_img_linear=False):
    result = yagiz_pipeline(
        yagiz_intrinsic_model,
        img,
        resize_conf=0.0,
        maintain_size=True,
        linear=is_inp_img_linear,
        device="cuda",
    )
    # we need to reshape to (H, W, 1) for shading
    raw_shading = uninvert(result["inv_shading"])
    raw_shading = raw_shading.reshape(raw_shading.shape[0], raw_shading.shape[1], 1)

    raw_albedo = result["albedo"]

    rgb_shading = tone_map_image(raw_shading)
    rgb_albedo = tone_map_image(raw_albedo)

    raw_render = result["image"]

    return [raw_shading], [raw_albedo], [rgb_shading], [rgb_albedo], raw_render


def get_shading_albedo_cIMLE_yagiz(
    img, number_of_samples, d_latent, is_inp_img_linear=False
):
    raw_shadingS = []
    raw_albedoS = []
    rgb_shadingS = []
    rgb_albedoS = []
    for s_index in range(number_of_samples):
        z_code = torch.randn(1, d_latent).to("cuda")
        result = cIMLE_yagiz_pipeline(
            cIMLE_yagiz_intrinsic_model,
            img,
            resize_conf=0.0,
            maintain_size=True,
            linear=is_inp_img_linear,
            device="cuda",
            stage=1,
            z_code=z_code,
        )

        raw_shading = uninvert(result["gry_shd"])
        raw_shading = raw_shading.reshape(raw_shading.shape[0], raw_shading.shape[1], 1)

        raw_albedo = result["gry_alb"]
        raw_render = result["image"]

        rgb_shading = tone_map_image(raw_shading)
        rgb_albedo = tone_map_image(raw_albedo)

        raw_shadingS.append(raw_shading)
        raw_albedoS.append(raw_albedo)
        rgb_shadingS.append(rgb_shading)
        rgb_albedoS.append(rgb_albedo)


    return raw_shadingS, raw_albedoS, rgb_shadingS, rgb_albedoS, raw_render


def get_shading_albedo(
    img, prior_model, args, is_inp_img_linear=False, cIMLE_number_of_samples=1
):
    if "yagiz" in prior_model and "cIMLE" not in prior_model:
        return get_shading_albedo_yagiz(img, is_inp_img_linear)
    elif "yagiz" in prior_model and "cIMLE" in prior_model:
        return get_shading_albedo_cIMLE_yagiz(
            img,
            is_inp_img_linear=is_inp_img_linear,
            number_of_samples=cIMLE_number_of_samples,
            d_latent=args.cIMLE_d_latent,
        )
    else:
        raise ValueError("Invalid prior model")


def load_image(img_path, args):
    if args.format == "exr":
        img, alpha_channel, _ = read_exr_with_alpha(file_path=img_path)
        img = np.dstack(img)
    else:
        img = imageio.imread(img_path)

        if args.resize_w is not None and args.resize_h is not None:
            new_w, new_h = args.resize_w, args.resize_h
        else:
            new_w, new_h = img.shape[1], img.shape[0]

        img = Image.fromarray(img).resize((new_w, new_h))
        img = (np.array(img) / 255.0).astype(np.float32)
        # check if it has an alpha channel
        if img.shape[-1] == 4:
            alpha_channel = img[..., 3]
        else:
            alpha_channel = None
        img = img[:, :, :3]


        if args.make_bg_white and alpha_channel is not None:
            img = (
                img[..., :3] * alpha_channel[..., None]
                + (1.0 - alpha_channel[..., None]) * 1
            )
    return img, alpha_channel


def update_statistics(raw_img, img_type, stats_dict, args):
    for img in raw_img:
        log_raw_img = np.log(img + args.eps)
        stats_dict[f"min_{img_type}_log"] = min(
            stats_dict[f"min_{img_type}_log"], np.min(log_raw_img)
        )
        stats_dict[f"max_{img_type}_log"] = max(
            stats_dict[f"max_{img_type}_log"], np.max(log_raw_img)
        )


def save_image(img, alpha_channel, img_name, args, save_raw_images=False):
    if alpha_channel is not None:
        img = np.dstack([img, alpha_channel])
    if save_raw_images:
        np.save(img_name.replace(".png", ".npy"), img)
    else:
        img = Image.fromarray((np.clip(img, 0, 1) * 255.0).astype(np.uint8))
        img.save(img_name)


def main():
    args = get_args()
    if args.prior_model != "GT":
        setup_and_load_models(args.prior_model, args=args)
    render_img_paths = get_image_paths(
        dataset_root=args.dataset_root,
        dataset_type=args.dataset_type,
        format=args.format,
        custom_image_path=args.custom_image_path,
    )
    if args.prior_model == "GT":
        albedo_img_paths = get_image_paths(
            dataset_root=args.dataset_root,
            dataset_type=args.dataset_type,
            format=args.format,
            postfix=args.dataset_albedo_postfix,
            custom_image_path=args.custom_image_path,
        )

    setup_directories(
        root_output=args.dataset_root,
        img_paths=render_img_paths.keys(),
        prior_model=args.prior_model,
        args=args,
    )

    statistics_dict = {
        "min_render_log": math.inf,
        "max_render_log": -math.inf,
        "min_albedo_log": math.inf,
        "max_albedo_log": -math.inf,
        "min_shading_log": math.inf,
        "max_shading_log": -math.inf,
    }
    for split in render_img_paths.keys():
        pbar = tqdm.tqdm(range(len(render_img_paths[split])))
        for img_index in pbar:
            pbar.set_description(f"Processing {split}")
            if args.prior_model == "GT":
                raw_render, alpha_channel = load_image(
                    render_img_paths[split][img_index], args
                )
                [raw_albedoS], _ = load_image(albedo_img_paths[split][img_index], args)
                [raw_shadingS] = calculate_shading_from_albedo_and_rendered_image(
                    albedo=raw_albedoS, rendered_img=raw_render, epsilon=args.eps
                )
            else:
                rgb_render, alpha_channel = load_image(
                    render_img_paths[split][img_index], args
                )
                n_samples = 1
                if "cIMLE" in args.prior_model and (
                    split == "train" or split == "rgb" or split == "custom"
                ):
                    n_samples = args.cIMLE_number_of_samples
                raw_shadingS, raw_albedoS, rgb_shadingS, rgb_albedoS, raw_render = (
                    get_shading_albedo(
                        rgb_render,
                        args.prior_model,
                        args=args,
                        cIMLE_number_of_samples=n_samples,
                    )
                )
            raw_render = [raw_render]
            update_statistics(raw_render, "render", statistics_dict, args)
            update_statistics(raw_albedoS, "albedo", statistics_dict, args)
            update_statistics(raw_shadingS, "shading", statistics_dict, args)

            if args.save_albedo and (
                split == "train" or split == "rgb" or split == "custom"
            ):
                n_samples = len(rgb_albedoS)
                for i_sample in range(n_samples):
                    img_name = os.path.basename(render_img_paths[split][img_index])
                    if n_samples > 1:
                        img_name = img_name.split(".")
                        img_name = img_name[0] + f"_sample_{i_sample}." + img_name[1]
                    save_image(
                        img=rgb_albedoS[i_sample],
                        alpha_channel=alpha_channel,
                        img_name=os.path.join(
                            args.dataset_root,
                            f"{split}_albedo_{args.prior_model}",
                            img_name,
                        ),
                        args=args,
                    )
                    if args.save_raw_images:
                        save_image(
                            img=raw_albedoS[i_sample],
                            alpha_channel=alpha_channel,
                            img_name=os.path.join(
                                args.dataset_root,
                                f"{split}_albedo_{args.prior_model}",
                                img_name,
                            ),
                            args=args,
                            save_raw_images=True,
                        )
            if args.save_shading and (
                split == "train" or split == "rgb" or split == "custom"
            ):
                n_samples = len(rgb_shadingS)
                for i_sample in range(n_samples):
                    img_name = os.path.basename(render_img_paths[split][img_index])
                    if n_samples > 1:
                        img_name = img_name.split(".")
                        img_name = img_name[0] + f"_sample_{i_sample}." + img_name[1]
                    save_image(
                        img=rgb_shadingS[i_sample],
                        alpha_channel=alpha_channel,
                        img_name=os.path.join(
                            args.dataset_root,
                            f"{split}_shading_{args.prior_model}",
                            img_name,
                        ),
                        args=args,
                    )
                    if args.save_raw_images:
                        save_image(
                            img=raw_shadingS[i_sample],
                            alpha_channel=alpha_channel,
                            img_name=os.path.join(
                                args.dataset_root,
                                f"{split}_shading_{args.prior_model}",
                                img_name,
                            ),
                            args=args,
                            save_raw_images=True,
                        )
            if args.save_render and args.save_raw_images:
                save_image(
                    img=raw_render[0],
                    alpha_channel=alpha_channel,
                    img_name=os.path.join(
                        args.dataset_root,
                        f"{split}",
                        os.path.basename(render_img_paths[split][img_index]),
                    ),
                    args=args,
                    save_raw_images=True,
                )

    # save the statistics
    eps = "{:.0e}".format(args.eps)
    statistics_dict["max_albedo_log"] = float(statistics_dict["max_albedo_log"])
    statistics_dict["min_albedo_log"] = float(statistics_dict["min_albedo_log"])
    statistics_dict["max_render_log"] = float(statistics_dict["max_render_log"])
    statistics_dict["min_render_log"] = float(statistics_dict["min_render_log"])
    statistics_dict["max_shading_log"] = float(statistics_dict["max_shading_log"])
    statistics_dict["min_shading_log"] = float(statistics_dict["min_shading_log"])

    with open(
        f"{args.dataset_root}_meta/raw_statistics_eps_{eps}_{args.prior_model}.json",
        "w",
    ) as f:
        json.dump(statistics_dict, f)


if __name__ == "__main__":
    main()

<h1 align="center">Intrinsic PAPR</h1>

<p align="center">
  <b>Tackling Misattribution in 3D Intrinsic Decomposition via Proximity Attention Point Rendering</b><br>
  ECCV 2026
</p>

<p align="center">
  <a href="https://amoazeni75.github.io/Intrinsic-PAPR/">Project page</a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2407.00500">Paper</a>
</p>

<p align="center">
  <img src="assets/teaser.webp" width="100%" alt="Render, albedo and shading decomposition">
</p>

<p align="center">
  <i>View-consistent albedo and shading on real and synthetic scenes.</i>
</p>

Intrinsic PAPR builds on [PAPR](https://zvict.github.io/papr/) (Proximity Attention Point
Rendering) and splits its point-based renderer into an albedo branch and a shading branch, so
that each 3D primitive carries its own intrinsic properties. Because the split happens per
point, albedo and shading can be edited directly on the point cloud.

---

## Contents

- [1. Install](#1-install)
- [2. Data](#2-data)
- [3. The 2D prior](#3-the-2d-prior)
- [4. Train](#4-train)
- [5. Evaluate](#5-evaluate)
- [6. Editing](#6-editing)
- [7. Repository layout](#7-repository-layout)
- [8. Credits](#8-credits)

---

## 1. Install

```bash
conda env create -f environment.yml
conda activate intrinsic-papr
```

Data preparation runs in its own environment. Use the pinned commit:

```bash
conda create --clone intrinsic-papr -n intrinsic-papr-extract
conda activate intrinsic-papr-extract
pip install "git+https://github.com/compphoto/Intrinsic@903e8ee"
```

Check both:

```bash
conda run -n intrinsic-papr python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
conda run -n intrinsic-papr-extract python -c "from intrinsic.model_util import load_models; print('extraction deps ok')"
```

Rendering new synthetic scenes with Blender uses `conda env create -f blender.yml`.

The first training run downloads about 530 MB of pretrained weights, so the machine needs
network access once.

Run all commands from the repository root.

## 2. Data

Three scene types are supported, selected by `scene_1.dataset.type`.

| `type` | Datasets | Layout expected |
|---|---|---|
| `synthetic` | NeRF Synthetic, TensoIR | `transforms_<split>.json`, `train/`, `test/` |
| `t2` | Tanks & Temples | `rgb/`, `pose/`, `intrinsics.txt` (NSVF layout) |
| `mip360` | Mip-NeRF 360 | `transforms.json`, `images_<factor>/` |

Download the sources from their own projects:
[NeRF Synthetic](https://github.com/bmild/nerf), [TensoIR](https://github.com/Haian-Jin/TensoIR),
[Tanks & Temples in NSVF layout](https://github.com/facebookresearch/NSVF#dataset),
[Mip-NeRF 360](https://jonbarron.info/mipnerf360/).

> **Splits.** Synthetic and Mip-NeRF 360 read `transforms_train.json` and
> `transforms_test.json`. Tanks & Temples selects by the first character of each filename:
> `0` train, `1` test.

> **Mip-NeRF 360 resolution.** Set `dataset.factor` on the three dataset blocks or on the
> command line; it has no default.

> **Tanks & Temples**: use the NSVF-preprocessed release, not the raw download.

Training reads albedo maps extracted offline and stored next to the images. Run the steps
below in the `intrinsic-papr-extract` environment.

**Step 1 — synthetic scenes only: render the scene and its ground-truth albedo.**

```bash
conda activate blender
python tools/Blender_render_objects.py --help
```

**Step 2 — Tanks & Temples only: normalise the layout.** Run before Step 3.

```bash
python tools/prepare_TT_for_albedo_extraction.py --root ./data/tanks_temples --scenes Truck
```

Resizes to 1088x640 and writes `pretrained_albedo_transparent_bg_<scene>/` and
`pretrained_albedo_white_bg_<scene>/`. Point Step 3 at the one matching your config.

**Step 3 — extract albedo with the 2D prior.**

```bash
python tools/intrinsic_decomposition_using_prior_model.py \
  --prior_model yagiz_v1 \
  --dataset_type nerf_synthetic --dataset_root ./data/nerf_synthetic/lego \
  --save_albedo --save_render --save_raw_images --eps 1e-3
```

All four flags are required; none is the default:

- **`--save_albedo`** — write the albedo maps.
- **`--save_raw_images`** — write the float `.npy` beside each `.png`. The synthetic and
  Tanks & Temples templates set `dataset.image_file_format: "npy"`.
- **`--save_render`** — write the render as `.npy` into the split directory (`train/r_0.npy`).
- **`--eps 1e-3`** — must match `models.predict_in_log_space_eps` in the config; the epsilon
  is part of the statistics filename.

`--save_shading` is optional; training does not read a shading directory.

> [!IMPORTANT]
> `configs/base.yml` ships with `training.albedo_space_carving_loss.use: true`, which needs
> the ambiguity-aware prior (section 3) and its `<stem>_sample_0` ... `<stem>_sample_9` files.
> With a single-estimate prior such as `yagiz_v1`, also pass
> `--scene_1.training.albedo_space_carving_loss.use false`.

`--dataset_type` accepts `nerf_synthetic`, `tanks_temples` and `custom`. Tanks & Temples runs
against the directory Step 2 produced:

```bash
python tools/intrinsic_decomposition_using_prior_model.py --prior_model yagiz_v1 \
  --dataset_type tanks_temples \
  --dataset_root ./data/tanks_temples/pretrained_albedo_transparent_bg_Truck \
  --save_albedo --save_render --save_raw_images --eps 1e-3
```

**Mip-NeRF 360.** `--dataset_type custom` takes one image at a time via
`--custom_image_path`, so loop over the resolution directory and rename the output
afterwards:

```bash
SCENE=./data/mipnerf360/garden
for img in "$SCENE"/images_4/*.JPG; do
  python tools/intrinsic_decomposition_using_prior_model.py --prior_model yagiz_v1 \
    --dataset_type custom --dataset_root "$SCENE/images_4" \
    --custom_image_path "$img" --save_albedo --save_render --save_raw_images --eps 1e-3
done
mv "$SCENE/images_4/custom_albedo_yagiz_v1" "$SCENE/images_4_albedo_yagiz_v1"
```

The result is `images_4/`, `images_4_albedo_yagiz_v1/` and `images_4_meta/` side by side,
with `scene_1.dataset.train_albedo_extraction_method: "_yagiz_v1"`.

Extraction writes `<split>_albedo_<prior>/` and a `_meta/` directory holding
`raw_statistics_eps_<eps>_<prior>.json`. Albedo is written only for the supervised split:
`train` for `nerf_synthetic`, `rgb` for `tanks_temples`, `custom` for `custom`.


Two things must match the config:

- **The folder suffix.** `--prior_model yagiz_v1` produces `train_albedo_yagiz_v1/`, so set
  `scene_1.dataset.train_albedo_extraction_method` (and the `test_` twin) to `"_yagiz_v1"`.
  The shipped templates leave these empty, pointing at a plain `train_albedo/`.
- **The statistics epsilon.** Must equal `models.predict_in_log_space_eps` (`1e-03` in the
  templates).

## 3. The 2D prior

`ambiguity_aware_prior/` holds the ambiguity-aware 2D intrinsic decomposition network: a fork
of Careaga and Aksoy's ordinal-shading network with AdaIN style modulation, trained with a
cIMLE objective so it produces several plausible albedo estimates per image.

Two priors are available:

```bash
# the public deterministic prior (no extra weights needed)
python tools/intrinsic_decomposition_using_prior_model.py \
  --prior_model yagiz_v1 --dataset_type nerf_synthetic \
  --dataset_root ./data/nerf_synthetic/lego --save_albedo --save_render --save_raw_images --eps 1e-3

# the ambiguity-aware prior, using weights you trained yourself
python tools/intrinsic_decomposition_using_prior_model.py \
  --prior_model cIMLE_yagiz_v1 --model_checkpoint <path to .pt> \
  --dataset_type nerf_synthetic --dataset_root ./data/nerf_synthetic/lego \
  --save_albedo --save_render --save_raw_images --eps 1e-3
```

**cIMLE checkpoint.** Download it, then pass it as `--model_checkpoint`:

```bash
python tools/download_prior_checkpoint.py        # -> ./checkpoints/cimle_yagiz_v1.pt
```

The ordinal half is fetched automatically, so this is the only checkpoint needed. To train
your own, use `ambiguity_aware_prior/gry_shd_train.py`, which needs the Hypersim and MID
Intrinsics datasets. These weights are a fine-tune of Careaga and Aksoy's ordinal-shading
network; see the credits.

`scene_1.dataset.train_albedo_extraction_method` selects which prior's output training reads:
`""` for `train_albedo/`, `"_cIMLE_yagiz_v1"` for the ambiguity-aware one.

### Multiple albedo estimates per view

The space carving loss needs several albedo samples per view for the training split. Request
them with `--cIMLE_number_of_samples`:

```bash
python tools/intrinsic_decomposition_using_prior_model.py \
  --prior_model cIMLE_yagiz_v1 --model_checkpoint <path to .pt> \
  --cIMLE_number_of_samples 10 --cIMLE_d_latent 32 \
  --dataset_type nerf_synthetic --dataset_root ./data/nerf_synthetic/lego \
  --save_albedo --save_render --save_raw_images --eps 1e-3
```

Sample files land beside the single-estimate ones with the index appended to the stem,
e.g. `r_0_sample_0.npy` ... `r_0_sample_9.npy`. Only the training split gets samples.

The loss is on by default; `num_samples` must match `--cIMLE_number_of_samples`:

```yaml
scene_1:
  training:
    albedo_space_carving_loss:
      use: true              # shipped default
      num_samples: 10        # must match --cIMLE_number_of_samples
```

The loader reads `_sample_0` through `_sample_<num_samples-1>`. With `use: false` it reads the
plain single albedo map and ignores any sample files.

Note the test split keeps a single albedo, so it reads the unsuffixed directory. A scene
therefore needs both `train_albedo_cIMLE_yagiz_v1/` (samples) and `test_albedo/` (single).

See [ambiguity_aware_prior/README.md](ambiguity_aware_prior/README.md) to fine-tune the prior.

## 4. Train

Three config templates ship, one per scene type; each declares `base: base.yml` and lists only
the fields that differ. Pick the scene on the command line:

```bash
python train.py --opt configs/synthetic.yml \
  --index Lego --scene_1.index lego \
  --dataset_root ./data/nerf_synthetic \
  --scene_1.dataset.path lego --scene_1.eval.dataset.path lego \
  --gpu_id 0
```

`configs/tanks_and_temples.yml` and `configs/mipnerf360.yml` work the same way. Every key of the
merged config is a `--key` flag, nested with dots, so anything can be overridden without editing
YAML. Booleans take `true`/`false`.

The templates ship the settings the paper reports: space carving on for the whole 250K-step
schedule, the learnable per-view albedo scalar on and initialised to 0.85, and the space carving
weight `albedo_loss_weight / rgb_loss_weight` — 0.085 on synthetic scenes, 0.125 on real-world.

> [!NOTE]
> `models.supervision_scaler.apply_interval_start` / `apply_interval_end` bound the steps over
> which the scalar receives gradients. The templates cover `0` to `250000`, so raising
> `training.steps` means raising `apply_interval_end` to match.

Results land in `<save_dir>/<index>/<scene>/`: `checkpoints/`, `train_main_plots/`,
`train_pcd_plots/`, and `metrics.jsonl`.

Mip-NeRF 360 scenes set `geoms.background.append_bkg_points`, adding one attention slot per ray
at its intersection with a background sphere. Set `sphere_center` and `sphere_radius` from the
scene's COLMAP point cloud for a new scene.

## 5. Evaluate

Render the test views:

```bash
python test.py --opt configs/synthetic.yml \
  --index Lego --scene_1.index lego \
  --dataset_root ./data/nerf_synthetic \
  --scene_1.dataset.path lego --test_dataset_path lego \
  --scene_1.load_path checkpoints-250000.pth \
  --test_action render --render_frame_type all --media_type image \
  --save_albedo_images --save_image_with_numpy --gpu_id 0
```

This writes the predicted renders and albedo maps, and prints PSNR, SSIM and LPIPS.

For albedo metrics, apply a per-channel scale alignment between prediction and ground truth
before computing PSNR, SSIM and LPIPS, as in NeRFactor and GS-IR.

`--test_action calculate_albedo_consistency` writes `albedo_consistency.json` (MACE).

## 6. Editing

Albedo and shading live on the points, so an edit made in one view carries to every other view.
A region is given as a stroke file: one pixel per line, `x,y`, integer image coordinates.

```bash
python test.py --opt configs/synthetic.yml ... \
  --test_action freeform_transfer_albedo \
  --source_target_area_selection_method freeform_pixels \
  --source_area_path <source_strokes>.txt \
  --target_area_path <target_strokes>.txt \
  --freeform_source_key_frame_index 0 --freeform_target_key_frame_index 0
```

The key frame indices name the view each stroke file was drawn on.
`--freeform_source_point_method` / `--freeform_target_point_method` take `all` (every point a
stroke pixel attends to) or `highest_attention` (only the strongest).

A region can also be an axis-aligned box, with
`--source_target_area_selection_method points_cloud_areas_boxes`. Boxes live in
`point_cloud_areas.json` at the root of the scene directory, keyed by area index;
`--source_area_indices` and `--target_area_indices` select them.

The full set of `--test_action` values is:

| Action | What it does | Needs |
|---|---|---|
| `render` | render the test views and report PSNR, SSIM and LPIPS | |
| `calculate_albedo_consistency` | MACE across views, to `albedo_consistency.json` | `--save_albedo_images`, a source region, and enough views that a point is seen twice |
| `transfer_albedo` / `transfer_shading` | copy the source region's albedo or shading features onto the target region | a source and target region |
| `freeform_transfer_albedo` / `freeform_transfer_shading` | the same, with the regions given as stroke files | the stroke flags above |
| `change_brightness` | scale the dominant shading direction of every point | `--shading_intensity`, or `--intensity_sweep` for a range |
| `interpolate_albedo` | blend the albedo features of two or more points into a region | `--interpolate_colors_indices`, `--interpolate_colors_name` |
| `2D_color_interpolation_with_UNet` | render a grid blending two saved colour features | `--color_1_feature`, `--color_2_feature` |
| `TSNE` | a t-SNE plot of the point features | `--TSEN_frames` |
| `render_depth_pcd_for_comparison` | per-view depth maps and point-cloud renders | |

The error metrics for a transfer are computed by `tools/calculate_transfer_losses.py`
(and `tools/calculate_transfer_losses_multi_samples.py` for the multi-sample case).

## 7. Repository layout

```
train.py                    entry point: build the scene, then run the training loop
test.py                     entry point: build the scene, then run one --test_action
intrinsic_papr/
  config/                   config merging (loader.py) and the command line (cli.py)
  data/                     dataset.py, the loaders under loaders/, the image
                            pipelines (pipeline.py), ray and camera maths,
                            and the per-scene statistics
  models/                   renderer.py (the point renderer), attention.py, unet.py,
                            mlp.py, points.py, encodings.py, lpips.py
  training/                 scene.py (SceneManager), trainer.py (the loop),
                            evaluation.py, losses.py, checkpoint.py, schedules.py,
                            plots.py, metrics.py
configs/                    base.yml plus one template per scene type
tools/                      data preparation, the 2D prior interface, transfer metrics
ambiguity_aware_prior/      the 2D intrinsic decomposition prior
```

`train.py` and `test.py` run straight from a clone; `pip install -e .` also works.

This release carries only the code paths the shipped configs use. A config value outside the
documented set raises `NotImplementedError` rather than silently falling back.

## 8. Credits

This code builds on work by others:

- **PAPR**, Zhang et al. — the point renderer this work extends.
  https://github.com/zvict/papr
- **Colorful Diffuse Intrinsic Image Decomposition in the Wild** and the ordinal shading network,
  Careaga and Aksoy — `ambiguity_aware_prior/` is a fork of their training code, and the
  released cIMLE checkpoint is a fine-tune of their weights.
  https://github.com/compphoto/Intrinsic
- **MiDaS**, Ranftl et al. — `ambiguity_aware_prior/models/midas_net_small.py`.
  https://github.com/isl-org/MiDaS
- **LPIPS**, Zhang et al. — `intrinsic_papr/models/lpips.py`. https://github.com/richzhang/PerceptualSimilarity
- **U-Net**, Ronneberger et al., via SNP and Pytorch-UNet — `intrinsic_papr/models/unet.py`.
  https://github.com/princeton-vl/SNP, https://github.com/milesial/Pytorch-UNet

## Citation

```bibtex
@inproceedings{moazeni2026intrinsicpapr,
  title     = {Intrinsic {PAPR}: Tackling Misattribution in 3D Intrinsic Decomposition
               via Proximity Attention Point Rendering},
  author    = {Moazeni, Alireza and Peng, Shichong and Zhang, Yanshu
               and Vashist, Chirag and Li, Ke},
  booktitle = {European Conference on Computer Vision ({ECCV})},
  year      = {2026}
}
```

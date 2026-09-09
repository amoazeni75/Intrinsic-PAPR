# Ambiguity-aware 2D intrinsic decomposition prior

This is the 2D prior used by Intrinsic PAPR. It produces the albedo and shading maps that the 3D
model trains against.

## What this is a fork of

The network and its training code come from **Chris Careaga and Yağız Aksoy**, "Intrinsic Image
Decomposition via Ordinal Shading" and "Colorful Diffuse Intrinsic Image Decomposition in the
Wild" (https://github.com/compphoto/Intrinsic). The dataset handling, the MiDaS-small backbone and
the ordinal-shading pipeline are theirs, kept close to the originals. Please cite their work if
you use this.

## What we changed

A single deterministic albedo estimate cannot express the ambiguity in intrinsic decomposition:
many albedo/shading splits explain the same photograph. We make the prior produce several
plausible estimates instead of one.

- **AdaIN style modulation** in the encoder (`models/midas_net_small.py`), driven by a latent
  code `z ~ N(0, I)` through a small MLP that predicts per-layer scale and shift.
- **A cIMLE training objective** (`gry_shd_train.py`): for each image we draw M latent codes,
  generate M candidate albedos, and backpropagate only through the one that best matches the
  target. This keeps the samples diverse instead of collapsing to the mean.
- **Multi-dataset sampling** in `dataset/custom_datasets.py`, and a curated exclusion list for
  frames where the ordinal input is degenerate.

Fine-tuning is done once and is scene-independent.

## Install

```bash
conda env create -f environment.yml
conda activate 2D_ID_prior
pip install "git+https://github.com/compphoto/Intrinsic@main"
```

Run every command below **from inside this directory** (`cd ambiguity_aware_prior`). The training
script resolves `splits/` and `checkpoints/` relative to the working directory.

## Data

Two datasets, both from their original authors:

- **Hypersim** — https://github.com/apple/ml-hypersim. Only the `final_hdf5` `color` and
  `diffuse_reflectance` components are needed.
- **MID-Intrinsics** — https://github.com/compphoto/MIDIntrinsics, built from the MIT
  multi-illumination captures plus the pseudo-ground-truth albedo the authors released.

You also need the ordinal-shading weights from the upstream release
(https://github.com/CCareaga/ordinal_training/releases/tag/v1); the `rendered_only` weights are
the ones used to generate the ordinal inputs.

Precompute the ordinal inputs:

```bash
HYPERSIM_PATH=/path/to/Hypersim python tools/prep_ord_shd.py \
  --workers 8 --dataset hypersim --weights_path /path/to/ordinal_weights.pt

python tools/prep_ord_shd_mi.py --mid_path /path/to/MIDIntrinsics \
  --weights_path /path/to/ordinal_weights.pt
```

Optional 5K subsets, which is what the paper's fine-tune used:

```bash
python tools/make_Hypersim_subset.py --root_dir /path/to/Hypersim --dest_dir /path/to/Hypersim_5K
python tools/make_MIDIntrinsics_subset.py --root_dir /path/to/MIDIntrinsics --dest_dir /path/to/MIDIntrinsics_5K
```

## Fine-tune

```bash
python gry_shd_train.py \
  --HYPERSIM_PATH /path/to/Hypersim_5K --MID_PATH /path/to/MIDIntrinsics_5K --is_subset \
  --checkpoint paper_weights --pretrain_mlp --use_IMLE \
  --weights_path ./experiments --exp_name ambiguity_prior \
  --ordinal_path /path/to/ordinal_weights.pt \
  --train_iters 200 --epochs 10 --pretrain_steps 100 \
  --base_lr 0.25e-5 --mlp_lr 1.75e-5 --mlp_lr2 1.75e-5
```

Weights are written to `<weights_path>/<exp_name>/checkpoints/`, and scalar metrics to
`<weights_path>/<exp_name>/results/metrics.jsonl`. Qualitative evaluation reads images from
`--qual_eval_dir` (default `qual_eval/`); create that directory and put a few test images in it,
or the qualitative step produces nothing.

Fine-tuning takes roughly an hour for 100 steps on a single RTX 3090.

## Using the result

Point the extraction tool in the parent repository at the checkpoint:

```bash
cd ..
python tools/intrinsic_decomposition_using_prior_model.py \
  --prior_model cIMLE_yagiz_v1 --model_checkpoint ambiguity_aware_prior/experiments/ambiguity_prior/checkpoints/best.pt \
  --dataset_type nerf_synthetic --dataset_root ./data/nerf_synthetic/lego
```

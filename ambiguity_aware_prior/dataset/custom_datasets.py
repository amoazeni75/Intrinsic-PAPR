# ================================================================================
import os
import pickle
import random
from collections import deque
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from chrislib.color_util import iuv2rgb
from chrislib.data_util import (
    load_image,
    random_color_shift,
    random_crop_and_resize,
    random_flip,
)
from chrislib.general import (
    get_brightness,
    get_tonemap_scale,
    invert,
    match_scale,
    to2np,
    uninvert,
)
from scipy import signal
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import load_exr, load_h5_image, match_albedo, random_degrade



MIN_VAL = 1.0 / 255.0


# this is a generic dataset class that has the boilerplate functionality that is
# common across all the synthetic datasets. It assumes that certain stubs are filled
# out (i.e. children classes inherit and add dataset specific behavior)

# there are different modes that determine which components are returned by the dataloader
# STAGE 0 - ordinal shading: just requires the input image, gt albedo and gry shd (colors baked into albedo)
# STAGE 1 - gry shading: uses a base and full resolution ordinal estimation to output the final gray shading
# STAGE 2 - low-res shd chroma: requires input gray shading (gt or pred) and ground truth albedo and color shading
# STAGE 3 - final color shd: requires the input clr shading (low-res chroma) and the ground truth albedo and color shading
# STAGE 4 - diffuse shading: requires the estimated albedo and the ground-truth diffuse shading, albedo and residual


class GenericColorfulDataset(Dataset):

    # across all datasets we need to initialize the file list, and whether
    # or not this dataset can be sampled from in order, etc.
    def __init__(
        self,
        stage,
        clip=100,
        cache_prb=0.0,
        cache_len=250,
        augment=True,
        color_shift=False,
        use_pred_shd=False,
        random=True,
        use_normals=False,
        is_subset=False,
        ignore_list=[],
    ):
        """
        Generic dataset functionality that can be used with all rendered datasets. The class defines a dataset specific
        interface to be implemented by a custom child class. The inherited methods represent functionality that is specific
        to a given rendered dataset (e.g. file naming conventions, file type, etc.)

        params:
            stage (int): what task stage to use [0, 1, 2, 3, 4]
            clip (float): max value of shading to determine masked pixels
            cache_prob (float): value between 0-1 that determines probability of drawing sample from cache
            cache_len (int): size of the cache of loaded samples
            augment (bool): whether or not to perform augmentation (crop, resize, flip, etc) (default: True)
            color_shift (bool): wheter to apply random color cast (used for low-res shading chroma training) (default: False)
            use_pred (bool): whether or not to use estimated results from previous stage as input, if false ground-truth is used (default: False)
            random (bool): whether or not to randomly sample examples, if False use index to sample (default: True)
        """
        self.stage = stage
        self.clip = clip
        self.random = random
        self.use_pred_shd = use_pred_shd
        self.color_shift = color_shift
        self.augment = augment
        self.use_normals = use_normals
        self.ignore_list = ignore_list

        if self.stage == 1 and self.augment:
            self.kern = np.outer(signal.gaussian(384, 40), signal.gaussian(384, 40))

        # if we are doing the grayscale shading portion of the pipeline
        # just make sure that use_pred_shd is False to avoid accidental weirdness
        if self.stage in [0, 1]:
            self.use_pred_shd = False

        if self.stage in [0, 1, 2] and self.augment:
            print("adding color shift augmentation for shading chroma estimation")
            self.color_shift = True

        # we assume the subclasses will implement this function, it takes
        # in the root dir of the dataset and returns the set of inputs images
        self.file_list = self.populate_file_list()


        if is_subset:
            original_len = len(self.file_list)
            for i in range(len(self.file_list) - 1, -1, -1):
                item = self.file_list[i]
                if not os.path.exists(
                    os.path.join(self.root_dir, item + ".ord_shd.png")
                ):
                    self.file_list.pop(i)
            new_len = len(self.file_list)
            print(
                f"Removed {original_len - new_len} items from the dataset, {new_len} remaining"
            )

        # now remove the ignore list from the file list
        for item in self.ignore_list:
            if item in self.file_list:
                self.file_list.remove(item)
                print(
                    f"removed {item} from the dataset because it's in the ignore list"
                )

        self.cache_p = cache_prb
        self.cache = deque(maxlen=cache_len)

        self.dataset_len = len(self.file_list)

    # this function should take have any functionality to create a list of all
    # the input images in a dataset (or even all the unique image IDs, etc.)
    def populate_file_list(self):
        raise NotImplementedError

    # this function should return the input, albedo and mask given a single entry from the
    # file list that is returned by populate_file_list. The outputs should be tensors
    def load_inp_alb_msk(self, fname):
        raise NotImplementedError

    # this function converts an entry from the file list into the name of the corresponding
    # gray shading filename, this will be different for each dataset
    def get_shd_fname(self, fname):
        raise NotImplementedError

    # this function converts an entry from the file list into the name of the corresponding
    # ordinal shading filename, this will be different for each dataset
    def get_ord_fname(self, fname):
        raise NotImplementedError

    # this function converts an entry from the file list into the name of the corresponding
    # predicted normals filename, this will be different for each dataset
    def get_nrm_fname(self, fname):
        raise NotImplementedError

    # this function converts an entry from the file list into the name of the corresponding
    # predicted albedo filename, this will be different for each dataset
    def get_alb_fname(self, fname):
        raise NotImplementedError

    # this function uses the dataset specific code to get the predicted shading filename
    # and then runs the logic to load this shading layer
    def load_pred_shd(self, fname):
        pred_shd_fname = self.get_shd_fname(fname)

        pred_shd = load_image(pred_shd_fname)

        if self.stage == 2:
            # if we are in stage 2 and the pred shd has color, just take first channel
            # otherwise, the image is a single channel and we return that
            if len(pred_shd.shape) == 3:
                return torch.from_numpy(pred_shd[:, :, 0]).unsqueeze(0)
            else:
                return torch.from_numpy(pred_shd).unsqueeze(0)

        if self.stage == 3:
            # if there are three channels we have to permute to torch format, otherwise
            # raise an error because stage 3 requires shading with color
            if len(pred_shd.shape) == 3:
                # convert the IUV shading to RGB shading while it's still a numpy array
                pred_shd = iuv2rgb(pred_shd)
                return torch.from_numpy(pred_shd).permute(2, 0, 1)
            else:
                raise Exception(
                    "dataset in stage two but predicted shading has no chroma channels!"
                )

    def load_ord_shd(self, fname):
        # the ordinal shading is stored as an RGB image but it's actually just three shading
        # estimations at different resolutions stacked together, so we can treat it like an image
        ord_shd_fname = self.get_ord_fname(fname)

        ord_shd = load_image(ord_shd_fname)
        return torch.from_numpy(ord_shd).permute(2, 0, 1)

    def load_pred_nrm(self, fname):
        pred_nrm_name = self.get_nrm_fname(fname)

        pred_nrm = load_image(pred_nrm_name)
        return torch.from_numpy(pred_nrm).permute(2, 0, 1)

    def load_pred_alb(self, fname):
        pred_alb_name = self.get_alb_fname(fname)

        pred_alb = load_image(pred_alb_name)
        return torch.from_numpy(pred_alb).permute(2, 0, 1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # only use caching if
        #   a) cache probability isn't zero
        # and
        #   b) self.random is True (i.e. we aren't doing sampling)
        use_caching = (self.cache_p != 0.0) and self.random

        # only sample from cache if it's populated and with a specified probability
        pop_cache = (len(self.cache) > 0) and (random.uniform(0, 1) < self.cache_p)

        if False:
            cached = random.choice(self.cache)

            inp = cached["inp"]
            alb = cached["alb"]
            msk = cached["msk"]
            pred_shd = cached["pred_shd"]
            ord_shd = cached["ord_shd"]
            dif_shd = cached["dif_shd"]
            pred_nrm = cached["pred_nrm"]
            fname = cached["fname"]

            source = "cache"

        # if we aren't using a cached datapoint, so either sample one or use the passed idx
        else:
            source = "disk"


            file_idx = idx
            fname = self.file_list[file_idx]


            try:
                if self.stage == 4:
                    inp, alb, msk, dif_shd = self.load_inp_alb_msk(fname)
                else:
                    inp, alb, msk = self.load_inp_alb_msk(fname)
                    dif_shd = torch.ones_like(msk)
            except Exception as e:
                print(f"\033[91mfailed to process {fname}: {e}\033[0m")
                return {
                    "input": torch.tensor(-1),
                    "fname": fname,
                }

            # if we are using the predicted shading, load it from the disk
            # if not just use a dummy image so that the code can stay the same
            if self.use_pred_shd:

                # if stage 4, then we load the predicted albedo and then just divide to get the shading
                if self.stage == 4:
                    pred_alb = self.load_pred_alb(fname)
                    pred_shd = inp / pred_alb.clip(1e-4)
                else:
                    pred_shd = self.load_pred_shd(fname)
            else:
                if self.stage in [0, 1, 2]:
                    pred_shd = torch.ones((1, inp.shape[0], inp.shape[1]))
                elif self.stage in [3, 4]:
                    pred_shd = torch.ones((3, inp.shape[0], inp.shape[1]))

            if self.stage == 1:
                try:
                    ord_shd = self.load_ord_shd(fname)
                    # TODO: these are coming out of the datasets as different types (float vs uint8)
                    # it's fixed in the common dataloder code below, but not sure why it's happening
                    int_msk = (msk == 1).to(torch.uint8)
                    alb = match_albedo(inp, alb, ord_shd, int_msk)
                except Exception as e:
                    print(f"\033[91mfailed to process {fname}: {e}\033[0m")
                    return {
                        "input": torch.tensor(-1),
                        "fname": fname,
                    }
            else:
                ord_shd = torch.ones_like(inp)

            if self.use_normals:
                pred_nrm = self.load_pred_nrm(fname)
            else:
                pred_nrm = torch.ones_like(inp)

            # since we sampled a new datapoint, add it to the cache
            # since the cache is a deque, it will automatically pop the oldest datapoint
            datapoint = {
                "inp": inp,
                "alb": alb,
                "msk": msk,
                "pred_shd": pred_shd,
                "ord_shd": ord_shd,
                "dif_shd": dif_shd,
                "pred_nrm": pred_nrm,
                "fname": fname,
            }

            self.cache.append(datapoint)

        if self.augment:
            if self.color_shift and random.uniform(0, 1) < 0.5:
                inp = random_color_shift(inp)

            # now do the random resize and crop for the input, shading and mask
            # do random flip with 50% probability
            inp, alb, pred_shd, ord_shd, dif_shd, pred_nrm, msk = (
                random_crop_and_resize(
                    [inp, alb, pred_shd, ord_shd, dif_shd, pred_nrm, msk]
                )
            )
            inp, alb, pred_shd, ord_shd, dif_shd, pred_nrm, msk = random_flip(
                [inp, alb, pred_shd, ord_shd, dif_shd, pred_nrm, msk], mode="h"
            )
            inp, alb, pred_shd, ord_shd, dif_shd, pred_nrm, msk = random_flip(
                [inp, alb, pred_shd, ord_shd, dif_shd, pred_nrm, msk], mode="v"
            )

        # fix any fractional values caused by up or down sampling
        msk = msk > 0.5

        alb = alb.clip(0.0001)

        # first we compute the original implied shading (gt colorful shading)
        gt_clr_shd = inp / alb

        # this is the ground truth shading without color
        gt_gry_shd = get_brightness(gt_clr_shd, mode="torch")

        # in stage one we load gray shading and bake colors into albedo
        if self.stage in [0, 1, 2]:
            if self.use_pred_shd:
                pred_gry_shd = uninvert(pred_shd)
            else:
                pred_gry_shd = gt_gry_shd

            pred_gry_alb = inp / pred_gry_shd.clip(1e-4)

        # in stage three we load our gry+chroma shading and convert to rgb
        # then we divide it out of the image to get our imperfect albedo
        elif self.stage in [3, 4]:
            if self.use_pred_shd:
                pred_clr_shd = pred_shd
            else:
                pred_clr_shd = gt_clr_shd

            pred_clr_alb = inp / pred_clr_shd.clip(1e-4)

        # clip large values in the shading layer
        hi_shd_msk = (gt_gry_shd < self.clip).all(dim=0, keepdims=True)
        msk = msk & hi_shd_msk

        # in case anything gets messed up during albedo augmentation
        msk = msk & (alb > MIN_VAL).any(dim=0, keepdims=True).bool()

        # make the mask numeric for loss computations
        msk = msk.float()

        # return a dictionary with the keys expected in each stage
        out_dict = {}
        out_dict["input"] = inp
        out_dict["mask"] = msk
        out_dict["fname"] = fname

        if self.stage == 0:
            # since use_pred_shd is False, these will actually just be the
            # ground truth gray shading and albedo with colors baked in
            out_dict["gt_shd"] = invert(pred_gry_shd)
            out_dict["gt_alb"] = pred_gry_alb

        elif self.stage == 1:
            ord_base = ord_shd[0, :, :].unsqueeze(0)
            ord_full = ord_shd[1, :, :].unsqueeze(0)

            if self.augment and random.uniform(0, 1) < 0.25:
                ord_full = random_degrade(ord_full, self.kern)

            out_dict["ord_base"] = ord_base
            out_dict["ord_full"] = ord_full
            out_dict["gt_shd"] = invert(pred_gry_shd)
            out_dict["gt_alb"] = pred_gry_alb

        # if we are in stage 2, we have the grayscale shading inputs
        # and just the ground-truth albedo, and colorful shading
        elif self.stage == 2:
            out_dict["in_gry_shd"] = invert(pred_gry_shd)
            out_dict["in_gry_alb"] = pred_gry_alb
            out_dict["gt_alb"] = alb
            out_dict["gt_shd"] = invert(gt_clr_shd)

        # if we are in stage 3, we use the imperfect clr shading and albedo
        # and try to estimate accurate ground-truth albedo. We need to do some scale matching
        # here to make sure the albedo can be easily estimated by the network
        elif self.stage == 3:

            if random.uniform(0, 1) > 0.5:
                # roughly match the scale by matching the medians of pred and gt
                gt_med = np.median(alb)
                pred_med = np.median(pred_clr_alb)

                if pred_med > 0.05 and gt_med > 0.05:
                    pred_clr_alb = match_scale(
                        to2np(pred_clr_alb),
                        to2np(alb),
                        mask=to2np(torch.cat([msk] * 3, 0)).astype(bool),
                    )
                    pred_clr_alb = torch.from_numpy(pred_clr_alb).permute(2, 0, 1)

                    # now recompute the implied shading layers for the new albedos
                    pred_clr_shd = inp / pred_clr_alb.clip(1e-3)
                else:
                    out_dict["mask"] = torch.zeros_like(msk)
            else:
                if random.uniform(0, 1) > 0.5:
                    pred_clr_shd = F.gaussian_blur(gt_clr_shd, 3)
                    pred_clr_alb = inp / pred_clr_shd.clip(1e-3)
                else:
                    pred_clr_alb = F.gaussian_blur(alb, 3)
                    pred_clr_shd = inp / alb.clip(1e-3)


            out_dict["in_clr_shd"] = invert(pred_clr_shd)
            out_dict["in_clr_alb"] = pred_clr_alb
            out_dict["gt_alb"] = alb
            out_dict["gt_shd"] = invert(gt_clr_shd)

        elif self.stage == 4:
            out_dict["in_clr_shd"] = invert(pred_clr_shd)
            out_dict["in_clr_alb"] = pred_clr_alb
            out_dict["gt_dif_shd"] = invert(dif_shd)

            if self.use_normals:
                out_dict["nrm"] = pred_nrm

        out_dict["fname"] = fname
        return out_dict


# DATASET SPECIFIC CODE ==============================================================


class HypersimColorfulDataset(GenericColorfulDataset):
    def __init__(self, root_dir, split_file, stage, **kwargs):
        self.root_dir = root_dir
        self.split_file = split_file
        self.stage = stage
        super().__init__(stage, **kwargs)

    def populate_file_list(self):
        # load the split file to determine the set of images to load
        with open(self.split_file, "rb") as f:
            file_list = pickle.load(f)

        return file_list

    def get_shd_fname(self, fname):
        return f"{self.root_dir}/{fname}.gry_shd.png"

    def get_ord_fname(self, fname):
        return f"{self.root_dir}/{fname}.ord_shd.png"

    def get_nrm_fname(self, fname):
        return f"{self.root_dir}/{fname}.pred_nrm.png"

    def get_alb_fname(self, fname):
        return f"{self.root_dir}/{fname}.pred_alb.png"

    def load_inp_alb_msk(self, fname):
        try:
            hdr_alb = load_h5_image(f"{self.root_dir}/{fname}.diffuse_reflectance.hdf5")
            hdr_inp = load_h5_image(f"{self.root_dir}/{fname}.color.hdf5")
        except:
            print(f"failed to read: {self.root_dir}/{fname}.color.hdf5")

        hdr_alb = hdr_alb.astype(np.float32)
        hdr_inp = hdr_inp.astype(np.float32)

        # first clip any zeros in the albedo
        hdr_alb = hdr_alb.clip(0.0001)

        # get the tonemap scale as defined by the hypersim repo (see chrislib docs for details)
        inp_tm_scale = get_tonemap_scale(hdr_inp)

        # tonemap both the albedo and the image using this scale
        inp = hdr_inp * inp_tm_scale
        alb = hdr_alb * inp_tm_scale

        # scale the albedo between 0-1 by dividing by the max, this won't lose
        # information from clipping, but could due to quantization/percision
        # should be negligble though as the albedo is already near 0-1 range
        max_alb = alb.max()
        alb /= max_alb

        if self.stage == 4:
            hdr_dif = load_h5_image(
                f"{self.root_dir}/{fname}.diffuse_illumination.hdf5"
            )
            hdr_dif = hdr_dif.astype(np.float32)
            hdr_dif = np.nan_to_num(hdr_dif)
            hdr_dif *= max_alb

        # clip the input image, this does lose information obviously
        # but is required to create an ldr input image
        inp = inp.clip(0, 1)

        # any pixels that have a very low value across all channels
        # should be masked out, we expand these regions a bit to be safe
        msk = (alb > MIN_VAL).any(axis=-1, keepdims=True).astype(np.uint8)

        # erode the mask to ensure invalid pixels on edges are fully masked out
        kernel = np.ones((5, 5), np.uint8)
        msk = cv2.erode(msk, kernel, iterations=1)

        inp = torch.from_numpy(inp).permute(2, 0, 1)
        alb = torch.from_numpy(alb).permute(2, 0, 1)
        msk = torch.from_numpy(msk).unsqueeze(0)

        if self.stage == 4:
            dif = torch.from_numpy(hdr_dif).permute(2, 0, 1)
            return (inp, alb, msk, dif)

        else:
            return (inp, alb, msk)


class MINoAugmentationDataset(GenericColorfulDataset):
    def __init__(self, root_dir, stage, **kwargs):
        self.root_dir = root_dir
        super().__init__(stage, **kwargs)

    def populate_file_list(self):
        file_list = glob(f"{self.root_dir}/*/dir_*_mip2.exr")
        return file_list

    def get_ord_fname(self, fname):
        stem = Path(fname).name
        idx = stem.split("_")[1]
        return fname.replace(stem, f"{idx}_ord_shd.png")

    def get_shd_fname(self, fname):
        stem = Path(fname).name
        idx = stem.split("_")[1]
        return fname.replace(stem, f"{idx}_gry_shd.png")

    def load_inp_alb_msk(self, fname):
        stem = Path(fname).name
        hdr_inp = load_exr(fname)
        hdr_alb = load_exr(fname.replace(stem, "albedo.exr"))

        # get the tonemap scale as defined by the hypersim repo (see chrislib docs for details)
        inp_tm_scale = get_tonemap_scale(hdr_inp)

        # tonemap both the albedo and the image using this scale
        inp = hdr_inp * inp_tm_scale
        alb = hdr_alb * inp_tm_scale

        # scale the albedo between 0-1 by dividing by the max, this won't lose
        # information from clipping, but could due to quantization/percision
        # should be negligble though as the albedo is already near 0-1 range
        alb /= alb.max()

        # clip the input image, this does lose information obviously
        # but is required to create an ldr input image
        inp = inp.clip(0, 1)

        alb = alb.clip(0.0001)

        # any pixels that have a very low value across all channels
        # should be masked out, we expand these regions a bit to be safe
        msk = (alb > MIN_VAL).any(axis=-1, keepdims=True).astype(np.uint8)

        # erode the mask to ensure invalid pixels on edges are fully masked out
        kernel = np.ones((5, 5), np.uint8)
        msk = cv2.erode(msk, kernel, iterations=1)

        inp = torch.from_numpy(inp).permute(2, 0, 1)
        alb = torch.from_numpy(alb).permute(2, 0, 1)
        msk = torch.from_numpy(msk).unsqueeze(0)

        return (inp, alb, msk)


class multiple_dataset_sampler:
    def __init__(self, num_datasets, total_number_of_data, probs, datasets=None):
        self.probs = probs
        self.num_datasets = num_datasets
        self.length = total_number_of_data
        self.datasets = datasets

        assert len(self.probs) == self.num_datasets

    def get_dataset_indices(self, debug=False):
        if debug:
            self.dataset_indices = []
            for i in range(self.num_datasets):
                # add the index of dataset by the lendth of the dataset
                self.dataset_indices += [i] * len(self.datasets[i])
        else:
            self.dataset_indices = random.choices(
                list(range(self.num_datasets)), weights=self.probs, k=self.length
            )
        return self.dataset_indices


class CustomMultiDataset(Dataset):
    def __init__(
        self,
        datasets,
        datasets_indices,
        total_num_requested_data,
        use_cashed_data=False,
        save_cashed_data=False,
    ):
        if use_cashed_data:
            save_cashed_data = True
        self.datasets = datasets
        self.total_num_requested_data = total_num_requested_data
        self.datasets_indices = datasets_indices
        # now we need to first shuffle the file_list whiting each dataset; then we need pick
        # sample indices in the same order as datasets_indices

        tmp_point_indices_per_dataset = []
        for db_index, db in enumerate(self.datasets):
            random.shuffle(db.file_list)
            num_point_we_need_to_pick = self.datasets_indices.count(db_index)
            tmp_point_indices_per_dataset.append(num_point_we_need_to_pick)

        self.db_data_indices = []
        for i, db_index in enumerate(self.datasets_indices):
            self.db_data_indices.append(
                (db_index, tmp_point_indices_per_dataset[db_index] - 1)
            )
            tmp_point_indices_per_dataset[db_index] -= 1
        self.z_codes = None
        self.use_cashed_data = use_cashed_data
        self.save_cashed_data = save_cashed_data
        self.cashed_data = []

    def __len__(self):
        return self.total_num_requested_data

    def set_z_codes(self, z_codes):
        self.z_codes = z_codes

    def set_use_cashed_data(self):
        self.use_cashed_data = True
        

    def __getitem__(self, idx):
        if self.use_cashed_data:
            datapoint = self.cashed_data[idx]
            z_c = self.z_codes[idx]
            return datapoint, z_c
        else:
            cur_dset, sample_idx = self.db_data_indices[idx]
            datapoint = self.datasets[cur_dset][sample_idx]
            z_c = torch.tensor([0])
            if self.save_cashed_data:
                self.cashed_data.append(datapoint)
            return datapoint, z_c

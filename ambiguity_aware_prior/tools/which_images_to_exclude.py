import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from dataset.custom_datasets import *
from models.tools import *

STAGE = 1

args = get_args()

ignore_list = [
    "ai_035_010/images/scene_cam_00_final_hdf5/frame.0007",
    "ai_016_001/images/scene_cam_01_final_hdf5/frame.0095",
    "ai_048_007/images/scene_cam_01_final_hdf5/frame.0097",
    "ai_051_002/images/scene_cam_04_final_hdf5/frame.0095",
    "ai_044_005/images/scene_cam_01_final_hdf5/frame.0055",
    "ai_048_009/images/scene_cam_00_final_hdf5/frame.0051",
    "ai_047_002/images/scene_cam_00_final_hdf5/frame.0034",
    "ai_029_001/images/scene_cam_01_final_hdf5/frame.0006",
    "ai_051_002/images/scene_cam_06_final_hdf5/frame.0022",
    "ai_016_001/images/scene_cam_01_final_hdf5/frame.0088",
    "ai_005_007/images/scene_cam_00_final_hdf5/frame.0075",
    "ai_051_005/images/scene_cam_02_final_hdf5/frame.0087",
    "ai_036_003/images/scene_cam_01_final_hdf5/frame.0029",
    "ai_055_002/images/scene_cam_01_final_hdf5/frame.0025",
    "ai_008_009/images/scene_cam_00_final_hdf5/frame.0062",
    "ai_031_009/images/scene_cam_00_final_hdf5/frame.0014",
    "ai_003_005/images/scene_cam_01_final_hdf5/frame.0023",
    "ai_048_003/images/scene_cam_01_final_hdf5/frame.0056",
    "ai_001_002/images/scene_cam_01_final_hdf5/frame.0027",
    "ai_019_007/images/scene_cam_00_final_hdf5/frame.0092",
    "ai_035_003/images/scene_cam_00_final_hdf5/frame.0006",
    "ai_052_010/images/scene_cam_00_final_hdf5/frame.0082",
]
train_datasets = [
    MINoAugmentationDataset(
        args.MID_PATH,
        stage=STAGE,
        clip=args.clip,
        cache_prb=0.0,
        cache_len=0,
        ignore_list=ignore_list,
    ),
    HypersimColorfulDataset(
        args.HYPERSIM_PATH,
        "splits/skimmed_train_list.p",
        stage=STAGE,
        clip=args.clip,
        cache_prb=0.0,
        cache_len=0,
        ignore_list=ignore_list,
        is_subset=args.is_subset,
    ),
]
TRN_DATASET_PROBS = [0.5, 0.5]

TRN_DATASET_PROBS = [x / sum(TRN_DATASET_PROBS) for x in TRN_DATASET_PROBS]
number_of_z_codes = sum([len(dataset) for dataset in train_datasets])
subset_train_between_2_interval_sampler = multiple_dataset_sampler(
    num_datasets=len(train_datasets),
    total_number_of_data=number_of_z_codes,
    probs=TRN_DATASET_PROBS,
    datasets=train_datasets,
)
subset_train_between_2_interval_dataset = CustomMultiDataset(
    datasets=train_datasets,
    datasets_indices=subset_train_between_2_interval_sampler.get_dataset_indices(
        debug=True
    ),
    total_num_requested_data=number_of_z_codes,
)
subset_train_between_2_interval_dataloader = DataLoader(
    subset_train_between_2_interval_dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
)
bad_batches = []
start = time.time()
for i, data in enumerate(subset_train_between_2_interval_dataloader):
    in_img = data["input"].to(args.device, non_blocking=True)
    if in_img.size() == torch.tensor([-1]).size():
        print("bad batch, skipping")
        bad_batches.append(data["fname"])
    else:
        ord_base = data["ord_base"].to(args.device, non_blocking=True)
        ord_full = data["ord_full"].to(args.device, non_blocking=True)
        gt_shd = data["gt_shd"].to(args.device, non_blocking=True)
        gt_alb = data["gt_alb"].to(args.device, non_blocking=True)
        msk = data["mask"].to(args.device, non_blocking=True)

        if msk.sum() == 0:
            print("bad batch, data:", data["fname"])
            bad_batches.append(data["fname"])

        if i % 100 == 0:
            print(i)
            print(time.time() - start, "seconds")
            start = time.time()

print("--------------------")
print("bad batches:")
print(bad_batches)


# python which_images_to_exclude.py --ordinal_path ./checkpoints/rendered_only_ord_model_latest_by_chris.pt --weights_path ./experiments/ --checkpoint paper_weights --MID_PATH ./datasets/MIDIntrinsics_5K --HYPERSIM_PATH ./datasets/Hypersim_subset_5K --is_subset --exp_name debug --batchsize 2 --workers 0 --train_iters 10

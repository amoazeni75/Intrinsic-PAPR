from torch.utils.data import DataLoader

from .dataset import RINDataset


def get_traindataset(
    dataset_args,
    scene_config,
    use_albedo,
    debug,
):
    return RINDataset(
        dataset_args=dataset_args,
        scene_config=scene_config,
        mode="train",
        use_albedo=use_albedo,
        debug=debug,
    )


def get_trainloader(dataset, dataset_args):
    return DataLoader(
        dataset,
        batch_size=dataset_args.batch_size,
        shuffle=dataset_args.shuffle,
        num_workers=dataset_args.num_workers,
        pin_memory=True,
    )


def get_testdataset(
    dataset_args,
    scene_config,
    use_albedo,
    debug,
):
    return RINDataset(
        dataset_args=dataset_args,
        scene_config=scene_config,
        mode="test",
        use_albedo=use_albedo,
        debug=debug,
    )


def get_testloader(dataset, dataset_args):
    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=dataset_args.num_workers
    )


def get_dataset(
    dataset_args,
    scene_config,
    mode,
    use_albedo,
    debug,
):
    if mode == "train":
        return get_traindataset(
            dataset_args=dataset_args,
            scene_config=scene_config,
            use_albedo=use_albedo,
            debug=debug,
        )
    elif mode == "test":
        return get_testdataset(
            dataset_args=dataset_args,
            scene_config=scene_config,
            use_albedo=use_albedo,
            debug=debug,
        )
    else:
        raise ValueError("Unknown mode: {}".format(mode))


def get_loader(dataset, dataset_args, mode):
    if mode == "train":
        return get_trainloader(dataset, dataset_args)
    elif mode == "test":
        return get_testloader(dataset, dataset_args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

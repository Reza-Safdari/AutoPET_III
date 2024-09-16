import os
import pandas as pd

from monai.data import Dataset
from torch.utils.data import DataLoader
from monai.data import partition_dataset, CacheDataset

from .transforms import get_transforms
from .dataset import RandomPatientDataset


def read_splits(splits_file, kfold, include_negatives=True):
    data_df = pd.read_csv(splits_file)

    if not include_negatives:
        data_df = data_df[data_df['Diagnosis'] != 'NEGATIVE']

    fold = 0 if kfold == 'all' else int(kfold)

    splits = dict({})
    splits["train"] = data_df[data_df['Fold'] != fold]
    splits["valid"] = data_df[data_df['Fold'] == fold]

    splits["train"] = list(splits["train"]["Study ID"])
    splits["valid"] = list(splits["valid"]["Study ID"])

    if kfold == 'all':
        splits['train'] += splits['valid']

    return splits


def get_file_dict(root, split, use_cascading=False):
    data = [
        {
            "ct_vol": os.path.join(root, 'imagesTr', f"{file_name}_0000.nii.gz"),
            "pet_vol": os.path.join(root, 'imagesTr', f"{file_name}_0001.nii.gz"),
            "mask": os.path.join(root, 'labelsTr', f"{file_name}.nii.gz"),
            **({"pred_mask": os.path.join(root, 'predMasks', f"{file_name}_pred.nii.gz")} if use_cascading else {})
        }
        for file_name in split
    ]
    return data


def partition_data_between_processes(data_list, shuffle, seed, rank, world_size):
    return partition_dataset(
        data=data_list,
        num_partitions=world_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
        even_divisible=shuffle
    )[rank]


def build_loaders(cfg, logging, rank=0, world_size=1):
    splits = read_splits(cfg.metadata_path, cfg.fold, cfg.include_negatives)
    if cfg.suffix == '.nii.gz':
        train_files = get_file_dict(cfg.data_dir, splits['train'], cfg.use_cascading)
    else:
        train_files = splits['train']
    valid_files = get_file_dict(cfg.data_dir, splits['valid'], cfg.use_cascading)
    train_files = partition_data_between_processes(train_files, True, cfg.seed, rank, world_size)
    valid_files = partition_data_between_processes(valid_files, False, cfg.seed, rank, world_size)
    train_files = [
        {
            "ct_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000.nii.gz",
            "pet_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0001.nii.gz",
            "mask": "data/AutoPET_III/labelsTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333.nii.gz",
            "pred_mask": "data/AutoPET_III/predMasks/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000_pred.nii.gz"
        },
        {
            "ct_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000.nii.gz",
            "pet_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0001.nii.gz",
            "mask": "data/AutoPET_III/labelsTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333.nii.gz",
            "pred_mask": "data/AutoPET_III/predMasks/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000_pred.nii.gz"
        }
    ]

    valid_files = [
        {
            "ct_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000.nii.gz",
            "pet_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0001.nii.gz",
            "mask": "data/AutoPET_III/labelsTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333.nii.gz",
            "pred_mask": "data/AutoPET_III/predMasks/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000_pred.nii.gz"
        },
        {
            "ct_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000.nii.gz",
            "pet_vol": "data/AutoPET_III/imagesTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0001.nii.gz",
            "mask": "data/AutoPET_III/labelsTr/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333.nii.gz",
            "pred_mask": "data/AutoPET_III/predMasks/fdg_0cda25453b_10-12-2003-NA-PET-CT Ganzkoerper  primaer mit KM-47333_0000_pred.nii.gz"
        }
    ]

    train_transforms = get_transforms(
            "train",
            cfg.patch_size,
            spacing=cfg.spacing,
            resample=True,
            patch_per_sample=cfg.patch_per_sample,
            use_cascading=cfg.use_cascading,
    )
    valid_transforms = get_transforms(
        "val",
        cfg.patch_size,
        spacing=cfg.spacing,
        resample=True,
        use_cascading=cfg.use_cascading
    )

    train_dataset = None
    if cfg.suffix == '.npz':
        train_dataset = RandomPatientDataset(cfg.data_dir, split=splits['train'])
    elif cfg.suffix == '.nii.gz':
        # train_dataset = Dataset(train_files, transform=train_transforms)
        train_dataset = CacheDataset(
            train_files,
            transform=train_transforms,
            cache_rate=cfg.train_cache_rate,
            num_workers=cfg.num_workers_train,
        )
    valid_dataset = CacheDataset(
        valid_files,
        transform=valid_transforms,
        cache_rate=1.0,
        num_workers=cfg.num_workers_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers_train,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers_val,
        pin_memory=True
    )

    logging.info(f"{rank}: Number of training samples: {len(train_dataset)}")
    logging.info(f"{rank}: Number of validation samples: {len(valid_dataset)}")
    return train_loader, valid_loader

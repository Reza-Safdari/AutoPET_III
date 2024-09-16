from easydict import EasyDict

config = EasyDict()

config.seed = 42
config.run_dir = None
config.dataset_name = "AutoPET"  # Possible values AutoPET, Spleen
config.data_dir = "data/AutoPET_III"
config.metadata_path = "data/metadata/metadata.csv"
config.suffix = ".nii.gz"  # .nii.gz, .npz
config.fold = 0  # 0-4 and all
config.include_negatives = False
config.model = "SwinUNETR"  # Possible values: UNet, SegResNet, DynUNet
config.use_cascading = True
# config.cascaded_models = "SegResNet"
config.loss = "DiceCELoss"
config.deep_supervision = True
config.optimizer = "AdamW"
config.lr = 1e-4
config.weight_decay = 1e-5
config.momentum = 0.99
config.nesterov = True
config.scheduler_name = "CosineAnnealingLR"
config.resume = ""
config.batch_size = 4
config.train_cache_rate = 1.0
config.num_workers_train = 15
config.num_workers_val = 15
config.max_epochs = 200
config.patch_size = (96, 96, 96)
config.spacing = (2.0364201068878174, 2.0364201068878174, 3.0)
config.use_random_spacing = False
config.patch_per_sample = 1
config.sw_batch_size = 2
config.sw_overlap = 0.5
config.sw_mode = "constant"
config.log_freq = 1
config.val_freq = 1


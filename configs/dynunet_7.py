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
config.model = "DynUNet"  # Possible values: UNet, SegResNet, DynUNet
config.loss = "DiceCELoss"
config.deep_supervision = True
config.optimizer = "SGD"
config.lr = 1e-3
config.weight_decay = 3e-5
config.momentum = 0.99
config.nesterov = True
config.scheduler_name = "PolyLR"
config.resume = "weights/ckpt_00492_dice0.5706.pth"
config.batch_size = 6
config.num_workers_train = 15
config.num_workers_val = 15
config.max_epochs = 774
config.patch_size = (128, 160, 112)
config.spacing = (8.0, 8.0, 8.0)
config.patch_per_sample = 1
config.sw_batch_size = 8
config.sw_overlap = 0.5
config.sw_mode = "constant"
config.log_freq = 1
config.val_freq = 1


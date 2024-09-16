from easydict import EasyDict
config = EasyDict()

config.seed = 42
config.run_dir = None
config.dataset_name = "AutoPET"  # Possible values AutoPET, Spleen
config.data_dir = "data/AutoPET_III/patches"
config.metadata_path = "data/metadata/sub_metadata.csv"
config.suffix = ".npz"  # .nii.gz, .npz
config.fold = 0  # 0-4 and all
config.include_negatives = False
config.model = "UxLSTMEnc_3d"  # Possible values: UNet, SegResNet, DynUNet
config.loss = "DiceCELoss"
config.deep_supervision = True
config.optimizer = "SGD"
config.lr = 1e-3
config.weight_decay = 3e-5
config.momentum = 0.99
config.nesterov = True
config.scheduler_name = "PolyLR"
config.resume = ""
config.batch_size = 2
config.num_workers_train = 15
config.num_workers_val = 10
config.max_epochs = 150
config.patch_size = (128, 160, 112)
config.patch_per_sample = 1
config.sw_batch_size = 4
config.sw_overlap = 0.5
config.sw_mode = "constant"
config.log_freq = 1
config.val_freq = 1


import os
import sys
import importlib
import logging
import datetime
import random

from monai.config import print_config

import torch


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = os.path.basename(config_file)
    temp_module_name = os.path.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.%s" % temp_module_name)
    cfg = config.config
    if cfg.run_dir is None:
        cfg.run_dir = os.path.join('runs', cfg.dataset_name, temp_module_name)
    c = 1
    run_dir = cfg.run_dir
    while os.path.exists(run_dir):
        run_dir = cfg.run_dir + f"_{c}"
        c += 1
    cfg.run_dir = run_dir
    cfg.use_cascading = cfg.get("use_cascading", False)
    cfg.spacing = cfg.get("spacing", (2.0364201068878174, 2.0364201068878174, 3.0))
    cfg.use_random_spacing = cfg.get("use_random_spacing", False)
    if cfg.use_random_spacing:
        random_spacing = random.uniform(2, 10)
        cfg.spacing = tuple([random_spacing] * 3)
    return cfg


def init_log(log_path):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    handler_file = logging.FileHandler(log_path, mode="a")
    handler_stream = logging.StreamHandler(sys.stdout)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)
    return log_root


def log_info(cfg):
    logging.info(f"Start training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 40)
    logging.info("Parameters: ")
    cfg_dict = cfg.__dict__
    longest_key = max(len(k) for k in cfg_dict.keys())
    for name, value in cfg_dict.items():
        logging.info(f"{name.ljust(longest_key)} = {value}")
    logging.info("-" * 40)
    print_config(logging.root.handlers[0].stream)
    logging.info("=" * 40)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveCheckpoint(object):
    def __init__(self, run_dir):
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, cfg, model, optimizer, epoch, dice):
        checkpoint_path = os.path.join(self.checkpoint_dir,
                                       f"ckpt_{epoch:05d}_dice{dice:.4f}.pth")
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        state = {
            'config': cfg.__dict__,
            'state_dict': model.module.state_dict(),
            'epoch': epoch,
            'dice': dice,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)



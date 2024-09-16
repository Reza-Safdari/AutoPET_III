import argparse
import logging
import os
import time

from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import interpolate

from training_utils.utils import (get_config, init_log, log_info,
                                  AverageMeter, SaveCheckpoint)

from models import build_model
from losses import Loss
from training_utils.optimizers import build_optimizer
from training_utils.schedulers import build_scheduler
from training_utils.metrics import AutoPETMetricAggregator
from data import build_loaders

# initialize the distributed training process, every GPU runs in a process
try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    world_size = 1
    rank = 0
    local_rank = 0

dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)


def train_batch(batch_data, model, criterion, optimizer, scaler, device, model_name):
    model.train()

    start = time.time()
    optimizer.zero_grad()
    images = batch_data[0].to(device)
    gt_masks = batch_data[1].to(device)

    with torch.amp.autocast('cuda'):
        prediction, vae_loss = model(images) if model_name == "segresnetvae" else (model(images), 0.0)
        if isinstance(prediction, list):
            out_shape = prediction[0].shape[2:]
            prediction = [interpolate(pred, out_shape) if pred.shape[2:] != out_shape else pred for pred in prediction]
            prediction = torch.stack(prediction, dim=1)
        loss = criterion.compute_loss(prediction, gt_masks) + 0.5 * vae_loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    lr = optimizer.param_groups[0]['lr']
    return loss.item(), vae_loss, lr, time.time() - start


def evaluate(val_loader, model, sw_size, sw_batch_size, sw_overlap, sw_mode, device, model_name):
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)
    confusion = ConfusionMatrixMetric(reduction="mean", metric_name="f1 score")
    # test_aggregator = AutoPETMetricAggregator()

    start = time.time()
    with torch.no_grad():
        for val_data in val_loader:
            with torch.amp.autocast('cuda'):
                images = val_data[0].to(device)
                gt_masks = val_data[1].to(device)
                print(f"Image Shape: {images.shape}")
                print(f"GT Mask Shape: {gt_masks.shape}")
                
                prediction = sliding_window_inference(
                    inputs=images,
                    roi_size=sw_size,
                    sw_batch_size=sw_batch_size,
                    predictor=lambda x: model(x)[0] if model_name == "segresnetvae" else model(x),
                    overlap=sw_overlap,
                    mode=sw_mode
                )
            prediction = torch.ge(torch.sigmoid(prediction), 0.5)
            dice_metric(y_pred=prediction, y=gt_masks)
            confusion(y_pred=prediction, y=gt_masks)
            # test_aggregator.update(prediction, gt_masks)

        mean_val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        mean_fp = confusion.aggregate()[0].item()
        confusion.reset()
        # fp_fn_dice = test_aggregator.compute()

    return mean_val_dice, mean_fp, None, time.time() - start


def main(cfg):
    # disable logging for processes except 0 on every node
    if rank != 0:
        writer = type('DummyWriter', (), {'add_scalar': lambda *a: None})()
        logging = type('DummyLogger', (), {'info': lambda *a: None})()
        save_checkpoint = type('DummySaveCheckpoint', (), {'save': lambda *a: None})()
    else:
        writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, "tensorboard"))
        logging = init_log(log_path=os.path.join(cfg.run_dir, "training.log"))
        save_checkpoint = SaveCheckpoint(cfg.run_dir)
        log_info(cfg)
        os.makedirs(cfg.run_dir, exist_ok=True)

    if cfg.seed is not None:
        # set deterministic training for reproducibility
        set_determinism(seed=cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # use amp to accelerate training
    scaler = torch.amp.GradScaler('cuda')

    # ------------- building data loaders -------------
    train_loader, valid_loader = build_loaders(cfg, logging, rank, world_size)

    # ---- building model, criterion and optimizer ----
    model = build_model(cfg.model, cfg.dataset_name, cfg.patch_size, cfg.use_cascading, cfg.deep_supervision).to(device)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device])
    criterion = Loss(cfg.loss, cfg.deep_supervision).to(device)
    optimizer = build_optimizer(
        cfg.optimizer,
        model.parameters(),
        cfg.lr,
        cfg.weight_decay,
        cfg.momentum,
        cfg.nesterov
    )
    lr_scheduler = build_scheduler(
        cfg.scheduler_name,
        optimizer,
        cfg.max_epochs
    )

    start_epoch = 1
    best_metric = -1
    best_metric_epoch = -1
    if cfg.get("resume", "") != "":
        if os.path.isfile(cfg.resume):
            logging.info(f"Loading checkpoint: '{cfg.resume}'")
            checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
            model.module.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_metric = checkpoint["dice"]
            best_metric_epoch = checkpoint["epoch"]
            logging.info(f"Loaded checkpoint '{cfg.resume}' @ {start_epoch} epoch.")
        else:
            logging.info("No checkpoint found at '{}'".format(cfg.resume))

    # ---------------- training loop -----------------
    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    vae_loss_meter = AverageMeter()
    for epoch in range(start_epoch, cfg.max_epochs):
        start_time_epoch = time.time()
        start_data = time.time()
        for step, batch_data in enumerate(train_loader, start=1):
            num_samples = batch_data[0].size(0)
            data_time.update(time.time() - start_data, num_samples)
            loss, vae_loss, lr, train_batch_time = train_batch(
                batch_data,
                model,
                criterion,
                optimizer,
                scaler,
                device,
                cfg.model.lower(),
            )
            loss_meter.update(loss, num_samples)
            vae_loss_meter.update(vae_loss, num_samples)
            batch_time.update(train_batch_time, num_samples)
            # ------------- logging train status ------------
            if step % cfg.log_freq == 0:
                logging.info(
                    f"Train Epoch [{epoch}/{cfg.max_epochs}] " +
                    f"Step [{step}/{len(train_loader)}]\t" +
                    f"Data-Time {data_time.avg:.3f} " +
                    f"Batch-Time {batch_time.avg:.3f} " +
                    f"Loss {loss_meter.avg:.5f} " +
                    f"VAE-Loss {vae_loss_meter.avg:.5f} " +
                    f"Learning-Rate {lr:.6f}"
                )
                current_step = (epoch - 1) * len(train_loader) + step
                writer.add_scalar("train/loss", loss_meter.avg, current_step)
                writer.add_scalar("train/learning_rate", lr, current_step)
                batch_time.reset()
                data_time.reset()
                loss_meter.reset()
                vae_loss_meter.reset()
            start_data = time.time()
        lr_scheduler.step()
        # -------------- validating model ---------------
        if epoch % cfg.val_freq == 0:
            logging.info("-" * 40)
            dice_metric, f1_metric, fg_fn_dicee, val_total_time = evaluate(
                valid_loader,
                model,
                cfg.patch_size,
                cfg.sw_batch_size,
                cfg.sw_overlap,
                cfg.sw_mode,
                device,
                cfg.model.lower(),
            )
            # --------- logging validation result ----------
            logging.info(
                f"Validation [{epoch}/{cfg.max_epochs}]\t" +
                f"Validation-Time {val_total_time:.3f} " +
                f"Mean-Dice {dice_metric:.4f} "
                f"F1 {f1_metric:.4f} ",
                # f"FP: {fg_fn_dicee['false_positives']} ",
                # f"FN: {fg_fn_dicee['false_negatives']}"
            )
            writer.add_scalar("valid/mean-dice", dice_metric, epoch)
            writer.add_scalar("valid/F1", f1_metric, epoch)

            if dice_metric > best_metric:
                best_metric = dice_metric
                best_metric_epoch = epoch
                save_checkpoint.save(cfg, model, optimizer, epoch, dice_metric)
                logging.info(f"Best Mean-Dice {best_metric:.4f} @ Epoch {best_metric_epoch}")
                logging.info("-" * 40)

        logging.info("-" * 40)
        logging.info(
            f"End of Epoch {epoch}, Total-Time (Train + Val): '{time.time() - start_time_epoch:.2f}' "
            f"Best Mean-Dice '{best_metric:.4f}' @ epoch '{best_metric_epoch}'."
        )
        logging.info("-" * 40)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoPET III Segmentation Training in Pytorch")
    parser.add_argument("--config", type=str, help="py config file")
    args = parser.parse_args()
    cfg = get_config(args.config)
    main(cfg)

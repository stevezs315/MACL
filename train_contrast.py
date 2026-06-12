import gc
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.BraTS import BraTS
from dataset.KiTS import KiTS
from dataset.chd import CHD
from experiment_log import PytorchExperimentLogger
from loss.contrast_loss import SupConLoss
from lr_scheduler import LR_Scheduler
from myconfig import get_config
from network.unet2d import UNet2D_classification_MACL
from utils import AverageMeter


SUPPORTED_DATASETS = ("chd", "BraTS", "KiTS")
epoch_context = ""


def is_main_process():
    return torch.distributed.get_rank() == 0

def setup_distributed(args):
    if args.local_rank == -1:
        raise ValueError("DDP launch did not provide --local-rank/LOCAL_RANK")

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")


def setup_seed(args):
    if args.seed is None:
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"seed: {args.seed}")


def build_model(args):
    in_channels = 4 if args.dataset == "BraTS" else 1
    return UNet2D_classification_MACL(
        in_channels=in_channels,
        initial_filter_size=args.initial_filter_size,
        kernel_size=3,
        classes=args.classes,
        scale_factor=args.scale_factor,
        do_instancenorm=True,
    )


def build_dataset(args):
    if args.dataset == "chd":
        training_keys = os.listdir(os.path.join(args.data_dir, "train"))
        training_keys.sort()
        return CHD(keys=training_keys, purpose="train", args=args)

    if args.dataset == "BraTS":
        training_keys = os.listdir(args.data_dir)
        training_keys.sort()
        return BraTS(keys=training_keys, args=args, dstw=192, dsth=192)

    if args.dataset == "KiTS":
        training_keys = os.listdir(os.path.join(args.data_dir, "imgs"))
        training_keys.sort()
        return KiTS(keys=training_keys, purpose="train", args=args)

    raise ValueError(f"Unsupported dataset {args.dataset!r}")


def build_dataloader(args):
    train_dataset = build_dataset(args)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_works,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, train_sampler


def build_save_context(args):
    if args.save == "":
        args.save = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    model_result_dir = os.path.join(save_path, "model")
    os.makedirs(model_result_dir, exist_ok=True)
    args.model_result_dir = model_result_dir

    logger = PytorchExperimentLogger(save_path, "elog", ShowTerminal=True)
    writer = SummaryWriter(os.path.join(args.runs_dir, args.experiment_name + args.save)) if is_main_process() else None
    return save_path, logger, writer


def save_checkpoint(model, optimizer, epoch, train_loss, args):
    checkpoint = {
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_lr": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1,
        "train_loss": train_loss,
    }
    path_checkpoint = os.path.join(args.model_result_dir, f"checkpoint_epoch{epoch + 1}.pkl")
    torch.save(checkpoint, path_checkpoint)


def save_model(model, args, filename):
    torch.save({"net": model.module.state_dict()}, os.path.join(args.model_result_dir, filename))


def train_MACL(data_loader, model, criterion_SupCon, criterion_L1, epoch, optimizer, scheduler, scaler, logger, args):
    model.train()
    losses = AverageMeter()

    for batch_idx, tup in tqdm(enumerate(data_loader), desc=epoch_context, total=len(data_loader), ncols=90):
        scheduler(optimizer, batch_idx, epoch)

        img1, img2, img3, img4, slice_position, _ = tup

        if not args.symmetric_net:
            img2 = F.interpolate(img2, scale_factor=args.scale_factor, mode="bilinear", align_corners=True)
            img4 = F.interpolate(img4, scale_factor=args.scale_factor, mode="bilinear", align_corners=True)

        img1[img1 != img1] = 0
        img2[img2 != img2] = 0
        img3[img3 != img3] = 0
        img4[img4 != img4] = 0

        image1_var = Variable(img1.float(), requires_grad=False).to(args.device)
        image2_var = Variable(img2.float(), requires_grad=False).to(args.device)
        image3_var = Variable(img3.float(), requires_grad=False).to(args.device)
        image4_var = Variable(img4.float(), requires_grad=False).to(args.device)
        bsz = img1.shape[0]

        with torch.cuda.amp.autocast(enabled=args.AMP):
            pixel_1_1, pixel_1_2, instance_1, pixel_2_1, pixel_2_2, instance_2 = model(
                image1_var,
                image2_var,
                image3_var,
                image4_var,
            )

            if not args.symmetric_net:
                pixel_1_down_sample = F.interpolate(
                    pixel_1_1,
                    scale_factor=args.scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                pixel_1_down_sample = pixel_1_1

            loss_ER = criterion_L1(pixel_1_down_sample, pixel_2_1)

            if args.pixel_use:
                batch_size, channels, height, width = pixel_1_2.size()
                f1_pixel = pixel_1_2.view(batch_size, channels, height * width).permute(2, 0, 1).contiguous()
                f2_pixel = pixel_2_2.view(batch_size, channels, height * width).permute(2, 0, 1).contiguous()

                if args.only_pixel:
                    loss_instance = None
                else:
                    features_instance = torch.cat([instance_1.unsqueeze(1), instance_2.unsqueeze(1)], dim=1)
                    loss_instance = criterion_SupCon(features_instance, labels=slice_position)

                loss_pixel = 0
                for pixel_index in range(height * width):
                    features_pixel = torch.cat(
                        [f1_pixel[pixel_index].unsqueeze(1), f2_pixel[pixel_index].unsqueeze(1)],
                        dim=1,
                    )
                    loss_pixel += criterion_SupCon(features_pixel, labels=slice_position)
                loss_pixel = loss_pixel / (height * width)

                if args.only_pixel:
                    loss = loss_pixel * args.alpha + loss_ER
                else:
                    loss = loss_instance + loss_pixel * args.alpha + loss_ER * args.alpha_ER
            else:
                features_instance = torch.cat([instance_1.unsqueeze(1), instance_2.unsqueeze(1)], dim=1)
                loss = criterion_SupCon(features_instance, labels=slice_position)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        losses.update(loss.item(), bsz)
        scaler.step(optimizer)
        scaler.update()

        if (batch_idx + 1) % len(data_loader) == 0 and is_main_process():
            logger.print(
                f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, "
                f"lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}"
            )

    return losses.avg


def main():
    args = get_config()
    setup_distributed(args)

    save_path, logger, writer = build_save_context(args)
    if is_main_process():
        logger.print(f"saving to {save_path}")
        logger.print("creating model ...")

    setup_seed(args)

    model = build_model(args)
    if is_main_process():
        logger.print(f"model: {model}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=args.find_unused_parameters,
    )

    if is_main_process():
        num_parameters = sum(param.nelement() for param in model.module.parameters())
        logger.print(f"use {torch.cuda.device_count()} gpus!")
        logger.print(f"number of parameters: {num_parameters}")
        logger.print(f"Parameters: {args}")

    train_loader, train_sampler = build_dataloader(args)

    criterion_SupCon = SupConLoss(
        threshold=args.slice_threshold,
        temperature=args.temp,
        contrastive_method=args.contrastive_method,
    ).to(args.device)
    criterion_L1 = nn.L1Loss(reduction="mean").to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.AMP)

    if is_main_process():
        logger.print(f"Optimizer_Original: {optimizer}")
        logger.print(f"Scheduler_Original: {scheduler}")

    best_loss = np.inf
    total_start_time = time.time()

    global epoch_context
    for epoch in range(args.epochs):
        gc.collect()
        torch.cuda.empty_cache()
        epoch_context = f"Epoch[{epoch}/{args.epochs}]"
        train_sampler.set_epoch(epoch)

        train_loss = train_MACL(
            train_loader,
            model,
            criterion_SupCon,
            criterion_L1,
            epoch,
            optimizer,
            scheduler,
            scaler,
            logger,
            args,
        )

        if is_main_process():
            logger.print(f"\n Epoch: {epoch + 1}\tTraining Loss {train_loss:.4f} \t")
            writer.add_scalar("training_loss", train_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if (epoch + 1) % args.checkpoint_pretrain_interval == 0:
                save_checkpoint(model, optimizer, epoch, train_loss, args)

            save_model(model, args, "latest.pth")
            if train_loss < best_loss:
                best_loss = train_loss
                save_model(model, args, "best.pth")

    if is_main_process():
        train_time = time.time() - total_start_time
        train_time_str = time.strftime("%H:%M:%S", time.gmtime(train_time))
        logger.print(f"Training time:  {train_time_str}")
        writer.close()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

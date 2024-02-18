#!/usr/bin/env python
import time

import torch
import torch.nn.parallel
import torch.distributed as dist
from numpy import inf

from utils import meters
from models.model_utils import epoch_wrapup


def pretrain(loader, model, optimizer, ckp_manager, args, epoch=None, accum_epoch=0,
             category_name='Train', the_best_metric=-inf, writer=None, lr_scheduler=None, loss_scaler=None, total_steps=0):

    # Setup meters
    phase = 'Train'
    batch_time = meters.AverageMeter(f'Time/{category_name}', ':6.3f')
    data_time = meters.AverageMeter(f'Data/{category_name}', ':6.3f')
    lr_time = meters.AverageMeter(f'Learning Rate/{category_name}', ':6.6f')
    total_loss_mtr = meters.AverageMeter(f'{phase}/Total Loss/{category_name}', ':.4e', device=args.environment.gpu)
    mtr_list = [batch_time, data_time, lr_time, total_loss_mtr]
    # Loss, accuracy meters
    loss_name_list = args.criterion.args.loss_names
    loss_mtr_dict = {}
    acc_mtr_dict = {}
    for loss_name in loss_name_list:
        if loss_name == "mae_audio" or loss_name == "mae_frame":
            loss_mtr_dict[loss_name] = meters.AverageMeter(f'{phase}/{loss_name}_loss/{category_name}', ':.4e', device=args.environment.gpu)
        else:
            loss_mtr_dict[loss_name] = meters.AverageMeter(f'{phase}/{loss_name}_loss/{category_name}', ':.4e', device=args.environment.gpu)
            acc_mtr_dict[loss_name] = meters.AverageAccuracy(f'{phase}/{loss_name}_acc/{category_name}', ':.4e', device=args.environment.gpu)

    # Log penalty loss
    if hasattr(model, 'module'):
        _requires_penalty = model.module._req_penalty
        if model.module._req_opt:
            model.module.optimizer = optimizer
    else:
        _requires_penalty = model._req_penalty
        if model._req_opt:
            model.optimizer = optimizer

    if _requires_penalty:
        loss_mtr_dict['penalty'] = meters.AverageMeter(f'{phase}/penalty_loss/{category_name}', ':.4e', device=args.environment.gpu)

    mtr_list.extend(list(loss_mtr_dict.values()))
    mtr_list.extend(list(acc_mtr_dict.values()))

    progress = meters.ProgressMeter(
        len(loader), mtr_list, prefix=f"[{phase}][{epoch}]", tbwriter=writer)

    # Make sure distributed sampler uses different samples in each process.
    loader.batch_sampler.set_epoch(epoch=epoch)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    optimizer.zero_grad()

    if args.optim.use_grad_clip:
        clip_grad = args.optim.clip_grad
    else:
        clip_grad = None

    # Accumulate gradient since the model is too big
    batch_size = args.optim.batch_size
    per_gpu_batchsize = args.optim.per_gpu_batchsize
    grad_steps = max(batch_size // per_gpu_batchsize, 1)
    accumulated_steps = 0
    valid_accum_steps = 0

    # Switch to train mode
    model.train()

    optimizer.zero_grad()
    end = time.time()

    for data_idx, data in enumerate(loader):
        batch_i = loader.batch_sampler.advance_batches_seen() - 1
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if data_idx != 0:
            data_time.update(time.time() - end)

        keys = set([k for k in data.keys() if "video" in k or "audio" in k or "label_idx" in k])
        data = {k: v.cuda(args.environment.gpu, non_blocking=True) for k,v in data.items() if k in keys}

        # Compute output and loss
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = sum([v for k, v in output.items() if "loss" in k])

        # Log loss
        total_loss_mtr.update(loss.item(), per_gpu_batchsize)
        for loss_name in loss_mtr_dict.keys():
            loss_mtr_dict[loss_name].update(output[f'{loss_name}_loss'].item(), per_gpu_batchsize)
        # Log acc
        for loss_name in acc_mtr_dict.keys():
            acc_mtr_dict[loss_name].update(output[f'{loss_name}_logits'], output[f'{loss_name}_labels'])

        # normalize loss to account for batch accumulation
        loss = loss / grad_steps

        # weight update
        accumulated_steps += 1
        # Update weight when gradient is accumulated or when the last sample is out
        update_grad = accumulated_steps % grad_steps == 0 or batch_i == len(loader) - 1
        loss_scaler(loss, optimizer, clip_grad=clip_grad, model=model, update_grad=update_grad)

        torch.cuda.synchronize()
        if update_grad:
            optimizer.zero_grad()
            lr = optimizer.param_groups[0]["lr"]
            lr_time.update(lr)

            if valid_accum_steps % args.logging.print_freq == 0:
                tb_step = total_steps + valid_accum_steps * world_size
                progress.display(batch_i)
                progress.tbwrite(tb_step)

            # Update valid accumulated steps
            valid_accum_steps += 1

        # measure elapsed time, skip the first iteration since the first batch loading takes long time.
        if data_idx != 0:
            batch_time.update(time.time() - end)

        end = time.time()

        # Since we don't know the amount of real samples, update lr according to batch_i
        if lr_scheduler and args.optim.lr_scheduler == 'WarmupCosineSchedule':
            lr_scheduler.step()

        # Checkpoint
        ckp_manager.checkpoint(
            epoch=epoch,
            batch_i=batch_i,
            save_dict={'epoch': epoch,
                       'accum_epoch': accum_epoch,
                       'steps': total_steps,
            }
        )

    epoch_wrapup(phase, args, epoch, accum_epoch, loss_mtr_dict, acc_mtr_dict, model, writer)

    return valid_accum_steps





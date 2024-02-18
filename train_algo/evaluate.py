#!/usr/bin/env python
import os

import torch
import torch.nn.parallel
import torch.distributed as dist

from utils import meters
from models.model_utils import epoch_wrapup, record_acc_by_category


def evaluate(loader, model, ckp_manager, args, epoch=None, accum_epoch=0, category_name='Train', the_best_metric=None,
             phase=None, writer=None, total_steps=0):
    # Setup meters
    total_loss_mtr = meters.AverageMeter(f'{phase}/Total Loss/{category_name}', ':.4e', device=args.environment.gpu)
    mtr_list = [total_loss_mtr]
    # Loss, accuracy meters
    loss_name_list = args.criterion.args.loss_names
    loss_mtr_dict = {}
    acc_mtr_dict = {}
    for loss_name in loss_name_list:
        if loss_name == "mae_audio" or loss_name == "mae_frame":
            loss_mtr_dict[loss_name] = meters.AverageMeter(f'{phase}/{loss_name}_loss/{category_name}', ':.4e', device=args.environment.gpu)
        else:
            loss_mtr_dict[loss_name] = meters.AverageMeter(f'{phase}/{loss_name}_loss/{category_name}', ':.4e', device=args.environment.gpu)
            acc_mtr_dict[loss_name] = meters.AverageAccuracy(f'{phase}/{loss_name}_acc/{category_name}', ':.4e', phase=phase, device=args.environment.gpu)
    mtr_list.extend(list(loss_mtr_dict.values()))
    mtr_list.extend(list(acc_mtr_dict.values()))

    progress = meters.ProgressMeter(
        len(loader), mtr_list,
        prefix=f"[{phase}][{epoch}]", tbwriter=writer)

    # Make sure distributed sampler uses different samples in each process.
    loader.batch_sampler.set_epoch(epoch=epoch)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # switch to eval mode
    model.eval()
    enable_writer = phase == 'Eval'

    for data in loader:
        batch_i = loader.batch_sampler.advance_batches_seen() - 1
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        keys = set([k for k in data.keys() if "video" in k or "audio" in k or "label_idx" in k])
        data = {k: v.cuda(args.environment.gpu, non_blocking=True) for k,v in data.items() if k in keys}

        # Compute output and loss
        with torch.no_grad():
            output = model(data)
            loss = sum([v for k, v in output.items() if "loss" in k])
            batch_size = data['video_data'].shape[0]
            # Log loss
            total_loss_mtr.update(loss.item(), batch_size)
            for loss_name in loss_mtr_dict.keys():
                loss_mtr_dict[loss_name].update(output[f'{loss_name}_loss'].item(), batch_size)
            # Log acc
            for loss_name in acc_mtr_dict.keys():
                acc_mtr_dict[loss_name].update(output[f'{loss_name}_logits'], output[f'{loss_name}_labels'])

        if batch_i % args.logging.print_freq == 0:
            tb_step = (
                    epoch * len(loader.dataset) // args.optim.batch_size +
                    batch_i * world_size)
            if enable_writer:
                progress.tbwrite(tb_step)
            progress.display(batch_i)

    the_metric = epoch_wrapup(phase, args, epoch, accum_epoch, loss_mtr_dict, acc_mtr_dict, model, writer)

    if phase == 'Eval':
        # Save best performing model
        if the_best_metric is not None and the_best_metric < the_metric:
            the_best_metric = the_metric
            ckp_manager.force_checkpoint(
                ckpt_fname='checkpoint_best.pth',
                save_dict={'epoch':epoch,
                           'accum_epoch': accum_epoch,
                           'steps':total_steps,
                           'the_best_metric':the_best_metric,}
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return the_best_metric

    elif phase == 'Test':
        if 'vacls' in loss_name_list:
            record_acc_by_category(acc_mtr_dict['vacls'], classes=loader.dataset.classes, categories=loader.dataset.categories,
                                   save_path=os.path.join(args.logging.ckpt_dir, args.logging.name + args.logging.suffix, category_name), topk=(1,5),
                                   measure=args.data.measure)

    return

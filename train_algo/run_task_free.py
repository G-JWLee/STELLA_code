#!/usr/bin/env python
import builtins
import os

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models
import optimizers
import data
from utils import checkpointing, distributed_ops, amp
from train_algo.finetune import finetune
from train_algo.pretrain import pretrain
from train_algo.evaluate import evaluate
from train_algo.retrieval import retrieval
from models import objectives

def main_worker(gpu, ngpus_per_node, args):
    args.environment.gpu = gpu if ngpus_per_node > 0 and args.environment.distributed else None

    # suppress printing if not master
    if not args.data.args.debug and args.environment.multiprocessing_distributed and args.environment.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # setup distributed environment
    if args.environment.distributed:
        args = distributed_ops.init_distributed_environment(
            gpu, ngpus_per_node, args
        )
    def pretty_print(cfg, t=0):
        for k in cfg:
            if isinstance(cfg[k], type(cfg)):
                print(' '*t + k)
                pretty_print(cfg[k], t+2)
            else:
                print(' '*t + k, str(cfg[k]))
    pretty_print(args)

    if hasattr(args.cl_algo.args, 'mem_args'):
        _requires_buffer = True
    else:
        _requires_buffer = False

    if _requires_buffer:
        if args.environment.distributed:
            args.cl_algo.args.mem_args.memory_size = args.cl_algo.args.mem_args.memory_size // args.environment.ngpu

    # Create data loaders
    print(f"=> creating dataloaders")
    loaders = data.build_video_data_loaders(
        cfg=args.data,
        augm_cfg=args.data_augm,
        batch_size=args.optim.per_gpu_batchsize,
        workers=args.environment.workers,
        distributed=args.environment.distributed,
    )
    max_num_classes = 0
    for category_name in loaders:
        for mode in loaders[category_name]:
            print(category_name, mode, len(loaders[category_name][mode]),
                  loaders[category_name][mode].dataset.num_videos,
                  loaders[category_name][mode].dataset)
        if args.train_algo.trainer == 'finetune':
            max_num_classes = max(max_num_classes, len(loaders[category_name][mode].dataset.classes))
        else:
            max_num_classes = 0

    # Create model, criterion
    backbone_args = dict(args.backbone.args)
    backbone_args['num_classes'] = max_num_classes
    backbone_args['tasks'] = args.data.target_task
    backbone = getattr(models, args.backbone.arch)(**backbone_args)
    model_args = dict(args.criterion.args)
    model_args['backbone'] = backbone
    model = getattr(models, args.criterion.arch)(**model_args)
    # Wrap CL algorithm
    cl_args = dict(args.cl_algo.args)
    cl_args['model'] = model
    cl_args['device'] = args.environment.gpu
    cl_args['use_audio'] = args.data.args.use_audio

    model = getattr(models.cl_methods, args.cl_algo.name)(**cl_args)
    model = distributed_ops.send_to_device(
        model,
        distributed=args.environment.distributed,
        device=args.environment.gpu,
        unused_parameters=False,
    )
    print(model)
    model_without_ddp = model if not args.environment.distributed else model.module

    # Create optimizer & scheduler
    if args.optim.layer_decay < 1.0:
        assigner = optimizers.LayerDecayValueAssigner(
            list(args.optim.layer_decay ** (model_without_ddp.backbone.transformer.encoder_depth + 1 - i)
                 for i in range(model_without_ddp.backbone.transformer.encoder_depth + 2)),
            mid_fusion_depth=model_without_ddp.backbone.transformer.mid_fusion_depth)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    optimizer = optimizers.build_optimizer(args.optim, model_without_ddp,
                                           get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                           get_layer_sacle=assigner.get_scale if assigner is not None else None,)
    print(f"=> optimizer '{args.optim.method}'\n" + str(optimizer) + '\n')
    # AMP update from VideoMAE
    loss_scaler = amp.NativeScalerWithGradNormCount()
    lr_scheduler = None

    # Create TB loggers
    writer = None
    if args.logging.log_tb and (args.environment.gpu == 0 or args.environment.gpu is None):
        logdir = os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix)
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    cudnn.benchmark = True
    accum_epoch = 0
    train_init = True

    if args.train_algo.retrieval:
        loader = data.build_retrieval_loader(
            cfg=args.data,
            augm_cfg=args.data_augm,
            target_task='All',
            batch_size=args.optim.per_gpu_batchsize,
            workers=args.environment.workers,
            distributed=args.environment.distributed,
        )
        retrieval(loader,
                  model,
                  args,
                  epoch=args.optim.epochs,
                  writer=writer)

        print("finished!")
        return

    # Iteratively learn on the model. For finetuning, category_name is 'Finetune'
    for idx, category_name in enumerate(loaders):

        # Skip the explicit skip task list
        if category_name in args.data.skip_task:
            continue
        print(category_name)

        # Initialize classifier weight when learning new task
        if hasattr(model_without_ddp.backbone.transformer, "vacls_classifier"):
            model_without_ddp.backbone.transformer.vacls_classifier.apply(objectives.init_weights)

        # Optionally resume from a checkpoint
        modules = {'model': model, 'optimizer': optimizer,
                   'loss_scaler': loss_scaler, 'sampler': loaders[category_name]['train'].batch_sampler}
        if _requires_buffer:
            modules.update({'buffer': model_without_ddp.buffer})
        ckp_manager = checkpointing.CheckpointManager(
            modules=modules,
            ckpt_dir=os.path.join(args.logging.ckpt_dir, args.logging.name + args.logging.suffix, category_name),
            epoch_size=len(loaders[category_name]['train']),
            save_freq_mints=args.logging.save_freq_mints,
            save_freq_epoch=args.logging.save_freq,
        )

        total_steps = 0
        start_epoch = 1
        the_best_metric = None

        if train_init and args.environment.resume and ckp_manager is not None:
            start_epoch, accum_epoch, total_steps, the_best_metric = ckp_manager.resume(resume_path=args.logging.resume_path)
            the_best_metric = None
            train_init = False

        if args.train_algo.test_only:
            # Load best model
            ckp_manager.resume_best(**args.train_algo.test_args)
            evaluate(loaders[category_name]['test'],
                     model,
                     ckp_manager,
                     args,
                     epoch=0,  # For consistent test result
                     phase='Test',
                     category_name=category_name,
                     writer=writer)
        else:
            train = pretrain if args.train_algo.trainer == 'pretrain' else finetune
            epoch = None
            max_epoch = args.optim.epochs

            for epoch in range(start_epoch, max_epoch + 1):

                print('{} Epoch {}'.format(category_name, epoch))
                model_without_ddp.current_epoch = epoch

                steps = train(loaders[category_name]['train'],
                              model,
                              optimizer,
                              ckp_manager,
                              args,
                              epoch=epoch,
                              accum_epoch=accum_epoch,
                              category_name=category_name,
                              the_best_metric=the_best_metric,
                              writer=writer,
                              lr_scheduler=lr_scheduler,
                              loss_scaler=loss_scaler,
                              total_steps=total_steps)


                # Evaluate at every eval_freq epoch or at the end
                if (epoch % args.logging.eval_freq == 0) or (epoch == max_epoch):
                     evaluate(loaders[category_name]['eval'],
                               model,
                               ckp_manager,
                               args,
                               epoch=epoch,
                               accum_epoch=accum_epoch,
                               category_name=category_name,
                               the_best_metric=the_best_metric,
                               phase='Eval',
                               writer=writer,
                               total_steps=total_steps)

                total_steps += steps

                if _requires_buffer:
                    if hasattr(model_without_ddp, 'end_epoch'):
                        model_without_ddp.end_epoch()

                ckp_manager.checkpoint(
                    epoch=epoch,
                    save_dict={'epoch': epoch,
                               'accum_epoch': accum_epoch,
                               'steps': total_steps,
                               }
                )

            # For resuming at checkpoint at end_of_epoch
            if epoch is not None:
                accum_epoch += epoch
            else:
                accum_epoch += start_epoch

    print("finished!")
    return


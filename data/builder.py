import copy
from collections import OrderedDict

from data import video_datasets
from data import video_transforms
from data import audio_transforms
from data.resumable_samplers import ResumableBatchSampler
from torch.utils.data import DataLoader


class Transform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for k in x:
            x[k] = self.transforms[k](x[k])

        return x


def build_transforms(cfg, training):
    transforms = {}
    for k in cfg:
        # When extracting not augmented sample, do it only in training phase
        augm_args = dict(cfg[k].args)
        if cfg[k].name in vars(video_transforms):
            transforms[k] = video_transforms.__dict__[cfg[k].name](
                **augm_args, training=training)
        elif cfg[k].name in vars(audio_transforms):
            transforms[k] = audio_transforms.__dict__[cfg[k].name](
                **augm_args, training=training)
        else:
            raise NotImplementedError(f"Transform {cfg.name} not found.")
    return Transform(transforms)


def build_video_dataset(cfg, target_task, subset, transform=None):
    if cfg.name in vars(video_datasets):
        cfg_args = dict(cfg.args)
        dataset = video_datasets.__dict__[cfg.name](
            transform=transform,
            subset=subset,
            target_task=target_task,
            **cfg_args
        )
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not found.")

    return dataset

def build_video_data_loaders(cfg, augm_cfg, batch_size, workers=0, distributed=True):
    loaders = OrderedDict()

    for target_task in cfg.target_task:
        loaders[target_task] = {}
        for mode in cfg.splits:
            training = mode == 'train'

            transform = build_transforms(augm_cfg, training=training)
            db = build_video_dataset(cfg, target_task=target_task, subset=cfg.splits[mode], transform=transform)

            if mode == 'test':
                batch_sampler = ResumableBatchSampler(
                    batch_size=1, db=db,
                    distributed=distributed,
                    drop_last=training
                )
            else:
                batch_sampler = ResumableBatchSampler(
                    batch_size=batch_size, db=db,
                    distributed=distributed,
                    drop_last=training
                )
            loaders[target_task][mode] = DataLoader(
                db,
                batch_sampler=batch_sampler,
                num_workers=workers,
                pin_memory=True,
                persistent_workers=False,
                collate_fn=db.collate,
            )

    return loaders

def build_retrieval_loader(cfg, augm_cfg, target_task, batch_size, workers=0, distributed=True):
    transform = build_transforms(augm_cfg, training=False)
    cfg_ = copy.deepcopy(cfg)
    cfg_.name = cfg_.retrieval.name
    db = build_video_dataset(cfg_, target_task=target_task, subset='eval', transform=transform)
    batch_sampler = ResumableBatchSampler(
        batch_size=batch_size, db=db,
        distributed=distributed,
        drop_last=False
    )
    loader = DataLoader(
        db,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=db.collate,
    )

    return loader


def build_throughput_loader(cfg, batch_size, workers=0, distributed=True):
    db = build_video_dataset(cfg, target_task=None, subset=None, transform=None)
    batch_sampler = ResumableBatchSampler(
        batch_size=batch_size, db=db,
        distributed=distributed,
        drop_last=False
    )
    loader = DataLoader(
        db,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=db.collate,
    )

    return loader

def build_frames_loader(cfg, augm_cfg, batch_size, workers=0, distributed=True):
    transform = build_transforms(augm_cfg, training=False)
    db = build_video_dataset(cfg, target_task=None, subset=None, transform=transform)
    batch_sampler = ResumableBatchSampler(
        batch_size=batch_size, db=db,
        distributed=distributed,
        drop_last=False
    )
    loader = DataLoader(
        db,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=db.collate,
    )

    return loader
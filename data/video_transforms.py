import torch
from PIL import Image
import PIL
from torchvision import transforms as T
from pytorchvideo import transforms as vT
from data.transforms import video as vT2


class VideoMAE_transform:
    def __init__(self, input_size, training=False):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

        if training:
            transforms = [
                vT2.GroupMultiScaleCrop(input_size, [1, .875, .75]),
            ]
        else:
            transforms = [
                vT2.Resize(input_size),
            ]
        transforms += [
            vT2.ClipToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.t = T.Compose(transforms)

    def __call__(self, x):
        return self.t(x)


class CAV_Video_transform:

    def __init__(self, input_size, training=False):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        self.t = T.Compose(
            [
                T.Resize(input_size, interpolation=PIL.Image.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250])
            ]
        )

    def __call__(self, x):
        return self.t(x)


class Attention_vis_transform:
    def __init__(self, input_size, training=False):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

        transforms = [
            vT2.CenterCrop(input_size),
            vT2.ClipToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.t = T.Compose(transforms)

    def __call__(self, x):

        return self.t(x)

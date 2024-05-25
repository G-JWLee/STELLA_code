import os
import random
import glob
from typing import Any, Callable, Optional, List
import pandas as pd
import numpy as np
import torchvision
import torch
import h5py
import json

from PIL import Image
from decord import AudioReader

class AVSFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 4.0,
                 audio_duration: float = 10.0,
                 decode_audio: bool = True,
                 debug: bool = False,
                 **kwargs,
                 ) -> None:
        self._decode_audio = decode_audio
        self._num_frames = num_frames
        self._video_duration = video_duration  # TODO: check the time.
        self.audio_length = audio_duration
        self.print_error = False
        self._transform = transform
        self.base_path = base_path
        self.meta_path = meta_path

        print("loading metadata for audiovisual segmentation")
        self.split = 'train' if subset == 'train' else 'test'
        self.mask_num = 1 if self.split == 'train' else 4 # TODO: since our encoder get 4 frames, we might have to change it into 4
        df_all = pd.read_csv(os.path.join(meta_path, 's4_meta_data.csv'))
        self.df_split = df_all[df_all['split'] == self.split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


        if debug:
            self.df_split = self.df_split[:100]

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.df_split)


    def load_audio_clip(self, audio_path, audio_start=0, audio_duration=10):
        audio = AudioReader(audio_path)
        try:
            assert np.abs(audio._array).mean() != 0
        except:
            del audio
            raise ValueError(f'no audio in {audio_path}')
        audio_dict = {}
        audio_clip = audio._array.mean(0)[int(audio_start * 16000):int(audio_start + audio_duration * 16000)]
        audio_clip = audio_clip - audio_clip.mean()

        del audio

        audio_clip = {'audio_data': audio_clip}
        audio_clip = self._transform(audio_clip)

        audio_dict["audio_data"] = audio_clip["audio_data"]

        return audio_dict


    def load_video_clip(self, video_path, video_name, video_duration=4):

        video_dict = {}
        video_clip = [Image.open(os.path.join(video_path, "%s_%d.png"%(video_name, img_id))) for img_id in range(1, int(1+video_duration))]

        video_clip = {'video_data': video_clip}
        video_clip = self._transform(video_clip)

        video_dict["video_data"] = video_clip["video_data"]

        return video_dict


    def load_mask(self, mask_path, video_name):

        masks = []
        for mask_id in range(1, self.mask_num + 1):
            mask = Image.open(os.path.join(mask_path, "%s_%d.png"%(video_name, mask_id))).convert('1')
            mask = self.mask_transform(mask)
            masks.append(mask)
        mask_dict = {'mask_data': torch.stack(masks, dim=0)}

        return mask_dict


    def __getitem__(self, video_index):
        ret = dict()
        ret.update(self._get_video_audio(video_index, self._decode_audio))

        return ret

    def __len__(self):
        return len(self.df_split)


    def _get_video_audio(self, index, decode_audio):

        try:
            df_one_video = self.df_split.iloc[index]
            video_name, category = df_one_video[0], df_one_video[2]
            img_base_path = os.path.join(self.base_path, 'visual_frames', self.split, category, video_name)
            mask_base_path = os.path.join(self.base_path, 'gt_masks', self.split, category, video_name)

            sample_dict = {
                'vid': video_name,
                "index": index,
                "category": category,
            }

            video_dict = self.load_video_clip(img_base_path, video_name, self._video_duration)
            sample_dict.update(video_dict)

            mask_dict = self.load_mask(mask_base_path, video_name)
            sample_dict.update(mask_dict)

            if decode_audio:
                audio_path = os.path.join(self.base_path, 'audio_wav', self.split, category, video_name + '.wav')
                audio_dict = self.load_audio_clip(audio_path, 0, self.audio_length)
                sample_dict.update(audio_dict)

            return sample_dict

        except Exception as e:
            if self.print_error:
                print(e)
            video_index = random.sample(range(len(self)), k=1)[0]
            return self._get_video_audio(video_index, decode_audio)


    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        video_keys = set([k for k in keys if "video" in k])
        audio_keys = set([k for k in keys if "audio" in k])
        mask_keys = set([k for k in keys if "mask" in k])

        batch_size = len(batch)
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]

        if len(video_keys) != 0:
            max_video_length = self._num_frames
            max_height = max([i[2] for i in video_sizes])
            max_width = max([i[3] for i in video_sizes])

        for video_key in video_keys:
            video = dict_batch[video_key]
            new_videos = torch.ones(batch_size, max_video_length, 3, max_height, max_width) * -1.0
            for bi in range(batch_size):
                orig_batch = video[bi]
                if orig_batch is None:
                    new_videos[bi] = None
                else:
                    orig = video[bi]
                    orig = orig.transpose(0, 1)  # 10 x 3 x T x H x W -> 10 x T x 3 x H x W
                    new_videos[bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig
            dict_batch[video_key] = new_videos

        audio_sizes = list()
        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            for audio_i in audio:
                audio_sizes += [audio_i.shape]

        if len(audio_keys) != 0:
            max_height = max([i[1] for i in audio_sizes])
            max_width = max([i[2] for i in audio_sizes])

        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            new_audios = torch.ones(batch_size, 1, max_height, max_width) * -1.0
            for bi in range(batch_size):
                orig_batch = audio[bi]
                if orig_batch is None:
                    new_audios[bi] = None
                else:
                    orig = audio[bi]
                    new_audios[bi, : orig.shape[0], : orig.shape[1], : orig.shape[2]] = orig
            dict_batch[audio_key] = new_audios

        for mask_key in mask_keys:  # TODO: check the data format
            mask = dict_batch[mask_key]
            new_masks = torch.zeros(batch_size, self.mask_num, 1, 224, 224,)
            for bi in range(batch_size):
                orig_batch = mask[bi]
                if orig_batch is None:
                    new_masks[bi] = None
                else:
                    orig = mask[bi]
                    new_masks[bi] = orig
            dict_batch[mask_key] = new_masks

        return dict_batch

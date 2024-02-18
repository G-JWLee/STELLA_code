import os
import random
from typing import Any, Callable, Optional, List
import pandas as pd
import numpy as np
import torchvision
import torch
import h5py

from PIL import Image
from decord import AudioReader

class AVEDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subset: str = 'train',
                 meta_path: str = "./",
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 audio_duration: float = 1.0,
                 decode_audio: bool = True,
                 debug: bool = False,
                 **kwargs,
                 ) -> None:

        self._decode_audio = decode_audio
        self._num_frames = num_frames
        self.video_len = 10
        self.audio_length = audio_duration
        self.fps = 4
        self.print_error = False
        self._transform = transform
        self.data_path = meta_path

        print("loading metadata for ave")
        with h5py.File(os.path.join(meta_path, 'labels.h5'), 'r') as hf:
            self.classes = hf['avadataset'][:]

        if subset == 'train':
            with h5py.File(os.path.join(meta_path, 'train_order.h5'), 'r') as hf:
                order = hf['order'][:]

        else:
            with h5py.File(os.path.join(meta_path, 'test_order.h5'), 'r') as hf:
                order = hf['order'][:]

        if debug:
            order = order[:500]

        self.lis = order
        self.raw_gt = pd.read_csv(os.path.join(meta_path, 'Annotations.txt'), sep="&", header=0)
        self.label_num = self.classes.shape[-1]

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.lis)


    def load_audio_clip(self, audio_path):
        audio = AudioReader(audio_path)
        try:
            assert np.abs(audio._array).mean() != 0
        except:
            del audio
            raise ValueError(f'no audio in {audio_path}')

        total_audio = []
        for audio_sec in range(self.video_len):
            ## yb: align ##
            sample_indx = np.linspace(0, audio._array.shape[1] - 16000*(self.audio_length+0.1), num=10, dtype=int)
            audio_clip = audio._array.mean(0)[sample_indx[audio_sec]:sample_indx[audio_sec]+int(16000*self.audio_length)]
            audio_clip = audio_clip - audio_clip.mean()

            audio_clip = {'audio_data': audio_clip}
            audio_clip = self._transform(audio_clip)

            total_audio.append(audio_clip['audio_data'])

        total_audio = torch.stack(total_audio)
        del audio

        audio_dict = {"audio_data": total_audio}

        return audio_dict


    def load_video_clip(self, video_path):

        total_vid = []

        for video_sec in range(self.video_len):
            frame_start_idx = int(video_sec * self.fps)
            frame_end_idx = frame_start_idx + int(self.audio_length * self.fps)
            downsample_indices = np.linspace(frame_start_idx, frame_end_idx, self._num_frames, endpoint=False).astype(
                np.int)
            video_clip = [Image.open(os.path.join(video_path, 'frame_{:d}.jpg'.format(i))) for i in downsample_indices]
            video_clip = {'video_data': video_clip}
            video_clip = self._transform(video_clip)
            total_vid.append(video_clip['video_data'])

        total_vid = torch.stack(total_vid)
        video_dict = {'video_data': total_vid}

        return video_dict

    def __getitem__(self, video_index):
        ret = dict()
        ret.update(self._get_video_audio(video_index, self._decode_audio))

        return ret

    def __len__(self):
        return len(self.lis)

    def _get_video_audio(self, index, decode_audio):

        real_idx = self.lis[index] - 1 # start from 1
        try:
            file_name = self.raw_gt.iloc[real_idx][1]
            frame_path = os.path.join(self.data_path, 'frames', file_name)
            wav = os.path.join(self.data_path, 'audio', file_name + '.mp3')

            sample_dict = {
                'vid': file_name,
                "index": index,
            }

            video_dict = self.load_video_clip(frame_path)
            sample_dict.update(video_dict)

            if decode_audio:
                audio_dict = self.load_audio_clip(wav)
                sample_dict.update(audio_dict)

            sample_dict.update({'label_idx': torch.tensor(self.classes[real_idx])})

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
        label_keys = set([k for k in keys if "label_idx" in k])
        other_keys = keys - video_keys - audio_keys - label_keys

        batch_size = len(batch)
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]

        if len(video_keys) != 0:
            max_video_length = self._num_frames
            max_height = max([i[3] for i in video_sizes])
            max_width = max([i[4] for i in video_sizes])

        for video_key in video_keys:
            video = dict_batch[video_key]
            new_videos = torch.ones(batch_size, self.video_len, max_video_length, 3, max_height, max_width) * -1.0
            for bi in range(batch_size):
                orig_batch = video[bi]
                if orig_batch is None:
                    new_videos[bi] = None
                else:
                    orig = video[bi]
                    orig = orig.transpose(1, 2)  # 10 x 3 x T x H x W -> 10 x T x 3 x H x W
                    new_videos[bi, :, : orig.shape[1], :, : orig.shape[3], : orig.shape[4]] = orig
            dict_batch[video_key] = new_videos

        audio_sizes = list()
        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            for audio_i in audio:
                audio_sizes += [audio_i.shape]

        if len(audio_keys) != 0:
            max_height = max([i[2] for i in audio_sizes])
            max_width = max([i[3] for i in audio_sizes])

        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            new_audios = torch.ones(batch_size, self.video_len, 1, max_height, max_width) * -1.0
            for bi in range(batch_size):
                orig_batch = audio[bi]
                if orig_batch is None:
                    new_audios[bi] = None
                else:
                    orig = audio[bi]
                    new_audios[bi, :, : orig.shape[1], : orig.shape[2], : orig.shape[3]] = orig
            dict_batch[audio_key] = new_audios


        for label_key in label_keys:
            label = dict_batch[label_key]
            new_labels = torch.ones(batch_size, self.video_len, self.label_num, dtype=torch.float)
            for bi in range(batch_size):
                orig_batch = label[bi]
                if orig_batch is None:
                    new_labels[bi] = None
                else:
                    orig = label[bi]
                    new_labels[bi] = orig
            dict_batch[label_key] = new_labels

        task_keys = [k for k in list(dict_batch.keys()) if "task_id" in k]
        for task_key in task_keys:
            task = dict_batch[task_key]
            new_tasks = torch.ones(batch_size, dtype=torch.long)
            for bi in range(batch_size):
                orig_batch = int(task[bi])
                if orig_batch is None:
                    new_tasks[bi] = None
                else:
                    orig = int(task[bi])
                    new_tasks[bi] = orig
            dict_batch[task_key] = new_tasks

        return dict_batch



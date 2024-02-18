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

from data.base_video_dataset import BaseFinetuneDataset

class MSRVTTRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self,
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
        self.video_len = 10
        self._video_duration = video_duration
        self.audio_length = audio_duration
        self.fps = 4
        self.print_error = False
        self.sample_type = 'middle'
        self._transform = transform
        self.base_path = base_path
        self.meta_path = meta_path

        print("loading metadata for msrvtt")
        # Code from TVLT
        file_list = []
        json_fp = os.path.join(meta_path, 'annotation', 'MSR_VTT.json')
        data = json.load(open(json_fp, 'r'))
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(meta_path, 'high-quality', 'structured-symlinks', "test_list_miech.txt")
        test_df = pd.read_csv(split_dir, names=['videoid'])

        df = df[df['image_id'].isin(test_df['videoid'])]

        metadata = df.groupby(['image_id'])['caption'].apply(list)
        for i, key in enumerate(metadata.keys()):
            file_list.append(key)

        valid_file_list = glob.glob(os.path.join(self.meta_path, 'audio', '*.mp3'))
        valid_file_list = [os.path.basename(file_name).split('.')[0] for file_name in valid_file_list]
        self.data = [file_name for file_name in valid_file_list if file_name in file_list]

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.data)


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


    def load_video_clip(self, video_path, video_start=0, video_duration=4):
        # Since we sampled frames with fps=8, frames exist at every 0.125 second
        frame_start_idx = int(video_start * self.fps)
        frame_end_idx = frame_start_idx + int(video_duration * self.fps)
        downsample_indices = np.linspace(frame_start_idx, frame_end_idx, self._num_frames, endpoint=False).astype(np.int)
        video_dict = {}
        video_clip = [Image.open(os.path.join(video_path, 'frame_{:d}.jpg'.format(i))) for i in downsample_indices]

        video_clip = {'video_data': video_clip}
        video_clip = self._transform(video_clip)

        video_dict["video_data"] = video_clip["video_data"]

        return video_dict

    def __getitem__(self, video_index):
        ret = dict()
        ret.update(self._get_video_audio(video_index, self._decode_audio, self.sample_type))

        return ret

    def __len__(self):
        return len(self.data)

    def _get_video_audio(self, index, decode_audio, sample_type):
        video_id = self.data[index]

        try:
            frame_path = os.path.join(self.meta_path, 'frames', video_id)
            wav = os.path.join(self.meta_path, 'audio', video_id + '.mp3')

            sample_dict = {
                'vid': video_id,
                "index": index,
            }

            if sample_type == 'middle':
                start = np.array([self.video_len / 2 - self._video_duration//2])
            else:
                raise  ValueError(f'Undefined sampling type {sample_type}')


            video_dict = self.load_video_clip(frame_path, start, self._video_duration)
            sample_dict.update(video_dict)

            if decode_audio:
                audio_dict = self.load_audio_clip(wav, 0, self.audio_length)
                sample_dict.update(audio_dict)

            return sample_dict

        except Exception as e:
            if self.print_error:
                print(e)
            video_index = random.sample(range(len(self)), k=1)[0]
            return self._get_video_audio(video_index, decode_audio, sample_type)


    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        video_keys = set([k for k in keys if "video" in k])
        audio_keys = set([k for k in keys if "audio" in k])

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

        return dict_batch



class MSRVTTPretrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 4.0,
                 audio_duration: float = 10.0,
                 decode_audio: bool = True,
                 sample_type: str = "",
                 debug: bool = False,
                 **kwargs,
                 ) -> None:

        self._decode_audio = decode_audio
        self._num_frames = num_frames
        self.video_len = 10
        self._video_duration = video_duration
        self.audio_length = audio_duration
        self.fps = 4
        self.print_error = False
        self._transform = transform
        self.base_path = base_path
        self.meta_path = meta_path

        if sample_type == "":
            if subset == 'train':
                sample_type = 'random'
            elif subset == 'eval':
                sample_type = 'middle'
            else:
                raise  ValueError(f"Undefined sample type {sample_type}")
        self.sample_type = sample_type


        print("loading metadata for msrvtt")
        # Code from TVLT
        file_list = []
        json_fp = os.path.join(meta_path, 'annotation', 'MSR_VTT.json')
        data = json.load(open(json_fp, 'r'))
        df = pd.DataFrame(data['annotations'])

        if subset == 'train':
            subset_meta = "train_list_miech.txt"
        else:
            subset_meta = "test_list_miech.txt"

        split_dir = os.path.join(meta_path, 'high-quality', 'structured-symlinks', subset_meta)
        subset_df = pd.read_csv(split_dir, names=['videoid'])

        df = df[df['image_id'].isin(subset_df['videoid'])]

        metadata = df.groupby(['image_id'])['caption'].apply(list)
        for i, key in enumerate(metadata.keys()):
            file_list.append(key)

        valid_file_list = glob.glob(os.path.join(self.meta_path, 'audio', '*.mp3'))
        valid_file_list = [os.path.basename(file_name).split('.')[0] for file_name in valid_file_list]
        self.data = [file_name for file_name in valid_file_list if file_name in file_list]

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.data)


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


    def load_video_clip(self, video_path, video_start=0, video_duration=4):
        # Since we sampled frames with fps=8, frames exist at every 0.125 second
        frame_start_idx = int(video_start * self.fps)
        frame_end_idx = frame_start_idx + int(video_duration * self.fps)
        downsample_indices = np.linspace(frame_start_idx, frame_end_idx, self._num_frames, endpoint=False).astype(np.int)
        video_dict = {}
        video_clip = [Image.open(os.path.join(video_path, 'frame_{:d}.jpg'.format(i))) for i in downsample_indices]

        video_clip = {'video_data': video_clip}
        video_clip = self._transform(video_clip)

        video_dict["video_data"] = video_clip["video_data"]

        return video_dict

    def __getitem__(self, video_index):
        ret = dict()
        ret.update(self._get_video_audio(video_index, self._decode_audio, self.sample_type))

        return ret

    def __len__(self):
        return len(self.data)

    def _get_video_audio(self, index, decode_audio, sample_type):
        video_id = self.data[index]

        try:
            frame_path = os.path.join(self.meta_path, 'frames', video_id)
            wav = os.path.join(self.meta_path, 'audio', video_id + '.mp3')

            sample_dict = {
                'vid': video_id,
                "index": index,
            }

            if sample_type == 'random':
                start_cand = np.arange(0, self.video_len - self._video_duration, step=1/self.fps)
                start = np.random.choice(start_cand, 1)

            elif sample_type == 'middle':
                start = np.array([self.video_len / 2 - self._video_duration//2])
            else:
                raise ValueError(f'Undefined sampling type {sample_type}')


            video_dict = self.load_video_clip(frame_path, start, self._video_duration)
            sample_dict.update(video_dict)

            if decode_audio:
                audio_dict = self.load_audio_clip(wav, 0, self.audio_length)
                sample_dict.update(audio_dict)

            return sample_dict

        except Exception as e:
            if self.print_error:
                print(e)
            video_index = random.sample(range(len(self)), k=1)[0]
            return self._get_video_audio(video_index, decode_audio, sample_type)


    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        video_keys = set([k for k in keys if "video" in k])
        audio_keys = set([k for k in keys if "audio" in k])

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

        return dict_batch

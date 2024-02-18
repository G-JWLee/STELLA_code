import os
import random
import numpy as np
import torch
import torch.utils.data
import logging

from PIL import Image
from decord import AudioReader

logger = logging.getLogger(__name__)


class BaseFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 video_info,
                 transform,
                 num_frames: int = 4,
                 video_duration: float = 1.0,
                 audio_duration: float = 1.0,
                 decode_audio: bool = True,
                 decode_video: bool = True,
                 sample_type: str = 'random',
                 label_smooth: float = 0,
                 **kwargs,
                 ):

        self._video_info = video_info
        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._num_frames = num_frames
        self._video_duration = video_duration
        self._audio_duration = audio_duration
        self._transform = transform
        self.video_len = 10
        self.fps = 8
        self.print_error = False

        assert sample_type in ['random', 'middle', 'left_most', 'right_most', 'uniform']
        self.sample_type = sample_type
        self.label_smooth = label_smooth

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self._video_info)

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
        ret.update(self._get_video_audio(video_index, self._decode_audio, self._decode_video, self.sample_type))

        return ret

    def __len__(self):
        return len(self._video_info)

    def _get_video_audio(self, index, decode_audio, decode_video, sample_type):
        datum = self._video_info[index]
        datum = self.decode_data(datum)
        try:
            sample_dict = {
                **datum,
                "index": index,
            }

            if decode_audio:
                sample_dict.update({"audio_data": []})

            if decode_video:
                sample_dict.update({"video_data": []})

            if 'labels' in datum:
                sample_dict.update({"label_idx": []})

            if sample_type == 'random':
                start_cand = np.arange(0, self.video_len - self._video_duration, step=1/self.fps)
                start = np.random.choice(start_cand, 1)

            elif sample_type == 'middle':
                start = np.array([self.video_len / 2 - self._video_duration//2])

            elif sample_type == 'left_most':
                start = np.array([0])

            elif sample_type == 'right_most':
                start = np.array([self.video_len - self._video_duration])

            elif self.sample_type == 'uniform':
                start = np.arange(0, self.video_len - self._video_duration +0.01, step=self._video_duration/2)

            else:
                raise ValueError(f'Undefined sampling type {sample_type}')

            for start_time in start:

                # Video
                if decode_video:
                    video_dict = self.load_video_clip(datum['frames_path'],
                                                            start_time, self._video_duration)
                    for key in video_dict:
                        sample_dict[key].append(video_dict[key])

                # Audio
                if decode_audio:
                    audio_dict = self.load_audio_clip(datum['wav'],
                                                            0, self._audio_duration)
                    for key in audio_dict:
                        sample_dict[key].append(audio_dict[key])

                # Label, for multi-label classification
                if 'labels' in datum:
                    label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
                    for label_str in datum['labels'].split(','):
                        label_indices[int(self.classes[label_str])] = 1.0 - self.label_smooth
                    label_indices = torch.FloatTensor(label_indices)
                    sample_dict['label_idx'].append(label_indices)

            return sample_dict

        except Exception as e:
            if self.print_error:
                print(e)
            video_index = random.sample(range(len(self)), k=1)[0]
            return self._get_video_audio(video_index, decode_audio, decode_video, sample_type)

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        video_keys = set([k for k in keys if "video" in k])
        audio_keys = set([k for k in keys if "audio" in k])
        label_keys = set([k for k in keys if "label_idx" in k])
        other_keys = keys - video_keys - audio_keys - label_keys

        # Change list formed data into tensor, extend batch size if more than one data in sample
        new_batch = []
        for sample in batch:
            if 'video_data' in sample:
                while len(sample['video_data']) != 0:
                    copied_dict = {k: sample[k] if k in other_keys else sample[k].pop() for k in keys}
                    new_batch.append(copied_dict)
            else:
                while len(sample['audio_data']) != 0:
                    copied_dict = {k: sample[k] if k in other_keys else sample[k].pop() for k in keys}
                    new_batch.append(copied_dict)

        batch_size = len(new_batch)
        dict_batch = {k: [dic[k] if k in dic else None for dic in new_batch] for k in keys}

        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]

        for size in video_sizes:
            assert (
                    len(size) == 4
            ), f"Collate error, an video should be in shape of (T, 3, H, W), instead of given {size}"

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
                    orig = orig.transpose(0, 1)  # 3 x T x H x W -> T x 3 x H x W
                    new_videos[bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig
            dict_batch[video_key] = new_videos

        audio_sizes = list()
        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            for audio_i in audio:
                audio_sizes += [audio_i.shape]
        for size in audio_sizes:
            assert (
                    len(size) == 3
            ), f"Collate error, an audio should be in shape of (1, H, W), instead of given {size}"
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


        for label_key in label_keys:
            label = dict_batch[label_key]
            new_labels = torch.ones(batch_size, self.label_num, dtype=torch.float)
            for bi in range(batch_size):
                orig_batch = label[bi]
                if orig_batch is None:
                    new_labels[bi] = None
                else:
                    orig = label[bi]
                    new_labels[bi] = orig
            dict_batch[label_key] = new_labels

        return dict_batch
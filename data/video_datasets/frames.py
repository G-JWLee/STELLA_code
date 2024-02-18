import os
import random
import torch
import torch.utils.data
import torchvision
import numpy as np

from PIL import Image
from decord import AudioReader


class BaseFinetuneFramesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 base_path: str = '',
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 1.0,
                 audio_duration: float = 1.0,
                 decode_audio: bool = True,
                 **kwargs,
                 ):
        video_info = [
            ['S1Kbym7WYzs', os.path.join(base_path, 'vggsound/data/frames/S1Kbym7WYzs_82000_92000'),
             os.path.join(base_path, 'vggsound/data/audio/S1Kbym7WYzs_82000_92000.mp3')],
        ]

        self._video_info = video_info
        self._decode_audio = decode_audio
        self._num_frames = num_frames
        self._video_duration = video_duration
        self._audio_duration = audio_duration
        self._transform = transform
        self.video_len = 10
        self.fps = 8
        self.print_error = False

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

    def load_video_clip(self, video_path, starts=-1, eval=False):
        if eval:
            # starting from the given start time, extract num_frames frames.
            frame_start_idx = int(starts * self.fps)
            frame_end_idx = frame_start_idx + int(self._video_duration * self.fps)
            downsample_indices = np.linspace(frame_start_idx, frame_end_idx, self._num_frames, endpoint=False).astype(np.int)
        else:
            # starting from the given start time, extract frames in 1s(self.fps frames) interval
            frame_start_idx = int(starts * self.fps)
            downsample_indices = np.arange(frame_start_idx, self._audio_duration * self.fps, step=self.fps).astype(np.int)

        video_clip = [Image.open(os.path.join(video_path, 'frame_{:d}.jpg'.format(i))) for i in downsample_indices]
        video_dict = {}

        video_clip = {'video_data': video_clip}
        video_clip = self._transform(video_clip)

        video_dict["video_data"] = video_clip["video_data"]

        return video_dict

    def decode_data(self, np_data):
        dataum = {}
        dataum['vid'] = np_data[0]
        dataum['frames_path'] = np_data[1]
        dataum['wav'] = np_data[2]

        return dataum

    def __getitem__(self, video_index):
        ret = dict()
        ret.update(self._get_video_audio(video_index, self._decode_audio))

        return ret

    def __len__(self):
        return len(self._video_info)

    def _get_video_audio(self, index, decode_audio):
        datum = self._video_info[index]
        datum = self.decode_data(datum)
        try:
            sample_dict = {
                **datum,
                "video_data": [],
            }

            if decode_audio:
                sample_dict.update({"audio_data": []})

            if 'labels' in datum:
                sample_dict.update({"label_idx": []})

            start = 7 * (1/self.fps)
            eval = False

            # Video
            video_dict = self.load_video_clip(datum['frames_path'], starts=start, eval=eval)
            for key in video_dict:
                sample_dict[key].append(video_dict[key])

            # Audio
            if decode_audio:
                audio_dict = self.load_audio_clip(datum['wav'],
                                                        0, self._audio_duration)
                for key in audio_dict:
                    sample_dict[key].append(audio_dict[key])

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
        other_keys = keys - video_keys - audio_keys

        # Change list formed data into tensor, extend batch size if more than one data in sample
        new_batch = []
        for sample in batch:
            while len(sample['video_data']) != 0:
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
            max_video_length = max([i[1] for i in video_sizes])
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
                    new_videos[bi, : orig.shape[0], : orig.shape[1], : orig.shape[2], : orig.shape[3]] = orig
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

        return dict_batch



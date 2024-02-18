import os
import pandas as pd
import numpy as np
import torchvision
from data.base_video_dataset import BaseFinetuneDataset



class AudioSet20kFinetuneDataset(BaseFinetuneDataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 label_smooth: float = 0,
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 4.0,
                 audio_duration: float = 10.0,
                 decode_audio: bool = True,
                 sample_type: str = "",
                 debug: bool = False,
                 **kwargs,
                 ) -> None:

        print("loading metadata for audioset")

        if subset == 'train':
            meta_data = pd.read_csv(os.path.join(meta_path, 'audioset_20k_cleaned.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label'])
        else:
            meta_data = pd.read_csv(os.path.join(meta_path, 'audioset_eval_cleaned.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label'])
            label_smooth = 0
        label_csv = pd.read_csv(os.path.join(meta_path, 'class_labels_indices.csv'), header=0)
        # No category information in AudioSet
        self.classes, self.categories = self.make_index_dict(label_csv)
        self.label_num = len(self.classes)
        print('number of classes is {:d}'.format(self.label_num))
        sample_meta = self.pro_data(meta_data, base_path)

        if debug:
            sample_meta = sample_meta[:100]

        if sample_type == "":
            if subset == 'train':
                sample_type = 'random'
            elif subset == 'eval':
                sample_type = 'middle'
            elif subset == 'test':
                sample_type = 'uniform'
            else:
                raise ValueError(f"Undefined sample type {sample_type}")

        super().__init__(
            video_info=sample_meta,
            transform=transform,
            num_frames=num_frames,
            video_duration=video_duration,
            audio_duration=audio_duration,
            decode_audio=decode_audio,
            sample_type=sample_type,
            label_smooth=label_smooth,
        )

    def pro_data(self, meta_data, base_path):
        sample_meta = []
        for data_info in meta_data.itertuples():
            sample_meta.append(
                [data_info[1], os.path.join(base_path, data_info[2]), os.path.join(base_path, data_info[3]),
                 data_info[4]])
        sample_meta = np.array(sample_meta, dtype=str)
        return sample_meta

    def decode_data(self, np_data):
        dataum = {}
        dataum['vid'] = np_data[0]
        dataum['frames_path'] = np_data[1]
        dataum['wav'] = np_data[2]
        dataum['labels'] = np_data[3]

        return dataum

    def make_index_dict(self, label_csv):
        index_lookup = {}
        category_lookup = {}
        for index, row in label_csv.iterrows():
            index_lookup[row['mid']] = index
            category_lookup[row['mid']] = 'Finetune'

        return index_lookup, category_lookup


class AudioSet20kPretrainDataset(BaseFinetuneDataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 label_smooth: float = 0,
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 4.0,
                 audio_duration: float = 10.0,
                 decode_audio: bool = True,
                 sample_type: str = "",
                 debug: bool = False,
                 **kwargs,
                 ) -> None:

        print("loading metadata for audioset")

        if subset == 'train':
            meta_data_as = pd.read_csv(os.path.join(meta_path, 'audioset_20k_cleaned.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label'])
            meta_data_vgg = pd.read_csv(os.path.join(meta_path, 'vgg_unused_cleaned_shrink8.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label', 'category'])
            print("Merged!")
            meta_data = pd.concat([meta_data_as, meta_data_vgg])
        else:
            meta_data_as = pd.read_csv(os.path.join(meta_path, 'audioset_eval_cleaned.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label'])
            meta_data_vgg = pd.read_csv(os.path.join(meta_path, 'vgg_unused_test_cleaned_shrink8.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label', 'category'])
            print("Merged!")
            meta_data = pd.concat([meta_data_as, meta_data_vgg])
            label_smooth = 0

        sample_meta = self.pro_data(meta_data, base_path)

        if sample_type == "":
            if subset == 'train':
                sample_type = 'random'
            elif subset == 'eval':
                sample_type = 'middle'
            else:
                raise ValueError(f"Undefined sample type {sample_type}")
        print(sample_type)

        super().__init__(
            video_info=sample_meta,
            transform=transform,
            num_frames=num_frames,
            video_duration=video_duration,
            audio_duration=audio_duration,
            decode_audio=decode_audio,
            sample_type=sample_type,
            label_smooth=label_smooth,
        )

    def pro_data(self, meta_data, base_path):
        sample_meta = []
        for data_info in meta_data.itertuples():
            sample_meta.append(
                [data_info[1], os.path.join(base_path, data_info[2]), os.path.join(base_path, data_info[3])])
        sample_meta = np.array(sample_meta, dtype=str)
        return sample_meta

    def decode_data(self, np_data):
        dataum = {}
        dataum['vid'] = np_data[0]
        dataum['frames_path'] = np_data[1]
        dataum['wav'] = np_data[2]

        return dataum

    def make_index_dict(self, label_csv):
        index_lookup = {}
        category_lookup = {}
        for index, row in label_csv.iterrows():
            index_lookup[row['mid']] = index
            category_lookup[row['mid']] = 'Pretrain'

        return index_lookup, category_lookup


class AudioSet20kRetrievalDataset(BaseFinetuneDataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 label_smooth: float = 0,
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 4.0,
                 audio_duration: float = 10.0,
                 decode_audio: bool = True,
                 sample_type = None,
                 debug: bool = False,
                 **kwargs,
    ) -> None:

        print("loading metadata for audioset")
        meta_data = pd.read_csv(os.path.join(meta_path, 'audioset_eval_5_per_class_for_retrieval_cleaned.csv'),
                                names=['vid', 'frames_path', 'wav', 'label'])
        sample_meta = self.pro_data(meta_data, base_path)
        self.category_list = ['All']

        # if debug:
        #     sample_meta = sample_meta[:100]

        if sample_type == "":
            sample_type = 'middle'

        super().__init__(
            video_info=sample_meta,
            transform=transform,
            num_frames=num_frames,
            video_duration=video_duration,
            audio_duration=audio_duration,
            decode_audio=decode_audio,
            sample_type=sample_type,
            label_smooth=label_smooth,
        )

    def pro_data(self, meta_data, base_path):
        sample_meta = []
        for data_info in meta_data.itertuples():
            sample_meta.append([data_info[1], os.path.join(base_path, data_info[2]),
                                os.path.join(base_path, data_info[3]), 'All'])
        sample_meta = np.array(sample_meta, dtype=str)
        return sample_meta

    def decode_data(self, np_data):
        dataum = {}
        dataum['vid'] = np_data[0]
        dataum['frames_path'] = np_data[1]
        dataum['wav'] = np_data[2]
        dataum['category'] = np_data[3]

        return dataum

    def make_index_dict(self, label_csv):
        index_lookup = {}
        category_lookup = {}
        for index, row in label_csv.iterrows():
            index_lookup[row['mid']] = index
            category_lookup[row['mid']] = 'Retrieval'

        return index_lookup, category_lookup

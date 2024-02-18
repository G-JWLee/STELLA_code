import os
import pandas as pd
import numpy as np
import torchvision
from data.base_video_dataset import BaseFinetuneDataset


class VGGSoundFinetuneDataset(BaseFinetuneDataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 target_task: str = 'All',
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

        print("loading metadata for vggsound")
        # Three options for target_task 1. Origin 2. All 3. target task name
        # Use all VGGSound dataset
        if target_task == 'Origin':
            if subset == 'train':
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_train_cleaned.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
            else:
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_test_cleaned.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
                label_smooth = 0
            label_csv = pd.read_csv(os.path.join(meta_path, 'class_labels_indices.csv'), header=0)
        # Use subset of VGGSound dataset
        else:
            if subset == 'train':
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_train_cleaned_shrink8.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
            else:
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_test_cleaned_shrink8.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
                label_smooth = 0
            label_csv = pd.read_csv(os.path.join(meta_path, 'class_labels_indices_shrink8.csv'), header=0)

        if not target_task in ['All', 'Origin']:
            meta_data = meta_data.query(f"category=='{target_task}'")
            label_csv = label_csv.query(f"category=='{target_task}'")
        self.classes, self.categories = self.make_index_dict(label_csv)
        self.label_num = len(self.classes)
        print('number of classes is {:d}'.format(self.label_num))
        sample_meta = self.pro_data(meta_data, base_path)

        if debug:
            sample_meta = sample_meta[:400]

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
            category_lookup[row['mid']] = row['category']

        return index_lookup, category_lookup


class VGGSoundPretrainDataset(BaseFinetuneDataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 target_task: str = 'All',
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

        print("loading metadata for vggsound")

        # Three options for target_task 1. Origin 2. All 3. target task name
        # Use all VGGSound dataset
        if target_task == 'Origin':
            if subset == 'train':
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_train_cleaned.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
            else:
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_test_cleaned.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
        # Use subset of VGGSound
        else:
            if subset == 'train':
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_train_cleaned_shrink8.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
            else:
                meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_test_cleaned_shrink8.csv'),
                                        names=['vid', 'frames_path', 'wav', 'label', 'category'])
        if not target_task in ['All', 'Origin']:
            meta_data = meta_data.query(f"category=='{target_task}'")
        sample_meta = self.pro_data(meta_data, base_path)

        if debug:
            sample_meta = sample_meta[:1000]

        if sample_type == "":
            if subset == 'train':
                sample_type = 'random'
            elif subset == 'eval':
                sample_type = 'middle'
            else:
                raise ValueError(f"Undefined sample type {sample_type}")

        if kwargs['mats'] and (subset == 'train'):
            video_duration = video_duration * 1.5
            num_frames = int(num_frames * 1.5)

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
            category_lookup[row['mid']] = row['category']

        return index_lookup, category_lookup



class VGGSoundRetrievalDataset(BaseFinetuneDataset):
    def __init__(self,
                 subset: str = 'train',
                 base_path: str = "./",
                 meta_path: str = "./",
                 target_task: str = 'All',
                 label_smooth: float = 0,
                 transform: torchvision.transforms = None,
                 num_frames: int = 4,
                 video_duration: float = 4.0,
                 audio_duration: float = 10.0,
                 decode_audio: bool = True,
                 decode_video: bool = True,
                 sample_type: str = "",
                 debug: bool = False,
                 **kwargs,
                 ) -> None:

        print("loading metadata for vggsound")
        # Three options for target_task 1. Origin 2. All 3. target task name
        # Use all VGGSound dataset
        if target_task == 'Origin':
            meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_test_5_per_class_for_retrieval_cleaned.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label', 'category'])
        # Use subset of VGGSound dataset
        else:
            meta_data = pd.read_csv(os.path.join(meta_path, 'vgg_test_5_per_class_for_retrieval_cleaned_shrink8.csv'),
                                    names=['vid', 'frames_path', 'wav', 'label', 'category'])

        if not target_task in ['All', 'Origin']:
            meta_data = meta_data.query(f"category=='{target_task}'")
        sample_meta = self.pro_data(meta_data, base_path)
        self.category_list = meta_data['category'].unique()

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
            decode_video=decode_video,
            sample_type=sample_type,
            label_smooth=label_smooth,
        )

    def pro_data(self, meta_data, base_path):
        sample_meta = []
        for data_info in meta_data.itertuples():
            sample_meta.append(
                [data_info[1], os.path.join(base_path, data_info[2]),
                 os.path.join(base_path, data_info[3]), data_info[5]])
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
            category_lookup[row['mid']] = row['category']

        return index_lookup, category_lookup

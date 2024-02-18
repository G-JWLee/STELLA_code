
import pandas as pd
import os

ground_dir = '/dataset_path'

label_info_file = '/d1/dataset/AudioSet/data/class_labels_indices_audioset.csv'

split_file_list = [
    ground_dir + '/AudioSet/data/valid_train_uni.csv',
    ground_dir + '/AudioSet/data/valid_test_uni.csv',
]
save_path_list = [
    ground_dir + '/AudioSet/data/audioset_2m_cleaned.csv',
    ground_dir + '/AudioSet/data/audioset_2m_eval_cleaned.csv',
]
time_info_file_list = [
    ground_dir + '/AudioSet/data/unbalanced_train_segments.csv',
    ground_dir + '/AudioSet/data/eval_segments.csv',
]

# Follow the change format of the VGGSound experiment's csv files to
def valid_mp4_2_valid_split(label_file, time_info_file, split_file, save_path):
    meta_data = pd.read_csv(split_file, names=['videoid', 'video_path', 'label'])

    label_info = pd.read_csv(label_file, header=0)
    time_info = pd.read_csv(time_info_file, sep=', ', header=2)
    time_list = []
    for ytid, start, end, labels in time_info.values:
        time_list.append({'video_id': ytid, 'start': start, 'end': end, 'labels': labels})
    time_info = pd.DataFrame(time_list)

    valid_meta_data = pd.DataFrame(columns = ['vid', 'frames_path', 'wav', 'label', 'category'])

    for data_info in meta_data.itertuples():
        video_id = data_info[1]
        label = data_info[3]  # Single label to find category
        category = label_info.query(f"display_name == @label")['category'].item()

        info = time_info.query(f"video_id == '{video_id}'")
        video_start = info["start"].item()
        video_end = info["end"].item()
        multi_label = info["labels"].item().replace('"', '')  # AudioSet is multi-label dataset
        valid_label = []
        for label in multi_label.split(','):
            if not label_info.query(f'mid == "{label}"').empty:
                valid_label.append(label)
        assert len(valid_label) != 0
        valid_label = ','.join(valid_label)

        universal_video_path = os.path.join('AudioSet/data', 'frames', video_id + '_' +
                                            f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}")
        universal_audio_path = os.path.join('AudioSet/data', 'audio', video_id + '_' +
                                            f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}" + '.mp3')
        video_path = os.path.join(ground_dir, universal_video_path)
        audio_path = os.path.join(ground_dir, universal_audio_path)
        # Since some videos are not accessible, take only valid ones.
        if os.path.isdir(video_path) and os.path.isfile(audio_path):
            new_df = pd.DataFrame.from_dict([{'vid': video_id, 'frames_path': universal_video_path, 'wav': universal_audio_path,
                                              'label': valid_label, 'category': category}])
            valid_meta_data = pd.concat([valid_meta_data, new_df])
        else:
            print(video_path, audio_path, "not exist!")

    valid_meta_data.to_csv(save_path, header=False)

if __name__ == '__main__':
    for split_file, save_path, time_info_file in zip(split_file_list, save_path_list, time_info_file_list):
        valid_mp4_2_valid_split(label_info_file, time_info_file, split_file, save_path)

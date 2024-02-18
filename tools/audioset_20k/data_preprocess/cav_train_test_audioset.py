
import pandas as pd
import os
import json

ground_dir = '/dataset_path'
dataset = 'AudioSet-20k'
# dataset = 'AudioSet'

label_info_file = '/d1/dataset/AudioSet-20k/data/class_labels_indices.csv'

json_file_list = [
    ground_dir + '/AudioSet-20k/data/audioset_20k_cleaned.json',
    ground_dir + '/AudioSet-20k/data/audioset_eval_cleaned.json',
    ground_dir + '/AudioSet-20k/data/audioset_eval_5_per_class_for_retrieval_cleaned.json',
]
save_path_list = [
    ground_dir + '/AudioSet-20k/data/audioset_20k_cleaned.csv',
    ground_dir + '/AudioSet-20k/data/audioset_eval_cleaned.csv',
    ground_dir + '/AudioSet-20k/data/audioset_eval_5_per_class_for_retrieval_cleaned.csv',
]
time_info_file_list = [
    ground_dir + '/AudioSet-20k/balanced_train_segments.csv',
    ground_dir + '/AudioSet-20k/eval_segments.csv',
    ground_dir + '/AudioSet-20k/eval_segments.csv',
]

# Make valid csv file from cav json file while checking if the data actually exists.
def cav_json_2_valid_csv(label_file, time_info_file, json_file, save_path):
    with open(json_file, 'r') as f:
        data_json = json.load(f)

    data_list = data_json['data']
    label_info = pd.read_csv(label_file, header=0)
    time_info = pd.read_csv(time_info_file, sep=', ', header=2)
    time_list = []
    for ytid, start, end, labels in time_info.values:
        time_list.append({'video_id': ytid, 'start': start, 'end': end, 'labels': labels})
    time_info = pd.DataFrame(time_list)

    valid_meta_data = pd.DataFrame(columns = ['vid', 'frames_path', 'wav', 'label'])

    for data_info in data_list:
        video_id = data_info['video_id']
        label = data_info['labels']
        info = time_info.query(f"video_id == '{video_id}'")
        video_start = info["start"].item()
        video_end = info["end"].item()
        universal_video_path = os.path.join('AudioSet-20k/data', 'frames', video_id + '_' +
                                            f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}")
        universal_audio_path = os.path.join('AudioSet-20k/data', 'audio', video_id + '_' +
                                            f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}" + '.mp3')
        video_path = os.path.join('/d1/dataset', universal_video_path)
        audio_path = os.path.join('/d1/dataset', universal_audio_path)
        # Since some videos are not accessible, take only valid ones.
        if os.path.isdir(video_path) and os.path.isfile(audio_path):
            new_df = pd.DataFrame.from_dict([{'vid': video_id, 'frames_path': universal_video_path, 'wav': universal_audio_path,
                                              'label': label}])
            valid_meta_data = pd.concat([valid_meta_data, new_df])
        else:
            print(video_path, audio_path, "not exist!")

    valid_meta_data.to_csv(save_path, header=False)


if __name__ == '__main__':
    for json_file, save_path, time_info_file in zip(json_file_list, save_path_list, time_info_file_list):
        cav_json_2_valid_csv(label_info_file, time_info_file, json_file, save_path)

import os
import json
import pandas as pd
from multiprocessing import Pool

ground_dir = '/dataset_path'

train_json_file = ground_dir + '/AudioSet-20k/data/audioset_20k_cleaned.json'
test_json_file = ground_dir + '/AudioSet-20k/data/audioset_eval_cleaned.json'
train_time_info_file = ground_dir + '/AudioSet-20k/balanced_train_segments.csv'
test_time_info_file = ground_dir + '/AudioSet-20k/eval_segments.csv'
target_fold = ground_dir + '/AudioSet-20k/data/train_audio'

os.makedirs(target_fold, exist_ok=True)

def extract_audio(videoinfos):
    input_video, videoname, split = videoinfos
    # first resample audio
    mp3_path = target_fold + '/' + videoname + '.mp3'
    if os.path.isfile(mp3_path):
        return

    os.system('ffmpeg -i {:s} -vn -ar 16000 {:s} -ac 1'.format(input_video, mp3_path))  # save an intermediate file

# Extract audio from clips for efficient pre-training
if __name__ == "__main__":
    file_list = []
    with open(train_json_file, 'r') as f:
        train_data_list = json.load(f)["data"]
    train_time_info = pd.read_csv(train_time_info_file, sep=', ', header=2)
    train_time_list = []
    for ytid, start, end, labels in train_time_info.values:
        train_time_list.append({'video_id': ytid, 'start': start, 'end': end, 'labels': labels})
    train_time_info = pd.DataFrame(train_time_list)

    with open(test_json_file, 'r') as f:
        test_data_list = json.load(f)["data"]
    test_time_info = pd.read_csv(test_time_info_file, sep=', ', header=2)
    test_time_list = []
    for ytid, start, end, labels in test_time_info.values:
        test_time_list.append({'video_id': ytid, 'start': start, 'end': end, 'labels': labels})
    test_time_info = pd.DataFrame(test_time_list)

    for data_info in train_data_list:
        video_id = data_info['video_id']
        info = train_time_info.query(f"video_id == '{video_id}'")
        video_start = info["start"].item()
        video_end = info["end"].item()
        video_id = video_id + '_' + f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}"
        uni_path = os.path.join('AudioSet-20k/data/balanced_train_segments/video', video_id + '.mp4')
        input_f = os.path.join(ground_dir, uni_path)
        if os.path.isfile(input_f):
            file_list.append([input_f, video_id, 'train'])
        else:
            continue

    for data_info in test_data_list:
        video_id = data_info['video_id']
        info = test_time_info.query(f"video_id == '{video_id}'")
        video_start = info["start"].item()
        video_end = info["end"].item()
        video_id = video_id + '_' +f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}"
        uni_path = os.path.join('AudioSet-20k/data/eval_segments/video', video_id + '.mp4')
        input_f = os.path.join(ground_dir, uni_path)
        if os.path.isfile(input_f):
            file_list.append([input_f, video_id, 'test'])
        else:
            continue

    pool = Pool(32)
    pool.map(extract_audio, tuple(file_list))


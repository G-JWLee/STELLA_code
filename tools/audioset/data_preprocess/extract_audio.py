import os
import pandas as pd
from multiprocessing import Pool

ground_dir = '/dataset_path'

train_csv_file = ground_dir + '/AudioSet/data/valid_train_uni.csv'
test_csv_file = ground_dir + '/AudioSet/data/valid_test_uni.csv'
retri_csv_file = ground_dir + '/AudioSet/data/valid_retri_uni.csv'
target_fold = ground_dir + '/AudioSet/data/audio'

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
    train_meta_data = pd.read_csv(train_csv_file, names=['videoid', 'video_path', 'label'])
    test_meta_data = pd.read_csv(test_csv_file, names=['videoid', 'video_path', 'label'])
    retri_meta_data = pd.read_csv(retri_csv_file, names=['videoid', 'video_path', 'label'])

    for data in train_meta_data.itertuples():
        split = data[2].split('/')
        uni_path = '/'.join(split)
        input_f = os.path.join(ground_dir, uni_path)
        videoid = split[-1].split('.')[0]
        file_list.append([input_f, videoid, 'train'])

    for data in test_meta_data.itertuples():
        split = data[2].split('/')
        uni_path = '/'.join(split)
        input_f = os.path.join(ground_dir, uni_path)
        videoid = split[-1].split('.')[0]
        file_list.append([input_f, videoid, 'test'])

    for data in retri_meta_data.itertuples():
        split = data[2].split('/')
        uni_path = '/'.join(split)
        input_f = os.path.join(ground_dir, uni_path)
        videoid = split[-1].split('.')[0]
        file_list.append([input_f, videoid, 'test'])

    pool = Pool(32)
    pool.map(extract_audio, tuple(file_list))

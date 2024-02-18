import os
import time
import sys
import json
import subprocess
import pandas as pd
import shutil
from multiprocessing import Pool, Value

ground_dir = '/dataset_path'
raw_data_dir = ground_dir + '/MSRVTT/data'
save_path = raw_data_dir + '/audio'
# subset = "test_list_miech.txt"  # Test
subset = "train_list_miech.txt"  # Train

os.makedirs(save_path, exist_ok=True)

def extract_audio(videoinfos):
    video_name = videoinfos
    # first resample audio
    input_video = raw_data_dir + '/videos/all/' + video_name + '.mp4'
    mp3_path = save_path + '/' + video_name + '.mp3'


    if os.path.isfile(mp3_path):
        return

    # Extract only the <10s audio.
    os.system('ffmpeg -i {:s} -vn -ar 16000 {:s} -ac 1 -t 10'.format(input_video, mp3_path))  # save an intermediate file


if __name__ == "__main__":
    file_list = []
    json_fp = os.path.join(raw_data_dir, 'annotation', 'MSR_VTT.json')
    data = json.load(open(json_fp, 'r'))
    df = pd.DataFrame(data['annotations'])

    split_dir = os.path.join(raw_data_dir, 'high-quality', 'structured-symlinks', subset)
    subset_df = pd.read_csv(split_dir, names=['videoid'])

    df = df[df['image_id'].isin(subset_df['videoid'])]

    metadata = df.groupby(['image_id'])['caption'].apply(list)
    for i, key in enumerate(metadata.keys()):
        file_list.append(key)

    # extract_audio(file_list[0])
    pool = Pool(32)
    pool.map(extract_audio, tuple(file_list))

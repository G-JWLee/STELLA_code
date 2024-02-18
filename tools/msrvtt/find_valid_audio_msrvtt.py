import os
import time
import sys
import json
import glob
import subprocess
import pandas as pd
import shutil
from multiprocessing import Pool, Value
from decord import AudioReader
import numpy as np

# Erase invalid videos(or audios) in vggsound.

ground_dir = '/dataset_path'
raw_data_dir = ground_dir + '/MSRVTT/data'
# subset = "test_list_miech.txt"  # Test
subset = "train_list_miech.txt"  # Train

def log_invalid_audios(infos):
    video_name = infos
    audio_path = os.path.join(raw_data_dir, 'audio', video_name + '.mp3')

    assert os.path.exists(audio_path)

    cmd = f'ffmpeg -v error -i {audio_path} -f null -'
    if subprocess.call(cmd, shell=True):
       # Failed to read the video file.
        remove_cmd = f'rm -rf {audio_path}'
        print(remove_cmd)
        subprocess.call(remove_cmd, shell=True)
        return
    try:
        audio = AudioReader(audio_path)
    except:
        remove_cmd = f'rm -rf {audio_path}'
        print('invalid audio')
        print(remove_cmd)
        subprocess.call(remove_cmd, shell=True)
        del audio
        return

    return

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

    valid_file_list = glob.glob(os.path.join(raw_data_dir, 'audio', '*.mp3'))
    valid_file_list = [os.path.basename(file_name).split('.')[0] for file_name in valid_file_list]

    file_list = [file_name for file_name in valid_file_list if file_name in file_list]
    # log_invalid_audios(file_list[0])
    pool = Pool(32)
    pool.map(log_invalid_audios, tuple(valid_file_list))

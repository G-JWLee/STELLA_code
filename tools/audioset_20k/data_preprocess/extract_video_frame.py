# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_video_frame.py

import os
import glob
import json
import pandas as pd
from multiprocessing import Pool
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import subprocess

ground_dir = '/dataset_path'

train_json_file = ground_dir + '/AudioSet-20k/data/audioset_20k_cleaned.json'
test_json_file = ground_dir + '/AudioSet-20k/data/audioset_eval_cleaned.json'
train_time_info_file = ground_dir + '/AudioSet-20k/balanced_train_segments.csv'
test_time_info_file = ground_dir + '/AudioSet-20k/eval_segments.csv'
target_train_fold = ground_dir + '/AudioSet-20k/data/train_frames'
target_test_fold = ground_dir + '/AudioSet-20k/data/test_frames'

os.makedirs(target_train_fold, exist_ok=True)
os.makedirs(target_test_fold, exist_ok=True)

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()])



def extract_frame(videoinfos):
    input_video_path, video_id, split = videoinfos
    extract_frame_num = 80
    if split == 'train':
        target_fold_path = target_train_fold
    else:
        target_fold_path = target_test_fold
    target_fold = target_fold_path

    # Skip already existing extracted videos
    directory = target_fold + '/' + video_id
    if os.path.isdir(directory):
        file_count = len(glob.glob(directory + '/*.jpg'))
        if file_count >= extract_frame_num:
            return

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

    os.makedirs(directory, exist_ok=True)
    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        print('Extract frame {:d} from original frame {:d}, total video frame {:d} at frame rate {:d}.'.format(i, frame_idx, total_frame_num, int(fps)))
        try:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            _, frame = vidcap.read()
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            remove_cmd = f'rm -rf {directory}'  # Remove invalid video frames directory
            print(remove_cmd)
            subprocess.call(remove_cmd, shell=True)
            print(directory, 'not exist')
            return
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        save_image(image_tensor, os.path.join(target_fold, video_id, 'frame_{:d}.jpg'.format(i)))


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
    pool.map(extract_frame, tuple(file_list))


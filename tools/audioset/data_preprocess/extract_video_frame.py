# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_video_frame.py

import os
import glob
import pandas as pd
from multiprocessing import Pool
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import subprocess

ground_dir = '/dataset_path'

train_csv_file = ground_dir + '/AudioSet/data/valid_train_uni.csv'
test_csv_file = ground_dir + '/AudioSet/data/valid_test_uni.csv'
retri_csv_file = ground_dir + '/AudioSet/data/valid_retri_uni.csv'
target_fold_path = ground_dir + '/AudioSet/data/frames'

os.makedirs(target_fold_path, exist_ok=True)

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()])

def extract_frame(videoinfos):
    input_video_path, video_id, split = videoinfos
    extract_frame_num = 80
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
    train_meta_data = pd.read_csv(train_csv_file, names=['videoid', 'video_path', 'label'])
    train_meta_data = train_meta_data[100000:]
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
    pool.map(extract_frame, tuple(file_list))

# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_video_frame.py

import os
import time
import sys
import glob
import json
import subprocess
import pandas as pd
import shutil
from multiprocessing import Pool, Value
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

ground_dir = '/dataset_path'
raw_data_dir = ground_dir + '/MSRVTT/data'
save_path = raw_data_dir + '/frames'
# subset = "test_list_miech.txt"  # Test
subset = "train_list_miech.txt"  # Train

os.makedirs(save_path, exist_ok=True)

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()])

def extract_frame(videoinfos):
    video_name = videoinfos
    # first resample audio
    input_video_path = raw_data_dir + '/videos/all/' + video_name + '.mp4'
    extract_frame_num = 40

    # Skip already existing extracted videos
    directory = save_path + '/' + video_name
    if os.path.isdir(directory):
        file_count = len(glob.glob(directory + '/*.jpg'))
        if file_count >= extract_frame_num:
            return

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # this is to avoid vggsound video's bug on not accurate frame count
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
            remove_cmd = f'rm -rf {directory}'
            print(remove_cmd)
            subprocess.call(remove_cmd, shell=True)
            print(directory, 'not exist')
            return
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        save_image(image_tensor, os.path.join(save_path, video_name, 'frame_{:d}.jpg'.format(i)))


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

    pool = Pool(32)
    pool.map(extract_frame, tuple(file_list))
    # extract_frame(file_list[0])

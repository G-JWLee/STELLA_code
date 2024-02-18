# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_video_frame.py

import os
import glob
import pandas as pd
from multiprocessing import Pool, Value
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import subprocess

ground_dir = '/dataset_path'
raw_data_dir = ground_dir + '/AVE_Dataset/AVE'

data_file = ground_dir + '/AVE_Dataset/data/Annotations.txt'
target_fold = ground_dir + '/AVE_Dataset/data/frames'

os.makedirs(target_fold, exist_ok=True)

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()])

def extract_frame(videoinfos):
    # TODO: you can define your own way to extract video_id
    input_video_path, video_id = videoinfos
    extract_frame_num = 40

    # Skip already existing extracted videos
    directory = target_fold + '/' + video_id
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
        save_image(image_tensor, os.path.join(target_fold, video_id, 'frame_{:d}.jpg'.format(i)))


if __name__ == "__main__":
    file_list = []
    raw_gt = pd.read_csv(data_file, sep="&", header=0)

    for data_instance in raw_gt.iterrows():
        file_name = data_instance[1][1]
        input_f = os.path.join(raw_data_dir, file_name + '.mp4')
        file_list.append([input_f, file_name])

    pool = Pool(32)
    pool.map(extract_frame, tuple(file_list))
    # extract_frame(file_list[0])

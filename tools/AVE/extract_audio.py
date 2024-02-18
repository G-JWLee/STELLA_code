import os
import pandas as pd
from multiprocessing import Pool, Value

ground_dir = '/dataset_path'
raw_data_dir = ground_dir + '/AVE_Dataset/AVE'

data_file = ground_dir + '/AVE_Dataset/data/Annotations.txt'
target_fold = ground_dir + '/AVE_Dataset/data/audio'

os.makedirs(target_fold, exist_ok=True)

def extract_audio(videoinfos):
    input_video, videoname = videoinfos
    # first resample audio
    mp3_path = target_fold + '/' + videoname + '.mp3'
    if os.path.isfile(mp3_path):
        return

    os.system('ffmpeg -i {:s} -vn -ar 16000 {:s} -ac 1'.format(input_video, mp3_path))  # save an intermediate file


if __name__ == "__main__":
    file_list = []
    raw_gt = pd.read_csv(data_file, sep="&", header=0)

    for data_instance in raw_gt.iterrows():
        file_name = data_instance[1][1]
        input_f = os.path.join(raw_data_dir, file_name + '.mp4')
        file_list.append([input_f, file_name])

    pool = Pool(32)
    pool.map(extract_audio, tuple(file_list))

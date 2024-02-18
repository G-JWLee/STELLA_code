import os
from multiprocessing import Pool
import subprocess

video_dir = '/dataset_path/AudioSet/data/video'

# Erase invalid extracted frames in the AudioSet.
def log_invalid_videos(infos):
    index, uid = infos[0], infos[1]
    video_path = os.path.join(video_dir, uid)

    assert os.path.exists(video_path)

    cmd = f'ffmpeg -v error -i {video_path} -f null -'
    if subprocess.call(cmd, shell=True):
       # Failed to read the video file.
        remove_cmd = f'rm -rf {video_path}'
        print(remove_cmd)
        subprocess.call(remove_cmd, shell=True)
        return

    return

if __name__ == "__main__":

    file_list = []
    downloaded = os.listdir(video_dir)
    for id, video in enumerate(downloaded):
        file_list.append([id, video])

    pool = Pool(32)
    pool.map(log_invalid_videos, tuple(file_list))

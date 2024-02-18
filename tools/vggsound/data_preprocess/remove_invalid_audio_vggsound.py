import os
import numpy as np
from multiprocessing import Pool
import subprocess
from decord import AudioReader

video_dir = '/dataset_path/vggsound/data/audio'

# Erase invalid extracted audio in the VGGSound.
def log_invalid_audios(infos):
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
    try:
        audio = AudioReader(video_path)
    except:
        remove_cmd = f'rm -rf {video_path}'
        print('invalid audio')
        print(remove_cmd)
        subprocess.call(remove_cmd, shell=True)
        del audio
        return

    if np.abs(audio._array).mean() == 0:
        del audio
        remove_cmd = f'rm -rf {video_path}'
        print('no audio')
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
    pool.map(log_invalid_audios, tuple(file_list))

import os
import json
import pandas as pd


ground_dir = '/dataset_path'

label_info_file = ground_dir + '/AudioSet/data/class_labels_indices_audioset.csv'
retri_info_path = ground_dir + '/AudioSet/data/cl_retrieval_segments.csv'
time_info_path = ground_dir + '/AudioSet/data/eval_segments.csv'

retri_save_path = ground_dir + '/AudioSet/data/audioset_2m_eval_5_per_class_for_retrieval_cleaned.csv'

# Make valid csv file from cav json file (cl_retrieval_segments.csv, processed earlier stage) while checking if the data actually exists.
# Same as continual vggsound dataset, If the high-level category includes the class, we use them as retrieval task data.
def cav_category_retrieval_task(label_file, time_info_file, retri_info_file, save_path):

    retri_info = pd.read_csv(retri_info_file, header=0)
    label_info = pd.read_csv(label_file, header=0)

    valid_meta_data = pd.DataFrame(columns = ['vid', 'frames_path', 'wav', 'label', 'category'])

    for data_info in retri_info.itertuples():
        video_id = data_info[1]
        label = data_info[-1]
        label = label.split(',')
        category = label_info.query(f"mid == @label")['category']

        video_start = data_info[2]
        video_end = data_info[3]
        multi_label = ','.join(label)
        universal_video_path = os.path.join('AudioSet/data', 'frames', video_id + '_' +
                                            f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}")
        universal_audio_path = os.path.join('AudioSet/data', 'audio', video_id + '_' +
                                            f"{int(float(video_start) * 1000)}" + '_' + f"{int(float(video_end) * 1000)}" + '.mp3')
        video_path = os.path.join(ground_dir, universal_video_path)
        audio_path = os.path.join(ground_dir, universal_audio_path)
        # Since some videos are not accessible, take only valid ones.
        if os.path.isdir(video_path) and os.path.isfile(audio_path):
            new_df = pd.DataFrame.from_dict([{'vid': video_id, 'frames_path': universal_video_path, 'wav': universal_audio_path,
                                              'label': multi_label, 'category': category}])
            valid_meta_data = pd.concat([valid_meta_data, new_df])
        else:
            print(video_path, audio_path, "not exist!")

    valid_meta_data.to_csv(save_path, header=False)


if __name__ == '__main__':
    cav_category_retrieval_task(label_info_file, time_info_path, retri_info_path, retri_save_path)

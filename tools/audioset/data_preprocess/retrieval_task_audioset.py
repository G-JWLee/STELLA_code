import sys
import pandas as pd
import json

ground_dir = '/v6/jaewoo/dataset'
write_ground_dir = '/d1/dataset'

label_info_file = ground_dir + '/AudioSet/data/class_labels_indices_audioset.csv'
retri_json_file = ground_dir + '/AudioSet/data/audioset_eval_5_per_class_for_retrieval_cleaned.json'
retri_save_path = write_ground_dir + '/AudioSet/data/cl_retrieval_segments.csv'
time_info_path = ground_dir + '/AudioSet/data/eval_segments.csv'


# Make valid csv file from cav json file while checking if the data actually exists.
# Same as continual vggsound dataset, If the high-level category includes the class, we use them as retrieval task data.
# When downloading evaluation set, we
# 1.Excluded intersecting instances that live in more than two categories.
# 2.Downloaded class-wise
# Hence, we have to manually download data for retrieval task using audiosetdl
# Thus, we make cl_retrieval_segments.csv for audiosetdl following the same format of eval_segments
def cl_audioset_retrieval_csv(label_file, time_info_file, json_file, save_path):

    retri_meta_data = pd.DataFrame(columns=['# YTID', 'start_seconds', 'end_seconds', 'positive_labels'])

    with open(json_file, 'r') as f:
        data_json = json.load(f)

    data_list = data_json['data']
    label_info = pd.read_csv(label_file, header=0)
    time_info = pd.read_csv(time_info_file, sep=', ', header=2)
    time_list = []
    for ytid, start, end, labels in time_info.values:
        time_list.append({'video_id': ytid, 'start': start, 'end': end, 'labels': labels})
    time_info = pd.DataFrame(time_list)

    for data_info in data_list:
        video_id = data_info['video_id']
        label = data_info['labels']
        label = label.split(',')
        category = label_info.query(f"mid == @label")['category']
        if len(category) == 0:
            continue  # Skip the case where the class has no defined category
        else:
            category = category.to_list()
            if not all(x == category[0] for x in category):
                continue # Have to exclude sample that is located in the intersection between other classes.

        info = time_info.query(f"video_id == '{video_id}'")
        video_start = info["start"].item()
        video_end = info["end"].item()
        new_df = pd.DataFrame.from_dict([{'# YTID': video_id,
                                          'start_seconds': video_start,
                                          'end_seconds': video_end,
                                          'positive_labels': ','.join(label),
                                          }])
        retri_meta_data = pd.concat([retri_meta_data, new_df])

    retri_meta_data.to_csv(save_path, header=True, index=False)


if __name__ == '__main__':
    cl_audioset_retrieval_csv(label_info_file, time_info_path, retri_json_file, retri_save_path)

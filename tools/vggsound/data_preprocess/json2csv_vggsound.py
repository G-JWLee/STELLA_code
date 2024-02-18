
import pandas as pd
import os
import json

ground_dir = '/dataset_path'

label_info_file = ground_dir + '/vggsound/data/class_labels_indices_vgg.csv'
json_file_list = [
    ground_dir + '/vggsound/data/vgg_train_cleaned.json',
    ground_dir + '/vggsound/data/vgg_test_cleaned.json',
    ground_dir + '/vggsound/data/vgg_test_5_per_class_for_retrieval_cleaned.json',
]
save_path_list = [
    ground_dir + '/vggsound/data/vgg_train_cleaned.csv',
    ground_dir + '/vggsound/data/vgg_test_cleaned.csv',
    ground_dir + '/vggsound/data/vgg_test_5_per_class_for_retrieval_cleaned.csv'
]
category_info_file = ground_dir + '/vggsound/data/stat.csv'

# Make valid csv file from CAV json file while checking if the data actually exists.
def cav_json_2_valid_csv(label_file, category_file, json_file, save_path):
    with open(json_file, 'r') as f:
        data_json = json.load(f)

    # Label-key -> category-value matching dictionary
    categories = {}
    stat_csv_data = pd.read_csv(category_file, names=['label', 'num_samples', 'category'])
    for data_info in stat_csv_data.itertuples():
        categories[data_info[1]] = data_info[3]

    data_list = data_json['data']
    label_info = pd.read_csv(label_file, header=0)
    valid_meta_data = pd.DataFrame(columns = ['vid', 'frames_path', 'wav', 'label', 'category'])

    for data_info in data_list:
        video_id, video_start = '_'.join(data_info['video_id'].split('_')[:-1]), data_info['video_id'].split('_')[-1]
        video_start = int(video_start)
        label = data_info['labels']
        label_name = label_info.query(f"mid == '{label}'")['display_name'].item()
        original_label_name = ','.join(label_name.split('_'))
        category = categories[original_label_name]

        universal_video_path = os.path.join('vggsound/data', 'frames', video_id + '_' +
                                            f"{video_start * 1000}" + '_' f"{(video_start+10) * 1000}")
        universal_audio_path = os.path.join('vggsound/data', 'audio', video_id + '_' +
                                            f"{video_start * 1000}" + '_' + f"{(video_start+10) * 1000}.mp3")
        video_path = os.path.join(ground_dir, universal_video_path)
        audio_path = os.path.join(ground_dir, universal_audio_path)
        # Since some videos are not accessible, take only valid ones.
        if os.path.isdir(video_path) and os.path.isfile(audio_path):
            new_df = pd.DataFrame.from_dict([{'vid': video_id, 'frames_path': universal_video_path, 'wav': universal_audio_path,
                                              'label': label, 'category': category}])
            valid_meta_data = pd.concat([valid_meta_data, new_df])
        else:
            print(video_path, audio_path, "not exist!")

    valid_meta_data.to_csv(save_path, header=False)



if __name__ == '__main__':
    for json_file, save_path in zip(json_file_list, save_path_list):
        cav_json_2_valid_csv(label_info_file, category_info_file, json_file, save_path)











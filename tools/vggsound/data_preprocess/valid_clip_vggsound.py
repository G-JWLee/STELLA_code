import os
import pandas as pd

ground_dir = '/dataset_path'

train_split_path = ground_dir + '/vggsound/data/valid_train_uni.csv'
test_split_path = ground_dir + '/vggsound/data/valid_test_uni.csv'
vggsound_csv_path = ground_dir + '/vggsound/data/vggsound.csv'

# Make valid clip train/test csv file.
if __name__ == '__main__':

    meta_data = pd.read_csv(vggsound_csv_path, names=['video_uid', 'start', 'label', 'split'])

    valid_train_meta_data = pd.DataFrame(columns = ['video_uid', 'video_path', 'label'])
    valid_test_meta_data = pd.DataFrame(columns = ['video_uid', 'video_path', 'label'])

    for data_info in meta_data.itertuples():
        universal_video_path = os.path.join('vggsound/data', data_info[-1], 'video',
                                  data_info[1] + '_' + f"{int(data_info[2]) * 1000}" + '_' f"{(int(data_info[2])+10) * 1000}" + ".mp4")
        video_path = os.path.join(ground_dir, universal_video_path)
        # Since some videos are not accessible, take only valid ones.
        if os.path.isfile(video_path):
            new_df = pd.DataFrame.from_dict([{'video_uid': data_info[1], 'video_path': universal_video_path,
                                              'label': data_info[-2]}])
            if data_info[-1] == 'train':
                valid_train_meta_data = pd.concat([valid_train_meta_data, new_df])
            else:
                valid_test_meta_data = pd.concat([valid_test_meta_data, new_df])

    valid_train_meta_data.to_csv(train_split_path, header=False)
    valid_test_meta_data.to_csv(test_split_path, header=False)

    print("finished!")








import os
import pandas as pd
import re

ground_dir = '/dataset_path'


train_split_path = ground_dir + '/AudioSet/data/valid_train_uni.csv'
test_split_path = ground_dir + '/AudioSet/data/valid_test_uni.csv'
retri_split_path = ground_dir + '/AudioSet/data/valid_retri_uni.csv'
audioset_csv_path = ground_dir + '/AudioSet/data/class_labels_indices_audioset.csv'


def transform_class_name(class_name):
    # Extract the word inside parentheses
    word_in_paren = re.search(r'\((.*?)\)', class_name)
    if word_in_paren:
        word_in_paren = word_in_paren.group(1)

    # Remove parentheses and split the string into words
    words = re.sub(r'\(.*?\)', '', class_name).split()
    # Concatenate the word inside parentheses if it exists
    if word_in_paren:
        word_in_paren = word_in_paren.split()
        words = words + word_in_paren

    # Concatenate the words with underscores
    transformed = '_'.join(words)

    return transformed

# Make valid split train/test csv file for audioset pre-train tasks.
if __name__ == '__main__':

    meta_data = pd.read_csv(audioset_csv_path, header=0)

    valid_train_meta_data = pd.DataFrame(columns = ['video_uid', 'video_uid_w_time', 'video_path', 'label'])
    valid_test_meta_data = pd.DataFrame(columns = ['video_uid', 'video_uid_w_time', 'video_path', 'label'])
    valid_retri_meta_data = pd.DataFrame(columns = ['video_uid', 'video_uid_w_time', 'video_path', 'label'])

    for data_info in meta_data.itertuples():
        class_name = data_info[3]
        class_name_ = transform_class_name(class_name)

        # Train dataset class
        universal_train_video_dir = os.path.join('AudioSet/data', class_name_ + '_train', 'video')
        train_video_dir = os.path.join(ground_dir, universal_train_video_dir)

        if os.path.isdir(train_video_dir):

            train_video_files = os.listdir(train_video_dir)
            for video_file in train_video_files:
                if video_file.endswith('.mp4'):
                    video_uid_w_time = os.path.basename(video_file).split('.')[0]
                    video_uid = '_'.join(video_uid_w_time.split('_')[:-2])
                    new_df = pd.DataFrame.from_dict([{'video_uid': video_uid,
                                                      'video_uid_w_time': video_uid_w_time,
                                                      'video_path': os.path.join(universal_train_video_dir, video_file),
                                                     'label': class_name}])
                    valid_train_meta_data = pd.concat([valid_train_meta_data, new_df])

        # Test dataset class
        universal_test_video_dir = os.path.join('AudioSet/data', class_name_ + '_eval', 'video')
        test_video_dir = os.path.join(ground_dir, universal_test_video_dir)

        if os.path.isdir(test_video_dir):

            test_video_files = os.listdir(test_video_dir)
            for video_file in test_video_files:
                if video_file.endswith('.mp4'):
                    video_uid_w_time = os.path.basename(video_file).split('.')[0]
                    video_uid = '_'.join(video_uid_w_time.split('_')[:-2])
                    new_df = pd.DataFrame.from_dict([{'video_uid': video_uid,
                                                      'video_uid_w_time': video_uid_w_time,
                                                      'video_path': os.path.join(universal_test_video_dir, video_file),
                                                     'label': class_name}])
                    valid_test_meta_data = pd.concat([valid_test_meta_data, new_df])

    # Retrieval dataset
    universal_retri_video_dir = os.path.join('AudioSet/data/cl_retrieval_segments/video')
    retri_video_dir = os.path.join(ground_dir, universal_retri_video_dir)

    if os.path.isdir(retri_video_dir):

        retri_video_files = os.listdir(retri_video_dir)
        for video_file in retri_video_files:
            video_uid_w_time = os.path.basename(video_file).split('.')[0]
            video_uid = '_'.join(video_uid_w_time.split('_')[:-2])
            new_df = pd.DataFrame.from_dict([{'video_uid': video_uid,
                                              'video_uid_w_time': video_uid_w_time,
                                              'video_path': os.path.join(universal_retri_video_dir, video_file),
                                              'label': "retri"}])
            valid_retri_meta_data = pd.concat([valid_retri_meta_data, new_df])


    # Since AudioSet is multi-label dataset, there are duplicated data, drop them.
    valid_train_meta_data = valid_train_meta_data.drop_duplicates('video_uid_w_time')
    valid_train_meta_data = valid_train_meta_data.drop('video_uid_w_time', axis=1)
    valid_test_meta_data = valid_test_meta_data.drop_duplicates('video_uid_w_time')
    valid_test_meta_data = valid_test_meta_data.drop('video_uid_w_time', axis=1)
    valid_retri_meta_data = valid_retri_meta_data.drop_duplicates('video_uid_w_time')
    valid_retri_meta_data = valid_retri_meta_data.drop('video_uid_w_time', axis=1)

    valid_train_meta_data.to_csv(train_split_path, header=False)
    valid_test_meta_data.to_csv(test_split_path, header=False)
    valid_retri_meta_data.to_csv(retri_split_path, header=False)

    print("finished!")








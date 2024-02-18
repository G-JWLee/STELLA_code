
import pandas as pd

label_info_file = '/dataset_path/vggsound/data/class_labels_indices_vgg.csv'
category_info = '/dataset_path/vggsound/data/stat.csv'
save_path = '/dataset_path/vggsound/data/class_labels_indices.csv'


# Include category information to class_labels_indices_vgg.csv
if __name__ == '__main__':

    categories = {}
    stat_csv_data = pd.read_csv(category_info, names=['label', 'num_samples', 'category'])
    for data_info in stat_csv_data.itertuples():
        categories[data_info[1]] = data_info[3]

    label_info = pd.read_csv(label_info_file, header=0)
    label_info['category'] = label_info.apply(lambda x: categories[','.join(x['display_name'].split('_'))], axis=1)
    label_info.to_csv(save_path, index=False)





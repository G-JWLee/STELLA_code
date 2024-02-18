import pandas as pd
from tools.audioset import category_info, category_2_simple

ground_dir = '/dataset_path'

label_info_file = ground_dir + '/AudioSet/data/class_labels_indices.csv'
save_path = ground_dir + '/AudioSet/data/class_labels_indices_audioset.csv'

# Include category information to class_labels_indices_audioset.csv
if __name__ == '__main__':

    categories_dict = {}
    classes = []
    for categories in category_info:
        simple_cat_name = category_2_simple[categories]

        for class_name in category_info[categories]:
            categories_dict[class_name] = simple_cat_name
            classes.append(class_name)
        categories_split = categories.split('&')
        for category in categories_split:
            classes.append(category)
            categories_dict[category] = simple_cat_name

    label_info = pd.read_csv(label_info_file, header=0)
    label_info = label_info.query('display_name==@classes')
    label_info['category'] = label_info.apply(lambda x: categories_dict[x['display_name']], axis=1)
    label_info.to_csv(save_path, index=False)
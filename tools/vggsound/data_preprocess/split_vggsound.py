
import pandas as pd
import os

ground_dir = '/dataset_path'

train_csv_file = ground_dir + '/vggsound/data/vgg_train_cleaned.csv'
test_csv_file = ground_dir + '/vggsound/data/vgg_test_cleaned.csv'
retrieval_csv_file = ground_dir + '/vggsound/data/vgg_test_5_per_class_for_retrieval_cleaned.csv'
label_csv_file = ground_dir + '/vggsound/data/class_labels_indices.csv'

train_save_path = ground_dir + '/vggsound/data/vgg_train_cleaned_shrink8.csv'
valid_save_path = ground_dir + '/vggsound/data/vgg_test_cleaned_shrink8.csv'
retrieval_save_path = ground_dir + '/vggsound/data/vgg_test_5_per_class_for_retrieval_cleaned_shrink8.csv'
label_save_path = ground_dir + '/vggsound/data/class_labels_indices_shrink8.csv'

rest_save_path = ground_dir + '/vggsound/data/vgg_unused_cleaned_shrink8.csv'  # Collect unused train dataset
rest_valid_save_path = ground_dir + '/vggsound/data/vgg_unused_test_cleaned_shrink8.csv'

others_part1 = ['fireworks banging', 'slot machine', 'machine gun shooting', 'civil defense siren', 'missile launch', 'cap gun shooting',
                'printer printing', 'lathe spinning'] # combined with tools category which has 12 classes

# Make subset vggsound dataset whose classes belong to one of category and consist of 400 samples.
if __name__ == '__main__':

    train_meta_data = pd.read_csv(train_csv_file,
                                  names=['vid', 'frames_path', 'wav', 'label', 'category'])
    test_meta_data = pd.read_csv(test_csv_file,
                                 names=['vid', 'frames_path', 'wav', 'label', 'category'])
    retrieval_meta_data = pd.read_csv(retrieval_csv_file,
                                      names=['vid', 'frames_path', 'wav', 'label', 'category'])
    label_csv = pd.read_csv(label_csv_file, header=0)


    min_data_num = 400
    num_classes = 20
    valid_categories = ['sports','music','vehicle','people','animals','home+nature', 'tools+others','others']

    new_train_meta_data = pd.DataFrame(columns=['vid', 'frames_path', 'wav', 'label', 'category'])
    new_test_meta_data = pd.DataFrame(columns=['vid', 'frames_path', 'wav', 'label', 'category'])
    new_rest_meta_data = pd.DataFrame(columns=['vid', 'frames_path', 'wav', 'label', 'category'])
    new_rest_test_meta_data = pd.DataFrame(columns=['vid', 'frames_path', 'wav', 'label', 'category'])
    new_retrieval_meta_data = pd.DataFrame(columns=['vid', 'frames_path', 'wav', 'label', 'category'])
    new_label_meta_data = pd.DataFrame(columns=['index', 'mid', 'display_name', 'category'])

    for category_name in valid_categories:
        if '+' in category_name:
            category_name = list(category_name.split('+'))  # More than two categories
        else:
            category_name = [category_name]

        category_label_csv = label_csv.query(f"category == @category_name")

        num_classes_count = 0
        for _, row in category_label_csv.iterrows():
            if category_name == ['others']:  # Others_part2
                if row['display_name'] in others_part1:  # Skip classes in Others_part1
                    continue
                row['category'] = 'others_part2'
            elif category_name == ['tools','others']: # Others_part1
                if row['category'] == 'others' and not row['display_name'] in others_part1:  # Skip classes in Others_part2
                    continue
                row['category'] = 'others_part1'
            elif category_name == ['home','nature']:  # Home_nature
                row['category'] = 'home_nature'

            if num_classes_count >= num_classes:
                class_rest_meta = train_meta_data.query(f"label == '{row['mid']}'")
                new_rest_meta_data = pd.concat([new_rest_meta_data, class_rest_meta])
                class_rest_test_meta = test_meta_data.query(f"label == '{row['mid']}'")
                new_rest_test_meta_data = pd.concat([new_rest_test_meta_data, class_rest_test_meta])
            else:
                class_train_meta = train_meta_data.query(f"label == '{row['mid']}'")
                class_train_meta['category'] = row['category']
                new_train_meta_data = pd.concat([new_train_meta_data, class_train_meta.iloc[:min_data_num]])
                new_rest_meta_data = pd.concat([new_rest_meta_data, class_train_meta.iloc[min_data_num:]])

                class_test_meta = test_meta_data.query(f"label == '{row['mid']}'")
                class_test_meta['category'] = row['category']
                new_test_meta_data = pd.concat([new_test_meta_data, class_test_meta])

                class_label_meta = pd.DataFrame.from_dict([{'index': row['index'], 'mid': row['mid'], 'display_name': row['display_name'], 'category': row['category']}])
                new_label_meta_data = pd.concat([new_label_meta_data, class_label_meta])
            class_ret_meta_data = retrieval_meta_data.query(f"label == '{row['mid']}'")
            class_ret_meta_data['category'] = row['category']
            new_retrieval_meta_data = pd.concat([new_retrieval_meta_data, class_ret_meta_data])

            num_classes_count += 1

    # Shuffle dataset
    new_train_meta_data = new_train_meta_data.sample(frac=1).reset_index(drop=True)
    new_valid_meta_data = new_test_meta_data.sample(frac=1).reset_index(drop=True)
    new_rest_meta_data = new_rest_meta_data.sample(frac=1).reset_index(drop=True)
    new_retrieval_meta_data = new_retrieval_meta_data.sample(frac=1).reset_index(drop=True)

    new_train_meta_data.to_csv(train_save_path, header=False)
    new_valid_meta_data.to_csv(valid_save_path, header=False)
    new_rest_meta_data.to_csv(rest_save_path, header=False)
    new_rest_test_meta_data.to_csv(rest_valid_save_path, header=False)
    new_retrieval_meta_data.to_csv(retrieval_save_path, header=False)
    new_label_meta_data.to_csv(label_save_path)

    print("finished!")
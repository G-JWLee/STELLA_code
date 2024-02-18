import pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

ground_dir = '/dataset_path'

label_info_file = ground_dir + '/vggsound/data/class_labels_indices_shrink8.csv'

data_info_path_list = [
    ground_dir + '/vggsound/data/vgg_train_cleaned_shrink8.csv',
]

# Summarizes the statistics of each pre-train task in VGGSound.

def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)


def bar_plot(df, task, x_name, y_name, row=False):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create bar plot
    orient = 'h' if row else None
    plt.figure()
    ax = sns.barplot(x=x_name, y=y_name, data=df, orient=orient, color='darkgray')

    maximum = df[y_name].max()
    maximum = maximum//1000 * 1000
    minimum = df[y_name].min()
    minimum = minimum//1000 * 1000

    plt.xticks(fontsize=13)
    plt.yticks(np.arange(int(minimum), int(maximum)+1000, 1000), fontsize=10)

    # Set plot labels and title
    plt.title(task, fontsize=15, weight='bold')
    plt.xlabel('Category', fontsize=12, weight='bold')
    plt.ylabel('Number of video clips', fontsize=12, weight='bold')
    plt.xticks(fontsize=10)

    change_width(ax, 0.5)

    plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
    plt.show()

def summarize_vggsound_stat(data_info_file):
    meta_data = pd.read_csv(data_info_file, names=['vid','frames_path','wav','label','category'])
    label_info = pd.read_csv(label_info_file, header=0)

    category_df = pd.DataFrame()
    category_info = label_info['category'].unique()

    for category_name in category_info:

        class_df = pd.DataFrame()

        category_meta_data = meta_data.query("category == @category_name")
        print(category_name, ":", len(category_meta_data))

        if category_name.startswith('others'):
            split = category_name.split('_')
            category_name_ = '\n'.join(split)
        elif category_name == 'home_nature':
            category_name_ = 'home&\nnature'
        else:
            category_name_ = category_name
        new_cat_df = pd.DataFrame([{"category": category_name_.capitalize(), "num_instances": len(category_meta_data)}])
        category_df = pd.concat([category_df, new_cat_df])

        cat_df = label_info.query("category == @category_name")
        cat_classes = cat_df['display_name'].unique()
        for class_name in cat_classes:
            mid_name = label_info.query("display_name == @class_name")['mid'].item()
            class_meta_data = category_meta_data.query("label == @mid_name")
            print(class_name, ":", len(class_meta_data))

            new_class_df = pd.DataFrame([{"class": class_name, "num_instances": len(class_meta_data)}])
            class_df = pd.concat([class_df, new_class_df])
        class_df = class_df.reset_index(drop=True)
        bar_plot(class_df, f"{category_name}_stat", x_name="num_instances", y_name="class", row=True)

    category_df = category_df.reset_index(drop=True)
    bar_plot(category_df, "VGGSound CL dataset stat", x_name="category", y_name="num_instances")


if __name__ == '__main__':
    for data_info_path in data_info_path_list:
        summarize_vggsound_stat(data_info_path)

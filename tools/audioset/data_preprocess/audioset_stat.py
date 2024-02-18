from tools.audioset import category_info, category_2_simple
import pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

ground_dir = '/dataset_path'
save_path = '/base_path'

label_info_file = ground_dir + '/AudioSet/data/class_labels_indices_audioset.csv'

data_info_path_list = [
    ground_dir + '/AudioSet/data/audioset_2m_cleaned.csv',
]

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
    maximum = maximum//10000 * 10000
    minimum = df[y_name].min()
    minimum = minimum//10000 * 10000

    plt.xticks(fontsize=13)
    plt.yticks(np.arange(int(minimum), int(maximum)+10000, 10000), fontsize=10)

    # Set plot labels and title
    plt.title(task, fontsize=15, weight='bold')
    plt.xlabel('Category', fontsize=12, weight='bold')
    plt.ylabel('Number of video clips', fontsize=12, weight='bold')
    plt.xticks(fontsize=10)

    change_width(ax, 0.5)

    plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
    # plt.show()
    plt.savefig(os.path.join(save_path, f'STELLA_code/experiments/visualize/audioset_stat.pdf'), format='pdf', bbox_inches='tight')



def summarize_audioset_stat(data_info_file):
    meta_data = pd.read_csv(data_info_file, names=['vid','frames_path','wav','label','category'])
    label_info = pd.read_csv(label_info_file, header=0)

    category_df = pd.DataFrame()

    for category_name in category_info.keys():

        class_df = pd.DataFrame()

        simple_cat_name = category_2_simple[category_name]
        category_meta_data = meta_data.query("category == @simple_cat_name")
        print(simple_cat_name, ":", len(category_meta_data))
        new_cat_df = pd.DataFrame([{"category": simple_cat_name.capitalize(), "num_instances": len(category_meta_data)}])
        category_df = pd.concat([category_df, new_cat_df])

        for class_name in category_info[category_name]:
            mid_name = label_info.query("display_name == @class_name")['mid'].item()
            class_meta_data = category_meta_data[category_meta_data['label'].str.contains(rf'{mid_name}')]
            print(class_name, ":", len(class_meta_data))

            new_class_df = pd.DataFrame([{"class": class_name, "num_instances": len(class_meta_data)}])
            class_df = pd.concat([class_df, new_class_df])
        class_df = class_df.reset_index(drop=True)
        # bar_plot(class_df, f"{simple_cat_name}_stat", x_name="num_instances", y_name="class", row=True)

    category_df = category_df.reset_index(drop=True)
    bar_plot(category_df, "AudioSet CL dataset stat", x_name="category", y_name="num_instances")

if __name__ == '__main__':
    for data_info_path in data_info_path_list:
        summarize_audioset_stat(data_info_path)

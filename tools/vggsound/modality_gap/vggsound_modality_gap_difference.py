# import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import json

from tools import method_name_dict

palette_list=["#FF4E50", "#8C564B", "#5DADE2", "#A569BD", "#F5B041", "#1F77B4", "#594F4F"]
base_path = '/base_path'

def desaturate_color(color):
    hls_color = sns.color_palette("husl", n_colors=1, desat=0.5)[0]
    return sns.set_hls_values(color, s=hls_color[1])

palette_list=[desaturate_color(color) if idx!=len(palette_list)-1 else color for idx, color in enumerate(palette_list)]

def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)



def bar_plot(df, task, task_name, palettes):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create bar plot
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 6.5))
    ax = sns.barplot(x="methods", y=task_name, data=df, palette=palettes, ci='sd')

    # Set plot labels and title
    plt.title(task, fontsize=27, weight='bold')
    plt.xlabel('Method', fontsize=25, weight='bold')
    plt.ylabel('Difference', fontsize=25, weight='bold')
    plt.xticks(fontsize=18, weight='bold')

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.yticks(np.arange(round(minimum,2)-0.01, round(maximum,2)+0.01, 0.01), fontsize=25)
    plt.ylim([minimum-0.005, maximum+0.005])

    change_width(ax, 0.5)

    plt.show()


if __name__=='__main__':

    targets = ['sports', 'music', 'vehicle', 'people', 'animals', 'home_nature', 'others_part1', 'others_part2']
    methods = []
    random_seeds = []
    buffer_size = ''

    palettes = palette_list[:len(methods)]

    method_dir = os.path.join(base_path, 'STELLA_code/experiments/tblogs/cav_base_vggsound_pretrain/cav_base_vggsound_pretrain')

    modality_gap_dict = {}

    for method in methods:
        modality_gap_dict[method] = {}

        for seed in random_seeds:
            modality_gap_dict[method][seed] = {}

    for method in methods:
        for seed in random_seeds:

            if method == 'finetune':
                modality_gap_path = os.path.join(method_dir + '_' + method + '_' + seed, 'modality_gap_result.json')
            else:
                modality_gap_path = os.path.join(method_dir + '_' + method + '_' + seed + '_' + buffer_size, 'modality_gap_result.json')

            if not os.path.isfile(modality_gap_path):
                print("no", modality_gap_path, 'in the directory')
                continue

            with open(modality_gap_path, 'r') as f:
                modality_gap_result = json.load(f)

            task_modality_gap = {}
            for target in targets:
                task_modality_gap[target] = []

            for i in range(10, 90, 10):
                epoch_modality_gap = modality_gap_result[str(i)]
                for target in targets:
                    target_epoch_modality_gap = epoch_modality_gap[target]
                    task_modality_gap[target].append(target_epoch_modality_gap)

            modality_gap_dict[method][seed] = task_modality_gap

    modality_gap_dff_dfs = pd.DataFrame()

    for method in methods:
        method_modality_gap_diff = []
        for target_idx, target in enumerate(targets):
            if target_idx == (len(targets)-1): # Skip the last task for computing difference
                continue
            target_modality_gap_diff = []
            for seed in random_seeds:
                gap_list = np.array(modality_gap_dict[method][seed][target])
                last_gap = gap_list[-1]
                gap_diff = gap_list[:-1] - last_gap

                # max
                target_modality_gap_diff.append(np.max(gap_diff))

            method_modality_gap_diff.append(np.mean(np.array(target_modality_gap_diff)))

        new_mod_gap_df = pd.DataFrame([{'methods': method_name_dict[method],
                          'modality_gap_difference': np.mean(method_modality_gap_diff)}])
        modality_gap_dff_dfs = pd.concat([modality_gap_dff_dfs, new_mod_gap_df])

    modality_gap_dff_dfs = modality_gap_dff_dfs.reset_index(drop=True)
    bar_plot(modality_gap_dff_dfs, 'VGGSound Modality Gap Difference', 'modality_gap_difference', palettes)


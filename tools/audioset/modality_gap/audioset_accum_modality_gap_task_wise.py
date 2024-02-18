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

def dot_plot(df, task, task_name, palettes, metric):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create dot plot
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 6.5))
    ax = sns.lineplot(x="task_id", y=task_name, marker='o', linewidth=4.0,
                      markersize=16, data=df, hue="methods", palette=palettes)
    legend = ax.legend(loc='upper right', fontsize=19.8, ncol=3)
    # Set the line width in the legend
    for line in legend.get_lines():
        line.set_linewidth(3.5)

    # Set plot labels and title
    plt.title(task, fontsize=27, weight='bold')
    plt.xlabel('Pre-train task', fontsize=25, weight='bold')
    plt.ylabel(metric, fontsize=25, weight='bold')

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.xticks(list(set(df["task_id"].to_list())), fontsize=28)
    plt.yticks(np.arange(round(minimum,1)-0.04, round(maximum,1)+0.06, 0.02), fontsize=28)

    plt.ylim([minimum-0.01, maximum+0.05])

    plt.show()

if __name__=='__main__':

    targets = ['human','vehicle','nature','animal','others','home','music']
    methods = []
    random_seeds = []
    buffer_size = ''

    palettes = palette_list[:len(methods)]

    method_dir = os.path.join(base_path, 'STELLA_code/experiments/tblogs/cav_base_audioset_pretrain/cav_base_audioset_pretrain')

    modality_gap_dict = {}

    for method in methods:
        modality_gap_dict[method] = {}
        for seed in random_seeds:
            modality_gap_dict[method][seed] = {}

    for method in methods:
        for seed in random_seeds:

            if method == 'finetune':
                modality_gap_path = os.path.join(method_dir + '_' + method + '_' + seed, 'accumulated_modality_gap_result.json')
            else:
                modality_gap_path = os.path.join(method_dir + '_' + method + '_' + seed + '_' + buffer_size, 'accumulated_modality_gap_result.json')

            if not os.path.isfile(modality_gap_path):
                print("no", modality_gap_path, 'in the directory')
                continue

            with open(modality_gap_path, 'r') as f:
                modality_gap_result = json.load(f)

            task_modality_gap = []

            for i in range(15, 120, 15):
                epoch_modality_gap = modality_gap_result[str(i)]
                target_idx = i//15 - 1
                target = targets[target_idx]

                accum_mod_gap = epoch_modality_gap[target]
                task_modality_gap.append(accum_mod_gap)

            modality_gap_dict[method][seed] = task_modality_gap

    modality_dfs = pd.DataFrame()

    for method in methods:

        modality_gap = []
        for seed in random_seeds:
            modality_gap.append(np.array(modality_gap_dict[method][seed]))
        modality_gap = np.stack(modality_gap, axis=-1)
        modality_gap = np.mean(modality_gap, axis=-1)

        new_mod_gap_df = pd.DataFrame({'methods': [method_name_dict[method]] * len(targets),
                          'modality_gap': modality_gap,
                          'task_id': np.arange(1, len(targets)+1)})
        print(f'Method: {method_name_dict[method]}')
        print('Modality_gap', modality_gap)
        modality_dfs = pd.concat([modality_dfs, new_mod_gap_df])

    modality_dfs = modality_dfs.reset_index(drop=True)
    dot_plot(modality_dfs, "AudioSet Modality Gap", "modality_gap",
             palettes, 'Distance')


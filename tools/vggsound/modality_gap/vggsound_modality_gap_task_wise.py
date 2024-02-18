# import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import json

from tools import method_name_dict

palette_list=["#FF703F", "#5DADE2", "#A569BD", "#4CAF50", "#F5B041"]
base_path = '/base_path'

def dot_plot(df, task, task_name, palettes, metric):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create dot plot
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(10, 6.5))
    ax = sns.lineplot(x="task_id", y=task_name, marker='o', markersize=30, data=df, hue="methods", palette=palettes)

    # Set plot labels and title
    plt.title(task, fontsize=32)
    plt.xlabel('Pre-train task ID', fontsize=35)
    plt.ylabel(metric, fontsize=35)

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.xticks(list(set(df["task_id"].to_list())), fontsize=28)
    plt.yticks(np.arange(int(minimum)-1, int(maximum)+3, 2), fontsize=28)

    plt.ylim([minimum-1.0, maximum+3.0])

    # Increase legend box size
    ax.legend(loc='upper right', bbox_to_anchor=(1.00, 1.00), borderaxespad=0., fontsize=30)

    plt.show()

def dot_plot_gap(df, task, task_name, palettes, metric):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create dot plot
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(10, 6.5))
    ax = sns.lineplot(x="task_id", y=task_name, marker='o', markersize=30, data=df, hue="methods", palette=palettes)

    # Set plot labels and title
    plt.title(task, fontsize=32)
    plt.xlabel('Pre-train task ID', fontsize=35)
    plt.ylabel(metric, fontsize=35)

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.xticks(list(set(df["task_id"].to_list())), fontsize=28)
    plt.yticks(np.arange(round(minimum,1)-0.1, round(maximum,1)+0.1, 0.1), fontsize=28)

    plt.ylim([minimum-0.1, maximum+0.1])

    # Increase legend box size
    ax.legend(loc='upper right', bbox_to_anchor=(1.00, 1.00), borderaxespad=0., fontsize=30)

    plt.show()

if __name__=='__main__':

    targets = ['sports', 'music', 'vehicle', 'people', 'animals', 'home_nature', 'others_part1', 'others_part2']
    methods = []
    random_seeds = []
    buffer_size = ''

    tasks = ['audio2video_R1', 'video2audio_R1']
    task_titles = {'audio2video_R1': "Audio -> Video R@1",
                   'audio2video_R5': "Audio -> Video R@5",
                   'audio2video_R10': "Audio -> Video R@10",
                   'video2audio_R1': "Video -> Audio R@1",
                   'video2audio_R5': "Video -> Audio R@5",
                   'video2audio_R10': "Video -> Audio R@10"
                   }

    palettes = palette_list[:len(methods)]
    method_dir = os.path.join(base_path, 'STELLA_code/experiments/tblogs/cav_base_vggsound_pretrain/cav_base_vggsound_pretrain')

    retrieval_acc_dict = {}
    modality_gap_dict = {}

    for method in methods:
        retrieval_acc_dict[method] = {}
        modality_gap_dict[method] = {}

        for seed in random_seeds:
            retrieval_acc_dict[method][seed] ={}
            modality_gap_dict[method][seed] = {}

    for method in methods:
        for seed in random_seeds:

            retrieval_path = os.path.join(method_dir + '_' + method + '_' + seed + '_' + buffer_size, 'retrieval_result.json')
            modality_gap_path = os.path.join(method_dir + '_' + method + '_' + seed + '_' + buffer_size, 'modality_gap_result.json')

            if not os.path.isfile(retrieval_path):
                print("no", retrieval_path, 'in the directory')
                continue

            with open(retrieval_path, 'r') as f:
                retrieval_result = json.load(f)

            if not os.path.isfile(modality_gap_path):
                print("no", modality_gap_path, 'in the directory')
                continue

            with open(modality_gap_path, 'r') as f:
                modality_gap_result = json.load(f)

            task_retrieval_acc = {}
            task_modality_gap = {}
            for target in targets:
                task_retrieval_acc[target] = {}
                task_modality_gap[target] = []
                for task in tasks:
                    task_retrieval_acc[target][task] = []

            for i in range(10, 90, 10):
                epoch_retrieval_result = retrieval_result[str(i)]
                epoch_modality_gap = modality_gap_result[str(i)]
                for target in targets:
                    target_epoch_retrieval_result = epoch_retrieval_result[target]
                    for task in tasks:
                        task_name, metric = task.split('_')
                        acc = target_epoch_retrieval_result[task_name][metric]
                        task_retrieval_acc[target][task].append(acc * 100)
                    target_epoch_modality_gap = epoch_modality_gap[target]
                    task_modality_gap[target].append(target_epoch_modality_gap)

            retrieval_acc_dict[method][seed] = task_retrieval_acc
            modality_gap_dict[method][seed] = task_modality_gap

    for target_idx, target in enumerate(targets):
        task_dfs = {}
        for task in tasks:
            task_dfs[task] = pd.DataFrame()
        task_dfs['modality_gap'] = pd.DataFrame()

        for method in methods:
            for task in tasks:
                task_acc = []
                for seed in random_seeds:
                    task_acc.append(np.array(retrieval_acc_dict[method][seed][target][task][target_idx:]))
                task_acc = np.stack(task_acc, axis=-1)
                task_acc = np.mean(task_acc, axis=-1)

                new_task_df = pd.DataFrame({'methods': [method_name_dict[method]] * (len(targets) - target_idx),
                               task: task_acc,
                               'task_id': np.arange(target_idx+1, len(targets)+1)})

                task_dfs[task] = pd.concat([task_dfs[task], new_task_df])

            modality_gap = []
            for seed in random_seeds:
                modality_gap .append(np.array(modality_gap_dict[method][seed][target][target_idx:]))
            modality_gap = np.stack(modality_gap, axis=-1)
            modality_gap = np.mean(modality_gap, axis=-1)

            new_mod_gap_df = pd.DataFrame({'methods': [method_name_dict[method]] * (len(targets) - target_idx),
                              'modality_gap': modality_gap,
                              'task_id': np.arange(target_idx+1, len(targets)+1)})
            task_dfs['modality_gap'] = pd.concat([task_dfs['modality_gap'], new_mod_gap_df])

        for task in tasks:
            task_dfs[task] = task_dfs[task].reset_index(drop=True)
            metric = task_titles[task].split()[-1]
            dot_plot(task_dfs[task], f"{target.capitalize()} "+task_titles[task], task, palettes[:len(methods)], metric)

        task_dfs['modality_gap'] = task_dfs['modality_gap'].reset_index(drop=True)
        dot_plot_gap(task_dfs['modality_gap'], f"{target.capitalize()} " + "modality gap", "modality_gap",
                 palettes[:len(methods)], 'Distance')


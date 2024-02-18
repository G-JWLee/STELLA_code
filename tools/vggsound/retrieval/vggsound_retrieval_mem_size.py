import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json

from tools import method_name_dict

palette_list=["#FF4E50", "#8C564B", "#5DADE2", "#A569BD", "#F5B041", "#1F77B4", "#594F4F"]
base_path = '/base_path'

def desaturate_color(color):
    hls_color = sns.color_palette("husl", n_colors=1, desat=0.5)[0]
    return sns.set_hls_values(color, s=hls_color[1])

palette_list=[desaturate_color(color) if idx!=len(palette_list)-1 else color for idx, color in enumerate(palette_list)]


def dot_plot(df, task, task_name, palettes):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create dot plot
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 6.5))
    ax = sns.lineplot(x="buffer_size", y=task_name, linewidth=4.0,
                      marker='o', markersize=16, data=df, hue="methods", palette=palettes)
    legend = ax.legend(fontsize=20, ncol=2)
    # Set the line width in the legend
    for line in legend.get_lines():
        line.set_linewidth(3.5)

    # Set plot labels and title
    plt.title(task, fontsize=27, weight='bold')
    plt.xlabel('Rehearsal memory size', fontsize=25, weight='bold')
    plt.ylabel('Average accuracy', fontsize=25, weight='bold')

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.xticks(list(set(df["buffer_size"].to_list())), fontsize=25)
    plt.yticks(np.arange(int(minimum)-1.0, int(maximum)+1.0, 1.0), fontsize=25)

    plt.ylim([minimum-0.5, maximum+0.5])

    plt.show()

if __name__ == '__main__':

    targets = ['sports', 'music', 'vehicle', 'people', 'animals', 'home_nature', 'others_part1', 'others_part2']
    methods = []
    seeds = []
    buffer_sizes = []

    palettes = palette_list[:len(methods)]

    tasks = ['audio2video_R1', 'audio2video_R5', 'audio2video_R10', 'video2audio_R1', 'video2audio_R5',
             'video2audio_R10']
    task_titles = {'audio2video_R1': "Audio -> Video R@1",
                   'audio2video_R5': "Audio -> Video R@5",
                   'audio2video_R10': "Audio -> Video R@10",
                   'video2audio_R1': "Video -> Audio R@1",
                   'video2audio_R5': "Video -> Audio R@5",
                   'video2audio_R10': "Video -> Audio R@10"
                   }

    retrieve_dir = os.path.join(base_path, 'STELLA_code/experiments/tblogs/cav_base_vggsound_pretrain/cav_base_vggsound_pretrain')
    summarize_df = pd.DataFrame()

    for buffer_size in buffer_sizes:

        acc_avg_dict = {}
        num_tasks = len(targets)

        for method in methods:
            acc_avg_dict[method] = {}

            for task in tasks:
                acc_avg_dict[method][task] = {}

                for seed in seeds:
                    acc_avg_dict[method][task][seed] = 0

        for method in methods:
            for seed in seeds:

                if method == 'finetune':
                    path = os.path.join(retrieve_dir + '_' + method + '_' + seed,
                        'retrieval_result.json')

                elif method == 'multitask':
                    path = os.path.join(retrieve_dir + '_' + method + '_' + seed,
                                                        'retrieval_result.json')
                else:
                    path = os.path.join(retrieve_dir + '_' + method + '_' + seed + '_' + buffer_size,
                        'retrieval_result.json')

                if not os.path.isfile(path):
                    print("no", path, 'in the directory')
                    continue

                with open(path, 'r') as f:
                    result = json.load(f)

                task_acc = {}
                for target in targets:
                    task_acc[target] = {}
                    for task in tasks:
                        task_acc[target][task] = []

                if method == 'multitask':
                    epoch_result = result['10']
                    for target in targets:
                        target_epoch_result = epoch_result[target]
                        for task in tasks:
                            task_name, metric = task.split('_')
                            acc = target_epoch_result[task_name][metric]
                            task_acc[target][task].append(acc * 100)

                else:
                    for i in range(10, 90, 10):
                        epoch_result = result[str(i)]

                        for target in targets:
                            target_epoch_result = epoch_result[target]
                            for task in tasks:
                                task_name, metric = task.split('_')
                                acc = target_epoch_result[task_name][metric]
                                task_acc[target][task].append(acc * 100)

                for target_idx, target in enumerate(targets):
                    for task in tasks:
                        acc_list = np.array(task_acc[target][task])
                        last_task_acc = acc_list[-1]
                        acc_avg_dict[method][task][seed] += last_task_acc

        dfs = []
        task_dfs = {}
        for task in tasks:
            task_dfs[task] = pd.DataFrame()

        for method in methods:
            new_df = {'methods': method_name_dict[method], 'buffer_size': buffer_size}
            for task in tasks:
                for seed in seeds:
                    if seed in acc_avg_dict[method][task]:
                        acc_avg_dict[method][task][seed] /= num_tasks

                acc_avg_list = np.array(list(acc_avg_dict[method][task].values()))
                acc_avg_mean = np.round(np.mean(acc_avg_list), 2)
                acc_avg_std = np.round(np.std(acc_avg_list), 2)

                new_task_df = pd.DataFrame({'methods': [method_name_dict[method]] * len(seeds),
                                       f'{task}_avg_acc': acc_avg_list})
                task_dfs[task] = pd.concat([task_dfs[task], new_task_df])

                new_df[f"{task}_avg_acc"] = acc_avg_mean

            dfs.append(new_df)

        buffer_summary_df = pd.DataFrame(dfs)
        buffer_summary_df['audio2video_avg_acc'] = buffer_summary_df[
            [f'{task}_avg_acc' for task in tasks if task.startswith('audio2video')]].mean(axis=1)
        buffer_summary_df['video2audio_avg_acc'] = buffer_summary_df[
            [f'{task}_avg_acc' for task in tasks if task.startswith('video2audio')]].mean(axis=1)

        summarize_df = pd.concat([summarize_df, buffer_summary_df])

    summarize_df = summarize_df.reset_index(drop=True)
    dot_plot(summarize_df, 'Audio -> Video Avg Retrieval', 'audio2video_avg_acc', palettes)
    dot_plot(summarize_df, 'Video -> Audio Avg Retrieval', 'video2audio_avg_acc', palettes)




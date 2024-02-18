import pylab as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import json

from tools import method_name_dict

palette_list=["#594F4F", "#FF4E50", "#FF703F", "#5DADE2", "#A569BD", "#4CAF50", "#F5B041"]
base_path = '/base_path'

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
    plt.figure()
    ax = sns.barplot(x="methods", y=task_name, data=df, palette=palettes)

    # Set plot labels and title
    plt.title(task, fontsize=15, weight='bold')
    plt.xlabel('Method', fontsize=15, weight='bold')
    plt.ylabel(task, fontsize=15, weight='bold')
    plt.xticks(fontsize=7)

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.yticks(np.arange(int(minimum), int(maximum)+2, 1), fontsize=15)
    plt.ylim([minimum-1, maximum+1])

    change_width(ax, 0.25)
    plt.show()

if __name__ == '__main__':

    targets = ['human','vehicle','nature','animal','others','home','music']
    methods= []
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

    retrieve_dir = os.path.join(base_path, 'STELLA_code/experiments/tblogs/cav_base_audioset_pretrain/cav_base_audioset_pretrain')

    for buffer_size in buffer_sizes:

        forget_avg_dict = {}
        acc_avg_dict = {}
        backward_avg_dict = {}
        num_tasks = len(targets)

        for method in methods:
            forget_avg_dict[method] = {}
            acc_avg_dict[method] = {}

            for task in tasks:
                forget_avg_dict[method][task] = {}
                acc_avg_dict[method][task] = {}

                for seed in seeds:
                    forget_avg_dict[method][task][seed] = 0
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
                    task_acc = {}
                    epoch_result = result['15']
                    for target in targets:
                        try:
                            target_epoch_result = epoch_result[target]
                        except:
                            target_ = method_name_dict[target]
                            target_epoch_result = epoch_result[target_]

                        task_acc[target] = {}
                        for task in tasks:
                            task_acc[target][task] = []

                        for task in tasks:
                            task_name, metric = task.split('_')
                            acc = target_epoch_result[task_name][metric]
                            task_acc[target][task].append(acc * 100)

                else:
                    task_acc = {}
                    for target in targets:
                        task_acc[target] = {}
                        for task in tasks:
                            task_acc[target][task] = []

                    for i in range(15, 120, 15):
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
                        acc_diff = acc_list[:-1] - last_task_acc
                        if len(acc_diff) != 0:
                            forget_avg_dict[method][task][seed] += np.max(acc_diff)
                        acc_avg_dict[method][task][seed] += last_task_acc

        dfs = []
        for method in methods:
            new_df = {'methods': method_name_dict[method]}
            for task in tasks:
                for seed in seeds:
                    if seed in forget_avg_dict[method][task]:
                        forget_avg_dict[method][task][seed] /= (num_tasks - 1)
                        acc_avg_dict[method][task][seed] /= num_tasks

                forget_avg_list = np.array(list(forget_avg_dict[method][task].values()))
                forget_mean = np.round(np.mean(forget_avg_list), 2)
                forget_std = np.round(np.std(forget_avg_list), 2)
                acc_avg_list = np.array(list(acc_avg_dict[method][task].values()))
                acc_avg_mean = np.round(np.mean(acc_avg_list), 2)
                acc_avg_std = np.round(np.std(acc_avg_list), 2)

                print(f"Environment: {task}, {buffer_size}")

                print(method, f"avg_acc_mean: {acc_avg_mean:.2f}, avg_acc_std: {acc_avg_std:.2f}")
                print(method, f"forget_mean: {forget_mean:.2f}, forget_std: {forget_std:.2f}")

                new_df[f"{task}_avg_acc"] = acc_avg_mean
                new_df[f"{task}_avg_fgt"] = forget_mean

            dfs.append(new_df)

        summarize_df = pd.DataFrame(dfs)
        summarize_df['audio2video_avg_acc'] = summarize_df[
            [f'{task}_avg_acc' for task in tasks if task.startswith('audio2video')]].mean(axis=1)
        summarize_df['audio2video_avg_fgt'] = summarize_df[
            [f'{task}_avg_fgt' for task in tasks if task.startswith('audio2video')]].mean(axis=1)

        summarize_df['video2audio_avg_acc'] = summarize_df[
            [f'{task}_avg_acc' for task in tasks if task.startswith('video2audio')]].mean(axis=1)
        summarize_df['video2audio_avg_fgt'] = summarize_df[
            [f'{task}_avg_fgt' for task in tasks if task.startswith('video2audio')]].mean(axis=1)

        bar_plot(summarize_df,'Audio -> Video Fgt', 'audio2video_avg_fgt', palettes)
        bar_plot(summarize_df, 'Audio -> Video Avg', 'audio2video_avg_acc', palettes)
        bar_plot(summarize_df,'Video -> Audio Fgt', 'video2audio_avg_fgt', palettes)
        bar_plot(summarize_df, 'Video -> Audio Avg', 'video2audio_avg_acc', palettes)

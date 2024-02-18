import pandas as pd
import torch
import torch.distributed
import random
import os
import json
from utils import distributed_ops
from train_algo.retrieval import retrieval
from data import build_retrieval_loader
import numpy as np
from scipy import stats
from sklearn import metrics

def epoch_wrapup(phase, args, epoch, accum_epoch, loss_mtr_dict, acc_mtr_dict, model, writer=None):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    the_metric = 0

    print("")
    print("=================================================")

    if phase == 'Eval' and 'contrastive' in loss_mtr_dict.keys() and args.criterion.args.get_va_recall_metric and \
            (epoch % args.logging.retrieve_freq == 0 or epoch == args.optim.epochs):
        if 'Origin' in args.data.target_task:
            target_task = 'Origin'
        else:
            target_task = 'All'
        loader = build_retrieval_loader(
            cfg=args.data,
            augm_cfg=args.data_augm,
            target_task=target_task,
            batch_size=args.optim.per_gpu_batchsize,
            workers=args.environment.workers,
            distributed=args.environment.distributed,
        )
        retrieval(loader,
                  model,
                  args,
                  epoch=epoch + accum_epoch,
                  writer=writer,
              )


    for loss_name in loss_mtr_dict.keys():

        value = 0

        if loss_name in ["vam", 'vacls', 'contrastive']:
            acc_mtr_dict[loss_name].synchronize_between_processes()
            loss_mtr_dict[loss_name].synchronize_between_processes()

            value = acc_mtr_dict[loss_name].avg

            print(f"{loss_name}/{phase}/accuracy_epoch", value)
            print(f"{loss_name}/{phase}/loss_epoch", loss_mtr_dict[loss_name].avg)
            if writer is not None:
                writer.add_scalar(f"{loss_name}/{phase}/loss_epoch", loss_mtr_dict[loss_name].avg, epoch + accum_epoch)
                writer.add_scalar(f"{loss_name}/{phase}/accuracy_epoch", value, epoch + accum_epoch)

        elif loss_name == "mae_audio" or loss_name == "mae_frame" or loss_name == 'penalty':
            loss_mtr_dict[loss_name].synchronize_between_processes()

            value = - loss_mtr_dict[loss_name].avg

            print(f"{loss_name}/{phase}/loss_epoch", loss_mtr_dict[loss_name].avg)
            if writer is not None:
                writer.add_scalar(f"{loss_name}/{phase}/loss_epoch", loss_mtr_dict[loss_name].avg, epoch + accum_epoch)

        else:
            raise ValueError(f'Unknown loss name {loss_name}')


        the_metric += value

    print("=================================================")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return the_metric


def record_acc_by_category(acc_mtr, classes, categories, save_path, topk=(1, 5), measure='acc'):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print("")
    print("=================================================")

    preds = torch.cat(acc_mtr.preds, dim=0)
    labels = torch.cat(acc_mtr.labels, dim=0)

    # Gather all labels from distributed processes.
    preds = distributed_ops.concat_all_gather(preds)
    labels = distributed_ops.concat_all_gather(labels)

    assert preds.shape[0] == labels.shape[0]
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    if measure == 'acc':

        labels = labels.argmax(dim=-1).squeeze()

        maxk = max(topk)
        _top_k_vals, top_k_inds = torch.topk(preds, maxk, dim=1, largest=True, sorted=True)

        # (batch_size, max_k) -> (max_k, batch_size).
        top_k_inds = top_k_inds.t()
        # (batch_size, ) -> (max_k, batch_size)
        rep_k_labels = labels.view(1, -1).expand_as(top_k_inds)
        # (i, j) = 1 if top i-th prediction for the j-th sample is correct
        top_k_correct = top_k_inds.eq(rep_k_labels)

        category_wise_dict = {}
        category_wise_dict["num_samples"] = 0
        for k in topk:
            category_wise_dict[k] = 0

        for class_name, category in categories.items():
            if category not in category_wise_dict.keys():
                category_wise_dict[category] = {}
                category_wise_dict[category]["num_samples"] = 0
                for k in topk:
                    category_wise_dict[category][k] = 0

            category_wise_dict[category][class_name] = {}
            class_idx = classes[class_name]
            for k in topk:
                category_wise_dict[category][class_name][k] = {}

            label_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
            label_num_samples = None
            for k in topk:
                label_topk_correct = top_k_correct[:k, label_indices]
                label_num_samples = label_topk_correct.size(-1)
                label_top_correct = label_topk_correct.reshape(-1).float().sum()
                category_wise_dict[category][k] += label_top_correct.item()
                category_wise_dict[k] += label_top_correct.item()
                label_top_acc = label_top_correct / label_num_samples
                category_wise_dict[category][class_name][k] = label_top_acc.item()
                category_wise_dict[category][class_name]["num_samples"] = label_num_samples
                print(f"{class_name} top_{k} acc:\t{label_top_acc.item()}")

            category_wise_dict[category]["num_samples"] += label_num_samples
            category_wise_dict["num_samples"] += label_num_samples


        for category in set(category_wise_dict.keys()) - set(['num_samples']) - set(topk):
            for k in topk:
                category_wise_dict[category][k] /= category_wise_dict[category]["num_samples"]
                print(f"{category} avg_top_{k} acc:\t{category_wise_dict[category][k]}")

        for k in topk:
            category_wise_dict[k] /= category_wise_dict["num_samples"]
            print(f"avg_top_{k} acc:\t{category_wise_dict[k]}")

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "category_wise_top.json"), "w") as f1:
            json.dump(category_wise_dict, f1)

    elif measure == 'mAP':

        category_wise_dict = {}
        # Merge category and index information
        class_info = [{'mid':k, 'class_index':classes[k], 'category':categories[k]} for k in categories]
        class_info = pd.DataFrame(class_info)
        category_list = class_info['category'].unique()

        # Entire mAP
        AP = []
        num_classes = labels.shape[-1]
        for k in range(num_classes):
            avg_precision = metrics.average_precision_score(labels[:, k], preds[:, k], average=None)
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print('mAP is {:.4f}'.format(mAP))
        category_wise_dict['mAP'] = mAP

        # Category-wise mAP
        for category in category_list:

            category_AP = []

            category_classes = class_info.query(f"category == '{category}'")['class_index'].to_list()  # extract class indices correspond to the category

            category_indices = (labels[:,category_classes] == 1).any(1) # indices that contain the label of current category.
            selected_labels = labels[category_indices]
            selected_preds = preds[category_indices]

            for class_idx in category_classes:
                avg_precision = metrics.average_precision_score(selected_labels[:, class_idx], selected_preds[:, class_idx], average=None)
                category_AP.append(avg_precision)
            category_mAP = np.mean(category_AP)
            print(f'{category} mAP is {category_mAP:.4f}')
            category_wise_dict[category] = category_mAP

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "mAP.json"), "w") as f1:
            json.dump(category_wise_dict, f1)


    print("=================================================")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

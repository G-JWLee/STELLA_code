#!/usr/bin/env python
import os
import json
import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm
from utils.distributed_ops import concat_all_gather
from utils.distributed_ops import all_gather

def get_sim_mat(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    sim_mat = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mat.cpu().numpy()


def compute_metrics(x, category_mask=None):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d

    if category_mask is not None:
        ind = ind[category_mask]

    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

def print_computed_metrics_no_mr(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}'.format(r1, r5, r10))

def retrieval(loader, model, args, epoch=0, writer=None):

    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    category_dict = {}
    for i, category in enumerate(loader.dataset.category_list): # for i, category in enumerate(args.data.target_task):
        category_dict[category] = i

    A_a_feat , A_v_feat, category_info = [], [], []
    with torch.no_grad():
        for sample in tqdm.tqdm(loader):
            batch_i = loader.batch_sampler.advance_batches_seen() - 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            keys = set([k for k in sample.keys() if "video" in k or "audio" in k or "label_idx" in k])
            input = {k: v.cuda(args.environment.gpu, non_blocking=True) for k,v in sample.items() if k in keys}
            input['masked_visual'] = False  # Do not mask when doing retrieval task
            input['masked_audio'] = False
            input['modality_token'] = True
            input['retrieval'] = True

            with torch.cuda.amp.autocast():
                output = model(input)
                # mean pool, normalization all patches
                audio_output = torch.mean(output['latent_c_a'], dim=1)
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.mean(output['latent_c_v'], dim=1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)

            audio_output = audio_output.detach()
            video_output = video_output.detach()

            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

            category_indices = [category_dict[category] for category in sample['category']]
            category_indices = torch.tensor(category_indices, device=audio_output.device)
            category_info.append(category_indices)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    A_a_feat = torch.cat(A_a_feat, dim=0)
    A_v_feat = torch.cat(A_v_feat, dim=0)
    category_info = torch.cat(category_info, dim=0)

    A_a_feat = concat_all_gather(A_a_feat)
    A_v_feat = concat_all_gather(A_v_feat)
    category_info = concat_all_gather(category_info)
    result_dict = {}
    if writer is not None:

        for direction in ['audio2video', 'video2audio']:
            if direction == 'audio2video':
                # audio -> video retrieval
                sim_mat = get_sim_mat(A_a_feat, A_v_feat)

            elif direction == 'video2audio':
                # video -> audio retrieval
                sim_mat = get_sim_mat(A_v_feat, A_a_feat)

            for category_name, category_idx in category_dict.items():
                if category_name not in result_dict.keys():
                    result_dict[category_name] = {}
                category_indices = category_info == category_idx
                category_indices = category_indices.detach().cpu()
                category_result = compute_metrics(sim_mat, category_indices)
                writer.add_scalar(f"{category_name}/{direction}_r1", category_result['R1'], epoch)
                writer.add_scalar(f"{category_name}/{direction}_r5", category_result['R5'], epoch)
                writer.add_scalar(f"{category_name}/{direction}_r10", category_result['R10'], epoch)
                writer.add_scalar(f"{category_name}/{direction}_MR", category_result['MR'], epoch)
                writer.flush()

                result_dict[category_name][direction] = category_result

            result = compute_metrics(sim_mat)
            print(direction)
            print_computed_metrics(result)
            writer.add_scalar(f"{direction}/r1", result['R1'], epoch)
            writer.add_scalar(f"{direction}/r5", result['R5'], epoch)
            writer.add_scalar(f"{direction}/r10", result['R10'], epoch)
            writer.add_scalar(f"{direction}/MR", result['MR'], epoch)
            writer.flush()
            result_dict[direction] = result

        if os.path.isfile(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json')):
            with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json'), 'r') as f:
                accum_result_dict = json.load(f)
            accum_result_dict.update({epoch: result_dict})
        else:
            accum_result_dict = {}
            accum_result_dict[epoch] = result_dict
        os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix), exist_ok=True)
        with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json'), 'w') as f:
            json.dump(accum_result_dict, f)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return result_dict

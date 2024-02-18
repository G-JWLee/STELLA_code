import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.distributed_ops import concat_all_gather


def patchify_frame(vids, p=16):
    """
    imgs: (N, T, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    t = vids.shape[1]
    h = vids.shape[3] // p
    w = vids.shape[4] // p
    x = vids.reshape(shape=(vids.shape[0], t, vids.shape[2], h, p, w, p))
    x = torch.einsum('ntchpwq->nthwpqc', x)
    x = x.reshape(shape=(vids.shape[0], h * w * t, p**2 * vids.shape[2]))
    return x


def patchify_audio(specs, audio_p=[16,16]):
    """
    specs: (N, 1, T, F) or (N, L_A, 1, 16, 16)
    x: (N, L, patch_size**2 *1)
    """
    h, w = specs.shape[-2]//audio_p[0], specs.shape[-1]//audio_p[1]
    c = 1
    if len(specs.shape) == 5:
        x = specs.reshape(shape=(specs.shape[0], specs.shape[1], c, h, audio_p[0], w, audio_p[1]))
        x = torch.einsum('ntchpwq->nthwpqc', x)
        x = x.reshape(shape=(specs.shape[0], specs.shape[1] * h * w, audio_p[0]*audio_p[1] * c))
    else:
        x = specs.reshape(shape=(specs.shape[0], c, h, audio_p[0], w, audio_p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(specs.shape[0], h * w, audio_p[0]*audio_p[1] * c))
    return x


def patchify(imgs, c, h, w, p=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
    return x


def unpatchify(x, c, h, w, p=16):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    return imgs

def unpatchify_video(x, t, c, h, w, p=16):
    x = x.reshape(shape=(x.shape[0], t, h, w, p, p, c))
    x = torch.einsum('nthwpqc->ntchpwq', x)
    videos = x.reshape(shape=(x.shape[0], t, c, h * p, w * p))
    return videos
    
def denormalize(x):
    return (np.clip(x, -1.0, 1.0) + 1.0)/2.0



def compute_mae_audio(model, batch, audio_patch_size=[2, 128], norm_pix_loss=False, mae_loss_weight=1.0, reduction='mean'):
    infer = model.infer(batch, mask_audio=True, use_mae=True, compute_joint_embedding=True)

    audio = infer['audio']
    # audio = audio.transpose(2, 3)
    target = patchify_audio(audio, audio_patch_size)

    input_mask, target_mask = infer['audio_masks']
    B, _, D = infer["audio_feats"].shape
    _, _, D_ = target.shape
    target = target[~target_mask].reshape(B, -1, D_)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    pred = model.transformer.mae_score_audio(
        infer["audio_feats"][input_mask].reshape(B, -1, D))
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    if reduction == 'none':
        loss = loss.sum(dim=-1) / input_mask.sum(dim=-1)  # [N], loss for each instance
    else:
        loss = loss.sum() / input_mask.sum()  # mean loss on removed patches

    loss = loss * mae_loss_weight

    return {
        "mae_audio_loss": loss,
    }

def compute_mae_frames(model, batch, patch_size=16, norm_pix_loss=False,
                      mae_loss_weight=1.0, reduction='mean'):

    if 'video_code_inputs' in batch and batch["video_code_inputs"]:
        compute_from_v_codes = True
    else:
        compute_from_v_codes = False

    infer = model.infer(batch, mask_audio=True, mask_visual=True, use_mae=True,
                        compute_from_v_codes=compute_from_v_codes,
                        compute_unimodal_embedding=True, compute_joint_embedding=True)

    if not compute_from_v_codes:
        # Video mask reconstruction
        target_v = patchify_frame(infer['video'], patch_size)
        input_mask_v, target_mask_v  = infer['video_masks']
        B, _, D = infer["video_feats"].shape
        _, _, D_ = target_v.shape
        target_v = target_v[~target_mask_v].reshape(B, -1, D_)
        if norm_pix_loss:
            mean_v = target_v.mean(dim=-1, keepdim=True)
            var_v = target_v.var(dim=-1, keepdim=True)
            target_v = (target_v - mean_v) / (var_v + 1.e-6) ** .5
        pred_v = model.transformer.mae_score_video(
            infer["video_feats"][input_mask_v].reshape(B, -1, D))
        loss_v = (pred_v - target_v) ** 2
        loss_v = loss_v.mean(dim=-1)  # [N, L], mean loss per patch

        if reduction == 'none':
            loss_v = loss_v.sum(dim=-1) / input_mask_v.sum(dim=-1)  # [N], loss for each instance
        else:
            loss_v = loss_v.sum() / input_mask_v.sum()  # mean loss on removed patches

        loss_v = loss_v * mae_loss_weight
        if input_mask_v.sum() == 0:
            print("zero mask video!")
    else:
        if reduction == 'none':
            B, _, _ = infer["video_feats"].shape
            loss_v = torch.zeros(B).cuda(batch["video_data"].device, non_blocking=True)
        else:
            loss_v = torch.Tensor([0]).cuda(batch["video_data"].device, non_blocking=True)

    mae_return = {
        "mae_frame_loss": loss_v,
    }

    return mae_return

def compute_mae_joint_frames(model, batch, patch_size=16, audio_patch_size=[2, 128], norm_pix_loss=False,
                      mae_loss_weight=1.0, cont_loss_weight=1.0, contrastive=False, tau=0.05, reduction='mean'):

    if 'audio_code_inputs' in batch and batch["audio_code_inputs"]:
        compute_from_a_codes = True
    else:
        compute_from_a_codes = False
    if 'video_code_inputs' in batch and batch["video_code_inputs"]:
        compute_from_v_codes = True
    else:
        compute_from_v_codes = False

    infer = model.infer(batch, mask_audio=True, mask_visual=True, use_mae=True, compute_embedding=contrastive,
                        compute_from_a_codes=compute_from_a_codes, compute_from_v_codes=compute_from_v_codes,
                        compute_unimodal_embedding=True, compute_joint_embedding=True)

    if not compute_from_a_codes:
        # Audio mask reconstruction
        audio = infer['audio']

        target_a = patchify_audio(audio, audio_patch_size)
        input_mask_a, target_mask_a = infer['audio_masks']
        B, _, D = infer["audio_feats"].shape
        _, _, D_ = target_a.shape
        target_a = target_a[~target_mask_a].reshape(B, -1, D_)
        if norm_pix_loss:
            mean_a = target_a.mean(dim=-1, keepdim=True)
            var_a = target_a.var(dim=-1, keepdim=True)
            target_a = (target_a - mean_a) / (var_a + 1.e-6) ** .5

        pred_a = model.transformer.mae_score_audio(
            infer["audio_feats"][input_mask_a].reshape(B, -1, D))
        loss_a = (pred_a - target_a) ** 2
        loss_a = loss_a.mean(dim=-1)  # [N, L], mean loss per patch

        if reduction == 'none':
            loss_a = loss_a.sum(dim=-1) / input_mask_a.sum(dim=-1)  # [N], loss for each instance
        else:
            loss_a = loss_a.sum() / input_mask_a.sum()  # mean loss on removed patches

        loss_a = loss_a * mae_loss_weight
        if input_mask_a.sum() == 0:
            print("zero mask audio!")
    else:
        if reduction == 'none':
            B, _, _ = infer["audio_feats"].shape
            loss_a = torch.zeros(B).cuda(batch["video_data"].device, non_blocking=True)
        else:
            loss_a = torch.Tensor([0]).cuda(batch["video_data"].device, non_blocking=True)

    if not compute_from_v_codes:
        # Video mask reconstruction
        target_v = patchify_frame(infer['video'], patch_size)
        input_mask_v, target_mask_v  = infer['video_masks']
        B, _, D = infer["video_feats"].shape
        _, _, D_ = target_v.shape
        target_v = target_v[~target_mask_v].reshape(B, -1, D_)
        if norm_pix_loss:
            mean_v = target_v.mean(dim=-1, keepdim=True)
            var_v = target_v.var(dim=-1, keepdim=True)
            target_v = (target_v - mean_v) / (var_v + 1.e-6) ** .5
        pred_v = model.transformer.mae_score_video(
            infer["video_feats"][input_mask_v].reshape(B, -1, D))
        loss_v = (pred_v - target_v) ** 2
        loss_v = loss_v.mean(dim=-1)  # [N, L], mean loss per patch

        if reduction == 'none':
            loss_v = loss_v.sum(dim=-1) / input_mask_v.sum(dim=-1)  # [N], loss for each instance
        else:
            loss_v = loss_v.sum() / input_mask_v.sum()  # mean loss on removed patches

        loss_v = loss_v * mae_loss_weight
        if input_mask_v.sum() == 0:
            print("zero mask video!")
    else:
        if reduction == 'none':
            B, _, _ = infer["video_feats"].shape
            loss_v = torch.zeros(B).cuda(batch["video_data"].device, non_blocking=True)
        else:
            loss_v = torch.Tensor([0]).cuda(batch["video_data"].device, non_blocking=True)

    mae_return = {
        "mae_audio_loss": loss_a,
        "mae_frame_loss": loss_v,
    }

    if contrastive:
        # Masked Video-Audio contrastive loss
        if 'latent_c_a' in infer and 'latent_c_v' in infer:
            mae_return.update(compute_vacon(infer, cont_loss_weight, bidirect_contrast=True, tau=tau, reduction=reduction))

    return mae_return


def compute_vacon(infer, loss_weight=1.0, bidirect_contrast=False, tau=0.05, reduction='mean'):
    # calculate nce loss for mean-visual representation and mean-audio representation
    # Compute similarity w.r.t video, s = c_v T c_a
    # audio samples become classes to predict, video samples become one instance of batch
    if len(infer['latent_c_a'].shape) == 3:
        audio_output = torch.mean(infer['latent_c_a'], dim=1)
    else:
        audio_output = infer['latent_c_a']
    if len(infer['latent_c_v'].shape) == 3:
        video_output = torch.mean(infer['latent_c_v'], dim=1)
    else:
        video_output = infer['latent_c_v']
    audio_rep = F.normalize(audio_output, dim=-1) # B x D
    video_rep = F.normalize(video_output, dim=-1) # B x D
    valid_bs, device = len(audio_rep), audio_rep.device
    bs = valid_bs

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    audio_rep_trg = concat_all_gather(audio_rep) # B*gpus x D
    # https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
    audio_rep_trg[bs*rank:bs*(rank+1),:] = audio_rep
    # In RepLAI, they just use xv, xv_trg and xa, xa_trg to compute bidrectional contrastive loss, without above.

    if not bidirect_contrast:
        total = torch.mm(video_rep, audio_rep_trg.transpose(0, 1)) / tau  # B x B*gpus
        # Blocks negatives to be computed as anchors
        if reduction == 'none':
            nce = -torch.diag(torch.nn.functional.log_softmax(total, dim=-1)[:,rank*bs:(rank+1)*bs])[:valid_bs] * loss_weight
        else:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=-1)[:,rank*bs:(rank+1)*bs])[:valid_bs]) * loss_weight

        logits = torch.nn.functional.softmax(total, dim=-1)
        labels = torch.arange(rank*bs, (rank+1)*bs, device=total.device)
    else:
        video_rep_trg = concat_all_gather(video_rep)
        video_rep_trg[bs*rank:bs*(rank+1),:] = video_rep

        total = torch.mm(video_rep, audio_rep_trg.transpose(0, 1)) / tau  # B x B*gpus
        total_av = torch.mm(audio_rep, video_rep_trg.transpose(0, 1)) / tau
        # Blocks negatives to be computed as anchors
        if reduction == 'none':
            nce_1 = -torch.diag(torch.nn.functional.log_softmax(total, dim=-1)[:, rank * bs:(rank + 1) * bs])[:valid_bs] * loss_weight
            nce_2 = -torch.diag(torch.nn.functional.log_softmax(total_av, dim=-1)[:, rank * bs:(rank + 1) * bs])[:valid_bs] * loss_weight
            nce = (nce_1 + nce_2) / 2
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=-1)[:, rank * bs:(rank + 1) * bs])[:valid_bs]) * loss_weight
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_av, dim=-1)[:, rank * bs:(rank + 1) * bs])[:valid_bs]) * loss_weight
            nce = (nce_1 + nce_2) / 2

        logits = torch.nn.functional.softmax(total, dim=-1)
        labels = torch.arange(rank*bs, (rank+1)*bs, device=total.device)

    return {
        "contrastive_loss": nce,
        "contrastive_logits": logits,
        "contrastive_labels": labels,
        "audio_output": audio_output[:valid_bs],
        "video_output": video_output[:valid_bs],
    }

def compute_vacls(model, batch):
    device = batch['video_data'].device
    infer = model.infer(batch, compute_embedding=False, compute_joint_embedding=True)
    vacls_logits = model.transformer.vacls_classifier(infer["cls_feats"])
    vacls_labels = infer['label_idx'].to(device)

    vacls_loss = model.transformer.vacls_criterion(vacls_logits, vacls_labels)

    return {
        "vacls_loss": vacls_loss,
        "vacls_logits": vacls_logits,
        "vacls_labels": vacls_labels,
    }

def compute_embedding(model, batch):
    # Class token extracted with masked data input
    if 'masked_visual' in batch and batch['masked_visual']:
        mask_visual = True
    else:
        mask_visual = False
    if 'masked_audio' in batch and batch['masked_audio']:
        mask_audio = True
    else:
        mask_audio = False

    if 'modality_token' in batch and batch['modality_token']:
        compute_modality_embedding = True
    else:
        compute_modality_embedding = False

    if 'joint_token' in batch and batch['joint_token']:
        compute_joint_embedding = True
    else:
        compute_joint_embedding = False

    if 'audio_code_inputs' in batch and batch['audio_code_inputs']:
        compute_from_a_codes = True
    else:
        compute_from_a_codes = False
    if 'video_code_inputs' in batch and batch['video_code_inputs']:
        compute_from_v_codes = True
    else:
        compute_from_v_codes = False

    infer = model.infer(batch, mask_audio=mask_audio, mask_visual=mask_visual, use_mae=False, compute_from_a_codes=compute_from_a_codes, compute_from_v_codes=compute_from_v_codes,
                        compute_unimodal_embedding=not compute_joint_embedding, compute_joint_embedding=compute_joint_embedding,
                        compute_embedding=compute_modality_embedding)

    return {
        "embedding_a": infer["audio_feats"],
        "embedding_v": infer["video_feats"],
        "label_idx": infer['label_idx'],
        "latent_c_a": infer['latent_c_a'],
        "latent_c_v": infer['latent_c_v'],
        "inter_c_a": infer['inter_c_a'],
        "inter_c_v": infer['inter_c_v'],
    }


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
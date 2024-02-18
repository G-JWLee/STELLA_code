import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from models.avm_module import CrossModalAttention
from utils.buffer import Buffer_task_free
from einops import rearrange
import random


class STELLA(nn.Module):

    def __init__(self, model: nn.Module, embedding_dim: int, batch_size: int, device, alpha: float,
                 avm_pretrain_path: str, matching_loss_weight: float, mem_args,
                 core_video_ratio: float, core_audio_ratio: float, num_core_audio_times: int,
                 att_temperature: float, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.device = device
        self.batch_size = batch_size

        # Audio-video matching module initialization
        self.avmatching_module = CrossModalAttention(
            dim=embedding_dim,
            pretrain_path=avm_pretrain_path,
        )
        self.matching_loss_weight = matching_loss_weight
        # Der hyperparameter
        self.alpha = alpha
        # Buffer initialization
        self.buffer = Buffer_task_free(**mem_args, device=self.device)

        self.core_video_ratio = core_video_ratio
        self.core_audio_ratio = core_audio_ratio
        self.backbone.transformer.mask_comp_ratio_v = core_video_ratio
        self.backbone.transformer.mask_comp_ratio_a = core_audio_ratio
        self.num_freq_tokens = 8
        self.num_core_audio_times = num_core_audio_times

        self.att_temperature = att_temperature

        self._req_penalty = True
        self._req_opt = False

    def forward(self, inputs):
        if 'retrieval' in inputs and inputs['retrieval']:
            output = self.backbone(inputs)

            return output

        if self.training:
            inputs['masked_visual'] = False
            inputs['masked_audio'] = False
            # Does not generate computational graph while extracting intermediate embeddings.
            self.backbone.requires_grad_(requires_grad=False)
            embed_output = self.backbone(inputs)
            audio_embeds = embed_output['inter_c_a']
            video_embeds = embed_output['inter_c_v']
            N, L_V, D = video_embeds.shape
            N, L_A, D = audio_embeds.shape
            num_cand_video_tokens = int(L_V * (1 - self.core_video_ratio))
            num_cand_audio_tokens = int(L_A * (1 - self.core_audio_ratio))

            # Extract multi-modal relationship
            att_av_ids, pos_cross_att_av, att_va_ids, pos_cross_att_va, pos_v_q, pos_v_k, pos_a_q, pos_a_k,\
                n_pos_cross_attn_av, n_pos_cross_attn_va = \
                self.extract_multimodal_relationship(video_embeds, audio_embeds)

            # Train AVM module
            vam_output = self.learn_multimodal_relationship(video_embeds, audio_embeds)
            self.backbone.requires_grad_(requires_grad=True)
            _, H, _, D_h = pos_v_k.shape

            # N x H x (L_v * k_v) x D
            discri_pos_v_k = torch.gather(pos_v_k, dim=2,
                                          index=att_av_ids[:, None, :num_cand_video_tokens, None].repeat(1, H, 1, D_h))
            # N x H x (L_v * k_v) x D
            pos_v_cls = torch.gather(pos_v_q, dim=2,
                                        index=att_av_ids[:, None, :num_cand_video_tokens, None].repeat(1, H, 1, D_h))
            discri_n_pos_cross_attn_av = torch.gather(n_pos_cross_attn_av, dim=1, index=att_av_ids[:,:num_cand_video_tokens])
            # N x H x D
            pos_v_cls = torch.sum(pos_v_cls * discri_n_pos_cross_attn_av[:,None,:,None], dim=2) / torch.sum(discri_n_pos_cross_attn_av, dim=1, keepdim=True).unsqueeze(dim=1)
            # N x H x (L_a * k_a) x D
            discri_pos_a_k = torch.gather(pos_a_k, dim=2,
                                          index=att_va_ids[:, None, :num_cand_audio_tokens, None].repeat(1, H, 1, D_h))
            # N x H x (L_a * k_a) x D
            pos_a_cls = torch.gather(pos_a_q, dim=2,
                                        index=att_va_ids[:, None, :num_cand_audio_tokens, None].repeat(1, H, 1, D_h))
            discri_n_pos_cross_attn_va = torch.gather(n_pos_cross_attn_va, dim=1, index=att_va_ids[:,:num_cand_audio_tokens])
            # N x H x D
            pos_a_cls = torch.sum(pos_a_cls * discri_n_pos_cross_attn_va[:,None,:,None], dim=2) / torch.sum(discri_n_pos_cross_attn_va, dim=1, keepdim=True).unsqueeze(dim=1)

            buf_inputs = None
            if not self.buffer.is_empty():

                # Load past core patches
                buf_inputs = self.buffer.get_data(self.batch_size)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k, v in buf_inputs.items()}

                # M x D
                spu_a_cls = buf_inputs["audio_query"][:N]
                v_spu_att = torch.einsum('nhld,nhd->nhl', discri_pos_v_k, spu_a_cls) * self.avmatching_module.scale

                v_pos_att = torch.gather(pos_cross_att_av, dim=-1,
                                         index=att_av_ids[:, None, None, :num_cand_video_tokens].repeat(1, H, L_A, 1))
                v_pos_att = torch.gather(v_pos_att, dim=2,
                                         index=att_va_ids[:, None, :num_cand_audio_tokens, None].repeat(1, H, 1,
                                                                                                          num_cand_video_tokens))
                v_pos_att = torch.sum(v_pos_att * discri_n_pos_cross_attn_va[:,None,:,None], dim=2)/ torch.sum(discri_n_pos_cross_attn_va, dim=1, keepdim=True).unsqueeze(dim=1)

                v_att = torch.stack([v_spu_att, v_pos_att], dim=3) # N x H x L_v' x (2)
                v_att = v_att.softmax(dim=-1)
                v_att = v_att.mean(dim=1)

                prob_v_att = v_att[:,:,0]
                att_av_ids_restore = torch.argsort(att_av_ids, dim=1)
                prob_v_att = torch.cat([prob_v_att, torch.zeros(N, L_V - num_cand_video_tokens, device=self.device)], dim=1)
                prob_v_att = torch.gather(prob_v_att, dim=1, index=att_av_ids_restore)

                v_prune_mat = torch.bernoulli(prob_v_att).bool()
                prune_n_pos_cross_attn_av = n_pos_cross_attn_av.clone()
                prune_n_pos_cross_attn_av[v_prune_mat] = .0

                # M x D
                spu_v_cls = buf_inputs["video_query"][:N]
                a_spu_att = torch.einsum('nhld,nhd->nhl', discri_pos_a_k, spu_v_cls) * self.avmatching_module.scale

                a_pos_att = torch.gather(pos_cross_att_va, dim=-1,
                                         index=att_va_ids[:, None, None, :num_cand_audio_tokens].repeat(1, H, L_V, 1))
                a_pos_att = torch.gather(a_pos_att, dim=2,
                                         index=att_av_ids[:, None, :num_cand_video_tokens, None].repeat(1, H, 1,
                                                                                                          num_cand_audio_tokens))
                a_pos_att = torch.sum(a_pos_att * discri_n_pos_cross_attn_av[:,None,:,None], dim=2) / torch.sum(discri_n_pos_cross_attn_av, dim=1, keepdim=True).unsqueeze(dim=1)

                a_att = torch.stack([a_spu_att, a_pos_att], dim=3)  # N x H x L_a' x (2)
                a_att = a_att.softmax(dim=-1)
                a_att = a_att.mean(dim=1)

                prob_a_att = a_att[:,:,0]
                att_va_ids_restore = torch.argsort(att_va_ids, dim=1)
                prob_a_att = torch.cat([prob_a_att, torch.zeros(N, L_A - num_cand_audio_tokens, device=self.device)], dim=1)
                prob_a_att = torch.gather(prob_a_att, dim=1, index=att_va_ids_restore)
                a_prune_mat = torch.bernoulli(prob_a_att).bool()

                # Past video data info
                buf_v_prune_mat = torch.bernoulli(buf_inputs['prob_v_att']).bool()
                buf_inputs['n_attn_av'][buf_v_prune_mat] = .0

                # Concatenate video data info
                concat_n_attn_av = torch.cat([prune_n_pos_cross_attn_av, buf_inputs['n_attn_av']], dim=0)
                att_av_ids = torch.multinomial(concat_n_attn_av, L_V, replacement=False)

                video_data = torch.cat([inputs['video_data'],buf_inputs['video_data']], dim=0)

                # Past audio data info
                buf_a_prune_mat = torch.bernoulli(buf_inputs['prob_a_att']).bool()

                # Concatenate audio data info
                concat_a_prune_mat = torch.cat([a_prune_mat, buf_a_prune_mat], dim=0)
                concat_n_attn_va = torch.cat([n_pos_cross_attn_va, buf_inputs['n_attn_va']], dim=0)

                concat_a_prune_mat = self.compute_core_audio_indices(concat_n_attn_va, concat_a_prune_mat, num_cand_audio_tokens)
                concat_a_prune_mat = concat_a_prune_mat.float()
                att_va_ids = torch.argsort(concat_a_prune_mat, descending=False, dim=1)

                audio_data = torch.cat([inputs['audio_data'],buf_inputs['audio_data']], dim=0)

                self.avmatching_module.requires_grad_(requires_grad=True)
                self.backbone.requires_grad_(requires_grad=True)
            else:
                prob_v_att = torch.zeros(N, L_V)
                prob_a_att = torch.zeros(N, L_A)

                video_data = inputs['video_data']
                audio_data = inputs['audio_data']

            # Extract core patches based on attention-based importance score
            core_video_patches, core_audio_patches = self.importance_based_patch_selection(
                att_av_ids, att_va_ids, video_data, audio_data
            )
            core_inputs = {
                "video_data": core_video_patches,
                "audio_data": core_audio_patches,
                "att_map_av_ids": att_av_ids,
                "att_map_va_ids": att_va_ids,
            }

            output = self.backbone(core_inputs)
            output.update(vam_output)

            if buf_inputs is not None:
                buf_logits_a = output["audio_output"][-self.batch_size:]
                buf_logits_v = output["video_output"][-self.batch_size:]

                penalty = self.alpha * F.mse_loss(buf_inputs['logits_a'], buf_logits_a) + \
                          self.alpha * F.mse_loss(buf_inputs['logits_v'], buf_logits_v)
            else:
                penalty = torch.Tensor([0]).cuda(self.device, non_blocking=True)

            output['penalty_loss'] = penalty

            self.buffer.add_data(video_data=inputs['video_data'], audio_data=inputs['audio_data'],
                                 logits_a=output["audio_output"][:N],
                                 logits_v=output["video_output"][:N],
                                 audio_query=pos_a_cls, video_query=pos_v_cls,
                                 prob_v_att=prob_v_att, prob_a_att=prob_a_att,
                                 n_attn_av=n_pos_cross_attn_av, n_attn_va=n_pos_cross_attn_va)

        else:
            inputs['masked_visual'] = False
            inputs['masked_audio'] = False
            embed_output = self.backbone(inputs)
            audio_embeds = embed_output['inter_c_a']
            video_embeds = embed_output['inter_c_v']

            # Audio-Video Matching objective
            vam_output  = self.learn_multimodal_relationship(video_embeds, audio_embeds)

            # Audio-Video Contrastive learning and mae
            del inputs['masked_visual']
            del inputs['masked_audio']
            output = self.backbone(inputs)
            output.update(vam_output)

        return output

    def learn_multimodal_relationship(self, video_code, audio_code):

        # Perform video-audio matching objective
        pos_len = len(audio_code) // 2

        if len(audio_code) <= 2: # in case when per_batch_size = 1 or 2, have positive pair
            vam_labels = torch.ones(len(audio_code), dtype=torch.float32).to(self.device)
        else:
            neg_len = len(audio_code) - pos_len
            vam_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(self.device)
            vam_labels = vam_labels[torch.randperm(vam_labels.size(0))]

        # Audio-Video correspondence (avc)
        zero_indices = (vam_labels == 0).nonzero().view(-1)
        video_indices = torch.arange(0, len(audio_code)).to(self.device)
        # Exchange videos among audio-video match = False samples
        if len(zero_indices) != 0:
            randomized_zero_indices = copy.deepcopy(zero_indices)
            unsatisfied = True
            while unsatisfied:
                randomized_zero_indices = randomized_zero_indices[torch.randperm(randomized_zero_indices.size(0))]
                unsatisfied = False
                for  a, b in zip(zero_indices, randomized_zero_indices):
                    if a == b:
                        unsatisfied = True
                        break
            video_indices[zero_indices] = randomized_zero_indices

        vam_video_code = torch.stack(
            [
                v for v in video_code[video_indices]
            ]
        )

        code_inputs = {
            "video_data": vam_video_code,
            "audio_data": audio_code,
            "audio_code_inputs": True,
            "video_code_inputs": True,
            "joint_token": True,
        }
        output = self.backbone(code_inputs)
        vam_logits = self.avmatching_module(output["embedding_a"], output["embedding_v"])
        vam_loss = F.binary_cross_entropy_with_logits(vam_logits.squeeze(), vam_labels.squeeze()) * self.matching_loss_weight

        vam_output = {
            "vam_loss": vam_loss,
            "vam_logits": vam_logits,
            "vam_labels": vam_labels,
        }

        return vam_output


    def extract_multimodal_relationship(self, video_code, audio_code):
        pos_code_inputs = {
            "video_data": video_code,
            "audio_data": audio_code,
            "audio_code_inputs": True,
            "video_code_inputs": True,
            "joint_token": True,
        }
        self.avmatching_module.requires_grad_(requires_grad=False)
        pos_output = self.backbone(pos_code_inputs)
        pos_cross_attn_av, pos_cross_attn_va, pos_v_q, pos_v_k, pos_a_q, pos_a_k = \
            self.avmatching_module.infer_attention(pos_output['embedding_a'], pos_output['embedding_v'],
                                                   True, True,
                                                   normalize=False)
        # Use audio2video attention information to select core video patches
        n_pos_cross_attn_av = (pos_cross_attn_av/self.att_temperature).softmax(dim=-1).mean(dim=1).mean(dim=1)
        core_v_code_indices = self.compute_core_video_indices(n_pos_cross_attn_av)
        # Use video2audio attention information to select important timeline
        n_pos_cross_attn_va = (pos_cross_attn_va/self.att_temperature).softmax(dim=-1).mean(dim=1).mean(dim=1)
        core_a_code_indices = self.compute_core_video_indices(n_pos_cross_attn_va)

        self.avmatching_module.requires_grad_(requires_grad=True)

        return core_v_code_indices, pos_cross_attn_av, core_a_code_indices, pos_cross_attn_va, \
            pos_v_q, pos_v_k, pos_a_q, pos_a_k, n_pos_cross_attn_av, n_pos_cross_attn_va

    def compute_core_video_indices(self, v_val):
        v_val_ids = torch.argsort(v_val, dim=1, descending=True)
        return v_val_ids


    def compute_core_audio_indices(self, a_val, a_prune, num_cand_audio):
        # Find important timeline (16 time indices) that is most related to the current video input
        a_val = a_val.reshape(len(a_val), -1, self.num_freq_tokens) # B x t x 8(f)
        a_val = a_val.sum(dim=-1) # B x t

        a_prune = a_prune.reshape(len(a_val), -1, self.num_freq_tokens)
        a_prune_sum = a_prune.sum(dim=-1)  # B x t

        avg_a_val = F.avg_pool1d(a_val.unsqueeze(dim=1), kernel_size=self.num_core_audio_times).squeeze(dim=1)
        important_timeline = torch.multinomial(avg_a_val, avg_a_val.shape[1], replacement=False)
        N, T = a_val.shape
        num_chunk = T // self.num_core_audio_times

        for b in range(N):
            num_tokens = 0
            for t_idx in range(num_chunk):
                time_chunk = important_timeline[b][t_idx]
                num_chunk_prune_sum = a_prune_sum[b,time_chunk*self.num_core_audio_times:(time_chunk+1)*self.num_core_audio_times].sum()
                num_tokens += (self.num_freq_tokens * self.num_core_audio_times - num_chunk_prune_sum)

                if num_tokens > num_cand_audio:
                    check_chunk = a_prune[b,time_chunk*self.num_core_audio_times:(time_chunk+1)*self.num_core_audio_times].reshape(-1)
                    if (time_chunk-1 in important_timeline[b][:t_idx]) and (time_chunk+1 not in important_timeline[b][:t_idx]):
                        cum_check_chunk = torch.cumsum(torch.flip(~check_chunk, dims=(0,)), dim=0)
                        add_prune_idx = torch.where(cum_check_chunk == (num_tokens - num_cand_audio))[0][0]
                        check_chunk[-(add_prune_idx+1):] = True
                    elif (time_chunk+1 in important_timeline[b][:t_idx]) and (time_chunk-1 not in important_timeline[b][:t_idx]):
                        cum_check_chunk = torch.cumsum(~check_chunk, dim=0)
                        add_prune_idx = torch.where(cum_check_chunk == (num_tokens - num_cand_audio))[0][0]
                        check_chunk[:add_prune_idx+1] = True
                    else:
                        if random.randint(0,1) == 0:
                            cum_check_chunk = torch.cumsum(torch.flip(~check_chunk, dims=(0,)), dim=0)
                            add_prune_idx = torch.where(cum_check_chunk == (num_tokens - num_cand_audio))[0][0]
                            check_chunk[-(add_prune_idx + 1):] = True
                        else:
                            cum_check_chunk = torch.cumsum(~check_chunk, dim=0)
                            add_prune_idx = torch.where(cum_check_chunk == (num_tokens - num_cand_audio))[0][0]
                            check_chunk[:add_prune_idx + 1] = True

                    a_prune[b, time_chunk * self.num_core_audio_times:(time_chunk + 1) * self.num_core_audio_times] = check_chunk.reshape(self.num_core_audio_times,
                                                                                                                                          self.num_freq_tokens)
                    for t_prune_idx in range(t_idx+1, num_chunk):
                        prune_time_chunk = important_timeline[b][t_prune_idx]
                        a_prune[b, prune_time_chunk * self.num_core_audio_times:(prune_time_chunk + 1) * self.num_core_audio_times] = True
                    break

        a_prune = a_prune.reshape(N, -1)

        return a_prune



    def importance_based_patch_selection(self, att_av_ids, att_va_ids, video_data, audio_data):
        """
        Given importance information and original data, extract core patches
        """
        # Extract core video patches
        video_patches = video_data.transpose(1, 2)
        video_patches = rearrange(video_patches, 'b c t (h p0) (w p1) -> b c (t h w) p0 p1', p0=16, p1=16)
        video_patches = video_patches.transpose(1, 2)  # B x patches x c x 16 x 16
        N, L_V = video_patches.shape[:2]
        num_core_video_tokens = int(L_V * (1 - self.core_video_ratio))
        video_keep_ids = att_av_ids[:, :num_core_video_tokens]
        core_video_patches = torch.gather(video_patches, dim=1,
                                          index=video_keep_ids[:, :, None, None, None].repeat(1, 1, 3, 16, 16))

        # Extract core audio patches
        audio_patches = audio_data
        audio_patches = rearrange(audio_patches, 'b c (t p0) (f p1) -> b c (t f) p0 p1', p0=16, p1=16)
        audio_patches = audio_patches.transpose(1, 2)
        N, L_A = audio_patches.shape[:2]
        num_core_audio_tokens = int(L_A * (1 - self.core_audio_ratio))
        audio_keep_ids = att_va_ids[:, :num_core_audio_tokens]
        core_audio_patches = torch.gather(audio_patches, dim=1,
                                          index=audio_keep_ids[:, :, None, None, None].repeat(1, 1, 1, 16, 16))

        return core_video_patches, core_audio_patches



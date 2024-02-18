import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from models import CrossModalAttention


class Finetune_submodule(nn.Module):

    def __init__(self, model: nn.Module, embedding_dim: int, device,
                 avm_pretrain_path: str, matching_loss_weight: float, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.backbone.requires_grad_(requires_grad=False)
        self.avmatching_module = CrossModalAttention(
            dim=embedding_dim,
            pretrain_path=avm_pretrain_path,
        )

        self.matching_loss_weight = matching_loss_weight
        self.device = device

        self._req_penalty = False
        self._req_opt = False

    def forward(self, inputs):
        if 'retrieval' in inputs and inputs['retrieval']:  # For retrieval task
            return self.backbone(inputs)

        with torch.no_grad():
            inputs['masked_visual'] = False
            inputs['masked_audio'] = False
            embed_output = self.backbone(inputs)
            audio_embeds = embed_output['inter_c_a']
            video_embeds = embed_output['inter_c_v']

        vam_output = self.compute_vam(audio_embeds, video_embeds)

        return vam_output

    def compute_vam(self, audio_code, video_code, compute_av_positive=False, compute_va_positive=False):
        pos_len = len(audio_code) // 2

        if pos_len == 0: # in case when per_batch_size = 1, have positive pair
            vam_labels = torch.ones(1, dtype=torch.float32).to(self.device)
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

        if compute_av_positive or compute_va_positive:
            pos_code_inputs = {
                "video_data": video_code,
                "audio_data": audio_code,
                "audio_code_inputs": True,
                "video_code_inputs": True,
                "joint_token": True,
            }
            pos_output = self.backbone(pos_code_inputs)
            pos_cross_attn_av, pos_cross_attn_va = self.avmatching_module.infer_attention(pos_output['embedding_a'],
                                                                                          pos_output['embedding_v'],
                                                                                          compute_av_positive,
                                                                                          compute_va_positive)
        else:
            pos_cross_attn_av = pos_cross_attn_va = None

        return {
            "vam_loss": vam_loss,
            "vam_logits": vam_logits,
            "vam_labels": vam_labels,
            "cross_attn_av": pos_cross_attn_av,
            "cross_attn_va": pos_cross_attn_va,
        }


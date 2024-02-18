import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
from utils.buffer import Buffer_task_free
"""
https://github.com/NeurAI-Lab/CLS-ER/blob/main/models/clser.py
"""

class Cls_Er(nn.Module):

    def __init__(self, model: nn.Module, batch_size: int, reg_weight: float,
                 plastic_model_update_freq: float, plastic_model_alpha: float, stable_model_update_freq: float, stable_model_alpha: float,
                 mem_args, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.backbone)
        self.plastic_model.requires_grad_(requires_grad=False)
        self.stable_model = deepcopy(self.backbone)
        self.stable_model.requires_grad_(requires_grad=False)
        # set regularization weight
        self.reg_weight = reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = plastic_model_update_freq
        self.plastic_model_alpha = plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = stable_model_update_freq
        self.stable_model_alpha = stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.global_step = 0

        self.batch_size = batch_size
        self.device = device
        self.buffer = Buffer_task_free(**mem_args, device=self.device)

        self._req_penalty = True
        self._req_opt = False

    def forward(self, inputs):

        if self.training:

            if not self.buffer.is_empty():
                buf_inputs = self.buffer.get_data(self.batch_size)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k,v in buf_inputs.items()}
                buf_inputs['modality_token'] = True
                buf_inputs['masked_visual'] = True
                buf_inputs['masked_audio'] = True
                stable_model_logits = self.stable_model(buf_inputs)
                stable_model_av_prob, stable_model_va_prob,\
                    stable_v_logits, stable_a_logits = self.compute_vacon(stable_model_logits)

                plastic_model_logits = self.plastic_model(buf_inputs)
                plastic_model_av_prob, plastic_model_va_prob,\
                    plastic_v_logits, plastic_a_logits = self.compute_vacon(plastic_model_logits)

                av_label_mask = torch.eye(self.batch_size) > 0
                va_label_mask = torch.eye(self.batch_size) > 0

                av_sel_idx = stable_model_av_prob[av_label_mask] > plastic_model_av_prob[av_label_mask]
                av_sel_idx = av_sel_idx.unsqueeze(1)
                va_sel_idx = stable_model_va_prob[va_label_mask] > plastic_model_va_prob[va_label_mask]
                va_sel_idx = va_sel_idx.unsqueeze(1)

                v_emb_logits = torch.where(
                    av_sel_idx,
                    stable_v_logits,
                    plastic_v_logits
                )
                a_emb_logits = torch.where(
                    va_sel_idx,
                    stable_a_logits,
                    plastic_a_logits,
                )

                inputs['video_data'] = torch.cat((inputs['video_data'], buf_inputs['video_data']), dim=0)
                inputs['audio_data'] = torch.cat((inputs['audio_data'], buf_inputs['audio_data']), dim=0)
                output = self.backbone(inputs)

                buf_logits_a = output["audio_output"][-self.batch_size:]
                buf_logits_v = output["video_output"][-self.batch_size:]

                l_cons_a = torch.mean(self.consistency_loss(buf_logits_a, a_emb_logits.detach()))
                l_cons_v = torch.mean(self.consistency_loss(buf_logits_v, v_emb_logits.detach()))

                output['penalty_loss'] = self.reg_weight * l_cons_a + self.reg_weight * l_cons_v
            else:
                output = self.backbone(inputs)
                output['penalty_loss'] = torch.Tensor([0]).cuda(self.device, non_blocking=True)

            # Store data to buffer
            self.buffer.add_data(video_data=inputs['video_data'][:self.batch_size], audio_data=inputs['audio_data'][:self.batch_size])

            # Update the ema model
            self.global_step += 1
            if torch.rand(1) < self.plastic_model_update_freq:
                self.update_plastic_model_variables()
            if torch.rand(1) < self.stable_model_update_freq:
                self.update_stable_model_variables()

            return output

        else:
            return self.backbone(inputs)

    def compute_vacon(self, output, tau=0.05):
        audio_output = torch.mean(output['latent_c_a'], dim=1)
        video_output = torch.mean(output['latent_c_v'], dim=1)

        audio_rep = F.normalize(audio_output, dim=-1)
        video_rep = F.normalize(video_output, dim=-1)

        total_av = torch.mm(audio_rep, video_rep.transpose(0,1)) / tau
        total_va = total_av.transpose(0,1)

        prob_av = F.softmax(total_av, dim=1)
        prob_va = F.softmax(total_va, dim=1)

        return prob_av, prob_va, video_output, audio_output

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.backbone.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.backbone.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
import numpy as np
import torch
import torch.nn as nn
import torchvision
from copy import deepcopy
from torch.nn import functional as F
from utils.buffer import Buffer_task_free
"""
https://github.com/NeurAI-Lab/ESMER/blob/main/models/esmer.py
"""

class ESMER(nn.Module):

    def __init__(self, model: nn.Module, batch_size: int, reg_weight: float,
                 ema_model_update_freq: float, ema_model_alpha: float, loss_margin: float, loss_alpha: float,
                 std_margin: float, mem_args, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model

        # Initialize EMA model
        self.ema_model = deepcopy(self.backbone)
        self.ema_model.requires_grad_(requires_grad=False)
        # set regularization weight
        self.reg_weight = reg_weight
        # set parameters for ema model
        self.ema_model_update_freq = ema_model_update_freq
        self.ema_model_alpha = ema_model_alpha
        # Set loss functions
        self.consistency_loss = nn.MSELoss(reduction='none')

        # Running estimates
        self.global_step = 0
        self.loss_running_mean = 0
        self.loss_running_std = 0
        self.loss_margin = loss_margin
        self.loss_alpha = loss_alpha
        self.std_margin = std_margin

        self.batch_size = batch_size
        self.device = device
        self.buffer = Buffer_task_free(**mem_args, device=self.device)

        self._req_penalty = True
        self._req_opt = False

    def forward(self, inputs):

        if self.training:
            # =====================================================================
            # Apply Selective Cross Entropy loss
            # =====================================================================
            ema_model_output = self.ema_model(inputs, reduction='none')
            ema_model_loss = sum([v for k, v in ema_model_output.items() if "loss" in k])  # Should have reduction = 'none'

            ignore_mask = torch.zeros_like(ema_model_loss) > 0

            if self.loss_running_mean > 0:
                sample_weight = torch.where(
                    ema_model_loss >= self.loss_margin * self.loss_running_mean,
                    self.loss_running_mean / ema_model_loss,
                    torch.ones_like(ema_model_loss)
                )
                ignore_mask = ema_model_loss > self.loss_margin * self.loss_running_mean
            else:
                sample_weight = torch.ones_like(ema_model_loss)

            # =====================================================================
            # Apply Buffer loss
            # =====================================================================
            if not self.buffer.is_empty():
                buf_inputs = self.buffer.get_data(self.batch_size)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k,v in buf_inputs.items()}
                buf_inputs['modality_token'] = True
                buf_inputs['masked_visual'] = True
                buf_inputs['masked_audio'] = True

                ema_model_logits = self.ema_model(buf_inputs)
                ema_model_a_logits = torch.mean(ema_model_logits['latent_c_a'], dim=1)
                ema_model_v_logits = torch.mean(ema_model_logits['latent_c_v'], dim=1)

                # Note that this is model contains contrastive loss. Hence, we generate mask and
                # calculate the contrastive loss together with buffer data for fair comparison with other baselines
                inputs['video_data'] = torch.cat((inputs['video_data'], buf_inputs['video_data']), dim=0)
                inputs['audio_data'] = torch.cat((inputs['audio_data'], buf_inputs['audio_data']), dim=0)
                output = self.backbone(inputs, reduction='none')

                for key in output.keys():
                    if "loss" in key:
                        output[key][:len(sample_weight)] = output[key][:len(sample_weight)] * sample_weight
                        output[key] = output[key].mean()

                buf_logits_a = output["audio_output"][-self.batch_size:]
                buf_logits_v = output["video_output"][-self.batch_size:]

                l_cons_a = torch.mean(self.consistency_loss(buf_logits_a, ema_model_a_logits.detach()))
                l_cons_v = torch.mean(self.consistency_loss(buf_logits_v, ema_model_v_logits.detach()))

                output['penalty_loss'] = self.reg_weight * l_cons_a + self.reg_weight * l_cons_v

            else:
                output = self.backbone(inputs)
                output['penalty_loss'] = torch.Tensor([0]).cuda(self.device, non_blocking=True)

            # Store data to buffer
            self.buffer.add_data(video_data=inputs['video_data'][:self.batch_size][~ignore_mask],
                                 audio_data=inputs['audio_data'][:self.batch_size][~ignore_mask])

            # Update the ema model
            # Note that we assume task-free CL: no task boundary info is given.
            self.global_step += 1
            if torch.rand(1) < self.ema_model_update_freq:
                self.update_ema_model_variables()

            loss_mean, loss_std = ema_model_loss.mean(), ema_model_loss.std()
            ignore_mask = ema_model_loss > (loss_mean + (self.std_margin * loss_std))
            ema_model_loss = ema_model_loss[~ignore_mask]

            self.update_running_loss_ema(ema_model_loss.detach())

            return output

        else:
            return self.backbone(inputs)


    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_model_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.backbone.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_running_loss_ema(self, batch_loss):
        alpha = min(1 - 1 / (self.global_step + 1), self.loss_alpha)
        self.loss_running_mean = alpha * self.loss_running_mean + (1 - alpha) * batch_loss.mean()
        self.loss_running_std = alpha * self.loss_running_std + (1 - alpha) * batch_loss.std()
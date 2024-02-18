import copy
import torch
import torch.nn as nn
from utils.buffer import Buffer_task_free
from utils.training_ops import step_wo_state_update_adam, step_wo_state_update_adamw, step_wo_state_update_sgd


class GMED(nn.Module):

    def __init__(self, model: nn.Module, batch_size: int, edit_decay: float, grad_stride: float, reg_strength: float,
                 mem_args, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.batch_size = batch_size
        self.device = device
        self.buffer = Buffer_task_free(**mem_args, device=self.device)
        self.optimizer = None

        self._req_penalty = False
        self._req_opt = True

        self.edit_decay = edit_decay
        self.grad_stride = grad_stride
        self.reg_strength = reg_strength

    def forward(self, inputs):

        if self.training:

            if not self.buffer.is_empty():

                # Load past instances, we don't augment the buffer instance since GMED itself gives augmentation
                buf_inputs, sample_indices = self.buffer.get_data(self.batch_size, return_indices=True)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k, v in buf_inputs.items()}

                # Store original model in cache
                self.store_cache()

                # Evaluate grad of L w.r.t buffer
                self.clear_mem_grad(buf_inputs['video_data'])
                self.clear_mem_grad(buf_inputs['audio_data'])

                # Buffer loss before update
                self.optimizer.zero_grad()
                pre_output = self.backbone(buf_inputs)
                pre_loss = sum([v for k, v in pre_output.items() if "loss" in k])
                grad_reg_video = -torch.autograd.grad(torch.sum(pre_loss), buf_inputs['video_data'], retain_graph=True)[0]
                grad_reg_audio = -torch.autograd.grad(torch.sum(pre_loss), buf_inputs['audio_data'], retain_graph=True)[0]
                pre_loss.backward()
                edit_video_grad1 = buf_inputs['video_data'].grad
                edit_audio_grad1 = buf_inputs['audio_data'].grad

                self.clear_mem_grad(buf_inputs['video_data'])
                self.clear_mem_grad(buf_inputs['audio_data'])

                # Virtual update
                self.get_loss_and_pseudo_update(inputs)

                # Buffer loss after update
                post_output = self.backbone(buf_inputs)
                post_loss = sum([v for k, v in post_output.items() if "loss" in k])
                post_loss.backward()
                edit_video_grad2 = buf_inputs['video_data'].grad
                edit_audio_grad2 = buf_inputs['audio_data'].grad

                grad_delta_video = edit_video_grad2 - edit_video_grad1
                grad_delta_audio = edit_audio_grad2 - edit_audio_grad1

                self.clear_mem_grad(buf_inputs['video_data'])
                self.clear_mem_grad(buf_inputs['audio_data'])

                grad_delta_video = - grad_reg_video * self.reg_strength + grad_delta_video
                grad_delta_audio = - grad_reg_audio * self.reg_strength + grad_delta_audio

                mem_ages = self.buffer.get_mem_ages(sample_indices, astype=buf_inputs['video_data'])
                stride_decayed = (1 - self.edit_decay) ** mem_ages

                buf_inputs['video_data'] = buf_inputs['video_data'] + self.grad_stride * stride_decayed.view(-1, 1, 1, 1, 1) * grad_delta_video
                buf_inputs['audio_data'] = buf_inputs['audio_data'] + self.grad_stride * stride_decayed.view(-1, 1, 1, 1) * grad_delta_audio

                buf_inputs['video_data'] = buf_inputs['video_data'].detach()
                buf_inputs['audio_data'] = buf_inputs['audio_data'].detach()


                # Replace the original buffer data
                self.buffer.add_data(video_data=buf_inputs['video_data'], audio_data=buf_inputs['audio_data'],
                                     sample_idx=sample_indices)

                # Load original model
                self.load_cache()

                # Concatenate buffer input and current input
                inputs['video_data'] = torch.cat((inputs['video_data'], buf_inputs['video_data']), dim=0)
                inputs['audio_data'] = torch.cat((inputs['audio_data'], buf_inputs['audio_data']), dim=0)

            output = self.backbone(inputs)
            # Store data to buffer
            self.buffer.add_data(video_data=inputs['video_data'][:self.batch_size], audio_data=inputs['audio_data'][:self.batch_size])

            return output

        else:
            return self.backbone(inputs)

    def clear_mem_grad(self, mem_x):
        mem_x.detach_()
        mem_x.grad = None
        mem_x.requires_grad = True

    def store_cache(self):
        self.cache = copy.deepcopy(self.backbone.state_dict())

    def load_cache(self):
        self.backbone.load_state_dict(self.cache)
        self.backbone.zero_grad()


    def get_loss_and_pseudo_update(self, inputs):
        pseudo_output = self.backbone(inputs)
        self.optimizer.zero_grad()
        pseudo_loss = sum([v for k, v in pseudo_output.items() if "loss" in k])
        pseudo_loss.backward(retain_graph=False)
        if isinstance(self.optimizer, torch.optim.SGD):
            step_wo_state_update_sgd(self.optimizer, amp=1.)
        elif isinstance(self.optimizer, torch.optim.Adam):
            step_wo_state_update_adam(self.optimizer, amp=1.)
        elif isinstance(self.optimizer, torch.optim.AdamW):
            step_wo_state_update_adamw(self.optimizer)
        else:
            raise NotImplementedError


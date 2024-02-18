import copy
import torch
import torch.nn as nn
from utils.buffer import Buffer_task_free
from utils.training_ops import step_wo_state_update_adam, step_wo_state_update_adamw, step_wo_state_update_sgd


class MIR(nn.Module):

    def __init__(self, model: nn.Module, batch_size: int, search_factor: int,
                 mem_args, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.batch_size = batch_size
        self.search_factor = search_factor # load search_factor * batch_size of samples from replay buffer
        self.device = device
        self.buffer = Buffer_task_free(**mem_args, device=self.device)
        self.optimizer = None

        self._req_penalty = False
        self._req_opt = True

    def forward(self, inputs):

        if self.training:

            if not self.buffer.is_empty():

                # Load past instances
                buf_inputs = self.buffer.get_data(self.search_factor * self.batch_size)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k, v in buf_inputs.items()}

                # Store original model in cache
                self.store_cache()

                # Buffer loss before update
                self.backbone.requires_grad_(requires_grad=False)
                pre_output = self.backbone(buf_inputs, reduction='none')
                pre_loss = sum([v for k, v in pre_output.items() if "loss" in k])
                self.backbone.requires_grad_(requires_grad=True)

                # Temporally(Virtually) update model
                self.get_loss_and_pseudo_update(inputs)

                # Buffer loss after update
                self.backbone.requires_grad_(requires_grad=False)
                post_output = self.backbone(buf_inputs, reduction='none')
                post_loss = sum([v for k, v in post_output.items() if "loss" in k])
                scores = post_loss - pre_loss
                self.backbone.requires_grad_(requires_grad=True)

                # Find instances that gives maximal inference retrieval
                big_ind = scores.sort(descending=True)[1][:self.batch_size]
                buf_inputs = {k: v[big_ind] for k, v in buf_inputs.items()}

                # Load original model
                self.load_cache()

                inputs['video_data'] = torch.cat((inputs['video_data'], buf_inputs['video_data']), dim=0)
                inputs['audio_data'] = torch.cat((inputs['audio_data'], buf_inputs['audio_data']), dim=0)

            # Train model
            output = self.backbone(inputs)
            # Store data to buffer
            self.buffer.add_data(video_data=inputs['video_data'][:self.batch_size], audio_data=inputs['audio_data'][:self.batch_size])

            return output

        else:
            return self.backbone(inputs)

    def store_cache(self):
        self.cache = copy.deepcopy(self.backbone.state_dict())

    def load_cache(self):
        self.backbone.load_state_dict(self.cache)
        self.backbone.zero_grad()
        del self.cache

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
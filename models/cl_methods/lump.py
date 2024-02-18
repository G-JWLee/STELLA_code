import numpy as np
import torch.nn as nn
from utils.buffer import Buffer_task_free


class Lump(nn.Module):

    def __init__(self, model: nn.Module, alpha: float, batch_size: int,
                 mem_args, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.buffer = Buffer_task_free(**mem_args, device=self.device)

        self._req_penalty = False
        self._req_opt = False

    def forward(self, inputs):

        if self.training:
            if self.buffer.is_empty():
                output = self.backbone(inputs)
            else:
                buf_inputs = self.buffer.get_data(self.batch_size)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k,v in buf_inputs.items()}
                lam = np.random.beta(self.alpha, self.alpha)
                mixed_inputs = {}
                for modality in buf_inputs:
                    if modality.endswith('data'):
                        mixed_inputs[modality] = lam * inputs[modality] + (1 - lam) * buf_inputs[modality][:inputs[modality].shape[0]]
                output = self.backbone(mixed_inputs)
            # Store data to buffer
            self.buffer.add_data(video_data=inputs['video_data'], audio_data=inputs['audio_data'])

            return output

        else:
            return self.backbone(inputs)
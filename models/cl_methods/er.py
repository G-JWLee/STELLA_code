import torch
import torch.nn as nn
from utils.buffer import Buffer_task_free


class ER(nn.Module):

    def __init__(self, model: nn.Module, batch_size: int,
                 mem_args, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.batch_size = batch_size
        self.device = device
        self.buffer = Buffer_task_free(**mem_args, device=self.device)

        self._req_penalty = False
        self._req_opt = False

    def forward(self, inputs):

        if self.training:

            if not self.buffer.is_empty():
                buf_inputs = self.buffer.get_data(self.batch_size)
                buf_inputs = {k: v.cuda(self.device, non_blocking=True) for k,v in buf_inputs.items()}
                inputs['video_data'] = torch.cat((inputs['video_data'], buf_inputs['video_data']), dim=0)
                inputs['audio_data'] = torch.cat((inputs['audio_data'], buf_inputs['audio_data']), dim=0)

            output = self.backbone(inputs)

            # Store data to buffer
            self.buffer.add_data(video_data=inputs['video_data'][:self.batch_size], audio_data=inputs['audio_data'][:self.batch_size])

            return output

        else:
            return self.backbone(inputs)

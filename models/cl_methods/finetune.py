import torch.nn as nn

class Finetune(nn.Module):

    def __init__(self, model: nn.Module, device, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.device = device

        self._req_penalty = False
        self._req_opt = False

    def forward(self, inputs):
        return self.backbone(inputs)

import torch
from numpy import inf


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, model=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                parameters = model.parameters()
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)

            # https://github.com/facebookresearch/fairscale/issues/180
            # scaler.step(optimizer) will already check for invalid gradients and if these are found then the internal
            # optimizer.step() call will be skipped and the scaler.update() operation will decrease the scaling factor
            # When detected, it modifies the scale so that inf or nan does not happen again.
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(model, norm_type: float = 2.0, check_valid=False) -> torch.Tensor:
    with torch.no_grad():
        if check_valid:
            parameters = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # At initial training stage, gradient in AVM module sometimes has inf gradient element,
                    # Due to softmax function and amp torch.float16 overflow.
                    nan_mask = torch.isnan(param.grad)
                    inf_mask = torch.isinf(param.grad)
                    valid_gradients = not (nan_mask.any() or inf_mask.any())
                    if not valid_gradients:
                        param.grad[nan_mask] = 0.0
                        param.grad[inf_mask] = 0.0

                    parameters.append(param)
        else:
            parameters = model.parameters()
            parameters = [p for p in parameters if p.grad is not None]

        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
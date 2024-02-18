
# Custom scheduler from https://gaussian37.github.io/dl-pytorch-lr_scheduler/

import torch
import math

class MinimumStepLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, step_size, gamma, minimum_multi):

        def lr_lambda(step):
            state = step // step_size
            lr_multi = gamma ** state
            if lr_multi < minimum_multi:
                return minimum_multi
            else:
                return lr_multi

        super(MinimumStepLR, self).__init__(optimizer, lr_lambda)


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup then constant.
    Linearly increases learning rate schedule from 0 to 1 over 'warmup_steps' training steps.
    Keeps learning rate schedule equal to 1, after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, final_lr_ratio, epochs, niter_per_ep, warmup_epochs=0,
                 warmup_steps=-1, last_epoch=-1, **kwargs):
        self.warmup_steps = warmup_epochs * niter_per_ep
        if warmup_steps > 0:
            self.warmup_steps = warmup_steps

        self.t_total = epochs * niter_per_ep
        self.ratio = final_lr_ratio
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, self.ratio + 0.5 * (1 - self.ratio) * (1.0 + math.cos(math.pi * progress))
        )
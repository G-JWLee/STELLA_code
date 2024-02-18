import torch
import math
from torch.optim import SGD, Adam, AdamW


def step_wo_state_update_adam(adam, closure=None, amp=1.):
    """Performs a single optimization step. Do not update optimizer states
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    if type(adam) is not Adam:
        raise ValueError
    loss = None
    if closure is not None:
        loss = closure()

    for group in adam.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = adam.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            if group['weight_decay'] != 0:
                grad.add_(p.data, alpha=group['weight_decay'])

            # Decay the first and second moment running average coefficient
            exp_avg = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])

            bias_correction1 = 1 - beta1 ** (state['step'] + 1) # Since it is for virtual update, increase step by one
            bias_correction2 = 1 - beta2 ** (state['step'] + 1)
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 * amp

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

    return loss


def step_wo_state_update_sgd(sgd, closure=None, amp=1.):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in sgd.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = sgd.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-group['lr'] * amp, d_p)

    return loss


def step_wo_state_update_adamw(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p]

            # State initialization
            if len(state) == 0:
               state['step'] = 0
               # Exponential moving average of gradient values
               state['exp_avg'] = torch.zeros_like(p.data)
               # Exponential moving average of squared gradient values
               state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add(group['eps'])

            step_size = group['lr']
            if group['correct_bias']:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** (state['step'] + 1)
                bias_correction2 = 1.0 - beta2 ** (state['step'] + 1)
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group['weight_decay'] > 0.0:
                p.data.add_(-group['lr'] * group['weight_decay'], p.data)

    return loss
import sys
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from utils.distributed_ops import concat_all_gather

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', tbname='', device=None):
        self.name = name
        self.tbname = tbname
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device=self.device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        try:
            self.avg = self.sum / self.count
        except:
            self.avg = .0


class AverageAccuracy(object):
    """
    Computes accuracy and stores the average and current value
    Refered to TVLT Accuracy torchmetric.
    """

    def __init__(self, name, fmt=':f', tbname='', phase='Train', device=None):
        self.name = name
        self.tbname = tbname
        self.fmt = fmt
        self.reset()
        self.phase = phase
        self.preds = []
        self.labels = []
        self.device = device

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, logits, target):
        if self.phase == 'Test':
            # Ensemble multiple predictions of the same view together.
            if logits.size(1) > 1:
                # Multi-class accuracy
                preds = F.softmax(logits, dim=-1)
                preds = preds.mean(dim=0).detach()  # Since mutliple clips from a sample are tested
                self.preds.append(preds.unsqueeze(dim=0))
                preds = preds.argmax(dim=0)
            else:
                # Binary classification
                # Since we use binary_cross_entropy_with_logits (sigmoid + binary_cross_entropy), >= 0 means close to label 1.
                preds = (logits >= 0)
                preds = preds.mean(dim=0).detach()
                self.preds.append(preds)
                preds = preds >= 0.5
            target = target[0].detach() # Only one label is needed
            self.labels.append(target.unsqueeze(dim=0))
            target = target.argmax(dim=0)

        else:
            if logits.size(1) > 1:
                # Multi-class accuracy
                # preds = logits.argmax(dim=-1).squeeze(dim=-1)
                preds = logits.argmax(dim=-1)
                if len(target.size()) > 1:  # One-hot encoding
                    target = target.argmax(dim=-1)
            else:
                # Binary classification
                # Since we use binary_cross_entropy_with_logits (sigmoid + binary_cross_entropy), >= 0 means close to label 1.
                preds = (logits >= 0).squeeze(dim=-1)
                # target = target.squeeze(dim=-1)

        assert preds.shape == target.shape, f"{preds.shape} and {target.shape}"

        self.val = torch.sum(preds == target).item()
        self.sum += self.val
        self.count += target.numel()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device=self.device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        try:
            self.avg = self.sum / self.count
        except:
            self.avg = .0

    def synchronize_pred_label(self):
        if not is_dist_avail_and_initialized():
            return
        preds = torch.stack(self.preds, dim=0).to(self.device)
        labels = torch.stack(self.labels, dim=0).to(self.device)
        self.preds = concat_all_gather(preds)
        self.labels = concat_all_gather(labels)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", tbwriter=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.tbwriter = tbwriter

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def tbwrite(self, batch):
        if self.tbwriter is None:
            return
        scalar_dict = self.tb_scalar_dict()
        # print(scalar_dict)
        for k, v in scalar_dict.items():
            self.tbwriter.add_scalar(k, v, batch)
        sys.stdout.flush()

    def tb_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            if not meter.tbname:
                meter.tbname = meter.name
                tag = meter.tbname
                sclrval = val
                out[tag] = sclrval
        return out



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
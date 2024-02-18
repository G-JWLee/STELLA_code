import glob
import os
import torch
import re
import numpy as np
from numpy import inf

import time
import torch.distributed as dist


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def resume_from_checkpoint(ckpt_fname, modules):
    print(">>>> loading checkpoint '{}'".format(ckpt_fname))
    checkpoint = torch.load(ckpt_fname, map_location='cpu')

    # Load state dict
    for k in modules:
        modules[k].load_state_dict(checkpoint[k])
        del checkpoint[k]
    print(">>>> loaded checkpoint '{}' (epoch {})".format(
        ckpt_fname, checkpoint['epoch']))
    return checkpoint


def resume(modules, args):
    all_ckpt_fnames = glob.glob(os.path.join(args.logging.ckpt_dir, args.logging.name, 'checkpoint_*.pth'))
    if not all_ckpt_fnames:
        return

    # Find last checkpoint
    epochs = [float(re.match('checkpoint_(\d+\.*\d*).pth', fn.split('/')[-1]).group(1)) for fn in all_ckpt_fnames]
    ckpt_fname = all_ckpt_fnames[np.argsort(-np.array(epochs))[-1]]

    # Load checkpoint
    resume_from_checkpoint(ckpt_fname, modules)


class CheckpointManager:
    def __init__(self, modules, ckpt_dir, epoch_size, save_freq_epoch=None, save_freq_mints=None):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epoch_size = epoch_size
        self.save_freq = save_freq_epoch
        self.save_freq_mints = save_freq_mints

        self.time = time.time()
        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

        os.makedirs(self.ckpt_dir, exist_ok=True)

    def resume(self, resume_path=""):
        start_epoch = 1
        accum_epoch = 0
        steps = 0
        the_best_metric = -inf
        if resume_path != "":
            ckpt_fname = resume_path
        else:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')
            # Load state dict
            for k in self.modules:
                if k == 'buffer':  # Split the buffer into /n_gpus
                    per_gpu_buffer_size = checkpoint[k]['buffer_size'] // self.world_size
                    splited_checkpoint = {}
                    for data_name in checkpoint[k]:
                        if data_name not in self.modules[k].attributes:
                            splited_checkpoint[data_name] = checkpoint[k][data_name] // self.world_size
                        else:
                            splited_checkpoint[data_name] = checkpoint[k][data_name][self.rank * per_gpu_buffer_size:
                                                                                 (self.rank+1) * per_gpu_buffer_size]
                    self.modules[k].load_state_dict(splited_checkpoint)
                elif k == 'model':
                    self.modules[k].load_state_dict(checkpoint[k], strict=False)
                else:
                    self.modules[k].load_state_dict(checkpoint[k])

            start_epoch = checkpoint['epoch']
            if 'steps' in checkpoint:
                steps = checkpoint['steps']
            if 'accum_epoch' in checkpoint:
                accum_epoch = checkpoint['accum_epoch']
            if 'the_best_metric' in checkpoint:
                the_best_metric = checkpoint['the_best_metric']

            print("=> loaded checkpoint '{}' (epoch {})".format(
                    ckpt_fname, checkpoint['epoch']))
        return start_epoch, accum_epoch, steps, the_best_metric

    def resume_best(self, wa=False, wa_start=0, wa_end=0):
        if wa:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:04d}.pth')
            assert os.path.isfile(ckpt_fname.format(wa_start)), f"{ckpt_fname.format(wa_start)} not exist."
            sdA = torch.load(ckpt_fname.format(wa_start), map_location='cpu')['model']
            model_cnt = 1
            for epoch in range(wa_start + self.save_freq, wa_end, self.save_freq):
                assert os.path.isfile(
                    ckpt_fname.format(epoch)), f"{ckpt_fname.format(epoch)} not exist."
                sdB = torch.load(ckpt_fname.format(epoch), map_location='cpu')['model']
                for key in sdA:
                    sdA[key] = sdA[key] + sdB[key]
                model_cnt += 1
            for key in sdA:
                sdA[key] = sdA[key] / float(model_cnt)
            self.modules['model'].load_state_dict(sdA)
            print('wa {:d} models from {:d} to {:d}'.format(model_cnt, wa_start, wa_end))
        else:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_best.pth')
            if os.path.isfile(ckpt_fname):
                checkpoint = torch.load(ckpt_fname, map_location='cpu')
                # Load state dict
                self.modules['model'].load_state_dict(checkpoint['model'])
                print("=> loaded best checkpoint '{}' (epoch {})".format(
                    ckpt_fname, checkpoint['epoch']))


    def timed_checkpoint(self, save_dict=None):
        if self.save_freq_mints is None or self.save_freq_mints <= 0:
            return
        t = time.time() - self.time
        t_all = [t for _ in range(self.world_size)]
        if self.world_size > 1:
            dist.all_gather_object(t_all, t)
        if min(t_all) > self.save_freq_mints * 60:
            self.time = time.time()
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, filename=ckpt_fname)
                print(f"Saved checkpoint '{ckpt_fname}")

    def midway_epoch_checkpoint(self, epoch, batch_i, save_dict=None):
        if self.save_freq is None:
            return
        if ((batch_i + 1) / float(self.epoch_size) %
                self.save_freq) < (
                    batch_i / float(self.epoch_size) %
                    self.save_freq):
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:010.4f}.pth')
            ckpt_fname = ckpt_fname.format(
                epoch + batch_i / float(self.epoch_size))

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, filename=ckpt_fname)
                print(f"Saved checkpoint '{ckpt_fname}' (epoch {epoch}, batch_i {batch_i})")

    def end_epoch_checkpoint(self, epoch, save_dict=None):
        if self.save_freq is None:
            return
        if (epoch % self.save_freq == 0) or self.save_freq < 1:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:04d}.pth')
            ckpt_fname = ckpt_fname.format(epoch)
            model_fname = os.path.join(self.ckpt_dir, 'model_checkpoint_{:04d}.pth')
            model_fname = model_fname.format(epoch)

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, filename=ckpt_fname)
                save_checkpoint(state['model'], filename=model_fname)
                print(f"Saved checkpoint '{ckpt_fname}'  (epoch {epoch})")

    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict()
                 for k in self.modules}
        # Concatenate buffer data
        # Since concat_all function requires data to be in gpu memory, it is hard to do with video-audio data
        # Hence, we manually save them in ckpt directory and load them manually.
        if 'buffer' in self.modules:
            torch.save(state['buffer'], os.path.join(self.ckpt_dir, f'buffer_{self.rank}.pt'))
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            buffer_files = glob.glob(os.path.join(self.ckpt_dir, f'buffer_*.pt'))
            buffer_files.sort()

            new_state = {}
            for data_name in state['buffer']:
                if data_name in self.modules['buffer'].attributes:
                    new_state[data_name] = torch.tensor([], dtype=state['buffer'][data_name].dtype)
                else:
                    new_state[data_name] = 0

            for buffer_file in buffer_files:
                buffer_dict = torch.load(buffer_file, map_location='cpu')

                for data_name in buffer_dict:
                    if data_name in self.modules['buffer'].attributes:
                        new_state[data_name] = torch.cat([new_state[data_name], buffer_dict[data_name]])
                    else:
                        new_state[data_name] += buffer_dict[data_name]

            state['buffer'] = new_state

        if save_dict is not None:
            state.update(save_dict)
        return state

    def checkpoint(self, epoch, batch_i=None, save_dict=None):
        if batch_i is None:
            self.end_epoch_checkpoint(epoch, save_dict)
        else:
            self.timed_checkpoint(save_dict)
            self.midway_epoch_checkpoint(epoch, batch_i, save_dict=save_dict)

    def force_checkpoint(self, ckpt_fname, save_dict=None):
        ckpt_fname = os.path.join(self.ckpt_dir, ckpt_fname)
        state = self.create_state_dict(save_dict)
        if self.rank == 0:
            save_checkpoint(state, filename=ckpt_fname)
            print(f"Saved checkpoint '{ckpt_fname}'")

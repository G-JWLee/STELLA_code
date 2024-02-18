# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple
import torch
import numpy as np


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer_task_free(object):
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, memory_size, device):
        self.buffer_size = memory_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['audio_data', 'video_data', 'logits_a', 'logits_v',
                           'video_query', 'audio_query', 'prob_v_att', 'prob_a_att', 'n_attn_av', 'n_attn_va']
        self.age = np.zeros(memory_size)

    def init_tensors(self, video_data: torch.Tensor, audio_data: torch.Tensor,
                     logits_a: torch.Tensor, logits_v: torch.Tensor, video_query: torch.Tensor, audio_query: torch.Tensor,
                     prob_v_att: torch.Tensor, prob_a_att: torch.Tensor, n_attn_av: torch.Tensor, n_attn_va: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=torch.float32))

    def add_data(self, video_data, audio_data=None,
                 logits_a=None, logits_v=None, sample_idx=None, video_query=None, audio_query=None,
                 prob_v_att=None, prob_a_att=None, n_attn_av=None, n_attn_va=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param logits: tensor containing the outputs of the network
        :return:
        """
        if not hasattr(self, 'video_data'):
            self.init_tensors(video_data=video_data, audio_data=audio_data,
                              logits_a=logits_a, logits_v=logits_v,
                              video_query=video_query, audio_query=audio_query,
                              prob_v_att=prob_v_att, prob_a_att=prob_a_att, n_attn_av=n_attn_av, n_attn_va=n_attn_va)

        for i in range(video_data.shape[0]):
            # Randomly select location to store new data when buffer is full
            if sample_idx is None:
                index = reservoir(self.num_seen_examples, self.buffer_size)
                self.num_seen_examples += 1
            # Sample idx is given. This function will be used in GMED for sample update
            else:
                index = sample_idx[i]

            if index >= 0:
                self.video_data[index] = video_data[i].detach().cpu()
                if audio_data is not None:
                    self.audio_data[index] = audio_data[i].detach().cpu()
                if logits_v is not None:
                    self.logits_v[index] = logits_v[i].detach().cpu()
                if logits_a is not None:
                    self.logits_a[index] = logits_a[i].detach().cpu()

                if video_query is not None:
                    self.video_query[index] = video_query[i].detach().cpu()
                if audio_query is not None:
                    self.audio_query[index] = audio_query[i].detach().cpu()

                if prob_v_att is not None:
                    self.prob_v_att[index] = prob_v_att[i].detach().cpu()
                if prob_a_att is not None:
                    self.prob_a_att[index] = prob_a_att[i].detach().cpu()

                if n_attn_av is not None:
                    self.n_attn_av[index] = n_attn_av[i].detach().cpu()
                if n_attn_va is not None:
                    self.n_attn_va[index] = n_attn_va[i].detach().cpu()

    def get_data(self, size: int, sample_idx=None, return_indices=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if sample_idx is None:
            if size > min(self.num_seen_examples, self.video_data.shape[0]):
                size = min(self.num_seen_examples, self.video_data.shape[0])

            choice = np.random.choice(min(self.num_seen_examples, self.video_data.shape[0]),
                                      size=size, replace=False)
        else:
            choice = sample_idx

        clips = {}
        # Bring data
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                clips.update({attr_str: getattr(self, attr_str)[choice]})

        # If need sample idx for updating sample, return sample idx
        if return_indices:
            return clips, choice
        else:
            return clips


    def get_mem_ages(self, indices, astype):
        ages = self.age[indices]
        if torch.is_tensor(astype):
            ages = torch.from_numpy(ages).float().to(astype.device)
        return ages


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False


    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def state_dict(self):
        state_dict = {'num_seen_examples': self.num_seen_examples,
                      'buffer_size': self.buffer_size,
                      'age': self.age}
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                state_dict.update({attr_str: getattr(self, attr_str)})

        return state_dict

    def load_state_dict(self, state):
        self.num_seen_examples = state['num_seen_examples']
        self.buffer_size = state['buffer_size']
        self.age = state['age']
        for attr_str in self.attributes:
            if attr_str in state:
                setattr(self, attr_str, state[attr_str])

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmseg.core.utils import sync_random_seed
from mmseg.utils import get_device
import pandas as pd


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    Args:
        datasets (Dataset): the dataset will be loaded.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, world_size is retrieved from the
            current distributed group.
        rank (int, optional):  Rank of the current process within num_replicas.
            By default, rank is retrieved from the current distributed group.
        shuffle (bool): If True (default), sampler will shuffle the indices.
        seed (int): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed=0) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        device = get_device()
        self.seed = sync_random_seed(seed, device)

    def __iter__(self) -> Iterator:
        """
         Yields:
            Iterator: iterator of indices for rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class SIIMDistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    Args:
        datasets (Dataset): the dataset will be loaded.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, world_size is retrieved from the
            current distributed group.
        rank (int, optional):  Rank of the current process within num_replicas.
            By default, rank is retrieved from the current distributed group.
        shuffle (bool): If True (default), sampler will shuffle the indices.
        seed (int): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed=0) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        device = get_device()
        self.seed = sync_random_seed(seed, device)

        self.filenames = []
        for i in range(len(self.dataset)):
            self.filenames.append(self.dataset.img_infos[i]['filename'].replace('.jpg',''))

        df = pd.read_csv('../DatasetsSplits/SIIM-ACR_Pneumothorax/siim.csv')
        df["class"] = df[" EncodedPixels"].apply(lambda x: x != " -1")
        self.pos_filenames = df[df["class"] == 1]['ImageId']
        self.neg_filenames = df[df["class"] == 0]['ImageId']

        element_indices = dict()
        for index, value in enumerate(self.filenames):
            element_indices.setdefault(value, []).append(index)

        self.pos_indices = [element_indices.get(index, [None])[0] for index in self.pos_filenames]
        self.pos_indices = [x for x in self.pos_indices if x is not None]
        self.pos_indices = list(set(self.pos_indices))
        self.neg_indices = [element_indices.get(index, [None])[0] for index in self.neg_filenames]
        self.neg_indices = [x for x in self.neg_indices if x is not None]
        self.neg_indices = list(set(self.neg_indices))
        self.nums_pos = len(self.pos_indices)

    def __iter__(self) -> Iterator:
        """
         Yields:
            Iterator: iterator of indices for rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)

            neg_select = torch.randperm(len(self.neg_indices), generator=g).tolist()[: round(self.nums_pos * 0.4)]
            neg_indices = [self.neg_indices[i] for i in neg_select]
            indices = self.pos_indices + neg_indices
            random_shuffle = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in random_shuffle]
            self.total_size = len(indices)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # assert len(indices) == self.num_samples
        self.num_samples = len(indices)

        return iter(indices)

#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Adapted by Timm Hess (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452


import copy
from pprint import pprint
from typing import Optional, List
from packaging.version import parse
import numpy as np

import torch

from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.benchmarks.utils.flat_data import FlatData
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ClassBalancedBuffer, 
    ReservoirSamplingBuffer, 
    BalancedExemplarsBuffer, 
    ReservoirSamplingBuffer,
    ExemplarsBuffer,
    ExperienceBalancedBuffer
)
from avalanche.training.utils import cycle
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from src.utils.util import safe_index



class ClassTaskBalancedBuffer(BalancedExemplarsBuffer):
    def __init__(
        self, 
        max_size: int, 
        adaptive_size: bool = True,
        total_num_classes: int = None
    ):
        """
        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        self.task_shift = 1000
        

    def update(self, strategy, **kwargs):
        # Access the current experience dataset
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}

        # Check and get the task_label
        assert len(np.unique(new_data.targets_task_labels)) == 1, "Only one task label is supported"
        task_label = np.unique(new_data.targets_task_labels)[0]
        
        for idx, target in enumerate(new_data.targets):
            target = int(target+task_label*self.task_shift) # NOTE: 1000 should be bigger than max number of tasks!
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheDataset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(num_groups=self.total_num_classes)#len(self.seen_classes)
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])



def get_storage_policy(
        name: str, 
        mem_size: int, 
        total_num_classes: int = None, 
        num_experiences: int = None,
        task_incremental: bool = False,
        domain_incremental: bool = False
) -> ExemplarsBuffer:
    if name == "reservoir":
        storage_policy = ReservoirSamplingBuffer(max_size=mem_size)
    elif name == "class_balanced":
        storage_policy = ClassBalancedBuffer(max_size=mem_size, adaptive_size=False, total_num_classes=total_num_classes)
    elif name == "experience_balanced":
        storage_policy = ExperienceBalancedBuffer(max_size=mem_size, adaptive_size=False, num_experiences=num_experiences)
    elif name == "class_task_balanced":
        storage_policy = ClassTaskBalancedBuffer(
            max_size=mem_size, 
            adaptive_size=False, 
            total_num_classes=total_num_classes*num_experiences
        )
        print("Using ClassTaskBalancedBuffer")
    else:
        raise ValueError(f"Unknown storage policy: {name}")
    return storage_policy


class ReplayPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(
        self,
        mem_size: int,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    # def before_training(self, strategy, **kwargs):
    #     """ 
    #     Omit reduction in criterion to be able to 
    #     separate losses from buffer and batch
    #     """
    #     strategy._criterion.reduction = 'none'  # Overwrite
    #     # Also overwrite the _make_empty_loss function because it does not work with non reduced losses
    #     def new_make_empty_loss(self):
    #         return 0
    #     strategy._make_empty_loss = new_make_empty_loss.__get__(strategy)  # instance.some_method = new_method.__get__(instance)
    #     return super().before_training(strategy, **kwargs)

    def before_training_exp(
        self,
        strategy,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )

    # def before_backward(self, strategy, **kwargs):
    #     nb_samples = strategy.loss.shape[0]   
    #     # Return default loss if not using replay data in the batch
    #     if not nb_samples > strategy.train_mb_size:
    #         strategy.loss = strategy.loss.mean()
    #         return
        
    #     # Disentangle losses
    #     #print("DEBUG: replay - loss new", strategy.loss[:self.nb_new_samples].mean(), self.lmbda)
    #     #loss_new = (1-self.curr_replay_loss_weight) * strategy.loss[:self.nb_new_samples].mean()
    #     loss_new = strategy.loss[:strategy.train_mb_size].mean()
    #     loss = loss_new

    #     # Mem loss
    #     if nb_samples > strategy.train_mb_size:
    #         #loss_reg = self.curr_replay_loss_weight * strategy.loss[self.nb_new_samples:].mean()
    #         loss_reg = strategy.loss[strategy.train_mb_size:].mean()
    #         loss = loss_new + loss_reg  

    #     # Writeback loss to strategy   
    #     strategy.loss = loss      
    #     return

    def before_training_iteration(self, strategy, **kwargs):
        
        print("\nDEBUG replay: batch_size:", strategy.mb_x.shape[0])
        print("\nDEBUG replay: current_group:", strategy.experience.dataset._flat_data._transform_groups.current_group)
        if strategy.experience.dataset._flat_data._transform_groups.current_group is None:
            print("No transform group active") 

        # Check buffer transform group
        buf = self.storage_policy.buffer
        print("Buffer transform group:", buf._flat_data._transform_groups.current_group)

        print("Buffer total size:", len(buf))
        print("Buffer groups:")
        for gid, group in self.storage_policy.buffer_groups.items():
            print(f"  group {gid}: {len(group.buffer)} samples (max_size={group.max_size})")

        # # visually compare a batch
        # if strategy.clock.train_exp_counter > 0:
        #     import torchvision
        #     from torch.utils.data import DataLoader
        #     imgs_current, *_ = next(iter(DataLoader(strategy.adapted_dataset, batch_size=16)))
        #     imgs_buffer, *_ = next(iter(DataLoader(buf, batch_size=16)))
        #     torchvision.utils.save_image(imgs_current, "current.png")
        #     torchvision.utils.save_image(imgs_buffer, "buffer.png")
        #     import sys; sys.exit()
        
        # if strategy.clock.train_exp_counter > 0:
        #     import torchvision
        #     from torch.utils.data import DataLoader

        #     # Adjust mean/std to match your normalization
        #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #     std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        #     def denorm(imgs):
        #         return (imgs.cpu() * std + mean).clamp(0, 1)

        #     imgs_current, *_ = next(iter(DataLoader(strategy.adapted_dataset, batch_size=16)))
        #     imgs_buffer,  *_ = next(iter(DataLoader(buf, batch_size=16)))
        #     torchvision.utils.save_image(denorm(imgs_current), "current.png")
        #     torchvision.utils.save_image(denorm(imgs_buffer),  "buffer.png")
        #     import sys; sys.exit()

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)


class CumulativeReplayPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Replay plugin that mimics cumulative training with a bounded memory buffer.

    Unlike ReplayPlugin, which samples separately from the current dataset and
    the buffer (doubling the effective batch size with a fixed 50:50 split),
    this plugin concatenates the buffer directly onto the current experience
    dataset and creates a single unified DataLoader over the result.

    The sampling ratio between old and new data is then naturally proportional
    to their relative sizes: a batch of size B drawn from a combined dataset of
    N current + M buffer samples contains on average B*N/(N+M) current and
    B*M/(N+M) buffer samples. This matches how cumulative training behaves.

    :param mem_size: total number of patterns stored in the buffer.
    :param storage_policy: controls how exemplars are added to memory.
                           Defaults to ExperienceBalancedBuffer.
    """

    def __init__(
        self,
        mem_size: int,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size

        if storage_policy is not None:
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    def before_training_exp(
        self,
        strategy,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            return

        assert strategy.adapted_dataset is not None

        combined = strategy.adapted_dataset.concat(self.storage_policy.buffer)

        dataloader_kwargs = dict(
            batch_size=strategy.train_mb_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=combined.collate_fn,
        )

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                dataloader_kwargs["persistent_workers"] = kwargs["persistent_workers"]

        strategy.dataloader = torch.utils.data.DataLoader(combined, **dataloader_kwargs)

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)

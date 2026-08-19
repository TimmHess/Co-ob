#  Copyright (c) 2021-2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import os
from pathlib import Path
from tqdm import tqdm

import PIL

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS 
from torch.utils.data import Subset

from torchvision.datasets.utils import extract_archive
from torchvision import transforms

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.mini_imagenet.mini_imagenet_data import (
        MINI_IMAGENET_WNIDS,
        MINI_IMAGENET_WNID_TO_IDX,
        MINI_IMAGENET_CLASSES,
        MINI_IMAGENET_CLASS_TO_IDX,
    )

from datasets import load_from_disk, concatenate_datasets 
from datasets import Image as HFImage

from src.benchmarks.arrow_loader import (
    load_hf_arrow_dataset, HFDatasetWrapper, load_multi_dataset
)

IMGNET_NAMES = [
    # TODO
]


_default_train_transform = transforms.Compose([
    #transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

_default_test_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])



"""
Generator function for the "full resolution" (224x224) MiniImageNet dataset.
"""
def SplitArrowScenario(
    rootpath,
    n_experiences=10, 
    return_task_id=False,
    class_ids_from_zero_in_each_exp=False,
    seed=0,
    fixed_class_order=None, 
    per_exp_classes=None,
    train_transform=_default_train_transform,
    test_transform=_default_test_transform
):
    # Initialize the ImageFolder on the directory
    train_set, test_set = load_hf_arrow_dataset(
        rootpath=rootpath,
        train_dir="train",
        eval_dir="val",
        train_transform=None,
        eval_transform=None
    )
    
    # Generate the benchmark
    if return_task_id:
        print("SplitArrowScenario - will return a MultiTaskScenario")
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes=per_exp_classes,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp, # default False
            train_transform=train_transform,
            eval_transform=test_transform)
    else:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes=per_exp_classes,
            train_transform=train_transform,
            eval_transform=test_transform)



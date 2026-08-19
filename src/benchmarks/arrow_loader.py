#  Copyright (c) 2021-2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import os
from pathlib import Path
import subprocess
import zipfile
from tqdm import tqdm

import numpy as np
import PIL

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS 
from torch.utils.data import Subset

from torchvision.datasets.utils import extract_archive
from torchvision import transforms

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import nc_benchmark

from datasets import load_from_disk, concatenate_datasets 
from datasets import Image as HFImage



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


def load_multi_dataset(parent_path):
    dataset_dirs = [d for d in Path(parent_path).iterdir() if d.is_dir()]
    datasets = [load_from_disk(str(d)) for d in dataset_dirs]
    hf_train_dataset = concatenate_datasets(datasets)
    return hf_train_dataset


class HFDatasetWrapper(Dataset):
    """
    A wrapper class to make a Hugging Face Dataset compatible with PyTorch's Dataset.
    This allows it to be used as a drop-in replacement in any code that expects
    a standard torch.utils.data.Dataset.

    It implements __len__ and __getitem__, applying a transform on-the-fly.
    """
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: An instance of a Hugging Face Dataset.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, `transforms.ToTensor()`.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

        self.targets = hf_dataset["label"]
        assert self.targets is not None, "Dataset must have a 'label' field."

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        """
        Fetches the sample at the given index, applies the transform, and returns
        it as a (image, label) tuple, the standard format for PyTorch datasets.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the corresponding integer label.
        """
        # Get the raw sample from the Hugging Face dataset.
        # This is usually a dictionary: {'image': <PIL.Image>, 'label': <int>}
        sample = self.hf_dataset[idx]
        
        image = sample['image']
        label = sample['label']

        # print(type(image))
        # import sys; sys.exit()

        # Apply the transform to the image if one is provided.
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_hf_arrow_dataset(
    rootpath,
    train_dir="train",
    eval_dir="test",
    train_transform=None,
    eval_transform=None
):
    # Load Huggingface Arrow datasets
    try:
        hf_train_dataset = load_from_disk(os.path.join(rootpath, train_dir))
        hf_val_dataset = load_from_disk(os.path.join(rootpath, eval_dir))
    except Exception as e:
        print("Attempting to load multiple datasets from subdirectories...")
        hf_train_dataset = load_multi_dataset(os.path.join(rootpath, train_dir))
        hf_val_dataset = load_multi_dataset(os.path.join(rootpath, eval_dir))

    # Wrap for torch
    train_set = HFDatasetWrapper(
        hf_train_dataset) # , transform=transforms.ToPILImage()
    test_set = HFDatasetWrapper( # val_set
        hf_val_dataset) # No transform applied by default # , transform=transforms.ToPILImage()

    # Wrap for avalanche
    # Convert to AvalancheDataset if transforms are given
    if train_transform is not None and eval_transform is not None:
        transform_groups = dict(
            #train=(assert_pil_transform(train_transform), None),
            #eval=(assert_pil_transform(eval_transform), None),
            train=(train_transform, None),
            eval=(eval_transform, None),
        )
        avl_train_set = AvalancheDataset(train_set, transform_groups=transform_groups)
        avl_test_set = AvalancheDataset(test_set, transform_groups=transform_groups)
        avl_train_set.targets = train_set.targets
        avl_test_set.targets = test_set.targets
        print("Transforms applied")

        return avl_train_set, avl_test_set
    return train_set, test_set


"""
Generator function for the "full resolution" (224x224) MiniImageNet dataset.
"""
def SplitHFArrowDataset(
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
        # train_transform=None,
        # eval_transform=None
    )
    
    # Generate the benchmark
    if return_task_id:
        print("SplitMiniImgnet - will return a MultiTaskScenario")
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


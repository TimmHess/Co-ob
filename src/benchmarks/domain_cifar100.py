from collections import defaultdict

from continuum.datasets import CIFAR100
#from continuum.scenarios import ContinualScenario

#from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_generic_benchmark_from_tensor_lists
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_generic_benchmark_from_tensor_lists,  # NOTE: No better way presented yet
    create_multi_dataset_generic_benchmark
)
#from avalanche.benchmarks.utils import AvalancheDatasetType
import numpy as np
import torch
import copy

#from src.benchmarks.utils import XYDataset
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset

from src.utils.transform_tensor_dataset import PILTensorDataset

"""
We can create a instance incremental setting with the coarse labels, i.e. 20 classes. 
Data are labeled with the coarse labels of CIFAR100. However, data are shared between tasks using 
the original label to ensure a domain drift between tasks , e.g., for the coarse label say{aquatic mammals} 
the data go from beavers to dolphins to otters to seals to finally whales in separate tasks.
"""

def get_domain_cifar100_dataset(rootpath, train_transform=None, eval_transform=None):
    # Load the dataset from continuum (because they happen to have it prepared)
    train_set = CIFAR100(rootpath,
                        train=True,
                        labels_type="category",
                        task_labels="lifelong")

    test_set = CIFAR100(rootpath,
                        train=False,
                        labels_type="category",
                        task_labels="lifelong")
    # Access the data
    train_data = train_set.get_data() # Returns (imgs(50000), labels(20), task_labels)
    test_data = test_set.get_data() # Returns (imgs(10000), labels(20), task_labels)

    #Convert to Tensors
    train_tensor = torch.from_numpy(train_data[0])  #targets = torch.tensor(train_data[0])
    train_tensor = train_tensor.permute(0,3,1,2)
    train_labels = torch.from_numpy(train_data[1])

    test_tensor = torch.from_numpy(test_data[0])
    test_tensor = test_tensor.permute(0,3,1,2)
    test_labels = torch.from_numpy(test_data[1])

    # Merge into dataset
    #train_set = TensorDataset(train_tensor, train_labels)
    #test_set = TensorDataset(test_tensor, test_labels)
    train_set = PILTensorDataset(train_tensor, train_labels)
    test_set = PILTensorDataset(test_tensor, test_labels)

    # Convert to AvalancheDataset if transforms are given
    print("train_transform", train_transform)
    print("eval_transform", eval_transform)
    if train_transform is not None and eval_transform is not None:
        transform_groups = dict(
            #train=(assert_pil_transform(train_transform), None),
            #eval=(assert_pil_transform(eval_transform), None),
            train=(train_transform, None),
            eval=(eval_transform, None),
        )
        train_set = AvalancheDataset(train_set, transform_groups=transform_groups)
        test_set = AvalancheDataset(test_set, transform_groups=transform_groups)
        print("Transforms applied")

    return train_set, test_set



def DomainCifar100(
    rootpath, 
    train_transform=None, 
    eval_transform=None, 
    seed=None,
    task_labels_per_exp=None
):
    # Load the dataset from continuum (because the happen to have it prepared)
    train_set = CIFAR100(rootpath,
                        train=True,
                        labels_type="category",
                        task_labels="lifelong")

    test_set = CIFAR100(rootpath,
                        train=False,
                        labels_type="category",
                        task_labels="lifelong")
    # Access the data
    train_data = train_set.get_data() # Returns (imgs(50000), labels(20), task_labels)
    test_data = test_set.get_data() # Returns (imgs(10000), labels(20), task_labels)

    if seed:
         # Gernerate permutations for classes in tasks
        perm = np.zeros((20,5), dtype=int)
        for class_idx in range(20):
            perm[class_idx] = np.random.permutation(5)
    
    # Prepare TrainSet Split it according to the task labels
    prepared_data = []
    for data in [train_data, test_data]:
        if seed:
            data_copy = copy.deepcopy(data)
            # Apply the permutation to the task labels
            for class_idx in range(20):
                data[2][data[1]==class_idx] = \
                    perm[class_idx][data_copy[2][data[1]==class_idx]]
        # Get all tasks labels
        task_labels = np.unique(data[2])
        
        tensors_datasets = []
        target_datasets = []
        for task_label in task_labels:
            imgs = data[0][data[2] == task_label]
            labels = data[1][data[2] == task_label]
            #t = data[2][data[2] == task_label]
            imgs = torch.from_numpy(imgs).permute(0,3,1,2)
            labels = torch.from_numpy(labels)
            #t = torch.from_numpy(np.asarray(t))
            #tensors_datasets.append(PILTensorDataset(imgs, labels))
            tensors_datasets.append(imgs)  #PILTensorDataset(imgs, labels)
            target_datasets.append(labels)
        
        pil_datasets = []
        if task_labels_per_exp is None:
            task_labels_per_exp = [1]*len(tensors_datasets)
        curr_idx = 0
        for i in task_labels_per_exp:
            imgs = torch.cat(tensors_datasets[curr_idx:(curr_idx+i)], dim=0)
            labels = torch.cat(target_datasets[curr_idx:(curr_idx+i)], dim=0)
            curr_idx+=i
            pil_datasets.append(PILTensorDataset(imgs, labels))
        prepared_data.append(pil_datasets)

    # Create the avalanche benchmark
    scenario = create_multi_dataset_generic_benchmark(
        train_datasets=prepared_data[0],
        test_datasets=prepared_data[1],
        #task_labels=[0,1,2,3,4],  # Not settable here
        complete_test_set_only=False,
        train_transform=train_transform,
        train_target_transform=None,
        eval_transform=eval_transform,
        eval_target_transform=None,
    )
    return scenario
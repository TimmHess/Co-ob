################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Optional, List, Sequence, Dict, Any, Union
from copy import deepcopy

import torch
from torch.utils.data import Subset

from avalanche.benchmarks.scenarios.deprecated.classification_scenario import (
    ClassificationScenario,
    ClassificationStream,
    ClassificationExperience,
)
from avalanche.benchmarks.utils.classification_dataset import (
    _taskaware_classification_subset,
    TaskAwareClassificationDataset,
    TaskAwareSupervisedClassificationDataset,
)
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_multi_dataset_generic_benchmark
)

#from avalanche.benchmarks.utils import AvalancheSubset, AvalancheDataset
from avalanche.benchmarks.utils.flat_data import ConstantSequence

#from avalanche.benchmarks.utils.avalanche_dataset import SupportedDataset, \
#    AvalancheDataset, AvalancheDatasetType, AvalancheSubset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.classification_dataset import SupportedDataset

from avalanche.benchmarks.scenarios.deprecated.new_instances.ni_scenario import (
    NIExperience,
    NIStream
)

class NIRepeatDatasetScenario(
        ClassificationScenario[
            "NIStream", "NIExperience", TaskAwareSupervisedClassificationDataset
        ]
):
    """
    This class defines a "New Instance" scenario.
    Once created, an instance of this class can be iterated in order to obtain
    the experience sequence under the form of instances of
    :class:`NIExperience`.

    Instances of this class can be created using the constructor directly.
    However, we recommend using facilities like
    :func:`avalanche.benchmarks.generators.ni_scenario`.

    Consider that every method from :class:`NIExperience` used to retrieve
    parts of the test set (past, current, future, cumulative) always return the
    complete test set. That is, they behave as the getter for the complete test
    set.
    """

    def __init__(
            self,
            train_dataset: AvalancheDataset,
            test_dataset: AvalancheDataset,
            n_experiences: int,
            train_transform = None,
            eval_transform = None,
            task_labels: bool = True,
            shuffle: bool = True,
            seed: Optional[int] = None
        ):
        """
        Creates a CorruptionScenario instance given the training and test Datasets and
        the number of experiences.

        :param train_dataset: The training dataset. The dataset must be an
            instance of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``train_dataset=AvalancheDataset(torchvision_dataset)``.
        :param test_dataset: The test dataset. The dataset must be a
            subclass of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``test_dataset=AvalancheDataset(torchvision_dataset)``.
        :param n_experiences: The number of experiences.
        :param task_labels: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences.
            Defaults to False.
        :param shuffle: If True, the patterns order will be shuffled. Defaults
            to True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            PyTorch random number generator state will be used.
            Defaults to None.
        """

        self._has_task_labels = task_labels

        if n_experiences < 1:
            raise ValueError('Invalid number of experiences (n_experiences '
                             'parameter): must be greater than 0')

        unique_targets, unique_count = torch.unique(
            torch.as_tensor(train_dataset.targets), return_counts=True)

        self.n_classes: int = len(unique_targets)

        ##############################
        # Define Train Experiences
        ##############################
        train_experiences = []
        train_task_labels = []
        for t_id in range(n_experiences):
            if self._has_task_labels:
                train_task_labels.append(t_id)
            else:
                train_task_labels.append(0)
            task_labels = ConstantSequence(train_task_labels[-1], len(train_dataset))
        
            transform_groups = dict(
                    train=(train_transform, None),
                    test=(eval_transform, None)
            )

            # Add corrupted experience
            train_experiences.append(
                AvalancheDataset(
                    train_dataset,
                    transform_groups=deepcopy(transform_groups),
                    task_labels=task_labels,
                )
            )

        ##############################
        # Define Test Experiences
        ##############################
        test_experiences = []
        test_task_labels = []
        num_test_experiences = n_experiences
        for t_id in range(num_test_experiences):        
            if self._has_task_labels:
                test_task_labels.append(t_id)
            else:
                test_task_labels.append(0)
            task_labels = ConstantSequence(test_task_labels[-1], len(test_dataset))

            transform_groups = dict(
                    train=(train_transform, None),
                    test=(eval_transform, None)
            )

            test_experiences.append(
                AvalancheDataset(
                    test_dataset,
                    transform_groups=deepcopy(transform_groups),
                    task_labels=task_labels,
                )
            )

        # Create the scenario
        super(NIRepeatDatasetScenario, self).__init__(
            stream_definitions = {
                'train': (train_experiences, train_task_labels, train_dataset),
                'test': (test_experiences, test_task_labels, test_dataset)
            },
            complete_test_set_only=False,
            stream_factory=NIStream,
            experience_factory=NIExperience)
        
        return

    # NOTE: deactivated because I don't really know how this should be utilized?
    # def get_reproducibility_data(self) -> Dict[str, Any]:
    #     reproducibility_data = {
    #         'exps_patterns_assignment': self.train_exps_patterns_assignment,
    #         'has_task_labels': bool(self._has_task_labels),

    #     }
    #     return reproducibility_data

    # def alter_transforms(self):
    #     print(train_dataset.transform_groups["train"])
    #     return


def ni_repeat_dataset_benchmark(
    train_dataset: AvalancheDataset,
    test_dataset: AvalancheDataset,
    n_experiences: int,
    percentage_exp_assignment: Optional[Sequence[float]] = None,
    use_single_test_set=False,
    *,
    task_labels: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
    train_transform=None,
    eval_transform=None,
):
    train_datasets = []
    test_datasets = []

    if percentage_exp_assignment is not None:
        # Prepare the indices for the experience assignment
        remaining_patterns = set(range(len(train_dataset)))
        if seed is not None:
            torch.random.manual_seed(seed)
        patterns_order = torch.as_tensor(list(remaining_patterns))[
                            torch.randperm(len(list(remaining_patterns)))
                        ].tolist()
        n_patterns = len(patterns_order)
        n_patterns_per_exp = [
            int(n_patterns * percentage)
            for percentage in percentage_exp_assignment
        ]
    count = 0
    for exp_idx in range(n_experiences):
        if percentage_exp_assignment is not None:
            train_dataset_subset = Subset(train_dataset, 
                                   indices=patterns_order[count:count + n_patterns_per_exp[exp_idx]])
            count += n_patterns_per_exp[exp_idx]
            print("train_dataset", len(train_dataset_subset))
            print(count)
            train_datasets.append(train_dataset_subset)
        else:
            train_datasets.append(train_dataset)
        if not use_single_test_set:  # Always add full test dataset
            test_datasets.append(test_dataset)  
    
    if use_single_test_set:  # Add full test set only once 
        test_datasets.append(test_dataset)  

    assert len(train_datasets) > 0
    assert len(test_datasets) > 0

    # Create the avalanche benchmark
    scenario = create_multi_dataset_generic_benchmark(
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        complete_test_set_only=False,
        train_transform=train_transform,
        train_target_transform=None,
        eval_transform=eval_transform,
        eval_target_transform=None,
    )
    return scenario

    # return NIRepeatDatasetScenario(
    #     train_dataset=seq_train_dataset, 
    #     test_dataset=seq_test_dataset,
    #     n_experiences=n_experiences,
    #     task_labels=task_labels,
    #     shuffle=shuffle, 
    #     seed=seed,
    #     train_transform=train_transform,  # Be careful - transforms should only be applied 1 time.. 
    #     eval_transform=eval_transform,  # Be careful - transforms should only be applied 1 time.. 
    # )


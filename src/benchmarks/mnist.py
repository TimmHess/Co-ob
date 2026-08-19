from pathlib import Path
from typing import Optional, Sequence, Union, Any

import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import RandomRotation

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils import (
    _make_taskaware_classification_dataset,
    DefaultTransformGroups,
)
import avalanche.benchmarks.datasets.external_datasets.mnist as mnist
from avalanche.benchmarks.classic.cmnist import _default_mnist_train_transform, _default_mnist_eval_transform
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import\
    create_multi_dataset_generic_benchmark

from avalanche.benchmarks import NCScenario, nc_benchmark
from src.utils.transform_tensor_dataset import PILTensorDataset

from src.utils.util import assert_pil_transform, safe_index



MNIST_classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]




def get_mnist_dataset(dataset_root, train_transform=None, eval_transform=None):
    train_set, test_set = mnist.get_mnist_dataset(dataset_root=dataset_root)

    train_set.data = train_set.data.unsqueeze(1).repeat(1, 3, 1, 1)
    test_set.data = test_set.data.unsqueeze(1).repeat(1, 3, 1, 1)

    train_set = PILTensorDataset(train_set.data, train_set.targets)
    test_set = PILTensorDataset(test_set.data, test_set.targets)

    if train_transform is not None and eval_transform is not None:
        # transform_groups = dict(
        #     train=(assert_pil_transform(train_transform), None),
        #     eval=(assert_pil_transform(eval_transform), None),
        # )
        transform_groups = dict(
            train=(train_transform, None),
            eval=(eval_transform, None),
        )
        train_set = AvalancheDataset(train_set, transform_groups=transform_groups)
        test_set = AvalancheDataset(test_set, transform_groups=transform_groups)
        print("Transforms applied")
    
    return train_set, test_set


def SplitMNIST(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_mnist_train_transform,
    eval_transform: Optional[Any] = _default_mnist_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    Creates a CL benchmark using the MNIST dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    
    mnist_train, mnist_test = get_mnist_dataset(dataset_root)

    return nc_benchmark(
        train_dataset=mnist_train,
        test_dataset=mnist_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )



def _split_classes_across_experiences(classes, n_experiences):
    """Split classes as evenly as possible across experiences."""
    classes_per_exp = len(classes) // n_experiences
    remainder = len(classes) % n_experiences
    
    result = []
    start_idx = 0
    
    for exp in range(n_experiences):
        # First 'remainder' experiences get one extra class
        current_exp_classes = classes_per_exp + (1 if exp < remainder else 0)
        end_idx = start_idx + current_exp_classes
        
        result.append(classes[start_idx:end_idx])
        start_idx = end_idx
    
    return result

def _filter_dataset_by_classes(dataset, target_classes):
    """Filter dataset to only include samples from target classes."""
    # Get all data and targets
    data = dataset.data
    targets = dataset.targets
    
    # Create mask for target classes
    mask = np.isin(targets, target_classes)
    
    # Filter data and targets
    filtered_data = data[mask]
    filtered_targets = targets[mask]
    
    return filtered_data, filtered_targets

def StaticCorruptedSplitMNIST(
    n_experiences: int,
    corruption_set: Optional[Sequence[str]],
    severities: Optional[Sequence[int]],
    *,
    return_task_id: bool = True,  # NOTE: default?
    seed: Optional[int] = None,
    class_order: Optional[Sequence[int]] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Optional[Union[str, Path]] = None,
):
    """Creates a Split MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The 10 MNIST classes are split across ``n_experiences`` different tasks.
    Each experience contains a subset of the original 10 MNIST classes.
    The classes are distributed as evenly as possible across experiences.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from 0, is applied to each experience.

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. It indicates how many tasks the MNIST dataset should be
        split into. Should be between 1 and 10.
    :param seed: A valid int used to initialize the random number generator
        for class ordering. Can be None.
    :param class_order: A list of class indices (0-9) that defines the order
        in which classes should be assigned to experiences. If None, classes
        will be assigned in order 0-9, or randomly if seed is provided.
        If provided, seed parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation).
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation).
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if class_order is not None and len(class_order) != 10:
        raise ValueError(
            "class_order should contain exactly 10 class indices (0-9)."
        )

    if class_order is not None and set(class_order) != set(range(10)):
        raise ValueError(
            "class_order should contain all class indices from 0 to 9 exactly once."
        )

    # Determine class order
    if class_order is not None:
        classes = list(class_order)
    elif seed is not None:
        rng = np.random.RandomState(seed)
        classes = list(range(10))
        rng.shuffle(classes)
    else:
        classes = list(range(10))

    # Split classes across experiences
    classes_per_experience = _split_classes_across_experiences(classes, n_experiences)

    list_train_dataset = []
    list_test_dataset = []

    # Get the original MNIST dataset
    #mnist_train, mnist_test = get_mnist_dataset(dataset_root)
    mnist_train, mnist_test = mnist.get_mnist_dataset(dataset_root=dataset_root)


    from src.utils.corruption.corruption_handler import discover_corruption_pipeline
    corruption_pipeline_train = discover_corruption_pipeline(train_transform.transforms)
    corruption_pipeline_eval = discover_corruption_pipeline(eval_transform.transforms)
    # Create each experience
    for idx, exp in enumerate(range(n_experiences)):
        print("Creating experience {} with classes: {}".format(
            idx, classes_per_experience[exp]
        ))
        exp_classes = classes_per_experience[exp]
        
        # Filter training data for current experience classes
        train_subset = _filter_dataset_by_classes(mnist_train, exp_classes)
        train_subset = PILTensorDataset(train_subset[0], train_subset[1])  # Convert to PILTensorDataset
        test_subset = _filter_dataset_by_classes(mnist_test, exp_classes)
        test_subset = PILTensorDataset(test_subset[0], test_subset[1])  # Convert to PILTensorDataset

        # Adjust the corruption in the transform
        corruption_pipeline_train.set_corruption(safe_index(corruption_set, idx))
        corruption_pipeline_train.set_severity(safe_index(severities, idx))
        corruption_pipeline_eval.set_corruption(safe_index(corruption_set, idx))
        corruption_pipeline_eval.set_severity(safe_index(severities, idx))
        print("Setting corruption for experience {}: {} with severity {}".format(
            idx, safe_index(corruption_set, idx), safe_index(severities, idx)
        ))

        # Do a deepcopy of the train and eval transforms to freeze the corruption pipeline
        curr_train_transform = deepcopy(train_transform)
        curr_eval_transform = deepcopy(eval_transform)

        print("DEBUG: StaticCorruptedSplitMNIST: Applying corruption")
        print(curr_train_transform)
        print(curr_eval_transform)

        # Create task-aware datasets
        train_dataset = make_avalanche_dataset(
            _make_taskaware_classification_dataset(train_subset),
            frozen_transform_groups=DefaultTransformGroups((curr_train_transform, None)),
        )

        test_dataset = make_avalanche_dataset(
            _make_taskaware_classification_dataset(test_subset),
            frozen_transform_groups=DefaultTransformGroups((curr_eval_transform, None)),
        )

        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return create_multi_dataset_generic_benchmark(
        train_datasets=list_train_dataset,
        test_datasets=list_test_dataset,
        complete_test_set_only=False,
        train_transform=None,
        eval_transform=None,
    )



def _remap_targets_odd_even(targets):
    """Remap targets to odd (1) vs even (0)."""
    return (targets % 2).type(targets.dtype)

def _remap_targets_low_high(targets):
    """Remap targets to low (0) for <=4 and high (1) for >4."""
    return (targets > 4).type(targets.dtype)

def _apply_target_remapping(dataset, remapping_func):
    """Apply target remapping to dataset."""
    data = dataset.data
    targets = dataset.targets
    
    # Apply remapping function
    remapped_targets = remapping_func(targets)
    
    return data, remapped_targets

def OddEven_LowHigh_MNIST(
    n_experiences: int,
    corruption_set: Optional[Sequence[str]],
    severities: Optional[Sequence[int]],
    *,
    return_task_id: bool = True,  # NOTE: default?
    seed: Optional[int] = None,
    class_order: Optional[Sequence[int]] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Optional[Union[str, Path]] = None,
    grayscale=True,
):
    """Creates a remapped MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Creates two experiences:
    - Experience 1: Odd vs Even classification (full MNIST dataset)
    - Experience 2: Low (<=4) vs High (>4) classification (full MNIST dataset)

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from 0, is applied to each experience.

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. Should be 2 for this implementation.
    :param seed: A valid int used to initialize the random number generator
        (not used in this implementation but kept for compatibility).
    :param class_order: A list of class indices (not used in this implementation
        but kept for compatibility).
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation).
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation).
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if n_experiences != 2:
        raise ValueError(
            "This implementation only supports n_experiences=2 (Odd vs Even, Low vs High)."
        )

    # Define the remapping functions for each experience
    remapping_functions = [
        _remap_targets_odd_even,   # Experience 0: Odd vs Even
        _remap_targets_low_high    # Experience 1: Low vs High
    ]
    
    experience_descriptions = [
        "Odd vs Even",
        "Low (<=4) vs High (>4)"
    ]

    list_train_dataset = []
    list_test_dataset = []

    # Get the original MNIST dataset
    mnist_train, mnist_test = mnist.get_mnist_dataset(dataset_root=dataset_root)

    # Convert grayscale to RGB by repeating the single channel 3 times
    if not grayscale:
        mnist_train.data = mnist_train.data.unsqueeze(1).repeat(1, 3, 1, 1)
        mnist_test.data = mnist_test.data.unsqueeze(1).repeat(1, 3, 1, 1)

    from src.utils.corruption.corruption_handler import discover_corruption_pipeline
    corruption_pipeline_train = discover_corruption_pipeline(train_transform.transforms)
    corruption_pipeline_eval = discover_corruption_pipeline(eval_transform.transforms)
    
    # Create each experience
    for idx in range(n_experiences):
        print("Creating experience {} with task: {}".format(
            idx, experience_descriptions[idx]
        ))
        
        # Apply target remapping for current experience
        train_data, train_targets = _apply_target_remapping(mnist_train, remapping_functions[idx])
        train_subset = PILTensorDataset(train_data, train_targets)
        
        test_data, test_targets = _apply_target_remapping(mnist_test, remapping_functions[idx])
        test_subset = PILTensorDataset(test_data, test_targets)

        # Adjust the corruption in the transform
        corruption_pipeline_train.set_corruption(safe_index(corruption_set, idx))
        corruption_pipeline_train.set_severity(safe_index(severities, idx))
        corruption_pipeline_eval.set_corruption(safe_index(corruption_set, idx))
        corruption_pipeline_eval.set_severity(safe_index(severities, idx))
        print("Setting corruption for experience {}: {} with severity {}".format(
            idx, safe_index(corruption_set, idx), safe_index(severities, idx)
        ))

        # Do a deepcopy of the train and eval transforms to freeze the corruption pipeline
        curr_train_transform = deepcopy(train_transform)
        curr_eval_transform = deepcopy(eval_transform)

        print("DEBUG: StaticCorruptedSplitMNIST: Applying corruption")
        print(curr_train_transform)
        print(curr_eval_transform)

        # Create task-aware datasets
        train_dataset = make_avalanche_dataset(
            _make_taskaware_classification_dataset(train_subset),
            frozen_transform_groups=DefaultTransformGroups((curr_train_transform, None)),
        )

        test_dataset = make_avalanche_dataset(
            _make_taskaware_classification_dataset(test_subset),
            frozen_transform_groups=DefaultTransformGroups((curr_eval_transform, None)),
        )

        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return create_multi_dataset_generic_benchmark(
        train_datasets=list_train_dataset,
        test_datasets=list_test_dataset,
        complete_test_set_only=False,
        train_transform=None,
        eval_transform=None,
    )

def OddEven_LowHigh_MNIST_3channel(*args, **kwargs):
    kwargs['grayscale'] = False
    return OddEven_LowHigh_MNIST(*args, **kwargs)


def RotatedMNIST(
    n_experiences: int,
    *,
    return_task_id: bool = False,
    seed: Optional[int] = None,
    rotations_list: Optional[Sequence[int]] = None,
    train_transform: Optional[Any] = _default_mnist_train_transform,
    eval_transform: Optional[Any] = _default_mnist_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    """Creates a Rotated MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Random angles are used to rotate the MNIST images in ``n_experiences``
    different manners. This means that each experience is composed of all the
    original 10 MNIST classes, but each image is rotated in a different way.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from 0, is applied to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. It indicates how many different rotations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param rotations_list: A list of rotations values in degrees (from -180 to
        180) used to define the rotations. The rotation specified in position
        0 of the list will be applied to the task 0, the rotation specified in
        position 1 will be applied to task 1 and so on.
        If None, value of ``seed`` will be used to define the rotations.
        If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if rotations_list is not None and len(rotations_list) != n_experiences:
        raise ValueError(
            "The number of rotations should match the number"
            " of incremental experiences."
        )

    if rotations_list is not None and any(
        180 < rotations_list[i] < -180 for i in range(len(rotations_list))
    ):
        raise ValueError(
            "The value of a rotation should be between -180" " and 180 degrees."
        )

    list_train_dataset = []
    list_test_dataset = []
    rng_rotate = np.random.RandomState(seed)

    mnist_train, mnist_test = get_mnist_dataset(dataset_root)

    # for every incremental experience
    for exp in range(n_experiences):
        if rotations_list is not None:
            rotation_angle = rotations_list[exp]
        else:
            # choose a random rotation of the pixels in the image
            rotation_angle = rng_rotate.randint(-180, 181)

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))

        # Freeze the rotation
        rotated_train = make_avalanche_dataset(
            _make_taskaware_classification_dataset(mnist_train),
            frozen_transform_groups=DefaultTransformGroups((rotation, None)),
        )

        rotated_test = make_avalanche_dataset(
            _make_taskaware_classification_dataset(mnist_test),
            frozen_transform_groups=DefaultTransformGroups((rotation, None)),
        )

        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    return create_multi_dataset_generic_benchmark(
        train_datasets=list_train_dataset,
        test_datasets=list_test_dataset,
        complete_test_set_only=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
        #train_target_transform=None,
        #eval_target_transform=None,
    )

    # return nc_benchmark(
    #     list_train_dataset,
    #     list_test_dataset,
    #     n_experiences=len(list_train_dataset),
    #     task_labels=return_task_id,
    #     shuffle=False,
    #     class_ids_from_zero_in_each_exp=True,
    #     one_dataset_per_exp=True,
    #     train_transform=train_transform,
    #     eval_transform=eval_transform,
    # )
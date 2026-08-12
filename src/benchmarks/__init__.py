from .domain_cifar100 import *

def get_dataset_class_names(dataset_name):
    """
    Get the class names for a given dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        list: List of class names.
    """
    if dataset_name == "cifar100":
        from .cifar100 import CIFAR100_classes
        return CIFAR100_classes
    elif dataset_name == "domain_cifar100":
        from .cifar100 import CIFAR100_classes_super
        return CIFAR100_classes_super
    elif dataset_name == "cifar10":
        from .cifar100 import CIFAR10_classes
        return CIFAR10_classes
    elif dataset_name == "mnist":
        from .mnist import MNIST_classes
        return MNIST_classes
    elif dataset_name == "aircrafts":
        from .aircrafts import AIRCRAFTS_classes
        return AIRCRAFTS_classes
    elif dataset_name == "flowers-102":
        from .flowers102 import FLOWERS102_classes
        return FLOWERS102_classes
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
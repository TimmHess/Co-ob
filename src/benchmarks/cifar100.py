import numpy as np
import torch
import copy

from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
import avalanche.benchmarks.datasets.external_datasets.cifar as avl_cifar

from src.utils.transform_tensor_dataset import PILTensorDataset

CIFAR10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

CIFAR100_classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]
CIFAR100_classes_super = [
"aquatic mammals",
"fish",
"flowers",
"food containers",
"fruit and vegetables",
"household electrical devices",
"household furniture",
"insects",
"large carnivores",
"large man-made outdoor things",
"large natural outdoor scenes",
"large omnivores and herbivores",
"medium-sized mammals",
"non-insect invertebrates",
"people",
"reptiles",
"small mammals",
"trees",
"vehicles 1",
"vehicles 2"
]

"""
We can create a instance incremental setting with the coarse labels, i.e. 20 classes. 
Data are labeled with the coarse labels of CIFAR100. However, data are shared between tasks using 
the original label to ensure a domain drift between tasks , e.g., for the coarse label say{aquatic mammals} 
the data go from beavers to dolphins to otters to seals to finally whales in separate tasks.
"""

def get_cifar100_dataset(rootpath, train_transform=None, eval_transform=None):
    # Load the dataset from continuum (because the happen to have it prepared)
    # train_set = CIFAR100(rootpath,
    #                     train=True,
    #                     labels_type="category",
    #                     task_labels="lifelong")

    # test_set = CIFAR100(rootpath,
    #                     train=False,
    #                     labels_type="category",
    #                     task_labels="lifelong")
    # # Access the data
    # train_data = train_set.get_data() # Returns (imgs(50000), labels(20), task_labels)
    # test_data = test_set.get_data() # Returns (imgs(10000), labels(20), task_labels)

    # #Convert to Tensors
    # train_tensor = torch.from_numpy(train_data[0])  #targets = torch.tensor(train_data[0])
    # train_tensor = train_tensor.permute(0,3,1,2)
    # train_labels = torch.from_numpy(train_data[1])

    # test_tensor = torch.from_numpy(test_data[0])
    # test_tensor = test_tensor.permute(0,3,1,2)
    # test_labels = torch.from_numpy(test_data[1])

    cifar_train, cifar_test = avl_cifar.get_cifar100_dataset(rootpath)  # NOTE: CIFAR100 dataset
    train_data, train_targets = torch.from_numpy(cifar_train.data), torch.from_numpy(np.asarray(cifar_train.targets))
    train_data = train_data.permute(0,3,1,2)
    test_data, test_targets = torch.from_numpy(cifar_test.data), torch.from_numpy(np.asarray(cifar_test.targets))
    test_data = test_data.permute(0,3,1,2)

    # Merge into dataset
    #train_set = TensorDataset(train_data, train_targets)
    #test_set = TensorDataset(test_data, test_targets)
    train_set = PILTensorDataset(train_data, train_targets)
    test_set = PILTensorDataset(test_data, test_targets)

    # Convert to AvalancheDataset if transforms are given
    print("train_transform", train_transform)
    print("eval_transform", eval_transform)
    if train_transform is not None and eval_transform is not None:
        transform_groups = dict(
            # train=(assert_pil_transform(train_transform), None),
            # eval=(assert_pil_transform(eval_transform), None),
            train=(train_transform, None),
            eval=(eval_transform, None),
        )
        train_set = AvalancheDataset(train_set, transform_groups=transform_groups)
        test_set = AvalancheDataset(test_set, transform_groups=transform_groups)
        print("Transforms applied")

    return train_set, test_set
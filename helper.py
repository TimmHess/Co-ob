from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import torch
from torchvision import transforms
import numpy as np

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.scenarios.deprecated.generators import ni_benchmark

from src.benchmarks.mnist import (
    SplitMNIST, StaticCorruptedSplitMNIST, OddEven_LowHigh_MNIST
)
from src.benchmarks.domain_cifar100 import DomainCifar100
from src.benchmarks.repeat.repeat_dataset_scenario import (
    ni_repeat_dataset_benchmark
)
from src.benchmarks.imgnet1k_256x256 import SplitImageNet1k
from src.benchmarks.arrow_scenario import SplitArrowScenario

from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics,
)

from src.eval.continual_eval import ContinualEvaluationPhasePlugin
from src.eval.continual_eval_metrics import TaskTrackingLossPluginMetric,\
        TaskTrackingAccuracyPluginMetric,TaskTrackingMINAccuracyPluginMetric,\
        TaskAveragingPluginMetric, WCACCPluginMetric

from avalanche.training.supervised import Naive, Cumulative
from avalanche.training.plugins import FeatureDistillationPlugin
from src.methods.lwf import LwFPlugin
from src.methods.mas import MASPlugin

from src.utils.corruption.corruption_transforms import CorruptionPipeline

from src.methods.barlow_twins import TwoCropTransform

from src.methods.bn_freeze import BNFreezePlugin
from src.methods.freeze import FreezeModelPlugin
from src.methods.gradient_clipping import GradClipPlugin

from src.utils.training_iterations_controller import TrainingIterationControllerPlugin
from src.models.store_models import StoreModelsPlugin
from src.models.reinit_model import ReinitModelPlugin

from src.utils.onecyclelr_plugin import OneCycleSchedulerPlugin
 
from src.eval.initial_eval_stage import InitialEvalStage
from src.eval.linear_probing_metric import LinearProbingAccuracyMetric
from src.eval.knn_probing_metric import KNNProbingAccuracyMetric
from src.eval.concatenated_linear_probing_metric import ConcatenatedLinearProbingAccuracyMetric
from src.eval.downstream_linear import DownstreamLinearProbeAccuracyMetric, CorruptedDownstreamLinearProbeAccuracyMetric
from src.eval.downstream_concatenated_probing_metric import ConcatDownstreamProbingAccuracyMetric
from src.eval.downstream_knn import DownstreamKNNAccuracyMetric
from src.eval.downstream_concatenated_knn_probing import DownstreamConcatenatedKNNAccuracyMetric



def get_dataset(
    scenario_name,
    dset_rootpath,
    train_transform=None,
    eval_transform=None,
    eval_only=True,
):
    print("\nGetDataset: eval_transform: {}".format(eval_transform))
    train_set = None
    test_set = None
    n_classes = None
    dset_rootpath = deepcopy(dset_rootpath) # NOTE: just to be sure...

    eval_transform = deepcopy(eval_transform) # NOTE: because 'call by reference'
    train_transform = eval_transform if eval_only else deepcopy(train_transform) # NOTE: because 'call by reference'

    if scenario_name == "mnist":
        from src.benchmarks.mnist import get_mnist_dataset
        train_set, test_set = get_mnist_dataset(
            dataset_root=dset_rootpath, 
            train_transform=train_transform, 
            eval_transform=eval_transform
        )
        n_classes = 10
    elif scenario_name == 'cifar100':
        from src.benchmarks.cifar100 import get_cifar100_dataset
        train_set, test_set = get_cifar100_dataset(
            rootpath=dset_rootpath, 
            train_transform=train_transform,
            eval_transform=eval_transform
        )
        n_classes = 100
    elif scenario_name == 'domain_cifar100':
        from src.benchmarks.domain_cifar100 import get_domain_cifar100_dataset
        train_set, test_set = get_domain_cifar100_dataset(
            rootpath=dset_rootpath, 
            train_transform=train_transform,
            eval_transform=eval_transform
        )
        n_classes = 20
    elif scenario_name == 'imgnet100_256x256':
        from src.benchmarks.arrow_loader import load_hf_arrow_dataset
        train_set, test_set = load_hf_arrow_dataset(
            rootpath=dset_rootpath,
            train_transform=train_transform,
            eval_transform=eval_transform,
            eval_dir="val"
        )
        n_classes = 100
    elif scenario_name == 'imgnet1k_256x256':
        from src.benchmarks.arrow_loader import load_hf_arrow_dataset
        train_set, test_set = load_hf_arrow_dataset(
            rootpath=dset_rootpath,
            train_transform=train_transform,
            eval_transform=eval_transform,
            eval_dir="val"
        )
        n_classes = 1000
    else:
        try: 
            from src.benchmarks.arrow_loader import load_hf_arrow_dataset
            train_set, test_set = load_hf_arrow_dataset(
                rootpath=dset_rootpath,
                train_transform=train_transform,
                eval_transform=eval_transform
            )
            n_classes = len(set(test_set.targets))
            print("Inferred number of classes:", n_classes)
        except:
            raise ValueError("Unknown dataset name: {}".format(scenario_name))

    return train_set, test_set, n_classes


def _get_norm_from_transform(t):
    """Extract (mean, std) from the Normalize inside a Compose. Returns (None, None) if not found."""
    if isinstance(t, transforms.Compose):
        for sub_t in t.transforms:
            if isinstance(sub_t, transforms.Normalize):
                return tuple(sub_t.mean), tuple(sub_t.std)
    return None, None


def get_transforms(
    data_aug,
    input_size,
    norm_mean=(0.0, 0.0, 0.0),
    norm_std=(1.0, 1.0, 1.0),
    use_two_crop_transform=False,
    use_corruption=False,
    eval_protocol="default",
):
    """
    Single place in codebase where data transforms are defined.

    eval_protocol:
        "default"  - eval uses Resize(input_size) -> ToTensor -> Normalize
        "imagenet" - eval uses the standard ImageNet protocol:
                     Resize(256, BILINEAR) -> CenterCrop(224) -> ToTensor -> Normalize

    Return: train and test transforms
    """
    resize = transforms.Resize(size=(input_size[1], input_size[2]))

    crop_flip = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2]),
                                    scale=(0.1 if input_size[0]>=64 else 0.2, 1.),
                                    ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    sim_clr = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2]),
                                    scale=(0.1 if input_size[0]>=64 else 0.2, 1.),
                                    ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.2,
                                    hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=input_size[0]//20*2+1,
                    sigma=(0.1, 2.0)
                )],
            p=0.5 if input_size[0]>32 else 0.0),
    ])
    rand_crop_aug = transforms.Compose([
        transforms.RandomCrop(size=(input_size[1], input_size[2]), padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ])
    autoaug_cifar10 = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10)
    ])
    autoaug_imgnet = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
    ])
    rand_augment = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9)
    ])

    rand_grayscale = transforms.Compose([transforms.RandomGrayscale(p=0.2)]) # TODO
    rand_gauss_blur = transforms.Compose([
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=input_size[0]//20*2+1, sigma=(0.1, 2.0))
                    ], p=0.5 if input_size[0]>32 else 0.0
                    )
                ])

    normalize = transforms.Normalize(norm_mean, norm_std)
    to_tensor = transforms.ToTensor()

    # --- Eval transform: protocol-driven, independent of augmentation ---
    if eval_protocol == "imagenet":
        print("\nUsing ImageNet eval protocol: Resize(256) -> CenterCrop(224)\n")
        test_transforms = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
        ]
    else:
        test_transforms = [resize]

    # --- Train transform: augmentation-driven ---
    train_transforms = []

    print("selected data augmentation:", data_aug)
    if "simclr" in data_aug:
        train_transforms.append(sim_clr)
    elif "auto_cifar10" in data_aug:
        train_transforms.append(autoaug_cifar10)
    elif "auto_imgnet" in data_aug:
        train_transforms.append(autoaug_imgnet)
    elif "rand_augment" in data_aug:
        train_transforms.append(rand_augment)
    elif "crop_flip" in data_aug:
        train_transforms.append(crop_flip)
        train_transforms.insert(0, resize)
    elif "rand_crop" in data_aug:
        train_transforms.append(rand_crop_aug)
        train_transforms.insert(0, resize)
        print("Using random crop augmentation")
    else:
        train_transforms.append(resize)

    # Image Corruptions (inserted before ToTensor/Normalize)
    if "corruption" in data_aug or use_corruption: # NOTE: this allows to add corruption only to the test set
        corruption_pipeline = CorruptionPipeline()
        train_transforms.append(corruption_pipeline)
        test_transforms.append(corruption_pipeline)

    # Finalize with ToTensor and Normalize
    train_transforms.extend([to_tensor, normalize])
    test_transforms.extend([to_tensor, normalize])

    # Finally, compose everything into a single transform
    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    if use_two_crop_transform:
        train_transforms = TwoCropTransform(train_transforms)

    return train_transforms, test_transforms


def get_scenario(
    args, 
    scenario_name, 
    dset_rootpath,
    num_experiences=None, 
    use_data_aug=None, 
    seed=42
):
    print(f"\n[SCENARIO] {scenario_name}, Task Incr = {args.task_incr}")

    # Check for 'none' string #NOTE: this will happen when using default roots for downstream sets
    if not dset_rootpath is None:
        if dset_rootpath.lower() == "none":
            dset_rootpath = None

    # Prepare general transforms
    train_transform = None
    test_transform = None
    data_transforms = dict()

    if scenario_name in ["smnist", "incr_mnist", "static_corrupted_mnist", "OE_LH_mnist"]:
        input_size = (1, 28, 28)
    elif scenario_name in ["cifar100", "domain_cifar100",\
                           "incr_cifar100", "incr_domain_cifar100",\
                           "repeat_cifar100", "repeat_domain_cifar100"]:
        input_size = (3, 32, 32)
    elif scenario_name in ["imagenet_A",
                           "imgnet100_256x256",
                           "imgnet1k_256x256",
                           "repeat_dataset"]:
        input_size = (3, 224, 224)
    else:
        print("Using default input size of 224x224")
        input_size = (3, 224, 224)

    if not args.overwrite_input_size is None:
        input_size = (input_size[0], args.overwrite_input_size[0], args.overwrite_input_size[1])

    norm_mean = None
    norm_stddev = None
    if args.overwrite_mean and args.overwrite_stddev:
        norm_mean = tuple(args.overwrite_mean)
        norm_stddev = tuple(args.overwrite_stddev)
        print("Overwriting mean and stddev with", norm_mean, norm_stddev)

    n_classes = None
    fixed_class_order = args.fixed_class_order

    # Prepare datasets/scenarios
    if scenario_name == 'smnist':
        n_classes = 10
        n_experiences = 5

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=(0.1307,), 
            norm_std=(0.3081,),
            use_corruption=False if args.use_corruptions is None else True
        )
        new_dset_root = dset_rootpath

        scenario = SplitMNIST(
            n_experiences=n_experiences, 
            return_task_id=args.task_incr, 
            seed=seed,
            fixed_class_order=[i for i in range(n_classes)], 
            dataset_root=new_dset_root,
            train_transform=train_transform,
            eval_transform=test_transform
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes // n_experiences  # For Multi-Head

    # (Data-) Incremental MNIST
    elif scenario_name == 'incr_mnist':
        n_classes = 10
        n_experiences = 2

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=(0.1307,),  
            norm_std=(0.3081,),
            use_corruption=False if args.use_corruptions is None else True
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy  

        train_set, test_set, n_classes = get_dataset(
            scenario_name="mnist",
            dset_rootpath=dset_rootpath,
            train_transform=None,
            eval_transform=None,
        )

        scenario = ni_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            shuffle=False,
            seed=args.seed,
            balance_experiences=True,
            percentage_exp_assignment=args.data_percentage_per_exp,
            train_transform=train_transform,
            eval_transform=test_transform
        )
        
        scenario.n_classes = n_classes
        #scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes

    elif scenario_name == 'static_corrupted_mnist':
        n_classes = 10
        n_experiences = 1

        fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)
        
        n_experiences = num_experiences if args.rotations_list is None else len(args.rotations_list)
        print("n_experiences:", n_experiences)

        if args.rotations_list is None:
            args.rotations_list = torch.linspace(0.0, 160.0, n_experiences).tolist()

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=(0.1307,),  
            norm_std=(0.3081,),
            use_corruption=False if args.use_corruptions is None else True            
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy  

        new_dset_root = dset_rootpath

        scenario = StaticCorruptedSplitMNIST(
            n_experiences=n_experiences,
            corruption_set=args.use_corruptions,
            severities=args.corruption_severities,
            dataset_root=new_dset_root,
            return_task_id=True,
            train_transform=altered_train_transform,
            eval_transform=test_transform,
            seed=args.seed,
            class_order=fixed_class_order,
        )

        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes #// n_experiences # because it does not matter

    elif scenario_name == 'OE_LH_mnist':
        n_classes = 2
        n_experiences = 2  # NOTE: hardcode
        print("n_experiences:", n_experiences)

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=(0.1307,),  
            norm_std=(0.3081,),
            use_corruption=False if args.use_corruptions is None else True            
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy  

        new_dset_root = dset_rootpath

        scenario = OddEven_LowHigh_MNIST(
            n_experiences=n_experiences,
            corruption_set=args.use_corruptions,
            severities=args.corruption_severities,
            dataset_root=new_dset_root,
            return_task_id=True, #args.task_incr,
            train_transform=altered_train_transform,
            eval_transform=test_transform,
            seed=args.seed
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes
 
    # CIFAR100
    elif scenario_name == 'cifar100':
        n_classes = 100
        n_experiences = 10

        if not fixed_class_order:
            fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            assert not args.fixed_class_order, "Cannot use random class ordering with fixed class ordering!"
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        fixed_class_order = [int(x) for x in fixed_class_order]

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True
        )
        print("train transform:")
        print(train_transform)
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        new_dset_root = dset_rootpath

        scenario = SplitCIFAR100(
            n_experiences=n_experiences, 
            return_task_id=True,  #args.task_incr, 
            seed=seed,
            fixed_class_order=fixed_class_order,
            train_transform=altered_train_transform,
            eval_transform=test_transform,
            dataset_root=new_dset_root,
            class_ids_from_zero_in_each_exp=args.task_incr
        )
        
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes // n_experiences

    # Domain Cifar100
    elif scenario_name == 'domain_cifar100':
        n_classes = 20
        #n_experiences = 5

        assert num_experiences is None, "domain cifar100 will always give 5 experiences! Remove overwriting of experiences!"

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True
        )
        altered_train_transform = train_transform # NOTE: need this potentially redundant line for 'supcon' strategy

        new_dset_root = dset_rootpath
        scenario = DomainCifar100(
            rootpath=new_dset_root,
            train_transform=altered_train_transform,
            eval_transform=test_transform,
            seed=args.seed if args.use_rand_class_ordering else None,  # NOTE: None is default task order
            task_labels_per_exp=args.task_labels_per_exp
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes #// n_experiences # because its domain incremental!

    # Data-incremental (new-instances) CIFAR100
    elif scenario_name == 'incr_cifar100':
        n_classes = 100
        n_experiences = 5

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        train_set, test_set, n_classes = get_dataset(
                scenario_name="cifar100",
                dset_rootpath=dset_rootpath,
                train_transform=None,
                eval_transform=None,
            )

        scenario = ni_benchmark(
            train_dataset=train_set, test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            shuffle=False,
            seed=args.seed,
            balance_experiences=True,
            percentage_exp_assignment=args.data_percentage_per_exp,
            train_transform=train_transform,
            eval_transform=test_transform
        )
        
        scenario.n_classes = n_classes
        initial_out_features = n_classes

    elif scenario_name == 'repeat_cifar100':
        n_classes = 100
        n_experiences = 5

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        # Load data
        train_set, test_set, n_classes = get_dataset(
                scenario_name="cifar100",
                dset_rootpath=dset_rootpath,
                train_transform=None,
                eval_transform=None,
            )

        # Create the benchmark
        scenario = ni_repeat_dataset_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=num_experiences,
            percentage_exp_assignment=args.data_percentage_per_exp,
            task_labels = True,
            shuffle = True,
            seed = None,
            train_transform = train_transform,
            eval_transform = test_transform
        )
        scenario.n_classes = n_classes
        #scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes

    # Data-incremental (new-instances) Domain-CIFAR100
    elif scenario_name == 'incr_domain_cifar100':
        n_classes = 20
        n_experiences = 5

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        train_set, test_set, n_classes = get_dataset(
            scenario_name="domain_cifar100",
            dset_rootpath=dset_rootpath,
            train_transform=None,
            eval_transform=None,
        )

        scenario = ni_benchmark(
            train_dataset=train_set, test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            shuffle=True,
            seed=args.seed,
            balance_experiences=True,
            percentage_exp_assignment=args.data_percentage_per_exp,
            train_transform=train_transform,
            eval_transform=test_transform
        )
        
        scenario.n_classes = n_classes
        #scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes

    elif scenario_name == 'repeat_domain_cifar100':
        n_classes = 20
        n_experiences = 5

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True
        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        train_set, test_set, n_classes = get_dataset(
            scenario_name="domain_cifar100",
            dset_rootpath=dset_rootpath,
            train_transform=None,
            eval_transform=None,
        )

        scenario = ni_repeat_dataset_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=num_experiences,
            percentage_exp_assignment=args.data_percentage_per_exp,
            use_single_test_set=args.use_single_test_set,
            task_labels = True,
            shuffle = True,
            seed = None,
            train_transform = train_transform,
            eval_transform = test_transform
        )
        
        scenario.n_classes = n_classes
        initial_out_features = n_classes

    elif scenario_name == 'imgnet100_256x256':
        n_classes = 100
        n_experiences = 10

        fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            print("Using random class ordering...")
            #fixed_class_order = None
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug,
            input_size=input_size,
            norm_mean=norm_mean if norm_mean else (0.4914, 0.4822, 0.4465),
            norm_std=norm_stddev if norm_stddev else (0.2023, 0.1994, 0.2010),
            use_corruption=False if args.use_corruptions is None else True,
            use_two_crop_transform=args.use_two_view_transform,
            eval_protocol="imagenet",
        )

        print("DEBUG: transforms")
        print(train_transform)
        print(test_transform)

        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        new_dset_root = dset_rootpath
        print("class order debug:", fixed_class_order)
        scenario = SplitArrowScenario(
            new_dset_root,
            n_experiences=n_experiences, 
            return_task_id=True, # args.task_incr,
            class_ids_from_zero_in_each_exp=args.class_from_zero_in_each_exp, 
            seed=seed, 
            #per_exp_classes=args.per_exp_classes_dict,
            fixed_class_order=fixed_class_order, 
            train_transform=altered_train_transform, 
            test_transform=test_transform 
        )
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes  
        if args.class_from_zero_in_each_exp:
            initial_out_features = n_classes // n_experiences # For Multi-Head
        
    elif scenario_name == 'imgnet1k_256x256':
        n_classes = 1000
        n_experiences = 1

        fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            print("Using random class ordering...")
            #fixed_class_order = None
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
            use_data_aug,
            input_size=input_size,
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225),
            use_corruption=False if args.use_corruptions is None else True,
            use_two_crop_transform=args.use_two_view_transform,
            eval_protocol="imagenet",
        )

        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy
  
        new_dset_root = dset_rootpath

        scenario = SplitImageNet1k(
            new_dset_root,
            n_experiences=n_experiences, 
            return_task_id=True, # args.task_incr,
            class_ids_from_zero_in_each_exp=args.class_from_zero_in_each_exp, 
            seed=seed, 
            per_exp_classes=args.per_exp_classes_dict,
            fixed_class_order=fixed_class_order, 
            train_transform=altered_train_transform, 
            test_transform=test_transform 
        )
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes  
        if args.class_from_zero_in_each_exp:
            initial_out_features = n_classes // n_experiences # For Multi-Head
    

    elif scenario_name == 'split_dataset':
        from avalanche.benchmarks import nc_benchmark
        assert num_experiences is not None
        n_experiences = num_experiences
        
        assert args.scenario_datasets is not None
        
        # Load (base) dataset
        dataset_name = args.scenario_datasets[0]
        rootpath = dset_rootpath

        train_set, test_set, n_classes = get_dataset(
            scenario_name=dataset_name,
            dset_rootpath=rootpath,
            train_transform=None,
            eval_transform=None,
        )

        # Construct transforms
        train_transform, test_transform = get_transforms(
            use_data_aug,
            input_size=input_size,
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True,

        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        # Generate the benchmark
        scenario = nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=args.class_from_zero_in_each_exp, # default: False
            train_transform=train_transform,
            eval_transform=test_transform
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes

    elif scenario_name == 'repeat_dataset':
        assert num_experiences is not None
        n_experiences = num_experiences
        
        assert args.scenario_datasets is not None
        
        # Load (base) dataset
        dataset_name = args.scenario_datasets[0]
        rootpath = dset_rootpath

        train_set, test_set, n_classes = get_dataset(
            scenario_name=dataset_name,
            dset_rootpath=rootpath,
            train_transform=None,
            eval_transform=None,
        )

        # Construct transforms
        train_transform, test_transform = get_transforms(
            use_data_aug, 
            input_size=input_size, 
            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225), # (0.2675, 0.2565, 0.2761)
            use_two_crop_transform=args.use_two_view_transform,  #'BarlowTwins' in args.strategy,
            use_corruption=False if args.use_corruptions is None else True,

        )
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy

        # Create the benchmark
        scenario = ni_repeat_dataset_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=num_experiences,
            percentage_exp_assignment=args.data_percentage_per_exp,
            task_labels = True,
            shuffle = True,
            seed = None,
            train_transform = train_transform,
            eval_transform = test_transform
        )
        scenario.n_classes = n_classes
        initial_out_features = n_classes
    else:
        raise ValueError("Unknown scenario name.")


    # Get initial_out_features
    initial_out_features = scenario.n_classes
    if args.task_incr and not args.domain_incr:
        initial_out_features = n_classes // n_experiences

    # Cutoff if applicable
    if args.partial_num_tasks is not None:
        scenario.train_stream = scenario.train_stream[: args.partial_num_tasks]
        scenario.test_stream = scenario.test_stream[: args.partial_num_tasks]

    # Pack transforms #NOTE: this is necessary because I am unable to retrieve the transforms from the scenario object (or avalanche dataset)
    data_transforms["train"] = train_transform
    data_transforms["eval"] = test_transform
    data_transforms["input_size"] = input_size
    norm_mean_used, norm_std_used = _get_norm_from_transform(test_transform)
    data_transforms["norm_mean"] = norm_mean_used
    data_transforms["norm_std"] = norm_std_used
    
    print(f"Scenario = {scenario_name}")

    return scenario, data_transforms, input_size, initial_out_features



def get_continual_evaluation_plugins(args, scenario):
    """Plugins for per-iteration evaluation in Avalanche."""
    assert args.eval_periodicity >= 1, "Need positive "

    if args.eval_with_test_data:
        args.evalstream_during_training = scenario.test_stream  # Entire test stream
    else:
        args.evalstream_during_training = scenario.train_stream  # Entire train stream
    print(f"Evaluating on stream (eval={args.eval_with_test_data}): {args.evalstream_during_training}")

    # Metrics
    loss_tracking = TaskTrackingLossPluginMetric()
    
    # Expensive metrics
    gradnorm_tracking = None
    # if args.track_gradnorm:
    #     gradnorm_tracking = TaskTrackingGradnormPluginMetric() # if args.track_gradnorm else None  # Memory+compute expensive
    # featdrift_tracking = None
    # if args.track_featdrift:
    #     featdrift_tracking = TaskTrackingFeatureDriftPluginMetric() # if args.track_features else None  # Memory expensive

    # Acc derived plugins
    acc_tracking = TaskTrackingAccuracyPluginMetric()
    #lca = TrackingLCAPluginMetric()

    acc_min = TaskTrackingMINAccuracyPluginMetric()
    acc_min_avg = TaskAveragingPluginMetric(acc_min)
    wc_acc_avg = WCACCPluginMetric(acc_min)

    # wforg_10 = WindowedForgettingPluginMetric(window_size=10)
    # wforg_10_avg = TaskAveragingPluginMetric(wforg_10)
    # wforg_100 = WindowedForgettingPluginMetric(window_size=100)
    # wforg_100_avg = TaskAveragingPluginMetric(wforg_100)

    # wplast_10 = WindowedPlasticityPluginMetric(window_size=10)
    # wplast_10_avg = TaskAveragingPluginMetric(wplast_10)
    # wplast_100 = WindowedPlasticityPluginMetric(window_size=100)
    # wplast_100_avg = TaskAveragingPluginMetric(wplast_100)

    tracking_plugins = [
        loss_tracking, gradnorm_tracking, acc_tracking,
        acc_min, acc_min_avg, wc_acc_avg,  # min-acc/worst-case accuracy
        # wforg_10, wforg_10_avg,  # Per-task metric, than avging metric
        # wforg_100, wforg_100_avg,  # Per-task metric, than avging metric
        # wplast_10, wplast_10_avg,  # Per-task metric, than avging metric
        # wplast_100, wplast_100_avg,  # Per-task metric, than avging metric
    ]
    tracking_plugins = [p for p in tracking_plugins if p is not None]

    trackerphase_plugin = ContinualEvaluationPhasePlugin(
        tracking_plugins=tracking_plugins,
        max_task_subset_size=args.eval_task_subset_size,
        eval_stream=args.evalstream_during_training,
        eval_max_iterations=args.eval_max_iterations,
        mb_update_freq=args.eval_periodicity,
        num_workers=args.num_workers,
    )
    return [trackerphase_plugin, *tracking_plugins]



def get_metrics(scenario, args, data_transforms):
    """Metrics are calculated efficiently as running avgs."""

    # Pass plugins, but stat_collector must be called first
    minibatch_tracker_plugins = []

    # Stats on external tracking stream
    if args.enable_continual_eval:
        print("Enabling continual eval plugins...")
        tracking_plugins = get_continual_evaluation_plugins(args, scenario)
        minibatch_tracker_plugins.extend(tracking_plugins)

    metrics = [
        accuracy_metrics(minibatch=False, experience=True, stream=True),  
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=False),

        # CONTINUAL EVAL
        *minibatch_tracker_plugins,
    ]

    # LinearProbing
    if args.use_lp_eval is not None:
        assert isinstance(args.use_lp_eval, list), "use_lp_eval must be list"
        for lp_metric in args.use_lp_eval:
            if "linear" == lp_metric:
                metrics.append(
                    LinearProbingAccuracyMetric(
                        train_stream=scenario.train_stream, 
                        test_stream=scenario.test_stream,
                        eval_all=True, 
                        criterion=torch.nn.CrossEntropyLoss(),
                        train_mb_size=args.bs_eval,
                        num_finetune_epochs=args.lp_finetune_epochs,
                        num_finetune_itrs=args.lp_finetune_itrs,
                        num_workers=args.num_workers,
                        buffer_lp_dataset=True,
                        normalize_features=False,
                        record_loss=args.lp_record_loss,
                        eval_train_stream=args.lp_eval_train_stream
                    )
                )
            if "linear_force_mh" == lp_metric:
                metrics.append(
                    LinearProbingAccuracyMetric(
                        train_stream=scenario.train_stream, 
                        test_stream=scenario.test_stream,
                        eval_all=True, 
                        criterion=torch.nn.CrossEntropyLoss(),
                        train_mb_size=args.bs_eval,
                        num_finetune_epochs=args.lp_finetune_epochs,
                        num_finetune_itrs=args.lp_finetune_itrs,
                        num_workers=args.num_workers,
                        buffer_lp_dataset=True,
                        normalize_features=False,
                        force_mh_eval=True,
                        record_loss=args.lp_record_loss,
                        eval_train_stream=args.lp_eval_train_stream
                    )
                )
            if "concat_linear" == lp_metric:
                metrics.append(
                    ConcatenatedLinearProbingAccuracyMetric(
                        train_stream=scenario.train_stream, 
                        test_stream=scenario.test_stream,
                        eval_all=True, 
                        criterion=torch.nn.CrossEntropyLoss(),
                        train_mb_size=args.bs_eval,
                        num_finetune_epochs=args.lp_finetune_epochs,
                        num_workers=args.num_workers,
                        buffer_lp_dataset=True,
                        normalize_features=False,
                        record_loss=args.lp_record_loss,
                        eval_train_stream=args.lp_eval_train_stream
                    )
                )
            if "concat_linear_force_mh" == lp_metric:
                metrics.append(
                    ConcatenatedLinearProbingAccuracyMetric(
                        train_stream=scenario.train_stream, 
                        test_stream=scenario.test_stream,
                        eval_all=True, 
                        force_mh_eval=True,
                        criterion=torch.nn.CrossEntropyLoss(),
                        train_mb_size=args.bs_eval,
                        num_finetune_epochs=args.lp_finetune_epochs,
                        num_workers=args.num_workers,
                        buffer_lp_dataset=True,
                        normalize_features=False,
                        record_loss=args.lp_record_loss,
                        eval_train_stream=args.lp_eval_train_stream
                    )
                )
            if "knn" == lp_metric:
                print("\nUsing KNN Probing")
                metrics.append(
                    KNNProbingAccuracyMetric(
                        train_stream=scenario.train_stream, 
                        test_stream=scenario.test_stream,
                        k=args.knn_k,
                        eval_all=True,
                        force_task_eval=True,
                        train_mb_size=args.bs_eval,
                        num_workers=args.num_workers,
                    )
                )
            

    # Down Stream Evaluation (Full Sets)
    if args.downstream_sets is not None and args.downstream_method is not None:
        assert len(args.downstream_sets) == len(args.downstream_rootpaths), "Must specify a rootpath for each downstream set"

        for task_id in range(len(args.downstream_sets)):
            # Get the respective dataset
            print("\nPrepare dataset for downstream task")
            print("args.downstream_sets[task_id]", args.downstream_sets[task_id])
            print("args.downstream_rootpaths[task_id]", args.downstream_rootpaths[task_id])

            # Build a downstream-specific eval transform using the backbone's norm
            # stats, but with an eval protocol driven by the downstream dataset itself.
            downstream_eval_protocol = "imagenet" if "imgnet" in args.downstream_sets[task_id] else "default"
            _, downstream_eval_transform = get_transforms(
                data_aug=[],
                input_size=data_transforms["input_size"],
                norm_mean=data_transforms["norm_mean"] or (0.0, 0.0, 0.0),
                norm_std=data_transforms["norm_std"] or (1.0, 1.0, 1.0),
                eval_protocol=downstream_eval_protocol,
            )

            train_set, test_set, n_classes = get_dataset(
                scenario_name=args.downstream_sets[task_id],
                dset_rootpath=args.downstream_rootpaths[task_id],
                train_transform=downstream_eval_transform,
                eval_transform=downstream_eval_transform,
            )

            if "linear" in args.downstream_method:
                if args.ds_corruptions is None:
                    metrics.append(DownstreamLinearProbeAccuracyMetric(
                            args=args, 
                            criterion=torch.nn.CrossEntropyLoss(),
                            downstream_task=args.downstream_sets[task_id],
                            train_set=train_set, 
                            eval_set=test_set,
                            train_mb_size=args.bs,
                            eval_mb_size=args.bs_eval,
                            n_classes = n_classes,
                            buffer_lp_dataset=args.lp_buffer_dataset,
                        )
                    )
                else:
                    print("DEBUG: helper.py: Using CorruptedDownstreamLinearProbeAccuracyMetric")
                    metrics.append(CorruptedDownstreamLinearProbeAccuracyMetric(
                            corruptions_set=args.ds_corruptions,
                            severities=args.ds_corruption_severities,
                            args=args, 
                            criterion=torch.nn.CrossEntropyLoss(),
                            downstream_task=args.downstream_sets[task_id],
                            train_set=train_set, 
                            eval_set=test_set,
                            train_mb_size=args.bs,
                            eval_mb_size=args.bs_eval,
                            n_classes = n_classes,
                            buffer_lp_dataset=args.lp_buffer_dataset,
                        )
                    )
                print("Added linear downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_linear" in args.downstream_method:
                metrics.append(
                    ConcatDownstreamProbingAccuracyMetric(
                        args=args, 
                        train_set=train_set, 
                        eval_set=test_set,
                        eval_mb_size=args.bs_eval,
                        downstream_task_name=args.downstream_sets[task_id],
                        criterion=torch.nn.CrossEntropyLoss(),
                        n_classes = n_classes,
                    )
                )
                print("Added concat-linear downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_linear_pca_reduce" in args.downstream_method:
                metrics.append(
                    ConcatDownstreamProbingAccuracyMetric(
                        args=args, 
                        train_set=train_set, 
                        eval_set=test_set,
                        eval_mb_size=args.bs_eval,
                        downstream_task_name=args.downstream_sets[task_id],
                        criterion=torch.nn.CrossEntropyLoss(),
                        n_classes = n_classes,
                        reduce_dim_in_head='pca',
                        target_dim=args.reduce_target_dim  # TODO: have this automatic?
                    )
                )
                print(f"Added concat-linear downstream (with PCA dim reduction to {args.reduce_target_dim}) evaluation for", 
                      args.downstream_sets[task_id])
            
            if "concat_linear_gaussian_reduce" in args.downstream_method:
                metrics.append(
                    ConcatDownstreamProbingAccuracyMetric(
                        args=args, 
                        train_set=train_set, 
                        eval_set=test_set,
                        eval_mb_size=args.bs_eval,
                        downstream_task_name=args.downstream_sets[task_id],
                        criterion=torch.nn.CrossEntropyLoss(),
                        n_classes = n_classes,
                        reduce_dim_in_head='gaussian',
                        target_dim=args.reduce_target_dim  # TODO: have this automatic?
                    )
                )
                print(f"Added concat-linear downstream (with Gaussian dim reduction to {args.reduce_target_dim}) evaluation for", 
                      args.downstream_sets[task_id])

            if "knn" in args.downstream_method:
                metrics.append(
                    DownstreamKNNAccuracyMetric(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                        k=args.knn_k,
                    )
                )
                print("Added KNN downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_knn" in args.downstream_method:
                metrics.append(
                    DownstreamConcatenatedKNNAccuracyMetric(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                        k=args.knn_k
                    )
                )
                print("Added Concatenated KNN downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_knn_gaussian_reduce" in args.downstream_method:
                metrics.append(
                    DownstreamConcatenatedKNNAccuracyMetric(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                        k=args.knn_k,
                        reduce_dim_method='gaussian',
                        target_dim=args.reduce_target_dim
                    )
                )
                print(f"Added concat-linear downstream (with Gaussian dim reduction to {args.reduce_target_dim}) evaluation for", 
                      args.downstream_sets[task_id])

            
            if "pca" in args.downstream_method:
                from src.eval.pca_probing import DownstreamPCAProbing
                metrics.append(
                    DownstreamPCAProbing(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                    )
                )
                print("Added PCA downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_pca" in args.downstream_method:
                from src.eval.downstream_concatenated_pca_probing import DownstreamConcatPCAProbing
                metrics.append(
                    DownstreamConcatPCAProbing(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                    )
                )
                print("Added Concatenated PCA downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_pca_gaussian_reduce" in args.downstream_method:
                from src.eval.downstream_concatenated_pca_probing import DownstreamConcatPCAProbing
                metrics.append(
                    DownstreamConcatPCAProbing(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                        reduce_dim_method='gaussian',
                        target_dim=args.reduce_target_dim
                    )
                )
                print("Added Concatenated PCA downstream evaluation for", args.downstream_sets[task_id])
            
            if "concat_pca_pca_reduce" in args.downstream_method:
                from src.eval.downstream_concatenated_pca_probing import DownstreamConcatPCAProbing
                metrics.append(
                    DownstreamConcatPCAProbing(
                        args=args, 
                        downstream_task=args.downstream_sets[task_id], 
                        train_set=train_set,
                        eval_set=test_set,
                        reduce_dim_method='pca',
                        target_dim=args.reduce_target_dim
                    )
                )
                print("Added Concatenated PCA downstream evaluation for", args.downstream_sets[task_id])

    # Finally
    print("Plugins added...\n")
    return metrics


def get_criterion(criterion_name):
    """Get the criterion based on the name."""
    if criterion_name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif criterion_name == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    elif criterion_name == 'mse':
        return torch.nn.MSELoss()
    elif criterion_name == 'clip':
        from src.models.huggingface_wrapper import avl_clip_loss
        return avl_clip_loss
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def get_optimizer(
    optim_name, 
    model, 
    lr, 
    weight_decay=0.0, 
    betas=(0.9, 0.999), 
    momentum=0.9,
    head_lr=None,
):
    weight_decay = float(weight_decay)
    if weight_decay: # weight decay is > 0
        if head_lr is not None:
            params = [{"params": model.feature_extractor.parameters(), "lr": lr},
                      {"params": model.train_classifier.parameters(), "lr": head_lr}]
        else:
            # If there is weight decay we need to be protective of bias and norm params
            #params = get_param_groups(model, lr=lr, weight_decay=weight_decay)
            params = model.parameters()
    else: # weight decay is 0
        if head_lr is not None:
            params = [{"params": model.feature_extractor.parameters(), "lr": lr},
                      {"params": model.train_classifier.parameters(), "lr": head_lr}]
        else:
            params = [{"params": model.parameters(), "lr": lr}]
            #params = get_param_groups(model, lr=lr, weight_decay=weight_decay)
        
        
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    elif optim_name == 'lars':
        from src.utils.lars_optim import LARS
        optimizer = LARS(params, lr=lr, weight_decay=weight_decay,
                    weight_decay_filter=True, 
                    lars_adaptation_filter=True
        )
        print("\nUsing LARS Optim with weight decay filter and lars adaptation filter\n")
    
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=betas  # might be overwritten by params but good to have default values 
        )
    else:
        print("No optimizer found for name", optim_name)
        raise ValueError()
    return optimizer
    

def get_strategy(
    args, 
    model, 
    eval_plugin,
    scenario,
    device, 
    plugins=None,
    data_transforms=None
):
    plugins = [] if plugins is None else plugins

    # CRITERION
    criterion = torch.nn.CrossEntropyLoss()

    # OPTIMIZER
    optimizer = get_optimizer(
        args.optim, 
        model, 
        args.lr[0], 
        weight_decay=args.weight_decay, 
        momentum=args.momentum,
        head_lr=args.head_lr
    )
    initial_epochs = args.epochs[0]
    
    # STRATEGY
    print("Eval every:", args.eval_every_during_training)
    if args.strategy == 'finetune':
        strategy = Naive(
            model, 
            optimizer, 
            criterion,
            train_epochs=initial_epochs, 
            device=device,
            train_mb_size=args.bs, 
            eval_mb_size=args.bs_eval,
            eval_every=-1 if args.eval_every_during_training is None else args.eval_every_during_training,
            peval_mode="iteration",
            evaluator=eval_plugin,
            plugins=plugins
        )
    
    elif args.strategy == 'joint':
        strategy = Cumulative(
            model, 
            optimizer, 
            criterion, 
            train_mb_size=args.bs,
            train_epochs=initial_epochs, 
            eval_mb_size=args.bs_eval, 
            eval_every=-1, 
            evaluator=eval_plugin, 
            device=device,
            plugins=plugins
        )

    else:
        raise NotImplementedError(f"Non existing strategy arg: {args.strategy}")
    

    #################
    # Use AMP
    #################
    if args.use_amp:
        from src.methods.amp import AMPPlugin
        print("\nUsing AMP")
        strategy.plugins.append(
            AMPPlugin()
        )

    #################
    # Use Accelerator
    #################
    if args.use_accelerator:
        from src.methods.hf_accelerator import HFAcceleratorPlugin
        print("\nUsing Accelerator")
        strategy.plugins.append(
            HFAcceleratorPlugin(strategy=strategy)
        )


    #################
    # Training iterations controller plugin
    #################
    # Set training iterations controller
    strategy.training_iteration_controller = TrainingIterationControllerPlugin(
        epochs=args.epochs,
        iterations=args.iterations_per_task,
    )
    strategy.plugins.append(
        strategy.training_iteration_controller
    )

    ################
    # LwF
    ################
    if args.lwf:
        lwf_plugin = LwFPlugin(
            alpha=args.lwf_alpha,
            temperature=2
        )
        strategy.plugins.append(
            lwf_plugin
        )
        print("Using LwFPlugin!")

    ################
    # MAS
    ################
    if args.mas:
        mas_plugin = MASPlugin(
            lambda_reg=1.0,
            alpha=0.5,
            num_workers=args.num_workers
        )
        strategy.plugins.append(
            mas_plugin
        )
        print("Using MASPlugin!")

    ################
    # Replay
    ################
    if args.use_replay:
        from src.methods.replay import get_storage_policy
        from src.methods.replay import ReplayPlugin as ERPlugin
        strategy.plugins.append(
            ERPlugin(
                mem_size=args.mem_size,
                batch_size=args.bs,
                batch_size_mem=None,
                task_balanced_dataloader=False,
                storage_policy=get_storage_policy(
                    args.replay_storage_policy, 
                    args.mem_size, 
                    scenario.n_classes, 
                    scenario.n_experiences
                )
            )
        )
        print("Using ERPlugin!")
    elif args.use_cumulative_replay:
        from src.methods.replay import CumulativeReplayPlugin, get_storage_policy
        strategy.plugins.append(
            CumulativeReplayPlugin(
                mem_size=args.mem_size,
                storage_policy=get_storage_policy(
                    args.replay_storage_policy, 
                    args.mem_size, 
                    scenario.n_classes, 
                    scenario.n_experiences
                )
            )
        )


    ################
    # Ensemble
    ################
    if args.use_ensemble:
        from src.methods.ensemble import EnsemblePlugin
        strategy.plugins.append(EnsemblePlugin())
        print("Using EnsemblePlugin!")


    #################
    # Additional Feature Distillation
    #################
    if args.add_feature_distill:
        strategy.plugins.append(
            FeatureDistillationPlugin(
                alpha=args.lwf_alpha, 
                mode="mse"
            )
        )


    #################
    # Model Freezing
    #################
    elif args.freeze_bn >= 0:
        print("Freezing BN on experience:", args.freeze_bn)
        strategy.plugins.append(
            BNFreezePlugin(
                experience_idx=args.freeze_bn
            )
        )
    elif args.freeze_model is not None:
        strategy.plugins.append(
            FreezeModelPlugin(
                mode=args.freeze_model,
                exp_to_freeze_on=args.freeze_model_exp,
            )
        )
                            

    #################
    # Gradient Clipping
    #################
    if not args.grad_clipping is None:
        print("Using Gradient Clipping")
        strategy.plugins.append(
            GradClipPlugin(
                clip_value=args.grad_clipping
            )
        )


    #################
    # Evaluate stored models
    #################
    if args.eval_stored_weights is not None:
        from src.methods.eval_stored_weights import EvalStoredWeightsPlugin
        print("Appending EvalStoredWeightsPlugin")
        strategy.plugins.append(
            EvalStoredWeightsPlugin(
                path_to_weights=args.eval_stored_weights,
                replace_key_dict=args.weights_replace_key_dict
            )
        )


    #################
    # Pretrain the model head
    #################
    if args.pretrain_head:
        from src.methods.head_pre_adaptation import HeadPreAdaptationPlugin
        strategy.plugins.append(
            HeadPreAdaptationPlugin(
                train_stream=scenario.train_stream,
                train_all=False,
                num_finetune_epochs=args.lp_finetune_epochs,
                num_finetune_itrs=args.lp_finetune_itrs,
                train_mb_size=args.bs, 
                num_workers=args.num_workers,
                skip_initial_eval=True, 
                buffer_lp_dataset=True,
            )
        )


    #################
    # Initial Eval Stage (to probe random network performance)
    #################
    print("args.skip_initial_eval:", args.skip_initial_eval)
    if not args.skip_initial_eval:
        print("Prepending an evaluation stage to training")
        strategy.plugins.append(
            InitialEvalStage(
                test_stream=scenario.test_stream,
                only_initial_eval=args.only_initial_eval
            )
        )

    #################
    # Model Storing to Disk
    #################
    if args.store_models:
        strategy.plugins.append(
            StoreModelsPlugin(
                model_name=args.backbone, 
                model_store_path=args.results_path,
                store_on_intermediate_epoch=args.store_models_every_epoch
            )
        )

    #################
    # Re-Initialize Model
    #################
    if args.reinit_model_exp >= 0:
        strategy.plugins.append(
            ReinitModelPlugin(
                reinit_after_exp=args.reinit_model_exp,
                reinit_deterministic=args.reinit_deterministic
            )
        )

    #################
    # Learning rate scheduling
    #################
    if not args.lr_scheduling == "none":
        # OneCycle LR scheduler
        if args.lr_scheduling == "onecycle_cos":
            strategy.plugins.append(
                OneCycleSchedulerPlugin(
                    max_lr=args.lr,
                    div_factor=args.one_cycle_div_factor,
                    final_lr=args.one_cycle_final_lr, 
                    pct_start=args.one_cycle_pct_start
                )
            )
        elif args.lr_scheduling == "inf_const_decay":
            from src.utils.inf_constant_decay import InfiniteConstantDecaySchedulerPlugin
            strategy.plugins.append(
                InfiniteConstantDecaySchedulerPlugin(
                    warmup_percent=args.lr_warmup_pct, 
                    decay_percent=args.lr_decay_pct, 
                    decay_type='sqrt'
                )
            )

    # Finally
    print(f"Running strategy:{strategy}")
    if hasattr(strategy, 'plugins'):
        print(f"with Plugins: {strategy.plugins}")
    return strategy

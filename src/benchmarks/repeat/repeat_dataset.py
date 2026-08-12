from pathlib import Path
from typing import Sequence, Optional, Union, Any


from src.benchmarks.repeat.repeat_dataset_scenario import ni_repeat_dataset_benchmark

from helper import get_dataset

def RepeatDatasetBenchmark(
        dataset_name,
        rootpath, 
        num_experiences,
        train_transform=None, 
        eval_transform=None, 
        seed=None
):
    # Load the dataset (with transforms)
    train_set, test_set, _ = get_dataset(
        scenario_name=dataset_name,
        dset_rootpath=rootpath,
        train_transform=train_transform,
        eval_transform=eval_transform,
        #subsample_classes=None
        eval_only=True
    )

    # Create the benchmark
    scenario = ni_repeat_dataset_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=num_experiences,
        task_labels = True,
        shuffle = True,
        seed = None,
        #train_transform = None,
        #eval_transform = None
    )

    return scenario
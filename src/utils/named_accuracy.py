from typing import List, Optional, Union, Dict
from collections import defaultdict

import torch
from torch import Tensor
from avalanche.evaluation.metrics.mean import Mean

from avalanche.evaluation import Metric
from avalanche.evaluation.metrics.accuracy import Accuracy


class NameAwareAccuracy(Metric[Dict[int, float]]):
    def __init__(self):
        self._mean_accuracy = defaultdict(Accuracy)

        self.active_name = None
        return
    
    def set_name(self, name):
        assert isinstance(name, str)
        self.active_name = name
        return
    
    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        name: str = None,
    ) -> None:
    
        if name is None:
            name = self.active_name
        assert name is not None, "Name must be given or set before calling update!"
        assert isinstance(name, str)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        self._mean_accuracy[name].update(predicted_y, true_y)
        return

    def result(self, name: Optional[str] = None) -> Dict[int, float]:
        """
        Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        task label is ignored if `self.split_by_task=False`.

        :param task_label: if None, return the entire dictionary of accuracies
            for each task. Otherwise return the dictionary
            `{task_label: accuracy}`.
        :return: A dict of running accuracies for each task label,
            where each value is a float value between 0 and 1.
        """
        assert name is None or isinstance(name, str)

        if name is None:
            return {k: v.result() for k, v in self._mean_accuracy.items()}
        else:
            return {name: self._mean_accuracy[name].result()}

    def reset(self, name: Optional[str] = None) -> None:
        """
        Resets the metric.
        task label is ignored if `self.split_by_task=False`.

        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert name is None or isinstance(name, str)
        if name is None:
            self._mean_accuracy = defaultdict(Accuracy)
        else:
            self._mean_accuracy[name].reset()
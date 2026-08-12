import numpy as np
import random
import copy
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from typing import TYPE_CHECKING

from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.models.utils import (
    freeze,
    unfreeze,
    freeze_batchnorm_layers, 
    unfreeze_batchnorm_layers, 
    set_batchnorm_layers_to_eval,
    unfreeze_norm_layers
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class FreezeModelPlugin(SupervisedPlugin):
    """
    """
    def __init__(self, 
                 mode,
                 exp_to_freeze_on,       
        ):
        super().__init__()

        self.mode = mode
        self.exp_to_freeze_on = exp_to_freeze_on
        return
    
    def before_training_exp(self, strategy: 'SupervisedTemplate', **kwargs):
        """
        """
        if strategy.clock.train_exp_counter >= self.exp_to_freeze_on:
            if self.mode == "backbone":
                freeze(strategy.model.feature_extractor)
                print("FreezeModelPlugin: Freezing backbone")
            elif self.mode == "backbone_but_norm":
                freeze(strategy.model.feature_extractor)
                print("FreezeModelPlugin: Freezing backbone")
                #unfreeze_batchnorm_layers(strategy.model.feature_extractor)
                unfreeze_norm_layers(strategy.model.feature_extractor)
                print("FreezeModelPlugin: Unfreezing norm layers in the backbone")
            elif self.mode == "head":
                freeze(strategy.model.train_classifier)
                print("FreezeModelPlugin: Freezing head")
            elif self.mode == "all":
                freeze(strategy.model)
                print("FreezeModelPlugin: Freezing all")
        return

import numpy as np
import random
import copy
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from typing import TYPE_CHECKING

from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.models.utils import freeze_batchnorm_layers, unfreeze_batchnorm_layers, set_batchnorm_layers_to_eval
from src.models.utils import freeze, unfreeze

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class BNFreezePlugin(SupervisedPlugin):
    """
    """
    def __init__(
            self, 
            experience_idx: int,
        ):
        super().__init__()
        self.experience_idx = experience_idx
        return
    
    def before_training_exp(self, strategy: 'SupervisedTemplate', **kwargs):
        """
        """
        print(type(strategy.clock.train_exp_counter), type(self.experience_idx))
        if strategy.clock.train_exp_counter >= self.experience_idx:
            print("\nFreezing BN layers in the backbone.\n")
            freeze_batchnorm_layers(strategy.model)
        
        return
    

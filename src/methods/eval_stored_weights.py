import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from src.models import load_weights

import os
import math


class EvalStoredWeightsPlugin(SupervisedPlugin):
    def __init__(
        self, 
        path_to_weights: str,
        replace_key_dict: str = None
    ):
        super().__init__()
        print("\nUsing EvalStoredWeightsPlugin!\n")
        print("DEBUG: path_to_weights:", path_to_weights)
        self.path_to_weights = path_to_weights
        if not self.path_to_weights.endswith('/'):
            self.path_to_weights += '/'
            print("Adjusted path_to_weights to:", self.path_to_weights)
        print("DEBUG: path_to_weights:", self.path_to_weights)

        self.model_weights = self._discover_weight_files(self.path_to_weights)
        print("EvalStoredWeightsPlugin: Weights found:\n", self.model_weights)

        self.orig_train_epochs = None

        self.replace_key_dict = None
        if replace_key_dict is not None:
            if replace_key_dict == "cassle":
                self.replace_key_dict = {
                    "encoder": "feature_extractor",
                }
            elif replace_key_dict == "cassle_ijepa":
                # For I-JEPA checkpoints loaded into IJEPAViTWrapper:
                # checkpoint key 'encoder.<k>' must map to
                # 'feature_extractor.encoder.<k>' (the wrapper holds the
                # VisionTransformer under the 'encoder' attribute).
                # Using 'encoder.' (with dot) keeps the replacement prefix-safe.
                self.replace_key_dict = {
                    "encoder.": "feature_extractor.encoder.",
                }
        return
    
    def _discover_weight_files(self, path):
        """
        Return list of all .pth files in the path directory
        """
        # Discover weight files
        weight_files = [f for f in os.listdir(path) if f.endswith('.pth') or f.endswith('.ckpt')]
        # Stort them by task
        weight_files = sorted(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return weight_files
    
    def _load_state_dict(self, weight_file):
        ckpt = torch.load(weight_file, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            # Lightning .ckpt — strip the "model." prefix if present
            return {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
        return ckpt

    def before_training_exp(self, strategy, **kwargs):
        if self.orig_train_epochs is None:
            self.orig_train_epochs = strategy.train_epochs
            print("Original train epochs set to:", self.orig_train_epochs)

        # Load weights to model according to current experience
        weight_file = self.path_to_weights + self.model_weights[strategy.clock.train_exp_counter]
        try:
            # DEBUG: check the dtypes of the weights being loaded
            # tmp_state_dict = torch.load(weight_file)
            # print({k: v.dtype for k, v in list(tmp_state_dict.items())[:5]})
            # import sys; sys.exit(0)

            if hasattr(strategy.model, "_orig_mod"):
                load_weights(model=strategy.model._orig_mod, 
                    state_dict=self._load_state_dict(weight_file),
                    replace_key_dict=self.replace_key_dict
                )
            else:
                load_weights(model=strategy.model, 
                    state_dict=self._load_state_dict(weight_file),
                    replace_key_dict=self.replace_key_dict
                )
           

            print("Loaded weights from: ", weight_file)
        except Exception as e:
            print("COULD NOT LOAD WEIGHTS from: ", weight_file)
            print(e)
            raise ValueError("COULD NOT LOAD WEIGHTS from: ", weight_file)
        # Skip training by setting train_epochs to 0
        strategy.train_epochs = 0
        return 
    

    def after_training_exp(self, strategy, **kwargs):
        if hasattr(strategy, "curr_max_iterations"):
            print("DEBUG: using curr_max_iterations:", strategy.curr_max_iterations)
            strategy.clock.train_iterations += strategy.curr_max_iterations
        else:
            strategy.clock.train_iterations += self.orig_train_epochs * \
                math.ceil(len(strategy.adapted_dataset)/strategy.train_mb_size)
        return 


    
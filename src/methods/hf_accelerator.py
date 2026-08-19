import types

import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from src.auxilliary.supervised_templated_accelerator import SupervisedTemplateAccelerator


class HFAcceleratorPlugin(SupervisedPlugin):
    def __init__(self, strategy):
        super().__init__()
        self.is_initialized = False

        self.wrap(strategy)
        return

    def wrap(self, strategy):
        """
        Switch the __base__ or __class__ of the strategy to use AMP.
        """
        # Check that initialization only happens once
        if self.is_initialized:
            return
        self.is_initialized = True
        print("DEBUG: HFAcceleratorPlugin before_training called!")
        
        # Replace the strategy class with the Accelerator version 
        try:
            # NOTE: This is ugly and dangeous, but it maintains inhenitance structure
            bases = tuple(
                SupervisedTemplateAccelerator if base is SupervisedTemplate \
                    else base for base in strategy.__class__.__bases__
                )
            strategy.__class__.__bases__ = bases
            print("DEBUG: switched __bases__")
        except Exception as e:
            if isinstance(strategy, SupervisedTemplate):
                strategy.__class__ = SupervisedTemplateAccelerator
                print("DEBUG: switched __class__")
                print("Warning: could not switch __bases__, switched __class__ instead!",
                      "This can break inheritance structure!")
        # Verify
        assert isinstance(strategy, SupervisedTemplateAccelerator), \
            "Strategy is not an instance of SupervisedTemplateAccelerator after switching class!"

        
        # Accelerator wrapped forward (NOTE: should not be needed..)
        # from avalanche.models.utils import FeatureExtractorModel
        # def amp_forward(self, *args, **kwargs):
        #     with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #         return self.orig_forward(*args, **kwargs)
            
        # FeatureExtractorModel.orig_forward = FeatureExtractorModel.forward
        # FeatureExtractorModel.forward = amp_forward
        # print("DEBUG: wrapped (backbone) model.forward with amp_forward")

        # Finally, initialize the Accelerator
        strategy.init_accelerator() 
        print("\n---Switched strategy to use Accelerator!---\n")
        return
    

    def before_training_exp(self, strategy, **kwargs):
        print("DEBUG: HFAcceleratorPlugin before_training_exp called!")
        print(f"Proc {strategy.accelerator.process_index} - Model device: {next(strategy.model.parameters()).device}")
        print(type(strategy.model))
        # strategy.model, strategy.optimizer, strategy.dataloader = \
        #     strategy.accelerator.prepare(
        #         strategy.model, strategy.optimizer, strategy.dataloader
        #     )
        # Accelearte model
        strategy.model.feature_extractor = strategy.accelerator.prepare(
            strategy.model.feature_extractor
        )
        print("feature_extractor done!")
        strategy.model.train_classifier = strategy.accelerator.prepare(
            strategy.model.train_classifier
        )
        print("train_classifier done!")
        try:
            strategy.model.projection_head = strategy.accelerator.prepare(
                strategy.model.projection_head
            )
            print("projection_head done!")
        except:
            pass
        
        # Accelerate optimizer
        strategy.optimizer = strategy.accelerator.prepare(
            strategy.optimizer
        )
        print("optimizer done!")
        # Accelerate dataloader
        strategy.dataloader = strategy.accelerator.prepare(
            strategy.dataloader
        )
        print("dataloader done!")
        return
    

    def after_training_exp(self, strategy, **kwargs):
        print("DEBUG: HFAcceleratorPlugin after_training_exp called!")
        # Unwrap the model components (for safety)
        # Wait for all processes
        strategy.accelerator.wait_for_everyone()
        strategy.model.feature_extractor = strategy.accelerator.unwrap_model(
            strategy.model.feature_extractor
        )
        strategy.model.train_classifier = strategy.accelerator.unwrap_model(
            strategy.model.train_classifier
        )
        try:
            strategy.model.projection_head = strategy.accelerator.unwrap_model(
                strategy.model.projection_head
            )
        except:
            pass

        strategy.accelerator.wait_for_everyone()
        return


    def before_eval_exp(self, strategy, **kwargs):
        print("DEBUG: HFAcceleratorPlugin before_eval_exp called!")
        # Prepare the dataloader
        # NOTE: avalanche reuses the self.dataloader for eval
        strategy.dataloader = strategy.accelerator.prepare(
            strategy.dataloader
        )
        print("dataloader done!")
        return

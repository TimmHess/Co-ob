import types

import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from src.auxilliary.supervised_templated_amp import SupervisedTemplateAMP


class AMPPlugin(SupervisedPlugin):  # NOTE: actually it is a self-supervised plugin...
    def __init__(self):
        super().__init__()
        self.is_initialized = False

        return

    def before_training(self, strategy, **kwargs):
        """
        Switch the __base__ or __class__ of the strategy to use AMP.
        """
        # Check that initialization only happens once
        if self.is_initialized:
            return
        
        self.is_initialized = True
        print("DEBUG: AMPPlugin before_training called!")
        try:
            # NOTE: This is ugly and dangeous, but it maintains inhenitance structure
            bases = tuple(
                SupervisedTemplateAMP if base is SupervisedTemplate \
                    else base for base in strategy.__class__.__bases__
                )
            strategy.__class__.__bases__ = bases
            print("DEBUG: switched __bases__")
        except Exception as e:
            
            if isinstance(strategy, SupervisedTemplate):
                strategy.__class__ = SupervisedTemplateAMP
                print("DEBUG: switched __class__")
                print("Warning: could not switch __bases__, switched __class__ instead!",
                      "This can break inheritance structure!")
        # Old version:
        # Switch class if strategy is instance of SupervisedTemplate
        # if isinstance(strategy, SupervisedTemplate):
        #     strategy.__class__ = SupervisedTemplateAMP
        #     print("DEBUG: switched __class__")
        # else:
        #     # Try to switch SupervisedTemplate in __bases__
        #     bases = tuple(
        #         SupervisedTemplateAMP if base is SupervisedTemplate \
        #             else base for base in strategy.__class__.__bases__
        #         )
        #     strategy.__class__.__bases__ = bases
        #     print("DEBUG: switched __bases__")
        assert isinstance(strategy, SupervisedTemplateAMP), \
            "Strategy is not an instance of SupervisedTemplateAMP after switching class!"

        
        # AMP wrapped forward
        from avalanche.models.utils import FeatureExtractorModel
        def amp_forward(self, *args, **kwargs):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                return self.orig_forward(*args, **kwargs)
            
        FeatureExtractorModel.orig_forward = FeatureExtractorModel.forward
        FeatureExtractorModel.forward = amp_forward
        #strategy.model.feature_extractor.orig_forward = strategy.model.feature_extractor.forward
        #strategy.model.feature_extractor.forward = types.MethodType(amp_forward, strategy.model.feature_extractor)
        print("DEBUG: wrapped (backbone) model.forward with amp_forward")

        # Finally, initialize the GradScaler
        strategy.init_scaler()
        print("\n---Switched strategy to use AMP!---\n")
        return

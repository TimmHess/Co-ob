import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class EMAPlugin(SupervisedPlugin):
    def __init__(self, alpha=0.999):
        super().__init__()
        self.alpha = alpha

    def _wrap_model(self, model):
        # Compute exponential moving averages of the weights and buffers
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), 
            use_buffers=True # Handle for BN
        )
        return ema_model
    
    def before_training(self, strategy):
        strategy.model.feature_extractor = self._wrap_model(strategy.model.feature_extractor)
        return
    
    def after_training_iteration(self, strategy, **kwargs):
        strategy.model.update_parameters(strategy.model.model)
        return 


    
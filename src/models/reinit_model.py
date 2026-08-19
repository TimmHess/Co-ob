
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.models.models import weight_reset

from copy import deepcopy


class ReinitModelPlugin(SupervisedPlugin):
    def __init__(
        self,
        reinit_after_exp,
        reinit_deterministic=False,
    ):
        super().__init__()

        self.reinit_after_exp = reinit_after_exp
        self.reinit_deterministic = reinit_deterministic

        self.reinit_model = None
        return


    def before_training_exp(self, strategy, **kwargs):
        if self.reinit_after_exp == strategy.clock.train_exp_counter:
            self.reinit_model = deepcopy(strategy.model)

        if self.reinit_model is not None:
            print(f"ReinitModelPlugin: Reinitializing model")
            if self.reinit_deterministic:  # NOTE: Reset by loading previous weights
                try:
                    strategy.model.load_state_dict(self.reinit_model.state_dict())  
                except Exception as e:
                    print(f"ReinitModelPlugin: Error loading state dict: {e}")
                    print("Instead loding in with non-strict mode")
                    strategy.model.load_state_dict(self.reinit_model.state_dict(), strict=False)
            else:  # NOTE: Reset by random weight re-init
                strategy.model.apply(weight_reset)  
        return
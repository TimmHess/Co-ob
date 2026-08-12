
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.models.utils import replace_BN_by_GN


class BN2GNModelAdapterPlugin(SupervisedPlugin):
    def __init__(self, num_groups=32):
        super().__init__()

        self.num_groups = num_groups
        return

    def before_training(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter == 0:
            replace_BN_by_GN(
                strategy.model,
                max_groups=self.num_groups
            )
            print(strategy.model)
            import sys; sys.exit()
        return
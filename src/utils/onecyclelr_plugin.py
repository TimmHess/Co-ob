from copy import deepcopy
from torch.optim.lr_scheduler import OneCycleLR
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.utils.util import safe_index

class OneCycleSchedulerPlugin(SupervisedPlugin):
    """
    Basically a wrapper for the OneCycleLR scheduler that is able to handle changes 
    to the optimizer's parameter_groups that happen during begin_training
    """
    def __init__(
        self, 
        max_lr,
        div_factor,
        final_lr, 
        pct_start,  
    ):
        super().__init__()
        self.max_lr = max_lr  # NOTE: list
        self.div_factor = div_factor  # NOTE: list
        self.final_lr = final_lr
        self.pct_start = pct_start  # NOTE: list
        
        self.scheduler = None
        self.init_params = {}
        return
    
    def before_training_exp(self, strategy, **kwargs):
        print("exp counter", strategy.clock.train_exp_counter)
        assert strategy.training_iteration_controller.epochs is not None, "Strategy needs to be equiped with training_iteration_controller!"
        
        exp_idx = strategy.clock.train_exp_counter
        total_steps = strategy.training_iteration_controller.get_curr_max_iterations()
        
        self.init_params = {
            'max_lr': safe_index(self.max_lr, exp_idx),
            'div_factor': safe_index(self.div_factor, exp_idx),
            'final_div_factor': safe_index(self.final_lr, exp_idx),
            'pct_start': safe_index(self.pct_start, exp_idx),
            'anneal_strategy': 'cos', 
            'total_steps': total_steps
        }

        self.scheduler = OneCycleLR(
            strategy.optimizer, 
            max_lr=safe_index(self.max_lr, exp_idx), 
            div_factor=safe_index(self.div_factor, exp_idx), 
            final_div_factor=safe_index(self.max_lr, exp_idx)/safe_index(self.final_lr, exp_idx),
            pct_start=safe_index(self.pct_start, exp_idx), 
            anneal_strategy='cos',  # NOTE: cosin annaling
            total_steps=total_steps,
        )
        return super().before_training_exp(strategy, **kwargs)

    def after_training_iteration(self, strategy, **kwargs):
        self.scheduler.step()
        #print(self.scheduler.get_last_lr())
        return super().after_training_iteration(strategy, **kwargs)
    
    def after_training_epoch(self, strategy, **kwargs):
        print(self.scheduler.get_last_lr())
        return
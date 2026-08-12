from copy import deepcopy
from torch.optim.lr_scheduler import OneCycleLR
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


# class DynamicLRSchedulingPlugin(SupervisedPlugin):
#     """
#     Basically a wrapper for the OneCycleLR scheduler that is able to handle changes 
#     to the optimizer's parameter_groups that happen during begin_training
#     """
#     def __init__(self, 
#                  args
#         ):
#         super().__init__()
#         self.args = args
#         return
    
#     def before_training_exp(self, strategy, **kwargs):
#         exp_counter = strategy.clock.train_exp_counter
#         scheduler = get_lr_scheduler(strategy.optimizer, self.args.lr_scheduling)
#         print("exp counter", strategy.clock.train_exp_counter)

#         # Get the total steps to do withing the scheduler
#         total_steps = strategy.training_iteration_controller.get_curr_max_iterations
       
#         self.scheduler = get_lr_scheduler(
#             optimizer=strategy.optimizer,
#             lr_scheduling=self.args.lr_scheduling,
#             max_lr=self.args.max_lr[exp_counter],
#             div_factor=self.args.div_factor,
#             final_lr=self.args.final_lr,
#             pct_start=self.args.pct_start,
#             max_iterations=strategy.training_iteration_controller.get_curr_max_iterations(),
#             )

#         # self.scheduler = OneCycleLR(
#         #             self.optimizer, 
#         #             max_lr=self.max_lr, 
#         #             div_factor=self.div_factor, # NOTE: initial_lr = max_lr/div_factor
#         #             final_div_factor=(self.max_lr/self.final_lr), # NOTE: final_lr = max_lr/final_div_factor
#         #             pct_start=self.pct_start, 
#         #             anneal_strategy='cos', # NOTE: uses cosin annaling
#         #             total_steps=total_steps,
#         #             three_phase=False,
#         #         )
#         return super().before_training_exp(strategy, **kwargs)

#     def after_training_iteration(self, strategy, **kwargs):
#         self.scheduler.step()
#         #print(self.scheduler.get_last_lr())
#         return super().after_training_iteration(strategy, **kwargs)
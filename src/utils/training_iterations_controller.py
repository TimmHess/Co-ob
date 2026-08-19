import torch

from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.utils.util import safe_index


class TrainingIterationControllerPlugin(SupervisedPlugin):
    """
    Plugin that manages the use of epochs or iterations for training.
    When added to a strategy, all training will be expressed in terms of iterations.
    :param epochs: List of integers, each representing the number of epochs to train on a specific experience.
    :param iterations: List of integers, each representing the number of iterations to train on a specific experience.
        Overwrites iterations when set. 
    """
    def __init__(self, epochs=None, iterations=None):
        super().__init__()
 
        self.epochs = epochs
        self.iterations = iterations

        self.curr_max_iteration = None
        return

    def before_training_exp(self, strategy, **kwargs):
        """
        Set the training iterations for the experience.
        """
        train_exp_counter = strategy.clock.train_exp_counter
        
        if self.iterations is not None:  # if iterations is defined we use iterations istead of epochs
            strategy.train_epochs = torch.iinfo(torch.int32).max  #NOTE: set to int32 max value
            print("\nUsing", safe_index(self.iterations, train_exp_counter), " iterations during experience", train_exp_counter, "\n")
            self.curr_max_iteration = safe_index(self.iterations, train_exp_counter)
        else:  # if only epochs are defined then we convert them to iterations confirming the current experience data size
            print("\nUsing", safe_index(self.epochs, train_exp_counter), " epochs during experience", train_exp_counter, "\n")
            self.curr_max_iteration = int(safe_index(self.epochs, train_exp_counter)*(len(strategy.adapted_dataset)//strategy.train_mb_size))
            strategy.train_epochs = torch.iinfo(torch.int32).max  #NOTE: set to int32 max value
            print("Converted to", self.curr_max_iteration, "iterations\n")
        
        # Set the current max iterations for the strategy
        strategy.curr_max_iterations = self.curr_max_iteration

        # Check if epochs or iterations are 0
        if self.curr_max_iteration == 0:
            # Increase the counter of train_iterations by an arbitrary number to preven results from overwriting one-another
            print("Adjusting the number of training iterations to prevent overwriting of results...")
            strategy.clock.train_iterations += 100
        return
    

    def before_training_epoch(self, strategy, **kwargs):
        """
        Set the training iterations for the experience.
        Redundant check, but useful when wanting to omit training alltogether.
        """
        if strategy.clock.train_exp_iterations == self.curr_max_iteration:
            print(f"\nStopping training, reached max iterations: {self.curr_max_iteration}")
            strategy.stop_training()
        return


    def after_training_iteration(self, strategy, **kwargs):
        """
        Check after every iteration whether the training is to be stopped.
        """
        if strategy.clock.train_exp_iterations == self.curr_max_iteration:
            print(f"\nStopping training, reached max iterations: {self.curr_max_iteration}")
            strategy.stop_training()
        return

    def get_curr_max_iterations(self):
        return self.curr_max_iteration
    


# class IterationsInsteadOfEpochs(BaseSGDPlugin):
#     """
#     Stop training based on number of iterations instead of epochs.
#     """

#     def __init__(self, max_iterations: int):
#         super().__init__()
#         self.max_iterations = max_iterations -1 # -1 because we start at 0

#     def before_training_exp(self, strategy: 'SupervisedTemplate', **kwargs):
#         if self.max_iterations == 0:
#             strategy.stop_training()
#         return super().before_training_exp(strategy, **kwargs)

#     def after_training_iteration(self, strategy, **kwargs):
#         if strategy.clock.train_exp_iterations == self.max_iterations:
#             print(f"Stopping training, reached max iterations: {self.max_iterations}")
#             strategy.stop_training()
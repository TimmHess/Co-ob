from copy import deepcopy

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class ExperienceFilterPlugin(SupervisedPlugin):
    def __init__(self, experiences):
        super().__init__()

        self.experiences = experiences
        return

    def before_training_iteration(self, strategy, **kwargs):
        """
        Skip of training needs to happen here for the methods to work correctly.
        """
        # (Re-)Set skip_eval flag on strategy
        strategy.skip_eval = False
        if strategy.clock.train_exp_counter in self.experiences:
            pass
        else:
            strategy.stop_training()
            # Set skip_eval flag on strategy
            strategy.skip_eval = True
            print("\n========= Skipping training for experience", strategy.clock.train_exp_counter, "=========\n")
            # TODO: Increase the clock by the repsective number of iterations (?)
            # strategy.clock.train_iterations += strategy.curr_max_iterations
        return

    # def before_eval(self, strategy, **kwargs):
    #     print("Calling before_eval for experience", strategy.clock.train_exp_counter)
    #     if strategy.clock.train_exp_counter in self.experiences:
    #        pass
    #     else:
    #         # Empty the eval stream for this training stage
    #         strategy.current_eval_stream = []
    #         print("\n =========Emptied the eval stream=========\n")
    #     return
    
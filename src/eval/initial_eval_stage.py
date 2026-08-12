import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from src.eval.continual_eval import ContinualEvaluationPhasePlugin



class InitialEvalStage(SupervisedPlugin):
    def __init__(self, test_stream, only_initial_eval=False):
        super().__init__()

        self.test_stream = test_stream
        self.only_initial_eval = only_initial_eval

        self._prev_state = None
        self._prev_training_modes = None
        return

    def before_training(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter == 0:
            print(f"\n{'=' * 40} INITIAL EVAL {'=' * 40}")
            print("\nDoing linear probing on model initialization")
            # Need to store the state of the trainer and model befor running eval inside train-loop
            self._prev_state, self._prev_training_modes = ContinualEvaluationPhasePlugin.get_strategy_state(strategy)
            # Trigger the evaluation by calling strategy.eval()? -> This will run entire evaluation... 
            print("Triggering initial evaluation before training starts...")
            strategy.eval(self.test_stream)
        return

    def after_eval(self, strategy):
        if self.only_initial_eval:
            print("\n\nExiting after initial evaluation")
            import sys; sys.exit(0)  # NOTE: terminate the run
        
        # Reset the state of the continual learner
        if strategy.clock.train_exp_counter == 0:
            #assert(not self._prev_state is None)
            #assert(not self._prev_training_modes is None)
            if self._prev_state is not None and self._prev_training_modes is not None:
                ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
                self.is_initial_eval_run = False  # Reset flag
            return

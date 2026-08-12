
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

import copy



ENSEMBLE_MODEL_BUFFER = "model_buffer"

def get_ensemble_model_buffer(strategy):
    assert hasattr(strategy, ENSEMBLE_MODEL_BUFFER), \
        f"Strategy does not have attribute {ENSEMBLE_MODEL_BUFFER}." \
        "Make sure to add the EnsemblePlugin to the strategy."
    return getattr(strategy, ENSEMBLE_MODEL_BUFFER, None)


class EnsemblePlugin(SupervisedPlugin):
    """Ensemble plugin.

    This plugin implements an ensemble learning strategy, where multiple
    models are trained and their predictions are combined to improve
    performance.
    """

    def __init__(self):
        super().__init__()


    def before_training(self, strategy, **kwargs):
        """
        Initialize the model buffer at the beginning of training.
        """
        if not hasattr(strategy, ENSEMBLE_MODEL_BUFFER):
            setattr(strategy, ENSEMBLE_MODEL_BUFFER, [])
        return
    

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        model_buffer = getattr(strategy, ENSEMBLE_MODEL_BUFFER)
        model_buffer.append(copy.deepcopy(strategy.model.feature_extractor)) # reference to strategy.model_buffer
        print("DEBUG: added model to ensemble buffer. Buffer size:", len(model_buffer))
        return
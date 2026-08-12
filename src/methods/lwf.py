import copy
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models import MultiTaskModule, avalanche_forward
from avalanche._annotations import deprecated

from avalanche.training import RegularizationMethod


class LearningWithoutForgetting(RegularizationMethod):
    """Learning Without Forgetting.

    The method applies knowledge distilllation to mitigate forgetting.
    The teacher is the model checkpoint after the last experience.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """
        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None
        self.expcount = 0
        # count number of experiences (used to increase alpha)
        self.prev_classes_by_task = defaultdict(set)
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding
        to old classes. 
        """

    def _distillation_loss(self, out, prev_out, active_units):
        """Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)

        # some people use the crossentropy instead of the KL
        # They are equivalent. We compute
        # kl_div(log_p_curr, p_prev) = p_prev * (log (p_prev / p_curr)) =
        #   p_prev * log(p_prev) - p_prev * log(p_curr).
        # Now, the first term is constant (we don't optimize the teacher),
        # so optimizing the crossentropy and kl-div are equivalent.
        log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

    def _lwf_penalty(self, out, x, curr_model):
        """
        Compute weighted distillation loss.
        """
        if self.prev_model is None:
            return 0
        else:
            if isinstance(self.prev_model.train_classifier, MultiTaskModule):
                # output from previous output heads.
                with torch.no_grad():
                    y_prev = avalanche_forward(self.prev_model, x, None)
                y_prev = {k: v for k, v in y_prev.items()}
                # in a multitask scenario we need to compute the output
                # from all the heads, so we need to call forward again.
                # TODO: can we avoid this?
                y_curr = avalanche_forward(curr_model, x, None)
                y_curr = {k: v for k, v in y_curr.items()}
            else:  # no task labels. Single task LwF
                with torch.no_grad():
                    y_prev = {0: avalanche_forward(self.prev_model, x, None)}
                    #y_prev = {0: self.prev_model(x)}
                y_curr = {0: out}

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads and only for seen units.
                if task_id in self.prev_classes_by_task:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    au = self.prev_classes_by_task[task_id]
                    dist_loss += self._distillation_loss(yc, yp, au)
            return dist_loss

    def __call__(self, mb_x, mb_pred, model):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[self.expcount]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        return alpha * self._lwf_penalty(mb_pred, mb_x, model)

    def post_adapt(self, agent, exp):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.

        :param agent: agent state
        :param exp: current experience
        """
        self.expcount += 1
        self.prev_model = copy.deepcopy(agent.model)

        task_ids = exp.dataset.targets_task_labels.uniques

        for task_id in task_ids:
            task_data = exp.dataset.task_set[task_id]
            pc = set(task_data.targets.uniques)

            if task_id not in self.prev_classes_by_task:
                self.prev_classes_by_task[task_id] = pc
            else:
                self.prev_classes_by_task[task_id] = self.prev_classes_by_task[
                    task_id
                ].union(pc)

    @deprecated(0.7, "switch to post_udpate method")
    def update(self, experience, model):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.

        :param experience: current experience
        :param model: current model
        """
        self.expcount += 1
        self.prev_model = copy.deepcopy(model)
        task_ids = experience.dataset.targets_task_labels.uniques

        for task_id in task_ids:
            task_data = experience.dataset.task_set[task_id]
            pc = set(task_data.targets.uniques)

            if task_id not in self.prev_classes_by_task:
                self.prev_classes_by_task[task_id] = pc
            else:
                self.prev_classes_by_task[task_id] = self.prev_classes_by_task[
                    task_id
                ].union(pc)




class LwFPlugin(SupervisedPlugin):
    """Learning without Forgetting plugin.

    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """
        super().__init__()
        self.lwf = LearningWithoutForgetting(alpha, temperature)

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        lwf_loss = self.lwf(strategy.mb_x, strategy.mb_output, strategy.model)
        #print("DEBUG: lwf_loss:", lwf_loss)
        strategy.loss += lwf_loss

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.lwf.post_adapt(strategy, strategy.experience)
import torch
from torch.amp import GradScaler

from avalanche.models import avalanche_forward
from avalanche.training.templates import SupervisedTemplate

from avalanche.models.utils import is_multi_task_module


class SupervisedTemplateAMP(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def init_scaler(self):
        """
        Initialize the gradient scaler for AMP.
        """
        self.scaler = GradScaler()
        return

    def forward(self):
        # if hasattr(self.experience, "task_labels"):
        #     return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        # else:
        #     return self.model(self.mb_x)

        # New version: # NOTE: hacky fix for issues with MultiTaskClassifier..        
        if is_multi_task_module(self.model):
            if is_multi_task_module(self.model.train_classifier):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    x = avalanche_forward(self.model.feature_extractor, self.mb_x, self.mb_task_id).float()
                    self.model.features = x.clone().detach()
                return avalanche_forward(self.model.train_classifier, x, self.mb_task_id)
            else:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        else:
            if is_multi_task_module(self.model.train_classifier):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    x = avalanche_forward(self.model.feature_extractor, self.mb_x, self.mb_task_id).float()
                    self.model.features = x.clone().detach()
                return avalanche_forward(self.model.train_classifier, x, self.mb_task_id)
            else:
                # Old version
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        pass

    def backward(self):
        # self.loss.backward(retain_graph=self.retain_graph)
        self.scaler.scale(self.loss).backward()
        self.scaler.unscale_(self.optimizer)
        pass

    def optimizer_step(self, **kwargs):
        # self.optimizer.step()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
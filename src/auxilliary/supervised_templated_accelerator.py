import torch
#from torch.amp import GradScaler
from accelerate import Accelerator

from avalanche.models import avalanche_forward
from avalanche.training.templates import SupervisedTemplate

from avalanche.models.utils import is_multi_task_module


class SupervisedTemplateAccelerator(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def init_accelerator(self):
        """
        Initialize the accelerator
        """
        #self.accelerator = Accelerator() #mixed_precision="fp16"
        self.accelerator = None
        print("DEBUG: Initialized Accelerator with mixed_precision='fp16'")
        return
    
    def set_accelerator(self, accelerator):
        print("Assigning accelerator to strategy...")
        self.accelerator = accelerator
        return
    

    def forward(self):
        # NOTE: hacky fix for issues with MultiTaskClassifier..        
        # if is_multi_task_module(self.model):
        #     if is_multi_task_module(self.model.train_classifier):
        #         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #             x = avalanche_forward(self.model.feature_extractor, self.mb_x, self.mb_task_id).float()
        #             self.model.features = x.clone().detach()
        #         return avalanche_forward(self.model.train_classifier, x, self.mb_task_id)
        #     else:
        #         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #             return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        # else:
        #     if is_multi_task_module(self.model.train_classifier):
        #         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #             x = avalanche_forward(self.model.feature_extractor, self.mb_x, self.mb_task_id).float()
        #             self.model.features = x.clone().detach()
        #         return avalanche_forward(self.model.train_classifier, x, self.mb_task_id)
        #     else:
        #         # Old version
        #         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #             return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        # pass
        pass

    def backward(self):
        # self.loss.backward(retain_graph=self.retain_graph)
        self.accelerator.backward(self.loss)
        pass

    # def optimizer_step(self, **kwargs):
    #     # self.optimizer.step()
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    
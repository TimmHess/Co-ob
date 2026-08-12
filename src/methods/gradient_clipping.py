from typing import TYPE_CHECKING, Optional, List
import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

class GradClipPlugin(SupervisedPlugin):
    def __init__(self, clip_value: float):
        super().__init__()

        self.clip_value = clip_value
        
        print("Warning: error_if_nonfinite=False in GradClipPlugin")
        return

    # DEBUG: print learning rates for all param groups in optimizer 
    # def before_training_iteration(self, strategy: 'SupervisedTemplate', **kwargs):
    #     """
    #     Called before the training iteration.
    #     """
        
    #     for i, param_group in enumerate(strategy.optimizer.param_groups):
    #         print(f"Parameter Group {i}: lr = {param_group['lr']}")
        
    #     return


    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Apply gradient clipping to all gradients of the model.
        """
        # print("DEBUG:GradClip after backward", strategy.loss.item())
        # for p in strategy.model.parameters():
        #     if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
        #         print("Non-finite gradient detected")
        #         import sys; sys.exit()
        #         break
        # print("DEBUG: all gradients passed")
        # import sys; sys.exit()

        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), max_norm=self.clip_value,
            norm_type=2, error_if_nonfinite=False) # True
        return
    

    # DEBUG: Visualize the corruption transforms
    # def before_eval_iteration(self, strategy, **kwargs):
    #     # Define in which experience to visualize the corruption transforms
    #     if not (strategy.clock.train_exp_counter in [0]):
    #         return
        
    #     import torchvision
    #     import torch

    #     # Access the current minibatch
    #     tensors = strategy.mbatch[0]
    #     labels = strategy.mbatch[1]
    #     task_id = strategy.mbatch[2]

    #     print("DEBUG: CorruptionPipelineHandlerPlugin: before_training_iteration called")
    #     print("unique labels:", torch.unique(labels), labels.shape)
    #     print("unique task ids:", torch.unique(task_id))
    #     print("labels", labels)
    #     print("task_id", task_id)

    #     # Visualize the first 64 images in the minibatch
    #     #self.discover_normalization(strategy=strategy)
    #     grid_img = torchvision.utils.make_grid(tensors[:64], nrow=10).permute(1, 2, 0)
    #     #std = torch.tensor([0.229, 0.224, 0.225]).to(strategy.device)
    #     #mean = torch.tensor([0.485, 0.456, 0.406]).to(strategy.device)
    #     std = torch.tensor((0.1307,)).to(strategy.device)
    #     mean = torch.tensor((0.3081,)).to(strategy.device)
    #     grid_img = grid_img * std + mean
    #     import cv2
    #     cv2.imwrite("corruptions.png", cv2.cvtColor(grid_img.to("cpu").numpy() * 255, cv2.COLOR_BGR2RGB))
    #     print("saving image done..\n\n")
    #     import sys; sys.exit()
    #     return 
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from src.utils.corruption.corruption_transforms import CorruptionPipeline

from src.utils.util import safe_index

from copy import deepcopy

import torch

def discover_corruption_pipeline(source_transform):
    """
    Discover the corruption pipeline from the source transform.
    :param source_transform: The source transform to discover the corruption pipeline from.
    :return: The discovered corruption pipeline.
    """
    for trf in source_transform:
        if isinstance(trf, CorruptionPipeline):
            return trf
    raise ValueError("Could not discover corruption transforms from train transforms!")


class CorruptionHandler():
    def __init__(
            self, 
            transform_group_name:str, 
            corruption_set:list = None,  # TODO: are these needed?
            severities:list = None  # TODO: are these needed?
        ):
        self.transform_group_name = transform_group_name
        self.corruption_set = corruption_set
        self.severities = severities

        self.corruption_pipeline = None
        return
    
    def discover_corruption_pipeline(self, source_transform):
        """
        :param source_transform: The source transform to discover the corruption pipeline from.
        """
        for trf in source_transform:
            if isinstance(trf, CorruptionPipeline):
                self.corruption_pipeline = trf
                break
        assert self.corruption_pipeline is not None, "Could not discover corruption transforms from train transforms!"
        return
    
    def set_corruption(self, name):
        print("DEBUG: CorruptionHandler: Setting corruption to", name)
        self.corruption_pipeline.set_corruption(name)
        return
    
    def set_severity(self, severity):
        print("DEBUG: CorruptionHandler: Setting severity to", severity)
        self.corruption_pipeline.set_severity(severity)
        return
    
    def get_corruptions(self):
        return self.corruption_set





class CorruptionPipelineHandlerPlugin(SupervisedPlugin):
    def __init__(self, corruption_set:list, severities:list):
        super().__init__()

        self.corruption_set = corruption_set
        self.corruption_severities = severities
        
        self.train_corruption_handler = CorruptionHandler("train")
        self.eval_corruption_handler = CorruptionHandler("eval")
        print("\nDEBUG:corruption_handler: CorruptionPipelineHandlerPlugin initialized")
        return


    def discover_transform(self, experience):
        transform = None
        try:
            transform = experience.dataset.get_transforms(
                experience.dataset._flat_data
            )
            transform = transform["train"].transforms[0].transforms
        except:
            print("CorruptionPipelineHandlerPlugin: Could not get transform from experience... trying again")
            try: 
                transform = experience.dataset._datasets[0].\
                    _transform_groups["train"].transforms[0].transforms
            except:
                transform = experience.dataset._flat_data.\
                    _transform_groups["train"].transforms[0].transforms
        return transform
    

    # def before_training(self, strategy, **kwargs):
    #     # NOTE: This is needed purly for eval_ckpt_representations script
    #     try: 
    #         transform = strategy.experience.dataset.get_transforms(
    #             strategy.experience.dataset._flat_data
    #         )
    #         transform = transform["train"].transforms[0].transforms
            
    #         self.train_corruption_handler.discover_corruption_pipeline(transform)
    #     except:
    #         pass
    #     return


    def before_training_exp(self, strategy, **kwargs):
        # Get access to the corruption pipeline
        # print(type(strategy.experience.dataset._flat_data._transform_groups))
        # print(type(strategy.experience.dataset._flat_data._datasets[0]._transform_groups))
        # print(type(strategy.experience.dataset._flat_data._datasets[0]._datasets[0]._transform_groups))
        #print(type(strategy.experience.dataset._flat_data._datasets[0]._datasets[0]._transform_groups["train"]))
        try:
            transform = strategy.experience.dataset.get_transforms(
                strategy.experience.dataset._flat_data
            )
            transform = transform["train"].transforms[0].transforms
        except:
            print("CorruptionPipelineHandlerPlugin: Could not get transform from experience... trying again")
            try: 
                transform = strategy.experience.dataset._datasets[0].\
                    _transform_groups["train"].transforms[0].transforms
            except:
                transform = strategy.experience.dataset._flat_data.\
                    _transform_groups["train"].transforms[0].transforms
                
        self.train_corruption_handler.discover_corruption_pipeline(transform)
        # Set corruption for experience
        self.train_corruption_handler.set_corruption(safe_index(self.corruption_set, strategy.clock.train_exp_counter))
        self.train_corruption_handler.set_severity(safe_index(self.corruption_severities, strategy.clock.train_exp_counter))
        return super().before_training_exp(strategy, **kwargs)

    
    # DEBUG: Print training data statistics
    # def before_training_iteration(self, strategy, **kwargs):
    #     tensors = strategy.mbatch[0]
    #     labels = strategy.mbatch[1]
    #     task_id = strategy.mbatch[2]

    #     print("DEBUG: CorruptionPipelineHandlerPlugin: before_training_iteration called")
    #     print("unique labels:", torch.unique(labels))
    #     print("unique task ids:", torch.unique(task_id))
        
    #     #import sys; sys.exit()
    #     return 


    # DEBUG: Visualize the corruption transforms
    # def before_training_iteration(self, strategy, **kwargs):
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


    # def before_eval(self, strategy, **kwargs):
    #     # Reset local eval_counter to track the eval experiences
    #     self.eval_counter = 0
    #     return
    

    def before_eval_exp(self, strategy, **kwargs):
        # try:
        #     transform = strategy.experience.dataset.get_transforms(
        #         strategy.experience.dataset._flat_data
        #     )
        #     transform = transform["eval"].transforms[0].transforms
        # except:
        #     raise ValueError("CorruptionPipelineHandlerPlugin: Could not get transform from experience...")
        # # Get access to the corruption pipeline
        # self.eval_corruption_handler.discover_corruption_pipeline(transform)
        # # Disable the train corruption for safety reasons
        self.train_corruption_handler.set_corruption("none")
        # # Set corruption for experience
        # if self.eval_corruption_pipeline is not None:
        #     self.eval_corruption_pipeline.set_corruption(safe_index(self.corruption_set, self.eval_counter))
        # self.eval_counter += 1
        return super().before_eval(strategy, **kwargs) 

    

class StaticCorruptionPlugin(SupervisedPlugin):
    def __init__(
            self, 
            corruption_set:list, 
            severities:list,
            train_stream
        ):
        super().__init__()
        self.corruption_set = corruption_set
        self.corruption_severities = severities
        self.train_stream = train_stream  # NOTE: this is needed to access the train stream in the before_training_exp method
        
        #self.train_corruption_pipeline = None  # NOTE: there is only one CorruptionPiplineInstance currently
        #self.eval_corruption_pipeline = None  # NOTE: there is only one CorruptionPiplineInstance currently
        self.train_corruption_handler = CorruptionHandler("train")
        self.eval_corruption_handler = CorruptionHandler("eval")
        print("\nDEBUG:corruption_handler: CorruptionPipelineHandlerPlugin initialized")

    
    def discover_transform(self, experience):
        transform = None
        try:
            transform = experience.dataset.get_transforms(
                experience.dataset._flat_data
            )
            transform = transform["train"].transforms[0].transforms
        except:
            print("CorruptionPipelineHandlerPlugin: Could not get transform from experience... trying again")
            try: 
                transform = experience.dataset._datasets[0].\
                    _transform_groups["train"].transforms[0].transforms
            except:
                transform = experience.dataset._flat_data.\
                    _transform_groups["train"].transforms[0].transforms
        return transform


    def freeze_corruption(self, experience):
        try:
            flat_data_ref = experience.dataset.get_transforms(
                experience.dataset._flat_data
            )
            flat_data_ref["train"].transforms[0].transforms = deepcopy(flat_data_ref["train"].transforms[0].transforms)
            flat_data_ref["eval"].transforms[0].transforms = deepcopy(flat_data_ref["eval"].transforms[0].transforms)
            print("deepcopy of transforms done")
            # repeat for "eval"
            # transform = transform["eval"].transforms[0].transforms 
            # transform = deepcopy(transform)
        except:
            raise ValueError("CorruptionPipelineHandlerPlugin: Could not get transform from experience...")


    def before_training_exp(self, strategy, **kwargs):
        print("DEBUG: StaticCorruptionPlugin: before_training_exp called")
        # for idx, exp in enumerate(self.train_stream):
        #     # Get access to the corruption pipeline
        #     #transform = self.discover_transform(exp)
        #     #self.train_corruption_handler.discover_corruption_pipeline(transform)
        #     #print("transform", transform)
        #     # print("done...")

        #     # First test
        #     # test_copy = deepcopy(exp)
        #     # test_transform = self.discover_transform(test_copy)
        #     #strategy.experience = deepcopy(strategy.experience)  # NOTE: this is needed to avoid the corruption transform to be shared between experiences

        #     # NOTE: Line below is not working because NCStream does not allow for item assignment 
        #     # self.train_stream[idx] = deepcopy(exp) 
            
        #     self.freeze_corruption(self.train_stream[idx])

        #     # print(transform is test_transform)  # should be false        
        

        #     # Set corruption for experience
        #     #self.train_corruption_handler.set_corruption(safe_index(self.corruption_set, strategy.clock.train_exp_counter))
        #     #self.train_corruption_handler.set_severity(safe_index(self.corruption_severities, strategy.clock.train_exp_counter))
        
        # print("deepcopy of exp's transform complete\n")
        # #import sys; sys.exit()

        exp1 = self.train_stream[0]
        transform1 = self.discover_transform(exp1)
        CH1 = CorruptionHandler("train")
        CH1.discover_corruption_pipeline(transform1)

        exp2 = self.train_stream[1]
        transform2 = self.discover_transform(exp2)
        CH2 = CorruptionHandler("train")
        CH2.discover_corruption_pipeline(transform2)

        print(CH1.corruption_pipeline is CH2.corruption_pipeline)  # should be false
        print("done")
        import sys; sys.exit()

        return
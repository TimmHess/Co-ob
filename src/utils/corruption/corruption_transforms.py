import torch

from collections import OrderedDict

import numpy as np
import imgaug.augmenters as iaa



class CorruptionPipeline(torch.nn.Module):
    """
    :params name: The name of the corruption to apply
    :params severity: The severity of the corruption

    NOTE: Corruption require the input to be in range uint8!
    """
    def __init__(self, name="none", severity=1):
        super().__init__()
        self.active_corrpution_name = name
        self.severity = severity
        
        self.names, funcs = iaa.imgcorruptlike.get_corruption_names("all")  # 15 corruptions + 1 clean
        self.corruption_dict = OrderedDict(zip(self.names, funcs))
        return
    
    def set_severity(self, severity):
        self.severity = severity
        return
    
    def set_corruption(self, name):
        """
        :params idx: The index or name of a corruption
        """
        assert isinstance(name, str), "Corruptions need to be indexed by name"
        
        self.active_corrpution_name = name
        return
    
    def forward(self, x):
        # Check if any corruption should be applied    
        if self.active_corrpution_name == "none": 
            return x
        
        x = np.asarray(np.array(x), dtype=np.uint8) # NOTE: requirement for imaug
        x = self.corruption_dict[self.active_corrpution_name](x, severity=self.severity)
        return x
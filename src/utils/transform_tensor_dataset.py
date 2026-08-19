import torch
from torchvision import transforms

from torch.utils.data import TensorDataset


class TransformTensorDataset(TensorDataset):
    """
    Assumes the first tensor to be the image targets.
    """
    def __init__(self, *tensors, transform=None):
        super(TransformTensorDataset, self).__init__(*tensors)
        self.transform = transform
        return

    def __getitem__(self, index):
        x = list(super(TransformTensorDataset, self).__getitem__(index))
        if self.transform:
             x[0] = self.transform(x[0])
        x = tuple(x)
        return x


class PILTensorDataset(TensorDataset):
    """
    Assumes the first tensor to be the image targets.
    Converts the first tensor to PIL image.
    """
    def __init__(self, *tensors):
        super(PILTensorDataset, self).__init__(*tensors)
        self.transform = transforms.ToPILImage()
        return

    def __getitem__(self, index):
        x = list(super(PILTensorDataset, self).__getitem__(index))
        x[0] = self.transform(x[0])
        x = tuple(x)
        return x
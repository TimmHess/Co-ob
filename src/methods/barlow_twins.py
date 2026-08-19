import torch
import torch.nn as nn

from copy import deepcopy

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models.utils import FeatureExtractorModel

class TwoCropTransform(nn.Module):
    """
    Applies same transform to two different crops of the same image.
    """
    def __init__(self, crop_transform):
        super().__init__()
        self.crop_transform = crop_transform

    def forward(self, x):
        return torch.stack([self.crop_transform(x), self.crop_transform(x)])

    def __call__(self, x):
        return torch.stack([self.crop_transform(x), self.crop_transform(x)])

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.crop_transform.transforms:
            format_string += "\n"
            format_string += f"{t}"
        format_string += "\n)"
        return format_string


class BarlowTwinLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, projection_dim=128):
        super().__init__()

        self.z_dim = projection_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, projections, targets):
        # NOTE: we don't need the targets
        z1 = projections[0]
        z2 = projections[1]
        
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag



class BarlowTwinsPlugin(SupervisedPlugin):  # NOTE: actually it is a self-supervised plugin...
    def __init__(
            self,
            projection_sizes=[512,2048,2048,2048],
            lmbda_coeff=5e-4,
            train_mb_size=128,
            split_forward=False
        ):
        super().__init__()

        # Projection head
        sizes = projection_sizes

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True)) # originally bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projection_head = nn.Sequential(*layers)
        print("Projection head:")
        print(self.projection_head, "\n")        

        # Loss
        self._criterion = BarlowTwinLoss(
            batch_size=train_mb_size,
            projection_dim=projection_sizes[-1], 
            lambda_coeff=lmbda_coeff # 3.9e-3 - https://github.com/MaxLikesMath/Barlow-Twins-Pytorch/tree/main
        )

        self.split_forward = split_forward

        self.__original_forward = None
        self.__original_criterion = None
        return
    
    def switch_forward_and_criterion(self, strategy, is_training):
        if is_training:  # Training
            def forward(obj):
                return None
            strategy.forward = forward.__get__(strategy)
            strategy._criterion = deepcopy(self._criterion)
        else:  # Evaluation
            strategy.forward = self.__original_forward.__get__(strategy)  # NOTE: for some reason this must not be using deepcopy!
            strategy._criterion = deepcopy(self.__original_criterion)
        return

    def before_training(self, strategy, **kwargs):
        """
        Alter the supervised-strategy for self-supervised training according to BarlowTwins.
        Addmittedly, this is a hack, but since SSL is not fully supported in Avalanche,
        this is a relatively easy way to achieve the desired behavior.
        """
        # Attach projection head to model
        assert isinstance(strategy.model, FeatureExtractorModel)
        strategy.model.projection_head = self.projection_head
        # Add projection head to optimizer
        #strategy.optimizer.add_param_group({'params': self.projection_head.parameters()})
        strategy.optimizer.param_groups[0]['params'].extend(strategy.model.projection_head.parameters())

        # Store the original forward function
        if self.__original_forward is None:
            self.__original_forward = deepcopy(strategy.forward)
        # Store the original criterion
        if self.__original_criterion is None:
            self.__original_criterion = deepcopy(strategy._criterion)
        return

    def before_training_exp(self, strategy, **kwargs):
        self.switch_forward_and_criterion(strategy, is_training=True)
        return


    def after_forward(self, strategy, **kwargs):
        # NOTE: avalanche forward should have done nothing when using this model (reutn None)
        # Compute the forward pass of the model and the projection head
        if self.split_forward:
            embed_1 = strategy.model.feature_extractor(strategy.mb_x[:,0,:])
            embed_2 = strategy.model.feature_extractor(strategy.mb_x[:,1,:])
            proj_embed_1 = strategy.model.projection_head(embed_1)
            proj_embed_2 = strategy.model.projection_head(embed_2)    
            strategy.mb_output = (proj_embed_1, proj_embed_2)
        else:
            embed = strategy.model.feature_extractor(torch.cat((strategy.mb_x[:,0,:], strategy.mb_x[:,1,:])))
            proj_embed = strategy.model.projection_head(embed)
            strategy.mb_output = (proj_embed[:strategy.mb_x.shape[0]], proj_embed[strategy.mb_x.shape[0]:])
        return

    def before_backward(self, strategy, **kwargs):
        # TODO: If use grad_scaler - scale the gradients
        return

    def before_eval(self, strategy, **kwargs):
        # Restore the original forward function
        self.switch_forward_and_criterion(strategy, is_training=False)
        print(strategy._criterion)
        return


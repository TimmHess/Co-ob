"""
This module contains monkey patches for the avalnche library.
The main purpose is to avoid modifying the avalnche library itself, while 
already adapting to "soon-to-be" deprecated functions.
"""

def patch_drop_last_dataloader():
    from torch.utils.data import DataLoader
    from avalanche.training.templates.base_sgd import BaseSGDTemplate
    from avalanche.benchmarks.utils.data_loader import (
        SingleDatasetDataLoader,
        TaskBalancedDataLoader,
        collate_from_data_or_kwargs,
    )

    # Re-define the make_train_dataloader method to change the drop_last parameter default
    def make_train_dataloader(
        obj,
        num_workers=0,
        shuffle=True,
        pin_memory=None,
        persistent_workers=False,
        drop_last=True,
        **kwargs
    ):
        """Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """

        assert obj.adapted_dataset is not None

        other_dataloader_args = obj._obtain_common_dataloader_parameters(
            batch_size=obj.train_mb_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        )

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        # use task-balanced dataloader for task-aware benchmarks
        if hasattr(obj.experience, "task_labels"):
            obj.dataloader = TaskBalancedDataLoader(
                obj.adapted_dataset,
                oversample_small_groups=True,
                **other_dataloader_args
            )
        else:
            obj.dataloader = DataLoader(obj.adapted_dataset, **other_dataloader_args)
    
    # Inject into the BaseSGDTemplate class
    BaseSGDTemplate.make_train_dataloader = make_train_dataloader
    print("===========MonkeyPatch: Dataloader drop_last set to True===========")
    return


def patch_feature_extractor_model():
    """
    This patch enables feature extractor models to be used with deepcopy
    """
    from avalanche.models.dynamic_modules import MultiTaskModule
    from avalanche.models.utils import FeatureExtractorModel
    from avalanche.models.utils import avalanche_forward

    # def forward(obj, x):
    #     x = obj.feature_extractor(x)
    #     obj.features = x.clone().detach()  # NOTE: this is needed if you ever want to deepcopy this model
    #     x = obj.train_classifier(x)
    #     return x
    def forward(self, x, task_lables=None):
        x = avalanche_forward(self.feature_extractor, x, task_lables)
        self.features = x.clone().detach()  # NOTE: this is needed if you ever want to deepcopy this model
        x = avalanche_forward(self.train_classifier, x, task_lables)
        return x

    FeatureExtractorModel.__bases__ = (MultiTaskModule, )
    FeatureExtractorModel.forward = forward
    print("===========MonkeyPatch: FeatureExtractorModel.forward===========")
    return

# NOTE: Does not work..
# def patch_feature_extractor_model():
#     """
#     Full replacement of the FeatureExtractorModel class
#     to allow Multi-headed integration
#     """
#     import sys
#     import importlib
#     from avalanche.models.utils import FeatureExtractorModel
#     from avalanche.models.dynamic_modules import MultiTaskModule
#     from avalanche.models.utils import avalanche_forward

#     class FeatureExtractorModelPatched(MultiTaskModule):
#         """
#         Feature extractor that additionnaly stores the features
#         """

#         def __init__(self, feature_extractor, train_classifier):
#             print("DEBUG: INIT PATCHED MODEL")
#             super().__init__()
#             self.feature_extractor = feature_extractor
#             self.train_classifier = train_classifier
#             self.features = None

#         def forward(self, x, task_lables=None):
#             x = avalanche_forward(self.feature_extractor, x, task_lables)
#             self.features = x.clone().detach()  # NOTE: this is needed if you ever want to deepcopy this model
#             x = avalanche_forward(self.train_classifier, x, task_lables)
#             return x

#     # Remove the class from the cache if it's already imported
#     module = sys.modules.get("avalanche.models.utils")
#     if module and hasattr(module, 'FeatureExtractorModel'):
#         delattr(module, 'FeatureExtractorModel')

#     # Overwrite the class globally in its original module
#     module = sys.modules["avalanche.models.utils"]
#     module.FeatureExtractorModel = FeatureExtractorModelPatched
#     importlib.reload(module)
#     from avalanche.models.utils import FeatureExtractorModel
#     print(FeatureExtractorModel)
#     print("===========MonkeyPatch: FeatureExtractorModel===========")
#     return


def patch_avalanche_dataset_get_transforms():
    """
    This patch enables a recursive search though the AvalancheDataset class for 
    its hidden transforms
    """
    from avalanche.benchmarks.utils.data import AvalancheDataset, _FlatDataWithTransform
    from avalanche.benchmarks.utils.transform_groups import TransformGroups, EmptyTransformGroups
    from avalanche.benchmarks.utils.transforms import TupleTransform

    def get_transforms(obj, flat_data):
        """
        Recursively walks though the dataset until it finds a transform group.
        """
        if isinstance(flat_data._transform_groups, EmptyTransformGroups):
            for dd in flat_data._datasets:
                if isinstance(dd, _FlatDataWithTransform):
                    return obj.get_transforms(dd)
        elif isinstance(flat_data._transform_groups, TransformGroups):
            assert isinstance(flat_data._transform_groups["train"], TupleTransform)
            return flat_data._transform_groups
        else:
            raise ValueError("Invalid transform group type")
        
    AvalancheDataset.get_transforms = get_transforms
    print("===========MonkeyPatch: AvalancheDataset.get_transforms===========")
    return

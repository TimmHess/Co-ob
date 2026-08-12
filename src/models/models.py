import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision.models
from torchvision.models import resnet18

from typing import List
import math
from copy import deepcopy

from avalanche.models import FeatureExtractorModel
from avalanche.models.dynamic_modules import MultiHeadClassifier,\
                    MultiTaskModule, IncrementalClassifier

from src.benchmarks import get_dataset_class_names

from src.models.vgg import VGGFeat, cfgs
from src.models.vit import VisionTransformerFeat

from src.models.utils import replace_BN_by_GN, replace_BN

from src.models.huggingface_wrapper import (
    HuggingWrapper, HuggingClipWrapper, HFClipHead
)

'''
General Definitions
'''
def weight_reset(m):
    #if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #    m.reset_parameters()
    if hasattr(m, 'reset_parameters'):
       m.reset_parameters()


def load_weights(model, state_dict, replace_key_dict=None):
    print("Loading model weights...")
    if replace_key_dict is not None:
        for key in list(state_dict.keys()):
            # print("Original key:", key)
            new_key = deepcopy(key)
            for old_str, new_str in replace_key_dict.items():
                new_key = new_key.replace(old_str, new_str)
                # print("Replacing '{}' with '{}', new key: '{}'".format(old_str, new_str, new_key))
            state_dict[new_key] = state_dict.pop(key)
    load_result = model.load_state_dict(state_dict, strict=False)
    # Print missing keys
    if load_result.missing_keys:
        print("Missing keys:")
        for key in load_result.missing_keys:
            print(key)
    else:
        print("All keys loaded successfully.")
    # Print unexpected keys
    if load_result.unexpected_keys:
        print("Unexpected keys:")
        for key in load_result.unexpected_keys:
            print(key)
    else:
        print("No unexpected keys.")
    return


'''
Layer Definitions
'''
class L2NormalizeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return torch.nn.functional.normalize(x, p=2, dim=1)
    

class FeatAvgPoolLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """ This format to be compatible with OpenSelfSup and the classifiers expecting a list."""
        # Pool
        assert x.dim() == 4, \
            "Tensor must has 4 dims, got: {}".format(x.dim())
        x = self.avg_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        return x

    

'''
Backbones
'''
class SimpleCNNFeat(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNNFeat, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(p=0.25)
        )

        self.feature_size = self.calc_feature_size(input_size)
    
    def calc_feature_size(self, input_size):
        with torch.no_grad():
            self.feature_size = self.features(torch.zeros(1, *input_size)).view(1, -1).size(1)
        return self.feature_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    

class MLPfeat(nn.Module):
    def_hidden_size = 400

    def __init__(self, nonlinear_embedding: bool, input_size=28 * 28,
                 hidden_sizes: tuple = None, nb_layers=2, drop_grad=0.0):
        """
        :param nonlinear_embedding: Include non-linearity on last embedding layer.
        This is typically True for Linear classifiers on top. But is false for embedding based algorithms.
        :param input_size:
        :param hidden_size:
        :param nb_layers:
        """
        super().__init__()
        assert nb_layers >= 2
        if hidden_sizes is None:
            hidden_sizes = [self.def_hidden_size] * nb_layers
        else:
            assert len(hidden_sizes) == nb_layers
        self.feature_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes

        # Need at least one non-linear layer
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_sizes[0]),
                                 nn.ReLU(inplace=True),
                                 #DropGrad(rate=drop_grad) # NOTE: Add DropGrad layer
                                 ))

        for layer_idx in range(1, nb_layers - 1):  # Not first, not last
            layers.add_module(
                f"fc{layer_idx}", nn.Sequential(
                    *(nn.Linear(hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]),
                      nn.ReLU(inplace=True),
                      #DropGrad(rate=drop_grad) # NOTE: Add DropGrad layer
                      )))

        # Final layer
        layers.add_module(
            f"fc{nb_layers}", nn.Sequential(
                *(nn.Linear(hidden_sizes[nb_layers - 2],
                            hidden_sizes[nb_layers - 1]),
                  #DropGrad(rate=drop_grad) # NOTE: Add DropGrad layer  
                  )))

        # Optionally add final nonlinearity
        if nonlinear_embedding:
            layers.add_module(
                f"final_nonlinear", nn.Sequential(
                    *(nn.ReLU(inplace=True),
                    )))

        self.features = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        # x = self.classifier(x)
        return x
    

class MLPLNfeat(nn.Module):
    def_hidden_size = 400

    def __init__(self, nonlinear_embedding: bool, input_size=28 * 28,
                 hidden_sizes: tuple = None, nb_layers=2):
        """
        :param nonlinear_embedding: Include non-linearity on last embedding layer.
        This is typically True for Linear classifiers on top. But is false for embedding based algorithms.
        :param input_size:
        :param hidden_size:
        :param nb_layers:
        """
        super().__init__()
        assert nb_layers >= 2
        if hidden_sizes is None:
            hidden_sizes = [self.def_hidden_size] * nb_layers
        else:
            assert len(hidden_sizes) == nb_layers
        self.feature_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes

        # Need at least one non-linear layer
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_sizes[0]),
                                 nn.LayerNorm(hidden_sizes[0]),
                                 nn.ReLU(inplace=True),
                                 ))

        for layer_idx in range(1, nb_layers - 1):  # Not first, not last
            layers.add_module(
                f"fc{layer_idx}", nn.Sequential(
                    *(nn.Linear(hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]),
                      nn.LayerNorm(hidden_sizes[layer_idx]),
                      nn.ReLU(inplace=True),
                      )))

        # Final layer
        layers.add_module(
            f"fc{nb_layers}", nn.Sequential(
                *(nn.Linear(hidden_sizes[nb_layers - 2],
                            hidden_sizes[nb_layers - 1]),
                  nn.LayerNorm(hidden_sizes[nb_layers - 1]),
                  )))

        # Optionally add final nonlinearity
        if nonlinear_embedding:
            layers.add_module(
                f"final_nonlinear", nn.Sequential(
                    *(nn.ReLU(inplace=True),
                    )))

        self.features = nn.Sequential(*layers)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x


class MLPBNfeat(nn.Module):
    def_hidden_size = 400

    def __init__(self, nonlinear_embedding: bool, input_size=28 * 28,
                 hidden_sizes: tuple = None, nb_layers=2):
        """
        :param nonlinear_embedding: Include non-linearity on last embedding layer.
        This is typically True for Linear classifiers on top. But is false for embedding based algorithms.
        :param input_size:
        :param hidden_size:
        :param nb_layers:
        """
        super().__init__()
        assert nb_layers >= 2
        if hidden_sizes is None:
            hidden_sizes = [self.def_hidden_size] * nb_layers
        else:
            assert len(hidden_sizes) == nb_layers
        self.feature_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes

        # Need at least one non-linear layer
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_sizes[0]),
                                 nn.BatchNorm1d(hidden_sizes[0]),
                                 nn.ReLU(inplace=True),
                                 ))

        for layer_idx in range(1, nb_layers - 1):  # Not first, not last
            layers.add_module(
                f"fc{layer_idx}", nn.Sequential(
                    *(nn.Linear(hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]),
                      nn.BatchNorm1d(hidden_sizes[layer_idx]),
                      nn.ReLU(inplace=True),
                      )))

        # Final layer
        layers.add_module(
            f"fc{nb_layers}", nn.Sequential(
                *(nn.Linear(hidden_sizes[nb_layers - 2],
                            hidden_sizes[nb_layers - 1]),
                  nn.BatchNorm1d(hidden_sizes[nb_layers - 1]),
                  )))

        # Optionally add final nonlinearity
        if nonlinear_embedding:
            layers.add_module(
                f"final_nonlinear", nn.Sequential(
                    *(nn.ReLU(inplace=True),
                    )))

        self.features = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        # x = self.classifier(x)
        return x
    

#########
#Slim ResNet (Avalanche Copy)
#########
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SlimResnet18Feat(nn.Module):
    def __init__(self, input_size, block, num_blocks, nf, global_pooling=False):
        super(SlimResnet18Feat, self).__init__()
        self.in_planes = nf
        self.global_pooling = global_pooling

        if self.global_pooling:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.conv1 = conv3x3(3, nf * 1)
        if input_size[0] == 1:  # Grayscale input
            self.conv1 = conv3x3(1, nf * 1)
            
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        #self.linear = nn.Linear(nf * 8 * block.expansion, 100) # num_classes

        self.feature_size = self.calc_feature_size(input_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def calc_feature_size(self, input_size):
        with torch.no_grad():
            self.feature_size = self.forward(torch.zeros(1, *input_size)).view(1, -1).size(1)
        return self.feature_size

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.global_pooling:
            out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        return out


'''
Classifier
'''
class BiCClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(in_features=in_features, 
                                    out_features=num_classes)  
        self.bias_layer = None  # NOTE: will be set by bic plugin

    def forward(self, x):
        x = self.classifier(x)
        
        if not self.training and self.bias_layer is not None:
            x = self.bias_layer(x)
        return x


'''
Model Factory
'''
def _get_feat_extr(args, input_size):
    """ Get embedding network. """
    if args.backbone == "mlp":  # MNIST MLP
        nonlin_embedding = args.classifier in ['linear']
        feat_extr = MLPfeat(hidden_sizes=(400, 400, args.featsize), 
                            nb_layers=3,
                            nonlinear_embedding=nonlin_embedding, 
                            input_size=math.prod(input_size),
                            drop_grad=args.drop_grad)
    elif args.backbone == "mlp_200":  # MNIST MLP with 2 hidden layers
        nonlin_embedding = args.classifier in ['linear']
        feat_extr = MLPfeat(hidden_sizes=(200, 200), 
                            nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, 
                            input_size=math.prod(input_size),
                            drop_grad=args.drop_grad)
    elif args.backbone == "mlp_ln":  # MNIST MLP with LayerNorm
        nonlin_embedding = args.classifier in ['linear']
        feat_extr = MLPLNfeat(hidden_sizes=(400, args.featsize), nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, 
                            input_size=math.prod(input_size))
    elif args.backbone == "mlp_bn":
        nonlin_embedding = args.classifier in ['linear']
        feat_extr = MLPBNfeat(hidden_sizes=(400, args.featsize), nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, 
                            input_size=math.prod(input_size))
    elif args.backbone == 'simple_cnn':
        feat_extr = SimpleCNNFeat(input_size=input_size)
    elif args.backbone == 'VGG_DF':
        feat_extr = VGGFeat(input_size=input_size, 
                            cfg=cfgs["S_DF"],
                            batch_norm=False,
                            dense_dim=128)
    elif args.backbone == 'VGG_DF_BN':
        feat_extr = VGGFeat(input_size=input_size, 
                            cfg=cfgs["S_DF"],
                            batch_norm=True,
                            dense_dim=128)
    elif args.backbone == "VGG11":
        feat_extr = VGGFeat(input_size=input_size, 
                    cfg=cfgs["A"],
                    batch_norm=False,
                    dense_dim=1024)
    elif args.backbone == "VGG11_PT_ImgNet1k":
        from torchvision.models import VGG11_Weights
        vgg11 = torchvision.models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1, dropout=0.0)
        vgg11.classifier = vgg11.classifier[:-1]  # Remove classifier but keep dense parameters
        feat_extr = vgg11
        feat_extr.feature_size = 4096  # NOTE: determined by model
    elif args.backbone == 'SlimResNet18':
        feat_extr = SlimResnet18Feat(
            input_size=input_size,
            block=BasicBlock, 
            num_blocks=[2, 2, 2, 2],
            nf=20,
            global_pooling=args.use_GAP
        )
    elif args.backbone == 'SlimResNet18_nf64':
        feat_extr = SlimResnet18Feat(
            input_size=input_size,
            block=BasicBlock, 
            num_blocks=[2, 2, 2, 2],
            nf=64,
            global_pooling=args.use_GAP
        )
    elif args.backbone == 'SlimResNet34_nf64':
        feat_extr = SlimResnet18Feat(
            input_size=input_size,
            block=BasicBlock, 
            num_blocks=[3, 4, 6, 3],
            nf=64,
            global_pooling=args.use_GAP
        )
    elif args.backbone == 'ResNet18_PT':
        from torchvision.models import ResNet18_Weights
        feat_extr = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Change final layer to Flatten
        feat_extr.fc = torch.nn.Flatten() 
        # feat_extr = torch.nn.Sequential(*list(feat_extr.children())[:-1]) # Remove final layer
        feat_extr.feature_size = 512  # NOTE: determined by ResNet18 with GAP
    elif args.backbone == 'ResNet18':
        feat_extr = resnet18(weights=None)
        # Change final layer to Flatten
        feat_extr.fc = torch.nn.Flatten() 
        # feat_extr = torch.nn.Sequential(*list(feat_extr.children())[:-1]) # Remove final layer
        feat_extr.feature_size = 512  # NOTE: determined by ResNet18 with GAP
    elif args.backbone == 'ResNet18_rdim':
        assert args.force_repr_dim is not None, "force_repr_dim must be set for ResNet18_dim"
        feat_extr = resnet18(weights=None)
        def reduce_dimension(feat_extr, new_out_channels):
            feat_extr.layer4[0].conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=new_out_channels,  # Changed
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
            feat_extr.layer4[0].bn2 = nn.BatchNorm2d(
                num_features=new_out_channels,  # Must match conv2 out_channels
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
            # Add a downsample for the residual connection
            feat_extr.layer4[0].downsample = nn.Sequential(
                nn.Conv2d(256, new_out_channels, kernel_size=(1,1), stride=(2,2), bias=False),
                nn.BatchNorm2d(new_out_channels)
            )
            feat_extr.layer4[1].conv1 = nn.Conv2d(
                in_channels=new_out_channels,
                out_channels=new_out_channels,  # Changed
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False
            )
            feat_extr.layer4[1].bn1 = nn.BatchNorm2d(
                num_features=new_out_channels,  # Must match conv2 out_channels
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
            feat_extr.layer4[1].conv2 = nn.Conv2d(
                in_channels=new_out_channels,
                out_channels=new_out_channels,  # Changed
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False
            )
            # Replace bn2 to match the new number of channels
            feat_extr.layer4[1].bn2 = nn.BatchNorm2d(
                num_features=new_out_channels,  # Must match conv2 out_channels
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        reduce_dimension(feat_extr, args.force_repr_dim)
        feat_extr.fc = torch.nn.Flatten() #feat_extr.fc = nn.Linear(512, 50) # alternative but not as nice.. 
        # feat_extr = torch.nn.Sequential(*list(feat_extr.children())[:-1]) # Remove final layer
        feat_extr.feature_size = args.force_repr_dim  # NOTE: determined by ResNet18 with GAP
    elif args.backbone == 'ViT_tiny':
        feat_extr = VisionTransformerFeat(
            n_channels=input_size[0],
            image_size=input_size[1],
            embed_dim=64,
            patch_size=4,
            n_layers=6,
            n_attention_heads=4,
            forward_mul=2,
        )
    elif args.backbone == 'ViT-Lite-7-4':
        feat_extr = VisionTransformerFeat(
            n_channels=input_size[0],
            image_size=input_size[1],
            embed_dim=256,
            patch_size=4,
            n_layers=7,
            n_attention_heads=4,
            forward_mul=2,
        )
    elif args.backbone == 'ViT_B_16_PT':
        from torchvision.models import ViT_B_16_Weights
        # Load the model
        vit_b_model = torchvision.models.vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_V1
        )
        # MonkeyPatch the backbone of the model
        def forward(obj, x):
             # Reshape and permute the input tensor
            x = obj._process_input(x)
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = obj.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = obj.encoder(x)
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
            return x   
        vit_b_model.forward = forward.__get__(vit_b_model)
        del vit_b_model.heads
        feat_extr = vit_b_model
        feat_extr.feature_size = 768  # NOTE: determined by model
    elif args.backbone == 'ViT_tiny_PT_ImgNet21k':
        import timm
        vit_tiny = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=True
        ) # ImgNet21k
        # MonkeyPatch the backbone of the model
        def forward(obj, x):
            x = obj.forward_features(x)
            x = x[:, 0]  # NOTE: Only take the class token
            return x
        vit_tiny.forward = forward.__get__(vit_tiny)
        feat_extr = vit_tiny
        feat_extr.feature_size = 192  # NOTE: determined by model
    elif args.backbone == "HF_ViT-B_16_PT_ImgNet21k":
        from transformers import ViTModel
        model_name = "google/vit-base-patch16-224-in21k" # ImgNet21k #model_name = "google/vit-base-patch16-224" # ImgNet21k finetuned to ImgNet1k
        # Load the model from huggingface
        vit = ViTModel.from_pretrained(model_name, cache_dir="./HF_models/")
        # Assign feature_extractor
        feat_extr = HuggingWrapper(model=vit, device=args.device)  # input_processor=image_processor,
        feat_extr.feature_size = 768
    elif args.backbone == "TM_ViT-B_16_augreg_in1k":  # "timm/vit_base_patch16_224.augreg_in1k"
        from transformers import AutoModel, AutoConfig
        from transformers.utils.import_utils import clear_import_cache
        model_name = "timm/vit_base_patch16_224.augreg_in1k"
        config = AutoConfig.from_pretrained(model_name, cache_dir="./HF_models/")
        model = AutoModel.from_pretrained(model_name, config=config, cache_dir="./HF_models/")
        #print(model.timm_model.blocks[0].attn)
        #print(type(model.timm_model.blocks[0].attn))
        from src.models.huggingface_wrapper import split_qkv_in_timm_vit
        split_qkv_in_timm_vit(model)
        from src.models.huggingface_wrapper import TimmWrapper
        feat_extr = TimmWrapper(model, device=args.device, use_cls=True)  # use_mean_pooling=True
        feat_extr.feature_size = 768
    elif args.backbone == "HF_CLIP_ViT_B_32":
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir="./HF_models/"
        ).to(args.device)
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir="./HF_models/", 
            use_fast=True
        )
        # Convert to ClipWrapper
        feat_extr = HuggingClipWrapper(
            model=model, 
            device=args.device, 
            input_processor=processor
        )
        feat_extr.feature_size = 512  # NOTE: determined by model
    elif args.backbone == 'ConvNext_Tiny_PT':
        from torchvision.models.convnext import ConvNeXt_Tiny_Weights
        conv_next = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # MonkeyPatch the backbone of the model
        def forward(obj, x):
            x = obj.features(x)
            x = obj.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        conv_next.forward = forward.__get__(conv_next)
        del conv_next.classifier
        feat_extr = conv_next
        feat_extr.feature_size = 768
    elif args.backbone == 'ConvNext_Tiny':
        from torchvision.models.convnext import ConvNeXt_Tiny_Weights
        conv_next = torchvision.models.convnext_tiny()
        # MonkeyPatch the backbone of the model
        def forward(obj, x):
            x = obj.features(x)
            x = obj.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        conv_next.forward = forward.__get__(conv_next)
        del conv_next.classifier
        feat_extr = conv_next
        feat_extr.feature_size = 768
    elif args.backbone == "MobileNetV3_small":
        mobile_net = torchvision.models.mobilenet_v2(weights=None)
        # MonkeyPatch the backbone of the model
        def forward(obj, x):
            x = obj.features(x)
            x = obj.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        mobile_net.forward = forward.__get__(mobile_net)
        del mobile_net.classifier
        feat_extr = mobile_net
        feat_extr.feature_size = 576
    elif args.backbone == "ViTMAE_Small":
        from transformers import ViTMAEConfig
        from src.models.vitmae_wrapper import AVL_ViTMAEForPreTraining, HFVitMAEWrapper, get_mae_config
        print("DEBUG: input_size", input_size)
        config = get_mae_config(name="tiny", img_size=input_size[1], patch_size=4, mask_ratio=0.75)
        model = AVL_ViTMAEForPreTraining(config)  # NOTE: Altered version of HF code
        feat_extr = HFVitMAEWrapper(model)
        feat_extr.feature_size = 192
    elif args.backbone in ("ijepa_vit_base", "ijepa_vit_small"):
        from src.models.ijepa_vit_wrapper import ijepa_vit_base, ijepa_vit_small
        img_size = input_size[1] if len(input_size) > 1 else 224
        patch_size = getattr(args, 'vit_patch_size', 16)
        if args.backbone == "ijepa_vit_base":
            feat_extr = ijepa_vit_base(img_size=img_size, patch_size=patch_size)
        else:
            feat_extr = ijepa_vit_small(img_size=img_size, patch_size=patch_size)
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")
    return feat_extr


def _get_classifier(
    classifier_type,
    n_classes: int, 
    feat_size: int, 
    initial_out_features: int,
    task_incr: bool = False,
    backbone: nn.Module = None,
    scenario_name=None,
): 
    """ 
    Get classifier head. For embedding networks this is normalization or identity layer.
    
    feat_size: Input to the linear layer
    initial_out_features: (Initial) output of the linear layer. Potenitally growing with task.
    """
    # No prototypes, final linear layer for classification
    if classifier_type == 'linear':  # Linear layer
        print("_get_classifier::feat_size: ", feat_size)
        if task_incr:
            classifier = MultiHeadClassifier(
                in_features=feat_size,
                initial_out_features=initial_out_features,
                masking=False
            )  #use_bias=lin_bias # True by default
        else:
            classifier = nn.Sequential(
                nn.Linear(
                    in_features=feat_size, 
                    out_features=n_classes 
                ) #bias=lin_bias # True by default 
            )
    # Prototypes held in strategy
    elif classifier_type == 'norm_embed':  # Get feature normalization
        classifier = L2NormalizeLayer()
    # CLIP head
    elif classifier_type == 'clip':
        classifier = HFClipHead(
            text_inputs=get_dataset_class_names(scenario_name),
            model=backbone,             
            device=backbone.device, 
            input_processor=backbone.input_processor
        )
    # Identity layer, no classifier
    elif classifier_type == 'identity':  # Just extract embedding output
        classifier = torch.nn.Flatten()
    else:
        raise NotImplementedError()
    return classifier



def get_model(
    args, 
    n_classes, 
    input_size, 
    initial_out_features, 
    backbone_weights=None, 
    model_weights=None,
):
    """ 
    Build model from feature extractor and classifier.
    
    n_classes: total number of classes in the dataset
    initial_out_features: number of classes for the first task
    """

    # Feature extractor
    feat_extr = _get_feat_extr(
        args=args, 
        input_size=input_size
    )  
    # Classifier
    classifier = _get_classifier(
        classifier_type=args.classifier, 
        n_classes=n_classes, 
        feat_size=feat_extr.feature_size, 
        initial_out_features=initial_out_features, 
        task_incr=args.task_incr,
        backbone=feat_extr,
        scenario_name=args.scenario_datasets[0] if args.scenario_datasets is not None else args.scenario
    ) 

    # Build model
    model = FeatureExtractorModel(
        feature_extractor=feat_extr,
        train_classifier=classifier
    )

    # Replace BatchNorm layers
    if not args.replace_BN == "default":
        print(f"Replacing BatchNorm layers with {args.replace_BN} layers.")
        replace_BN(model=model, norm_type=args.replace_BN, max_groups=args.replace_BN_by_GN_groups)

    # Load weights for only the feature extractor (backbone)
    if backbone_weights:
        print("Loading backbone weights from: ", backbone_weights)
        if args.backbone.startswith("ijepa_"):
            # cassle Lightning checkpoints: keys are 'encoder.*'; load directly into wrapper
            from src.models.ijepa_vit_wrapper import load_ijepa_checkpoint
            load_ijepa_checkpoint(feat_extr, backbone_weights)
        else:
            state_dict = torch.load(backbone_weights)
            replace_dict = {
                "encoder": "features",
                "feature_extractor.": "",
                "_orig_mod.": "",
                "module.": ""
            }
            load_weights(model=feat_extr, state_dict=state_dict, replace_key_dict=replace_dict)
        print("Done loading backbone weights.")

    # Load weights for the entire model (backbone + heads)
    if model_weights:
        state_dict = torch.load(model_weights)
        load_weights(model=model, state_dict=state_dict, replace_key_dict=None)
    
    # Finally
    return model

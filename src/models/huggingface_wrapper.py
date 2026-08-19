
import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models import DynamicModule

from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import _get_vector_norm #,clip_loss

import timm


class HuggingWrapper(nn.Module):
    def __init__(self, model, device, input_processor=None):
        super().__init__()
        self.model = model
        self.input_processor = input_processor
        self.device = device
    
    def forward(self, x):
        if self.input_processor is not None:
            # If input processor is provided, use it to process the input
            x = self.input_processor(x, return_tensors="pt", device=self.device)
        
        # All
        x = self.model(x).last_hidden_state.mean(dim=1)

        # All but CLS
        # x = self.model(x).last_hidden_state[:, 1:].mean(dim=1)
        
        # CLS
        # x = self.model(x).last_hidden_state[:, 0]
        return x 


import torch
import torch.nn as nn
from transformers.utils.import_utils import clear_import_cache
from timm.models.vision_transformer import Attention


# class AttentionSplit(Attention, nn.Module):
#     def __init__(self, config):
#         super().__init__(config)
#         # remove combined qkv
#         del self.qkv
#         # separate q, k, v projections
#         self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
#         self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
#         self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
#         self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)

#     def split_q_k_v_load_hook(self, state_dict, prefix, *args):
#         keys_to_delete = []
#         for key in list(state_dict.keys()):
#             if "qkv." in key:
#                 # split q, k, v from the combined projection
#                 q, k, v = state_dict[key].chunk(3, dim=0)
#                 # replace with individual q, k, v projections
#                 state_dict[key.replace("qkv.", "q.")] = q
#                 state_dict[key.replace("qkv.", "k.")] = k
#                 state_dict[key.replace("qkv.", "v.")] = v
#                 # mark the old qkv key for deletion
#                 keys_to_delete.append(key)
        
#         # remove old qkv keys
#         for key in keys_to_delete:
#             del state_dict[key]

#     def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
#         batch_size, height, width, _ = hidden_states.shape
#         qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
#         query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
#         key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
#         value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

#         attn_weights = (query * self.scale) @ key.transpose(-2, -1)

#         attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
#         attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
#         attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
#         attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
#         attn_output = self.proj(attn_output)

#         if output_attentions:
#             outputs = (attn_output, attn_weights)
#         else:
#             outputs = (attn_output, None)
#         return outputs

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

def split_qkv_in_timm_vit(model: Any) -> None:
    """
    Modifies a Timm-style Vision Transformer model to replace the fused 'qkv' layers
    with separate 'q', 'k', and 'v' projections. It directly copies the weights
    from the existing fused layer to the new individual layers.

    This enables applying LoRA to individual Q/K/V layers.

    Args:
        model: A model object containing a Vision Transformer, expected to have a 'blocks' attribute.
               This can be the timm model itself or a wrapper.
    """
    # Accommodate both wrapped and unwrapped timm models
    if hasattr(model, "timm_model") and hasattr(model.timm_model, "blocks"):
        timm_model = model.timm_model
    elif hasattr(model, "blocks"):
        timm_model = model
    else:
        raise ValueError("Model does not appear to be a Timm Vision Transformer with a 'blocks' attribute.")

    # Iterate over all transformer blocks
    for block_idx, block in enumerate(timm_model.blocks):
        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
            continue  # Skip if no attention module or qkv layer

        attn = block.attn
        
        # --- Step 1: Capture original qkv layer and its properties ---
        original_qkv = attn.qkv
        hidden_size = original_qkv.in_features
        qkv_bias = original_qkv.bias is not None

        # --- Step 2: Create separate q, k, v layers ---
        attn.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        attn.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        attn.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # --- Step 3: Split the weights and biases and copy them over ---
        # Detach and clone to avoid any lingering references
        qkv_weight = original_qkv.weight.detach()
        
        # Split the weight tensor into three parts for q, k, and v
        q_w, k_w, v_w = torch.chunk(qkv_weight, 3, dim=0)

        # Copy the weights to the new layers
        attn.q.weight.data.copy_(q_w)
        attn.k.weight.data.copy_(k_w)
        attn.v.weight.data.copy_(v_w)

        if qkv_bias:
            qkv_bias_tensor = original_qkv.bias.detach()
            q_b, k_b, v_b = torch.chunk(qkv_bias_tensor, 3, dim=0)
            attn.q.bias.data.copy_(q_b)
            attn.k.bias.data.copy_(k_b)
            attn.v.bias.data.copy_(v_b)

        # --- Step 4: Delete the fused qkv layer ---
        del attn.qkv
        
        # --- Step 5: Modify forward pass to use separate q/k/v ---
        # The forward pass needs to be rebound to the instance of the class
        def new_forward(self, x):
            B, N, C = x.shape
            # Compute Q, K, V separately
            # The dimension calculation C // self.num_heads assumes head_dim is not explicitly stored
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # Attention computation (same as original)
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = self.attn_drop(attn_probs)

            x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)

            # Final projection
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Replace the forward method on the attn module instance
        attn.forward = new_forward.__get__(attn)

        print(f"Split qkv in block {block_idx} and copied weights.")
    print("All applicable qkv layers have been replaced with separate q/k/v projections.")


class TimmWrapper(HuggingWrapper):
    def __init__(
        self, 
        model, 
        device, 
        input_processor=None, 
        use_cls=True, 
        use_mean_pooling=False
    ):
        super().__init__(model, device, input_processor)
        self.use_cls = use_cls
        self.use_mean_pooling = use_mean_pooling
        
        # Modify the model to split qkv if it's a ViT
        #self.model.

        # Set model to eval mode if needed
        self.model.eval()

    def forward(self, x):
        # If input processor provided, process image
        if self.input_processor is not None:
            x = self.input_processor(x, return_tensors="pt", device=self.device)
            # timm expects [B, C, H, W], so make sure it's in correct shape
            if isinstance(x, dict):
                x = x['pixel_values']  # For HF-style processors

        # Forward pass through timm model
        # Most timm ViTs return [B, num_tokens, embed_dim]
        features = self.model(x)

        # Handle different output types:
        # Some timm models return tuple (features, aux_outputs), we take first
        #if isinstance(features, tuple):
        #features = features[0]
        features = features.last_hidden_state if hasattr(features, 'last_hidden_state') else features[0]

        # Extract penultimate representation
        if self.use_cls:
            # Use CLS token (token 0)
            x = features[:, 0]  # [B, embed_dim]
        elif self.use_mean_pooling:
            # Average all tokens (including CLS)
            x = features.mean(dim=1)  # [B, embed_dim]
        else:
            # Average all tokens EXCEPT CLS
            x = features[:, 1:].mean(dim=1)  # [B, embed_dim]

        return x
    


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
# def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
#     print("")
#     print("logits", logits, logits.shape)
#     return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
#     caption_loss = contrastive_loss(similarity, dim=0)
#     print("DEBUG: Caption loss computed:", caption_loss.item())
#     image_loss = contrastive_loss(similarity, dim=1)
#     print("DEBUG: Image loss computed:", image_loss.item())
#     import sys; sys.exit()
#     return (caption_loss + image_loss) / 2.0


# def avl_clip_loss(similarity, labels):
#     """
#     Compute the CLIP loss for a given similarity matrix. 
#     This is a contrastive loss, the definition of lables is merely for compatability with 
#     the  avalanche library.
    
#     Args:
#         similarity (torch.Tensor): Similarity matrix of shape (batch_size, num_classes).
#         labels (torch.Tensor): Never used.
        
#     Returns:
#         torch.Tensor: Computed CLIP loss.
#     """
#     # loss = clip_loss(similarity)
#     # print("DEBUG: CLIP loss computed:", loss.item())
    
#     # y = torch.arange(len(similarity)).to(similarity.device)
#     # img2cap_match_idx = similarity.argmax(dim=1)
#     # cap2img_match_idx = similarity.argmax(dim=0)

#     # img_acc = (img2cap_match_idx == y).float().mean()
#     # cap_acc = (cap2img_match_idx == y).float().mean()

#     # caption_loss = contrastive_loss(similarity, dim=0)
#     # image_loss = contrastive_loss(similarity, dim=1)
#     # loss = (caption_loss + image_loss) / 2.0
#     # print("DEBUG: CLIP loss computed:", loss.item())
#     import sys; sys.exit()

#     return loss



def clip_class_to_prompt(class_list):
    """
    Convert a list of class names to prompts for CLIP.
    
    Args:
        class_list (list): List of class names.
        
    Returns:
        list: List of prompts formatted for CLIP.
    """
    return [f"a photo of a {class_name}" for class_name in class_list]



class HuggingClipWrapper(HuggingWrapper):
    def __init__(
            self, 
            model, 
            device, 
            input_processor=None,
            use_input_processor=False
        ):
        super().__init__(model=model, input_processor=input_processor, device=device)
        
        assert(isinstance(model, CLIPModel)), "Model must be an instance of CLIPModel"

        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.use_input_processor = use_input_processor

        
    def forward(self, x):
        if self.use_input_processor:
            print("Processing input with input processor...")
            # If input processor is provided, use it to process the input
            x = self.input_processor(
                images=x, 
                return_tensors="pt",
                device=self.device,
            )

        x = x.to(self.device)
        #x = self.model.get_image_features(**x)
        x = self.model.get_image_features(pixel_values=x)
        return x
    


class HFClipHead(nn.Module):
    def __init__(
            self, 
            model, 
            text_inputs,
            device, 
            input_processor,
        ):
        super().__init__()
        
        assert(isinstance(model, HuggingClipWrapper)), "Model must be an instance of CLIPModel"

        self.logit_scale = model.model.logit_scale.detach().clone()  # Store copy of the logit scale from the model

        self.text_inputs = clip_class_to_prompt(text_inputs)
        self.text_embeds = self.set_text_inputs(self.text_inputs, model) if text_inputs is not None else None
        

    def set_text_inputs(self, text_inputs, model):
        text_inputs = model.input_processor.tokenizer(
            self.text_inputs,
            padding=True,
            return_tensors="pt"
        )

        text_inputs = text_inputs.to(model.device)
        text_embeds = model.model.get_text_features(**text_inputs).detach().clone()
        text_embeds = text_embeds / _get_vector_norm(text_embeds)
        return text_embeds


    def shuffle_text_inputs(self, permutation_list):
        self.text_embeds = self.text_embeds[permutation_list]        

    def extent_text_inputs(self, new_text_inputs, model):
        raise NotImplementedError("Extending text inputs is not implemented in HFClipHead.")
        
    def forward(self, x):
        """
        x: processed image inputs
        """
        assert self.text_embeds is not None, "Text inputs must be processed before forward pass is called."

        x = x / _get_vector_norm(x)

        logits_per_text = self.text_embeds @ x.t()
        logits_per_text = logits_per_text * self.logit_scale.exp()
        logits_per_image = logits_per_text.t()
        return logits_per_image
        
    

class HFDynamicClipHead(HFClipHead, DynamicModule):
    def __init__(self, model, text_inputs, device, input_processor):
        super().__init__(
            model=model, 
            text_inputs=text_inputs, 
            device=device, 
            input_processor=input_processor
        )

        return

    # def adaptation(self, experience):
    #     super().adaptation(experience)
    #     return
    
import torch
import torch.nn as nn
import torchvision.models as models
import json
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional
import timm
import numpy as np


class TextExtractor(nn.Module):
    def __init__(self, args):
        super(TextExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

    
def create_text_extractor(
        model_name: str,
        device: torch.device = torch.device('cpu'),
        override_image_size = None,
):

    if model_name == 'bioclinicalbert':
        print(f'Loading {model_name} model config for text extractor.')
        with open('./src/ctranspath_448_bioclinicalbert.json') as f:
            model_cfg = json.load(f)
            
    elif model_name == 'pubmedbert':
        with open('./src/ctranspath_448_pubmedbert.json') as f:
            model_cfg = json.load(f)
    elif model_name == 'scratch':
        class scratch_ext(nn.Module):
            def __init__(self):
                super().__init__()
            def encode_text(self, token_ids, attention_mask=None):
                return token_ids
        return scratch_ext()
    else:
        raise RuntimeError(f'Model config for {model_name} not found.')
    
    if override_image_size:
        model_cfg['vision_cfg']['image_size'] = override_image_size
        print(f'Created model {model_name} with image size of {override_image_size} instead')
        
    model = CLIP(**model_cfg)
    model.to(device=device)
    return model



class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim,
            vision_cfg,
            text_cfg
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)


        if text_cfg.model_type == 'gpt':
            self.transformer = create_textual_gpt(model_path = text_cfg.model_path,
                                            model_config_path = text_cfg.model_config_path,
                                            vocab_size = text_cfg.vocab_size)
        elif text_cfg.model_type == 'pretrained_bert':
            self.transformer = load_pretrained_bert(model_name = text_cfg.model_path)
        else:
            raise NotImplementedError
                             
        self.context_length = text_cfg.max_context_length
        self.vocab_size = self.transformer.vocab_size
        self.text_projection = nn.Parameter(torch.empty(self.transformer.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()

    def init_parameters(self):
        # we do not initialize the text encoder parameters here - handled by hugging face
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def lock_text_tower(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def lock_temperature(self):
        self.logit_scale.requires_grad = False


    def encode_text(self, token_ids, attention_mask=None):
        # token_ids: [batch_size, n_ctx]
        # attn_mask: [batch_size, n_ctx]
        autoregressive = self.transformer.autoregressive
        if attention_mask is None:
            attention_mask = torch.ones(token_ids.size(), dtype=torch.long, device=token_ids.device)

        
        x = self.transformer(input_ids = token_ids, 
                             attention_mask = attention_mask, 
                             output_hidden_states=True)['hidden_states'][-1]

        if autoregressive:
            # find last position
            # cls_index = attention_mask.argmin(dim=-1)
            cls_index = attention_mask.argmin(dim=-1) - 2
            cls_index[cls_index < 0] = attention_mask.size(1) - 2
            hsz = x.shape[-1]
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            x = x.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            if self.transformer.pool == 'avg':
                # average over non-padding locations
                # x: [B, S, C], attention_mask: [B, S]
                # [B, S, C], [B, S, 1] ---> [B, C]
                x = x * attention_mask.unsqueeze(-1)
                x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif self.transformer.pool == 'cls':
                #x = x[:, 0, :] # cls token is the 1st position
                x = x[:,:,:]
            
            else:
                raise NotImplementedError
        #x = x @ self.text_projection
        return x

    def forward(self, image, text, text_attention_mask):
        if image is None:
            return self.encode_text(text, text_attention_mask)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text, text_attention_mask)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()
    
    
@dataclass
class CLIPVisionCfg:
    image_size: Union[Tuple[int, int], int] = 224
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    custom_model_path: str = ''  # path to custom model

@dataclass
class CLIPTextCfg:
    model_path: str = ''
    model_config_path: str = ''
    vocab_size: int = 32005
    max_context_length: int = 128
    num_prefix_tokens: int = 0
    model_type: str = 'pretrained_bert'
        
        
def create_textual_gpt(model_path = '', model_config_path = '', vocab_size = 32005):
    if os.path.isdir(model_path):
        logging.info('initializing text encoder from pretrained ckpt')
        textual = AutoModelWithLMHead.from_pretrained(model_path).transformer
        assert textual.wte.num_embeddings == vocab_size
    elif os.path.isfile(model_config_path):
        logging.info('initializing text encoder from scratch using config file')
        config = AutoConfig.from_pretrained(model_config_path)
        textual = AutoModelWithLMHead.from_config(config).transformer
        textual.resize_token_embeddings(vocab_size)
    else:
        logging.error('No config for text model found')
    textual.width = textual.wte.embedding_dim
    textual.vocab_size = textual.wte.num_embeddings
    textual.autoregressive = True
    return textual 

from transformers import AutoModelForMaskedLM

def load_pretrained_bert(model_name):
    assert model_name in ['PubMedBert', 'BioClinicalBERT']
    textual = AutoModelForMaskedLM.from_pretrained('/chenpingyi/projects/WSI-GPT/WsiVQA/src/pubmedbert').bert
    textual.width = textual.embeddings.word_embeddings.embedding_dim
    textual.vocab_size = textual.embeddings.word_embeddings.num_embeddings
    textual.autoregressive = False
    textual.pool = 'cls'
    return textual


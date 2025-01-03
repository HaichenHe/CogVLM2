import torch
import torch.nn as nn
from . import clip

from .build import MODEL_REGISTRY
import os
import numpy as np
import json

@MODEL_REGISTRY.register()
class BasicClipVideo(nn.Module):
    """
    Clip visual encoder for space feature extraction. Adding various temporal fusion type.
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
            comments of the config file.
        """
        super(BasicClipVideo, self).__init__()
        self.cfg = cfg
        self.num_pathways = 1
        self.alpha = nn.Parameter(torch.ones(512), requires_grad=True)  
        self.beta = nn.Parameter(torch.ones(512), requires_grad=True)   
        self._construct_network(cfg)
        self.model.eval() 
        
        self.all_categories = self.get_all_categories(cfg) # list of all classes
        # get standard prompt encode
        self.txt_encode = self.get_standard_prompt_encode(self.all_categories) # num_classes, d


        if self.training:
            # if apply category_generic_prompt
            if self.cfg.APPLY_CATEGORY_GENERIC_PROMPT:
                category_generic_prompt = self.get_category_generic_prompt_encode(self.cfg) # (num_classes, P) [prompt_1, prompt2, ...., prompt_num_classes]
                
                # concact with standard_prompt
                # self.txt_encode = self.txt_encode.unsqueeze(1) # num_classes, 1, d
                category_generic_prompt = category_generic_prompt.view(self.cfg.MODEL.NUM_CLASSES, self.cfg.CATEGORY_GENERIC_PROMPT_NUM, 512) # num_classes, P, d
                category_generic_prompt = category_generic_prompt.mean(1) # num_classes, d
                # self.txt_encode = torch.cat((self.txt_encode, category_generic_prompt), dim=1)   # num_classes, P+1, d
                # self.txt_encode = (self.txt_encode + category_generic_prompt) / 2   # num_classes, d
                self.txt_encode = self.txt_encode * self.alpha + category_generic_prompt * self.beta   # num_classes, d

        else:
            if self.cfg.APPLY_CATEGORY_GENERIC_PROMPT:
                category_generic_prompt = self.get_category_generic_prompt_encode(self.cfg)
                category_generic_prompt = category_generic_prompt.view(self.cfg.MODEL.NUM_CLASSES, self.cfg.CATEGORY_GENERIC_PROMPT_NUM, 512)
                category_generic_prompt = category_generic_prompt.mean(1)
                self.txt_encode = self.txt_encode * self.alpha + category_generic_prompt * self.beta

        self.cls_num = len(self.all_categories)

        self.lr_factor = {
            "message": cfg.MODEL.FINETUNE_FACTOR,
            "stadapt": cfg.MODEL.ADAPT_FINETUNE_FACTOR,
            "mlp": cfg.MODEL.MLP_FINETUNE_FACTOR,
        }



    def _construct_network(self, cfg):
        if cfg.MODEL.ARCH == 'vitb32':
            self.model, self.preprocess = clip.load("ViT-B/32", jit=False, )
        elif cfg.MODEL.ARCH == 'vitb16':
            self.model, self.preprocess = clip.load("ViT-B/16", jit=False, )
        else:
            print("error loading arch")
            exit()
        self.model.float()   



    def get_all_categories(self, cfg):
        all_categories = []
        text_mapping = json.load(open(cfg.DATA.INDEX_LABEL_MAPPING_FILE, 'r'))
        for key in text_mapping:
            all_categories.append(text_mapping[key])
        return all_categories
    

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            self.cache_text_features = self.model.encode_text(text)
        self.train()
        return self.cache_text_features



    def get_category_generic_prompt_encode(self, cfg):
        category_generic_prompt = []
        text_mapping = json.load(open(cfg.CATEGORY_GENERIC_PROMPT_FILE, 'r'))
        for category in self.all_categories:
            category_generic_prompt.append(text_mapping[category])
    
        # for key, value in text_mapping.items():
        #     # category_generic_prompt.append(" ".join(value))
        #     category_generic_prompt.append(value)
        
        category_generic_prompt = self.text_tokenize(category_generic_prompt).cuda() # num_classes*P, 77
        category_generic_prompt = self.cache_text(category_generic_prompt) # num_classes*P, d

        return category_generic_prompt
    


    def get_standard_prompt_encode(self, categories):
        # standard_prompt = '{}'
        # standard_prompt = 'A video of a person {}.'
        standard_prompt = 'This is a video about {}.'
        standard_prompt = [standard_prompt.format(category) for category in categories]
        standard_prompt = self.text_tokenize(standard_prompt).cuda()
        standard_prompt_encode = self.cache_text(standard_prompt) # num_classes, d
        return standard_prompt_encode

    

    def text_tokenize(self, categories):
        texts = torch.cat([clip.tokenize(category) for category in categories])
        return texts



    def forward(self, x, labels, update=False):
        # shape of x(input) is (bz, channel, clip_len, h, w)
        # labels is the ground truth label, shape is (bz, )

    
        # image processing
        assert len(x) == self.num_pathways
        x = x[0]
        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)
        
        img_encode = self.model.encode_image(x)        
        
        if self.training:
            # txt_encode.shape = num_classes, P, d
            # txt_encode = self.txt_encode.mean(1) # num_classes, d

            img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)
            txt_encode = txt_encode / txt_encode.norm(dim=-1, keepdim=True)
            
            pred = self.model.logit_scale.exp() * img_encode @ txt_encode.T
            pred = pred.reshape(bz, clip_len, -1).mean(1)

            return pred

        else:
            # img_encode [bz, feat_size]
            # dynamic_clf shape [type_num * cls_num, feat_size]
            img_encode /= img_encode.norm(dim=-1, keepdim=True)
            txt_encode = self.txt_encode / self.txt_encode.norm(dim=-1, keepdim=True)

            pred = self.model.logit_scale.exp() * img_encode @ txt_encode.T
            pred = pred.reshape(bz, clip_len, -1).mean(1)

            return pred
    
 

if __name__ == '__main__':
    model, preprocess = clip.load("/share/home/jia/.cache/clip/ViT-B-32.pt", jit=False, )
    
    # model: text and vision






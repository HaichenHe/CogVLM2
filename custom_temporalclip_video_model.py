import torch
import torch.nn as nn
from . import clip

from .build import MODEL_REGISTRY
import os
import numpy as np
import json

from typing import Tuple, Union
from .clip.model import CLIP,LayerNorm,Transformer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .clip.model import convert_weights
from .clip.clip import _MODELS, _download

from . import customize_visiontransformer
from .customize_visiontransformer import TemporalVisionTransformer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


@MODEL_REGISTRY.register()
class TemporalClipVideo(nn.Module):
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
        super(TemporalClipVideo, self).__init__()
        self.cfg = cfg
        self.num_pathways = 1
        # self.alpha = nn.Parameter(torch.ones(512), requires_grad=True)  
        # self.beta = nn.Parameter(torch.ones(512), requires_grad=True)   
        self._construct_network(cfg)
        self.model.eval() 
        # self.raw_model.eval()

        if self.cfg.APPLY_VIDEO_DESCRIPTION:
            with open(self.cfg.VIDEO_DESCRIPTION_FILE, 'r') as f:
                self.video_descriptions = json.load(f)
            self.video_descriptions_list = list(self.video_descriptions.values()) #234584
        
        self.all_categories = self.get_all_categories(cfg) # list of all classes
        self.origin_category_encode = self.cache_text(self.all_categories) # num_classes, d


        # if not self.cfg.APPLY_DISTINGUISHING_PROMPT:
        #     # get standard prompt encode
        #     self.txt_encode = self.get_standard_prompt_encode(self.all_categories) # num_classes, d
        # self.txt_encode = self.get_standard_prompt_encode(self.all_categories) # num_classes, d

        if self.training:
            if self.cfg.APPLY_DISTINGUISHING_PROMPT:
                self.txt_encode = self.get_distinguishing_prompt_encode(self.cfg) # num_classes, d
            # if apply category_generic_prompt
            if self.cfg.APPLY_CATEGORY_GENERIC_PROMPT:
                category_generic_prompt = self.get_category_generic_prompt_encode(self.cfg) # (num_classes, P) [prompt_1, prompt2, ...., prompt_num_classes]
                category_generic_prompt = category_generic_prompt.view(self.cfg.MODEL.NUM_CLASSES, self.cfg.CATEGORY_GENERIC_PROMPT_NUM, 512) # num_classes, P, d
                category_generic_prompt = category_generic_prompt.mean(1) # num_classes, d
                self.txt_encode = (self.txt_encode + category_generic_prompt) / 2   # num_classes, d



        else:
            if self.cfg.APPLY_DISTINGUISHING_PROMPT:
                self.txt_encode = self.get_distinguishing_prompt_encode(self.cfg) # num_classes, d
            elif self.cfg.APPLY_CATEGORY_GENERIC_PROMPT:
                category_generic_prompt = self.get_category_generic_prompt_encode(self.cfg) # (num_classes, P) [prompt_1, prompt2, ...., prompt_num_classes]
                category_generic_prompt = category_generic_prompt.view(self.cfg.MODEL.NUM_CLASSES, self.cfg.CATEGORY_GENERIC_PROMPT_NUM, 512) # num_classes, P, d
                category_generic_prompt = category_generic_prompt.mean(1) # num_classes, d
                self.txt_encode = (self.txt_encode + category_generic_prompt) / 2   # num_classes, d

        
        
        
        self.cls_num = len(self.all_categories)

        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.record_routing = cfg.MODEL.RECORD_ROUTING
        self.keep_raw_model = cfg.MODEL.KEEP_RAW_MODEL
        self.ensemble_pred = cfg.MODEL.ENSEMBLE_PRED
        self.distillation = cfg.MODEL.RAW_MODEL_DISTILLATION

    
        
        
        # self.prompt_embed. -> token_prefix + prompt_embed + token_suffix

        # learning factor
        # if self.cfg and self.cfg.MODEL.FINETUNE_FACTOR != 1.0:
        # Indicate parameters for finetuning.
        self.lr_factor = {
            "message": cfg.MODEL.FINETUNE_FACTOR,
            "stadapt": cfg.MODEL.ADAPT_FINETUNE_FACTOR,
            "mlp": cfg.MODEL.MLP_FINETUNE_FACTOR,
            "experts": cfg.MODEL.EXPERT_FINETUNE_FACTOR,
            "routing": cfg.MODEL.ROUTING_FINETUNE_FACTOR,
        } 


    
    def get_all_categories(self, cfg):
        all_categories = []
        text_mapping = json.load(open(cfg.DATA.INDEX_LABEL_MAPPING_FILE, 'r'))
        for key in text_mapping:
            all_categories.append(text_mapping[key])
        return all_categories
    

    def text_tokenize(self, categories):
        texts = torch.cat([clip.tokenize(category) for category in categories])
        return texts

    
    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            text = self.text_tokenize(text).cuda()
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
        
        # category_generic_prompt = self.text_tokenize(category_generic_prompt).cuda() # num_classes*P, 77
        category_generic_prompt = self.cache_text(category_generic_prompt) # num_classes*P, d

        return category_generic_prompt
    
    def get_distinguishing_prompt_encode(self, cfg):
        full_prompts = []
        text_mapping = json.load(open(cfg.DISTINGUISHING_PROMPT_FILE, 'r'))
        for category in self.all_categories:
            distinguishing_prompt = text_mapping.get(category, "")
            full_prompt = f"A video about {category}. {distinguishing_prompt}"
            full_prompts.append(full_prompt)
        
        full_prompts= self.cache_text(full_prompts)
        return full_prompts

    def get_standard_prompt_encode(self, categories):
        # standard_prompt = '{}'
        # standard_prompt = 'A video of a person {}.'
        standard_prompt = 'A video about {}.'
        standard_prompt = [standard_prompt.format(category) for category in categories]
        # standard_prompt = self.text_tokenize(standard_prompt).cuda()
        standard_prompt_encode = self.cache_text(standard_prompt) # num_classes, d
        return standard_prompt_encode



    def get_video_description(self, txt_encode, labels, index):
        # txt_encode: b, n_cls, d
        # labels: b
        # index: b
        video_description = [self.video_descriptions_list[i] for i in index] #b
        video_description = self.cache_text(video_description) # b, d

        for i in range(len(labels)):
            txt_encode[i, labels[i]] += video_description[i]
        
        return txt_encode # b, n_cls, d


    def _construct_network(self, cfg):

        context_length = cfg.MODEL.CONTEXT_LENGTH
        if cfg.MODEL.ARCH == 'vitb32':
            self.model, self.preprocess = load("ViT-B/32", jit=False, 
                    T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                    )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load("ViT-B/32", jit=False, 
                        T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=None,
                        use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                        num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                        record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                        )
                for name, p in self.raw_model.named_parameters():
                    p.requires_grad = False

        elif cfg.MODEL.ARCH == 'vitb16':
            self.model, self.preprocess = load("ViT-B/16", jit=False, 
                    T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                    )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load("ViT-B/16", jit=False, 
                        T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=None,
                        use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                        num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                        record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                        )
                for name, p in self.raw_model.named_parameters():
                    p.requires_grad = False
                
        elif cfg.MODEL.ARCH == 'vitl14':
            self.model, self.preprocess = load("ViT-L/14", jit=False, 
                    T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                    )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load("ViT-L/14", jit=False, 
                        T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=None,
                        use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                        num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                        record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                        )
                for name, p in self.raw_model.named_parameters():
                    p.requires_grad = False


        else:
            print("error loading arch")
            exit()
        self.model.float() 
        if cfg.MODEL.KEEP_RAW_MODEL:
            self.raw_model.float()
    
    

    def forward(self, x=None, labels=None, index=None):
        # shape of x(input) is (bz, channel, clip_len, h, w)
        # labels is the index of the class, len(labels) = bz
        # index is the index of the video, len(index) = bz

        assert len(x) == self.num_pathways
        x = x[0]
        if len(x.shape) == 4:
            # image input
            x = x.unsqueeze(2)
        
        # ensure eval state all the time, cost time ?
        if self.keep_raw_model:
            self.raw_model.eval()

        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)
         
       
        img_encode = self.model.encode_image(x)        
         
        if self.training:
            img_encode = img_encode.reshape(bz, clip_len, -1) #b,t,d
            img_encode = img_encode.mean(1) #b,d
            img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)
            txt_encode1 = self.txt_encode / self.txt_encode.norm(dim=-1, keepdim=True) # num_classes, d
            pred1 = torch.einsum('bd, nd->bn', img_encode, txt_encode1) # b, n_cls
            pred1 = self.model.logit_scale.exp() * pred1 # b, n_cls


            if self.cfg.APPLY_VIDEO_DESCRIPTION:
                txt_encode2 = self.origin_category_encode.unsqueeze(0).expand(bz, -1, -1)  #(b, n_cls, d)
                txt_encode2 = self.get_video_description(txt_encode2, labels, index) # b, n_cls, d
                txt_encode2 = txt_encode2 / txt_encode2.norm(dim=-1, keepdim=True) # num_classes, d

                pred2 = torch.einsum('bd,bnd->bn', img_encode, txt_encode2) # b, n_cls
                pred2 = self.model.logit_scale.exp() * pred2 # b, n_cls
            
                return pred1, pred2
            else:
                return pred1
            

        else:
            img_encode = img_encode.reshape(bz, clip_len, -1) #b,t,d
            img_encode = img_encode.mean(1) #b,d
            img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)
            txt_encode1 = self.txt_encode / self.txt_encode.norm(dim=-1, keepdim=True) # num_classes, d
            pred1 = torch.einsum('bd, nd->bn', img_encode, txt_encode1) # b, n_cls
            pred1 = self.model.logit_scale.exp() * pred1 # b, n_cls


            if self.cfg.APPLY_VIDEO_DESCRIPTION:
                txt_encode2 = self.origin_category_encode.unsqueeze(0).expand(bz, -1, -1)  #(b, n_cls, d)
                txt_encode2 = self.get_video_description(txt_encode2, labels, index) # b, n_cls, d
                txt_encode2 = txt_encode2 / txt_encode2.norm(dim=-1, keepdim=True) # num_classes, d

                pred2 = torch.einsum('bd,bnd->bn', img_encode, txt_encode2) # b, n_cls
                pred2 = self.model.logit_scale.exp() * pred2 # b, n_cls
            
                return pred1, pred2
            else:
                return pred1

    
    

class WCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 T=8,
                 temporal_modeling_type=None,
                 # other
                 use_checkpoint=False,
                 num_experts=0,
                 expert_insert_layers=[],
                 record_routing=False,
                 routing_type = 'patch-level'
                ):
        super().__init__(
                embed_dim,
                image_resolution, vision_layers, vision_width, vision_patch_size,
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
            )
        
        vision_heads = vision_width // 64
        self.visual = TemporalVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            T=T,
            temporal_modeling_type=temporal_modeling_type,
            use_checkpoint=use_checkpoint,
            num_experts=num_experts,
            expert_insert_layers=expert_insert_layers,
            record_routing = record_routing,
            routing_type = routing_type,
        )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(max(self.context_length, 77), transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.temporal_modeling_type = temporal_modeling_type
        
        
    # ignore. copy from videoX
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}
    
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def prompt_encode_text(self, prompts, tokenized_prompts,):
        prompts = prompts.type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)[:self.context_length, :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x 


def build_model(state_dict: dict, T=8, temporal_modeling_type=None, use_checkpoint=False,
                context_length=None, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
    else:
        raise NotImplementedError
    
    embed_dim = state_dict["text_projection"].shape[1]
    if context_length:
        context_length = context_length
    else:
        context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64

    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    model = WCLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            T=T, temporal_modeling_type=temporal_modeling_type,
            use_checkpoint=use_checkpoint, num_experts=num_experts,
            expert_insert_layers=expert_insert_layers,
            record_routing=record_routing,
            routing_type=routing_type,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    convert_weights(model)
    if num_experts > 0:
        for key in list(state_dict.keys()):
            if 'mlp' in key and key.startswith('visual'):
                for expert_id in range(num_experts):
                    if 'c_fc' in key or 'gelu' in key:
                        new_key = key.replace('mlp', 'experts_head.%d'%expert_id)
                    else:
                        new_key = key.replace('mlp', 'experts_tail.%d'%expert_id)
                    state_dict[new_key] = state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print(f"load pretrained CLIP: {msg}")

    return model.eval()



def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit:bool = False, download_root: str = None, T=8, temporal_modeling_type=False, use_checkpoint=False, context_length = 77, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
    
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
     
    model = build_model(state_dict or model.state_dict(), 
            T=T, temporal_modeling_type=temporal_modeling_type, 
            use_checkpoint=use_checkpoint, context_length = context_length,
            num_experts=num_experts, expert_insert_layers=expert_insert_layers,
            record_routing=record_routing, routing_type=routing_type
            ).to(device)
    if str(device) == "cpu":
        model.float()

    return model, _transform(model.visual.input_resolution)

if __name__ == '__main__':
    model, preprocess = clip.load("/share/home/jia/.cache/clip/ViT-B-32.pt", jit=False, )
    
    # model: text and vision





    

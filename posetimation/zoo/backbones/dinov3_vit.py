import math
import warnings
import os
import logging
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from utils.utils_registry import MODEL_REGISTRY
import collections.abc
from itertools import repeat

# 导入DINOv3相关模块
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../thirdparty/dinov3-main')))
from dinov3.models.vision_transformer import DinoVisionTransformer


def strip_prefix(state_dict, prefix):
    """移除状态字典中的前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        new_state_dict[key] = value
    return new_state_dict


@MODEL_REGISTRY.register()
class DINOv3ViT(nn.Module):
    def __init__(self, cfg=None, **kwargs):
        super(DINOv3ViT, self).__init__()
        self.cfg = cfg
        
        # DINOv3 backbone parameters
        img_size = cfg.MODEL.IMAGE_SIZE[1]  # H
        patch_size = cfg.MODEL.VIT.PATCH_SIZE
        in_chans = 3
        embed_dim = cfg.MODEL.VIT.EMB_DIM
        depth = cfg.MODEL.VIT.DEPTH
        num_heads = cfg.MODEL.VIT.NUM_HEADS
        drop_path_rate = cfg.MODEL.VIT.DROP_PATH_RATE
        ffn_ratio = cfg.MODEL.VIT.FFN_RATIO
        ffn_layer = cfg.MODEL.VIT.FFN_LAYER
        norm_layer = cfg.MODEL.VIT.NORM_LAYER

        # DINOv3特有参数
        pos_embed_rope_base = cfg.MODEL.VIT.POS_EMBED_ROPE_BASE
        pos_embed_rope_normalize_coords = cfg.MODEL.VIT.POS_EMBED_ROPE_NORMALIZE_COORDS
        pos_embed_rope_rescale_coords = cfg.MODEL.VIT.POS_EMBED_ROPE_RESCALE_COORDS
        pos_embed_rope_dtype = cfg.MODEL.VIT.POS_EMBED_ROPE_DTYPE
        layerscale_init = cfg.MODEL.VIT.LAYERSCALE_INIT
        n_storage_tokens = cfg.MODEL.VIT.N_STORAGE_TOKENS
        mask_k_bias = cfg.MODEL.VIT.MASK_K_BIAS

        self.backbone = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            ffn_ratio=ffn_ratio,
            ffn_layer=ffn_layer,
            norm_layer=norm_layer,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            layerscale_init=layerscale_init,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
        )
        
        self.pretrained = cfg.MODEL.PRETRAINED

        # Pose estimation head (deconvolution layers + final conv)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cfg.MODEL.VIT.DECONV1_IN_CHN, 
                              out_channels=cfg.MODEL.VIT.DECONV1_OUT_CHN, 
                              kernel_size=4, stride=2, padding=1,
                              output_padding=0, bias=False),
            nn.BatchNorm2d(cfg.MODEL.VIT.DECONV1_OUT_CHN),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=cfg.MODEL.VIT.DECONV2_IN_CHN, 
                              out_channels=cfg.MODEL.VIT.DECONV2_OUT_CHN, 
                              kernel_size=4, stride=2, padding=1,
                              output_padding=0, bias=False),
            nn.BatchNorm2d(cfg.MODEL.VIT.DECONV2_OUT_CHN),
            nn.ReLU(inplace=True),
        )
        final_layer = []
        final_layer.append(nn.Conv2d(in_channels=cfg.MODEL.VIT.FINAL_CONV_IN_CHN,
                                     out_channels=cfg.MODEL.VIT.FINAL_CONV_OUT_CHN,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0
                                    ))
        self.final_layer = final_layer[0]

    def init_weights(self, *args, **kwargs):
        """Initialize the weights in backbone."""
        logger = logging.getLogger(__name__)
        if os.path.isfile(self.pretrained):
            logger.info('=> loading pretrained DINOv3 model: {}'.format(self.pretrained))
            # Load DINOv3 pretrained weights
            state_dict = torch.load(self.pretrained, map_location='cpu')
            
            # DINOv3 state dict might have different keys, need to adapt
            # For DINOv3, the state_dict usually contains 'model' key
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove 'backbone.' prefix if present (from some DINOv3 checkpoints)
            state_dict = strip_prefix(state_dict, "backbone.")
            
            model_state_dict = self.backbone.state_dict()
            
            # Filter out keys that do not match
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_state_dict and v.shape == model_state_dict[k].shape}
            
            if len(pretrained_dict) == 0:
                logger.warning("No matching keys found in DINOv3 pretrained model. Loading might fail or be incomplete.")
            
            model_state_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_state_dict, strict=False)
            logger.info('=> DINOv3 pretrained model loaded successfully.')
        elif self.pretrained:
            logger.warning('=> DINOv3 pretrained model file not found: {}'.format(self.pretrained))
            # raise ValueError('{} is not exist!'.format(self.pretrained))

    @torch.jit.ignore
    def no_weight_decay(self):
        """不需要权重衰减的参数"""
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        """前向传播特征提取"""
        B, C, H, W = x.shape
        
        # 使用DINOv3 backbone提取特征
        # DINOv3会自动处理位置编码插值，支持不同输入尺寸
        features = self.backbone.forward_features(x)
        
        # 获取patch tokens（去掉cls token）
        x = features['x_norm_patchtokens']  # [B, N, D]
        
        # 重塑为空间维度
        # 计算patch数量
        patch_size = self.backbone.patch_size
        Hp = H // patch_size
        Wp = W // patch_size
        
        # 重塑为 [B, D, Hp, Wp]
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        
        # 通过反卷积层
        xp = self.deconv_layers(xp)
        
        # 最终卷积层
        xp = self.final_layer(xp)
        
        return x, xp

    def forward(self, x):
        """前向传播"""
        _, xp = self.forward_features(x)
        return xp

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = DINOv3ViT(cfg=cfg)
        return model

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        hyper_parameters_setting = "dinov3"
        return hyper_parameters_setting

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
from collections import OrderedDict

from .base import BaseModel

import time
import logging
import os
import math
import random
from functools import partial

from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .base import BaseModel
from .backbones.vit import ViT , PatchEmbed
from .fusionnet import Temporal_Feat_Fusion_Module
from .decoder import get_LRS
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
logger = logging.getLogger(__name__)

__all__ = ["SLDPOSE"]



@MODEL_REGISTRY.register()
class SLDPOSE(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = SLDPOSE(cfg, phase, **kwargs)
        return model

    def exists(self, x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d

    def __init__(self, cfg, phase, **kwargs):
        super().__init__()

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.heatmap_h = cfg.MODEL.HEATMAP_SIZE[1]
        self.heatmap_w = cfg.MODEL.HEATMAP_SIZE[0]
        self.use_supp_targ_num = cfg.MODEL.USE_SUPP_TARG_NUM
        self.use_supp_direct = cfg.MODEL.USE_SUPP_DIRECT
        self.decoder_dim = 768
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS

        self.backbone = ViT.get_net(cfg , phase)

        self.TFF = Temporal_Feat_Fusion_Module(cfg)

        self.decoder = get_LRS(cfg,emded_dim = self.decoder_dim)


        self.htms_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 3, 64, 2)

        self.patch_embed = PatchEmbed(
                img_size=(64,48), patch_size=4, in_chans=64 * 2, embed_dim=self.decoder_dim, ratio=1)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , self.decoder_dim))
        self.head = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.decoder_dim, out_channels=384, kernel_size=4, stride=2, padding=1,
                                       output_padding=0, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=384, out_channels=128, kernel_size=4, stride=2, padding=1,
                                       output_padding=0, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128,out_channels=17,kernel_size=1,stride=1,padding=0 )
        )



        logger.info("=>Backbone 的参数量为:{}".format(sum(p.numel() for p in self.backbone.parameters())))
        logger.info("=>TFF 的参数量为:{}".format(sum(p.numel() for p in self.TFF.parameters())))
        logger.info("=>LRS 的参数量为:{}".format(sum(p.numel() for p in self.decoder.parameters())))



    def forward(self, input ,  **kwargs):
        true_batchsize , channel_num = input.shape[0] , input.shape[1]//3
        assert "margin" in kwargs
        margin = kwargs["margin"]

        frame_curr, frame_supp_A, frame_supp_B = input.split(channel_num, dim=1)
        img_cat_batch= torch.cat((frame_curr, frame_supp_A, frame_supp_B), dim=0)

        feat_cat_batch , htms_cat_batch = self.backbone(img_cat_batch)
        htms_curr, htms_supp_A, htms_supp_B = htms_cat_batch.split(true_batchsize, dim=0)


        if "heatmaps_sup_cat" in kwargs:
            targ_htms_supp_cat_chn=kwargs["heatmaps_sup_cat"]
            targ_supp_A, targ_supp_B = targ_htms_supp_cat_chn.split(self.num_joints, dim=1)
            if self.use_supp_targ_num == 2:
                htms_supp_A, htms_supp_B = targ_supp_A, targ_supp_B
            elif self.use_supp_targ_num == 1:
                if self.use_supp_direct == 'left':
                    htms_supp_A = targ_supp_A
                if self.use_supp_direct == 'right':
                    htms_supp_B = targ_supp_B


        feat_fused = self.TFF(feat_cat_batch)

        htms_cat = torch.cat((htms_curr, htms_supp_A, htms_supp_B), dim=1)
        htms_merg = self.htms_merg_conv(htms_cat)

        diff_left = htms_curr - htms_supp_A
        diff_right = htms_curr - htms_supp_B

        diff_left_summed = diff_left.sum(dim=1, keepdim=True)
        diff_right_summed = diff_right.sum(dim=1, keepdim=True)

        diff_sigmoid = torch.sigmoid(diff_left_summed) + torch.sigmoid(diff_right_summed )

        mask = diff_sigmoid.repeat(1, 64, 1, 1)

        htms_masked = htms_merg * mask

        x = torch.cat((htms_merg ,htms_masked) , dim=1)


        x , (Hp, Wp) = self.patch_embed(x)
        x = x + self.pos_embed
        out = self.decoder(x , feat_fused)
        out = out.permute(0, 2, 1).reshape(true_batchsize, -1, Hp, Wp)
        out = self.head(out)

        return out





    def init_weights(self):
        logger = logging.getLogger(__name__)

        for module_name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        self.backbone.init_weights()
        if self.freeze_backbone_weights:
            self.backbone.freeze_weight()


    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):

        hyper_parameters_setting = "ViT_TFF_LRS"
        return hyper_parameters_setting



if __name__ == "__main__":
    print("!!!")




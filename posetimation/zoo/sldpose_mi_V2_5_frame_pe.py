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
from .backbones.vit import ViT , PatchEmbed , trunc_normal_
from .fusionnet import Temporal_Feat_Fusion_Module
from .decoder import get_LRS
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
from utils.vis import vis_htms
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
logger = logging.getLogger(__name__)

__all__ = ["SLDPOSE_MI_V2_5_FRAME_PE"]





def modified_sigmoid(x, k=1.5, theta = 0.5):
    return  1 / (1 + torch.exp(-k * (torch.abs(x) - theta)))

# def modified_sigmoid(x, k = 1.5, theta = 0.5 ):
#
#     x = torch.where(x != 0, 1 / (torch.exp(-k * (torch.abs(x) - theta)) + 1), x)
#     return x




@MODEL_REGISTRY.register()
class SLDPOSE_MI_V2_5_FRAME_PE(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = SLDPOSE_MI_V2_5_FRAME_PE(cfg, phase, **kwargs)
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
        self.supp_num = 4
        self.supp_weight = 1
        # self.htms_merg_style = 'single-all'
        self.htms_merg_style = 'all-cat'
        self.htms_dict = dict()
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS

        self.backbone = ViT.get_net(cfg , phase)

        self.TFF = Temporal_Feat_Fusion_Module(cfg , frame=5 , dim = cfg.MODEL.VIT.EMB_DIM)

        self.decoder = get_LRS(cfg,emded_dim = self.decoder_dim,feat_dim=cfg.MODEL.VIT.EMB_DIM)

        if self.htms_merg_style == 'single-all':
            self.htms_temp_keypts_merg_conv = nn.ModuleList([
                CHAIN_RSB_BLOCKS(5 , 1 , 1)
                for i in range(self.num_joints)])
            self.htms_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 1  , 64 , 1 )

        elif self.htms_merg_style == 'all-cat':
            self.htms_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 5, 64, 1)

        self.htms_merg_mask_comb_conv = CHAIN_RSB_BLOCKS(64*2, 64*2, 1)

        self.patch_embed = PatchEmbed(
                img_size=(64,48), patch_size = 4 , in_chans= 64*2 , embed_dim = self.decoder_dim, ratio=1)
        num_patches = self.patch_embed.num_patches
        self.pos_embed_htms = nn.Parameter(torch.zeros(1, num_patches , self.decoder_dim))
        # self.pos_embed_feat = nn.Parameter(torch.zeros(1, (cfg.MODEL.IMAGE_SIZE[1]/16*cfg.MODEL.IMAGE_SIZE[0]/16) , cfg.MODEL.VIT.EMB_DIM))
        trunc_normal_(self.pos_embed_htms, std=.02)
        # trunc_normal_(self.pos_embed_feat, std=.02)
        self.head = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.decoder_dim, out_channels=256, kernel_size=4, stride=2, padding=1,
                                       output_padding=0, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1,
                                       output_padding=0, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels = 128,out_channels=17,kernel_size=1,stride=1,padding=0 )
        )

        logger.info(
            "=> Number of parameters in the Backbone: {}".format(sum(p.numel() for p in self.backbone.parameters())))
        logger.info("=> Number of parameters in the TFF: {}".format(sum(p.numel() for p in self.TFF.parameters())))
        logger.info("=> Number of parameters in the LRS: {}".format(sum(p.numel() for p in self.decoder.parameters())))



    def forward(self, input ,  **kwargs):
        true_batchsize , channel_num = input.shape[0] , input.shape[1] // (self.supp_num+1)


        frame_list = input.split(channel_num, dim=1)
        img_cat_batch= torch.cat((frame_list), dim=0)

        feat_cat_batch , htms_cat_batch = self.backbone(img_cat_batch)
        htms_curr, htms_supp_A, htms_supp_B , htms_supp_C, htms_supp_D= htms_cat_batch.split(true_batchsize, dim=0)
        feat_cuur = feat_cat_batch[0:true_batchsize,:,:]

        #temporal feature fusion-> fuse three feature to one fused feature
        feat_fused = self.TFF(feat_cat_batch)


        #heatmaps merger->firstly single keypoint temporal heatmap merger, then all keypoints spatially merger

        if self.htms_merg_style == 'single-all':
            htms_curr_list = htms_curr.split(1, dim=1)
            htms_supp_A_list = htms_supp_A.split(1, dim=1)
            htms_supp_B_list = htms_supp_B.split(1, dim=1)
            htms_supp_C_list = htms_supp_C.split(1, dim=1)
            htms_supp_D_list = htms_supp_D.split(1, dim=1)
            keypts_htms_list = []
            for joint_index in range(self.num_joints):
                keypoint_temp_cat = torch.cat((htms_curr_list[joint_index],
                                               htms_supp_A_list[joint_index] * self.supp_weight,
                                               htms_supp_B_list[joint_index] * self.supp_weight,
                                               htms_supp_C_list[joint_index] * self.supp_weight,
                                               htms_supp_D_list[joint_index] * self.supp_weight),
                                              dim=1)
                keypoint_temp_merg = self.htms_temp_keypts_merg_conv[joint_index](keypoint_temp_cat)
                keypts_htms_list.append(keypoint_temp_merg)

            htms_merg = self.htms_merg_conv(torch.cat(keypts_htms_list, dim=1)).cuda()

        elif self.htms_merg_style == 'all-cat':
            htms_cat = torch.cat((htms_curr, htms_supp_A * self.supp_weight, htms_supp_B * self.supp_weight,
                                  htms_supp_C * self.supp_weight, htms_supp_D * self.supp_weight), dim=1)
            htms_merg = self.htms_merg_conv(htms_cat)




        diff_A = htms_curr - htms_supp_A
        diff_B = htms_curr - htms_supp_B
        diff_C = htms_curr - htms_supp_C
        diff_D = htms_curr - htms_supp_D
        diff_A_sigm  = modified_sigmoid(diff_A)
        diff_B_sigm  = modified_sigmoid(diff_B)
        diff_C_sigm  = modified_sigmoid(diff_C)
        diff_D_sigm  = modified_sigmoid(diff_D)
        diff_A_summed = diff_A_sigm.sum(dim=1, keepdim=True)
        diff_B_summed = diff_B_sigm.sum(dim=1, keepdim=True)
        diff_C_summed = diff_C_sigm.sum(dim=1, keepdim=True)
        diff_D_summed = diff_D_sigm.sum(dim=1, keepdim=True)

        diff_comb = (diff_A_summed + diff_B_summed + diff_C_summed + diff_D_summed ) * self.supp_weight
        mask = F.softmax(diff_comb , dim=-1)
        # mask_shape = diff_comb.shape
        # mask = F.softmax(diff_comb.reshape(true_batchsize,self.num_joints,-1) , dim=-1).reshape(mask_shape)
        htms_masked = htms_merg * mask


        htms_merg_mask_cat = torch.cat((htms_merg, htms_masked), dim=1)
        htms_merg_mask_cat = self.htms_merg_mask_comb_conv(htms_merg_mask_cat)




        x , (Hp, Wp) = self.patch_embed(htms_merg_mask_cat)
        x = x + self.pos_embed_htms
        # feat_fused = feat_fused + self.pos_embed_feat
        out = self.decoder(x , feat_fused)
        out = out.permute(0, 2, 1).reshape(true_batchsize, -1, Hp, Wp)
        out = self.head(out)


        # self.htms_dict={
        #     'diff_left':diff_left,
        #     'diff_right':diff_right ,
        #     'diff_left_sigm':diff_left_sigm ,
        #     'diff_right_sigm':diff_right_sigm ,
        #     'diff_left_summed':diff_left_summed ,
        #     'diff_right_summed':diff_right_summed ,
        #     'diff_comb':diff_comb ,
        #     'mask':mask ,
        #
        # }
        return out , feat_cuur , feat_fused , htms_curr , htms_merg_mask_cat





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

        # hyper_parameters_setting = "MI_ViT_TFF_LRS_T_" + str(cfg.DATASET.TIME_INTERVAL)
        hyper_parameters_setting = "ViT_TFF_LRS"
        return hyper_parameters_setting



if __name__ == "__main__":
    print("!!!")




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
from collections import OrderedDict

from ..base import BaseModel

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


from ..backbones.vit_v1 import ViT_V1 , PatchEmbed , trunc_normal_
from ..fusionnet import Temporal_Feat_Fusion_Module
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
from ..base import BaseModel
from ..backbones.vit import Block , trunc_normal_
from ..layers.block_zoo import *
from ..layers.rope import *
from ..layers.spatiotemporal_attention import *
from .model import *
from ..decoder_v2 import CrossBlock
# from utils.vis import vis_htms
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
logger = logging.getLogger(__name__)

__all__ = ["MambaPose"]







@MODEL_REGISTRY.register()
class MambaPose(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = MambaPose(cfg, phase, **kwargs)
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
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS
        self.token_num = int((cfg.MODEL.IMAGE_SIZE[1] / cfg.MODEL.VIT.PATCH_SIZE) * (
                cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.VIT.PATCH_SIZE))
        self.map_size_up_1x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE)
        self.map_size_up_2x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 2, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 2)
        self.map_size_up_4x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 4, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 4)

        self.frame_num = cfg.MODEL.FRAME_NUM

        self.multi_frame_fusion = True
        # self.multi_frame_fusion = True


        self.htms_dict = dict()


        self.backbone = ViT_V1.get_net(cfg , phase)

        self.add_pos_emb = cfg.MODEL.EXTRA.ADD_POS_EMB
        self.pos_emb_list = []
        if self.add_pos_emb:
            for i in range(self.frame_num):
                pos_emb = nn.Parameter(torch.zeros(1, self.token_num, cfg.MODEL.VIT.EMB_DIM)).cuda()
                nn.init.trunc_normal_(pos_emb, std=0.02)
                self.pos_emb_list.append(pos_emb)


        # 加位置编码 True：
        #待调参数： 层数；head；mlp_ratio；
        #待加模块：是否用mlp ； 是否用旋转嵌入 ； 是否用rmsnorm
        #所有待加的操作都是false，即最初始的模型版本  layer=2,mlp=false (最好是87.39，后面貌似是过拟合了一直在86.几)  layer=4 , mlp=false (最好的还是87.3，依旧是过拟合了)
        #仅加上mlp： layer=2,mlp=true (依旧过拟合)   layer=4 , mlp=true ()
        #在mamba和temporal attn 前面加上空间注意力块 layer = 2 ratio=2  head = 8
        #只留下空间注意力块，其余的全部去掉，试试还会不会过拟合。依旧过拟合！！！
        #两个改进：给drop设置0。1和 0.2；再最后一层前和backbone的feat融合（已经不过拟合了）
        #现在 将drop全部恢复到0，保留最后的融合，试试（依旧不过拟合，这说明了主要功劳在于最后的融合）
        #将pre 空间注意力块去掉试试，只留后面的mamba（效果不如加上空间注意力，所以，还是把空间注意力加上？）
        #空间注意力+mamba，开始调参数：（1）MLP——ratio = 4 （87.8792） （2） layer = 4 (87.99目前最好的) (3) d_state = 64 expand = 2,use_mamba2 , layer = 4（87.93）
        #use_mamba2 ,layer =2 ,mlp = 4 (88.05)
        #将开始的空间注意力替换为mamba（87.90） ； 再开始的mamba后加上mlp,加上norm （87.96） (等于是，目前做的事：空间注意力（88.05）；mamba（87.90）；mamba+mlp+norm（87.96）)
        #换成5帧了，试试（87.988）
        #还想尝试开始空间注意力+mlp+norm 这里是五帧 v1: 88.0658
        #5帧 v2: 将时序注意力换成了时序mamba，使用pre spatial mamba ，87.98

        args = ModelArgs(d_model=cfg.MODEL.VIT.EMB_DIM, n_layer=2, d_state=64, expand=2, use_mlp=True, num_heads=8,mlp_ratio = 4 ,
                                                                      use_rotary=False  ,use_rmsnorm=False ,
                         use_spatial_attn = False,   use_start_spatial_attn = False, drop_mamba = False, use_mamba2=True,use_start_spatial_mamba = True ,
                                                                       frame=self.frame_num)
        self.mamba_blocks = Mamba_blocks(args)

        if self.multi_frame_fusion:
            self.fusion_mlp = nn.Sequential(
                    nn.Linear(self.token_num * self.frame_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                )

        self.out_backbone_fusion = CHAIN_RSB_BLOCKS(cfg.MODEL.VIT.DECONV2_OUT_CHN+cfg.MODEL.HEAD.DECONV2_OUT_CHN,
                                                    cfg.MODEL.HEAD.FINAL_CONV_IN_CHN , 2)


        self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cfg.MODEL.HEAD.DECONV1_IN_CHN,
                                   out_channels=cfg.MODEL.HEAD.DECONV1_OUT_CHN, kernel_size=4, stride=2, padding=1,
                                   output_padding=0, bias=False),
                nn.BatchNorm2d(cfg.MODEL.HEAD.DECONV1_OUT_CHN),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=cfg.MODEL.HEAD.DECONV2_IN_CHN,
                                   out_channels=cfg.MODEL.HEAD.DECONV2_OUT_CHN, kernel_size=4, stride=2, padding=1,
                                   output_padding=0, bias=False),
                nn.BatchNorm2d(cfg.MODEL.HEAD.DECONV2_OUT_CHN),
                nn.ReLU(inplace=True),

            )
        self.final_layer = nn.Conv2d(in_channels=cfg.MODEL.HEAD.FINAL_CONV_IN_CHN, out_channels=cfg.MODEL.HEAD.FINAL_CONV_OUT_CHN,
                          kernel_size=1, stride=1, padding=0)

        param_count = {}
        for name, module in self.mamba_blocks.named_modules():
            # 收集模块的参数
            params = list(module.parameters())
            # 累加参数量
            param_count[name] = sum(p.numel() for p in params)

        # 打印每个模块的参数量
        for name, count in param_count.items():
            print(f"{name}: {count}")

    def forward(self, input ,  **kwargs):
        true_batchsize , channel_num = input.shape[0] , input.shape[1]//self.frame_num



        img_cat_batch= torch.cat(input.split(channel_num, dim=1), dim=0)

        #vit_backbone forward->outputs are two parts: feature and heatmaps
        htms_backbone_out_cat_batch , feats_backbone_out_cat_batch , feats_up1_backbone_out_cat_batch , feats_up2_backbone_out_cat_batch , \
                                                                                        = self.backbone(img_cat_batch)

        feat_up2_backbone_out_curr = feats_up2_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]

        if self.frame_num==3:
            htms_curr, htms_supp_A, htms_supp_B = htms_backbone_out_cat_batch.split(true_batchsize, dim=0)
        elif self.frame_num ==5:
             htms_curr, htms_supp_A, htms_supp_B, htms_supp_C, htms_supp_D = htms_backbone_out_cat_batch.split(true_batchsize, dim=0)

        feats_backbone_out_list = list(feats_backbone_out_cat_batch.split(true_batchsize, dim=0))
        # if self.add_pos_emb:
        #     for i in range(self.frame_num):
        #         feats_backbone_out_list[i] = feats_backbone_out_list[i] + self.pos_emb_list[i]

        feats_backbone_out_cat_token = torch.cat(feats_backbone_out_list,dim=1)


        feats_backbone_out_cat_token = self.mamba_blocks(feats_backbone_out_cat_token)

        if self.multi_frame_fusion:

            feats_mamba_out = self.fusion_mlp(feats_backbone_out_cat_token.permute(0,2,1  ) )

        else:
            feats_mamba_out =  feats_backbone_out_cat_token.split(self.token_num, dim=1)[0].permute(0,2,1  )



        feats_mamba_out = self.up_sample(feats_mamba_out.reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1]))

        out_backbone_cat = torch.cat((feats_mamba_out,feat_up2_backbone_out_curr), dim=1)
        out = self.out_backbone_fusion(out_backbone_cat)
        out = self.final_layer(out)




        return out, htms_curr








        # if "heatmaps_sup_cat" in kwargs:
        #     targ_htms_supp_cat_chn=kwargs["heatmaps_sup_cat"]
        #     targ_supp_A, targ_supp_B = targ_htms_supp_cat_chn.split(self.num_joints, dim=1)
        #     if self.use_supp_targ_num == 2:
        #         htms_supp_A, htms_supp_B = targ_supp_A, targ_supp_B
        #     elif self.use_supp_targ_num == 1:
        #         if self.use_supp_direct == 'left':
        #             htms_supp_A = targ_supp_A
        #         if self.use_supp_direct == 'right':
        #             htms_supp_B = targ_supp_B




        # self.htms_dict={
        #     # 'diff_left':diff_left,
        #     # 'diff_right':diff_right ,
        #     # 'diff_left_sigm':diff_left_sigm ,
        #     # 'diff_right_sigm':diff_right_sigm ,
        #     # 'diff_left_summed':diff_left_summed ,
        #     # 'diff_right_summed':diff_right_summed ,
        #     # 'diff_comb':diff_comb ,
        #     'mask':mask ,
        #     'htms_curr':htms_curr ,
        #     'htms_supp_A':htms_supp_A ,
        #     'htms_supp_B':htms_supp_B ,
        #     'final_htms': out,
        # }

        # return out , feat_cat_batch[0:true_batchsize, :, :] , feat_fused , htms_curr , htms_merg_mask_cat





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

        hyper_parameters_setting = "ViT_Mamba"
        return hyper_parameters_setting



if __name__ == "__main__":

    print("!!!")




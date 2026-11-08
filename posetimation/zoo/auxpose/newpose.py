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
# from utils.vis import vis_htms
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
logger = logging.getLogger(__name__)

__all__ = ["NEWPOSE_V1"]





@MODEL_REGISTRY.register()
class NEWPOSE_V1(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = NEWPOSE_V1(cfg, phase, **kwargs)
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
        self.multi_scale_streams = cfg.MODEL.STREAMS

        self.htms_dict = dict()

        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS

        self.backbone = ViT_V1.get_net(cfg , phase)

        if self.multi_scale_streams == 1:
            self.stream1 = Temporal_Feat_Fusion_Module(cfg, 3, cfg.MODEL.VIT.EMB_DIM)
        elif self.multi_scale_streams == 2:

            self.human_token = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.VIT.EMB_DIM))
            trunc_normal_(self.human_token, std=.02)

            self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_joints, cfg.MODEL.VIT.EMB_DIM))
            trunc_normal_(self.keypoint_token, std=.02)

            dpr = [x.item() for x in torch.linspace(0, 0.06, 2)]
            self.blocks_human = nn.ModuleList([
                Block(
                    dim=cfg.MODEL.VIT.EMB_DIM, num_heads=16, mlp_ratio= 4 , qkv_bias=True, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=dpr[i],
                )
                for i in range( 2 )])

            self.blocks_keypoint = nn.ModuleList([
                Block(
                    dim=cfg.MODEL.VIT.EMB_DIM, num_heads=16, mlp_ratio = 4, qkv_bias=True, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=dpr[i],
                )
                for i in range( 2 )])
            self.pos_emb_keypoint_and_visual = nn.Parameter(torch.zeros(1, self.num_joints + 192, cfg.MODEL.VIT.EMB_DIM))
            trunc_normal_(self.pos_emb_keypoint_and_visual , std=.02)
            self.scale = cfg.MODEL.VIT.EMB_DIM ** -0.5


            self.stream1 = Temporal_Feat_Fusion_Module(cfg, 3, cfg.MODEL.VIT.EMB_DIM)



            self.patch_embed1 = PatchEmbed( img_size=(self.heatmap_h / 2  , self.heatmap_w / 2), patch_size = 2 ,
                                            in_chans = 256 , embed_dim = cfg.MODEL.VIT.EMB_DIM, ratio=1)

            # self.stream2 = Temporal_Feat_Fusion_Module(cfg, 3, cfg.MODEL.VIT.EMB_DIM)






        elif self.multi_scale_streams == 3:
            self.stream1 = Temporal_Feat_Fusion_Module(cfg, 3, cfg.MODEL.VIT.EMB_DIM)
            self.patch_embed1 = PatchEmbed( img_size=(self.heatmap_h / 2  , self.heatmap_w / 2), patch_size = 2 , in_chans = 256 , embed_dim = cfg.MODEL.VIT.EMB_DIM, ratio=1)
            self.stream2 = Temporal_Feat_Fusion_Module(cfg, 3, cfg.MODEL.VIT.EMB_DIM)
            self.patch_embed1 = PatchEmbed( img_size=(self.heatmap_h / 4 , self.heatmap_w / 4), patch_size = 4 , in_chans = 128 , embed_dim = cfg.MODEL.VIT.EMB_DIM, ratio=1)
            self.stream3 = Temporal_Feat_Fusion_Module(cfg, 3, cfg.MODEL.VIT.EMB_DIM)


        # self.htms_merg_mask_comb_conv = CHAIN_RSB_BLOCKS(64  * 2 , 64 * 2 , 1 )


        # self.patch_embed = PatchEmbed(
        #         img_size=(self.heatmap_h , self.heatmap_w), patch_size = self.decoder_patch_size , in_chans = 64 * 2 , embed_dim = self.decoder_dim, ratio=1)
        # num_patches = self.patch_embed.num_patches
        # self.pos_embed_htm = nn.Parameter(torch.zeros(1, num_patches , self.decoder_dim))
        # trunc_normal_(self.pos_embed_htm, std=.02)



        self.head = nn.Sequential(
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
            nn.Conv2d(in_channels=cfg.MODEL.HEAD.FINAL_CONV_IN_CHN, out_channels=cfg.MODEL.HEAD.FINAL_CONV_OUT_CHN,
                      kernel_size=1, stride=1, padding=0)
        )




        logger.info(
            "=> Number of parameters in the Backbone: {}".format(sum(p.numel() for p in self.backbone.parameters())))
        logger.info("=> Number of parameters in the stream1: {}".format(sum(p.numel() for p in self.stream1.parameters())))
        # logger.info("=> Number of parameters in the LRS: {}".format(sum(p.numel() for p in self.decoder.parameters())))



    def forward(self, input ,  **kwargs):
        true_batchsize , channel_num = input.shape[0] , input.shape[1]//3
        assert "margin" in kwargs
        margin = kwargs["margin"]

        interval = torch.sum(margin, dim=1, keepdim=True)
        margin = torch.div(margin.float(), interval.float())
        prev_weight, next_weight = margin[:, 1], margin[:, 0]
        prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shape = [1, 1 , self.heatmap_h,self.heatmap_w]
        prev_weight, next_weight = prev_weight.repeat(shape), next_weight.repeat(shape)

        frame_curr, frame_supp_A, frame_supp_B = input.split(channel_num, dim=1)
        img_cat_batch= torch.cat((frame_curr, frame_supp_A, frame_supp_B), dim=0)

        #vit_backbone forward->output is two: feature and heatmaps
        htms_backbone_out_cat_batch , feats_backbone_out_cat_batch , feats_up1_backbone_out_cat_batch , feats_up2_backbone_out_cat_batch , = self.backbone(img_cat_batch)
        htms_curr, htms_supp_A, htms_supp_B = htms_backbone_out_cat_batch.split(true_batchsize, dim=0)

        if self.multi_scale_streams == 1:
            feat_fused_1 = self.stream1(feats_backbone_out_cat_batch)
            feat_fused_1_shape = feat_fused_1.permute(0, 2, 1).reshape(true_batchsize, -1, 16 , 12)
            out = self.head(feat_fused_1_shape)
            return out , htms_curr

        if self.multi_scale_streams == 2:
            # ------ human token -------
            human_token = self.human_token.repeat(true_batchsize * 3, 1, 1)
            feat_fused_1 = torch.cat((human_token , feats_backbone_out_cat_batch), dim=1)
            for blk in self.blocks_human:
                feat_fused_1 = blk(feat_fused_1)
            human_token = feat_fused_1[:, 0:1 , : ]
            visual_token = feat_fused_1[:, 1: , : ]
            human_token = human_token.permute(0, 2, 1)
            mask_human = visual_token @ human_token  #(B*3,192,1)
            visual_token = visual_token * mask_human #广播机制(B*3,192,1024)


            # ------ keypoint token -------
            # 先扩充keypoint token-->拼起来-->再进行pos_embedding
            keypoint_token = self.keypoint_token.repeat(true_batchsize * 3, 1, 1)
            feat_fused_1 = torch.cat((keypoint_token , visual_token), dim=1)
            pos_emb_keypoint_and_visual = self.pos_emb_keypoint_and_visual.repeat(true_batchsize * 3, 1, 1)
            feat_fused_1 = feat_fused_1 + pos_emb_keypoint_and_visual
            # 先从batch维度拆开-->从token维度拼起来-->计算各类token的num
            feat_fused_1_curr , feat_fused_1_pre , feat_fused_1_next = feat_fused_1.split(true_batchsize, dim=0)
            feat_fused_1_cat = torch.cat((feat_fused_1_curr, feat_fused_1_pre, feat_fused_1_next), dim=1)
            token_num_one_frame = feat_fused_1_curr.shape[1]
            token_num_one_frame_keypoint = self.num_joints
            token_num_one_frame_visual = token_num_one_frame - token_num_one_frame_keypoint

            # 先进行Transformer block 计算-->再从token维度拆开得到三帧的 keypoint+visual token
            for blk in self.blocks_keypoint:
                feat_fused_1_cat = blk(feat_fused_1_cat)
            feat_fused_1_curr, feat_fused_1_pre, feat_fused_1_next = feat_fused_1_cat.split(token_num_one_frame, dim=1)

            # 每帧都从token维度拆开得到 keypoint、visual token
            feat_fused_1_curr_keypoint_token , feat_fused_1_curr_visual_token = feat_fused_1_curr[:, 0:self.num_joints , : ], feat_fused_1_curr[:, self.num_joints: , : ]
            feat_fused_1_pre_keypoint_token , feat_fused_1_pre_visual_token = feat_fused_1_pre[:, 0:self.num_joints , : ], feat_fused_1_pre[:, self.num_joints: , : ]
            feat_fused_1_next_keypoint_token , feat_fused_1_next_visual_token = feat_fused_1_next[:, 0:self.num_joints , : ], feat_fused_1_next[:, self.num_joints: , : ]
            # 将不同帧的keypoint token 和 不同帧的visual token 拼接起来
            feat_fused_1_keypoint_token_cat = torch.cat((feat_fused_1_curr_keypoint_token, feat_fused_1_pre_keypoint_token, feat_fused_1_next_keypoint_token), dim=1)
            feat_fused_1_visual_token_cat = torch.cat((feat_fused_1_curr_visual_token, feat_fused_1_pre_visual_token, feat_fused_1_next_visual_token), dim=1)
            # 将keypoint token 矩阵乘 visual token 的转置-->进行softmax-->得到confidence map-->压缩第一维度将17*3压缩至1-->调换后两维得到mask
            keypoint_confidence_map = feat_fused_1_keypoint_token_cat @ feat_fused_1_visual_token_cat.permute(0, 2, 1) # (B,17*3,192*3)
            keypoint_confidence_map = F.softmax(keypoint_confidence_map, dim=-1) * self.scale
            mask_keypoint = keypoint_confidence_map.sum(dim=1, keepdim=True) # (B,1,192*3)
            mask_keypoint = mask_keypoint.permute(0, 2, 1) # (B,192*3,1)
            # 将mask 广播到 visual token 维度-->进行mask-->将mask后的visual token 先从1维度拆开，再从batch维度拼接
            feat_fused_1_visual_token_cat = feat_fused_1_visual_token_cat * mask_keypoint # (B,192*3,1024)
            feat_fused_1_visual_token_cat_batch = torch.cat(feat_fused_1_visual_token_cat.split(token_num_one_frame_visual , dim = 1) , dim = 0) # (B*3,192,1024)
            #丢入TFF
            feat_fused_1 = self.stream1(feat_fused_1_visual_token_cat_batch)



            feat_fused_2 , (hp , wp)= self.patch_embed1(feats_up1_backbone_out_cat_batch)
            feat_fused_2 = self.stream1(feat_fused_2)


            feat_fused_add = feat_fused_1 + feat_fused_2

            feat_fused_add_shape = feat_fused_add.permute(0, 2, 1).reshape(true_batchsize, -1, hp, wp)
            out = self.head(feat_fused_add_shape)
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





        # x , (Hp, Wp) = self.patch_embed(htms_merg_mask_cat)
        # x = x + self.pos_embed_htm
        # out = self.decoder(x , feat_fused)
        # out = out.permute(0, 2, 1).reshape(true_batchsize, -1, Hp, Wp)
        # out = self.head(out)


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

        hyper_parameters_setting = f"MI_ALPHA_{str(cfg.TRAIN.MI_ALPHA)}" \
                                   f"_BETA_{str(cfg.TRAIN.MI_BETA)}" \
                                   f"_GAMMA_{str(cfg.TRAIN.MI_GAMMA)}" \
                                   f"_ViT_TFF_LRS"
        # hyper_parameters_setting = "ViT_TFF_LRS"
        return hyper_parameters_setting



if __name__ == "__main__":
    print("!!!")




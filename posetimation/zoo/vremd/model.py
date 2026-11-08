from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



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

from ..base import BaseModel
from ..layers.block_zoo import *
from ..layers.spatiotemporal_attention import *
from posetimation.layers import CHAIN_RSB_BLOCKS
logger = logging.getLogger(__name__)

__all__ = ["VREMD"]







@MODEL_REGISTRY.register()
class VREMD(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = VREMD(cfg, phase, **kwargs)
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
        self.difference_type = cfg.MODEL.EXTRA.DIFFERENCE_TYPE
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS
        self.token_num = int((cfg.MODEL.IMAGE_SIZE[1] / cfg.MODEL.VIT.PATCH_SIZE) * (
                cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.VIT.PATCH_SIZE))
        self.map_size_up_1x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE)
        self.map_size_up_2x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 2, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 2)
        self.map_size_up_4x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 4, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 4)

        self.frame_num = cfg.MODEL.FRAME_NUM

        self.order_type = cfg.MODEL.ORDER_TYPE
        self.use_doform_cross = cfg.MODEL.EXTRA.USE_DEFOEM_CROSS
        self.use_diff_final_fusion_conv = cfg.MODEL.EXTRA.USE_DIFF_FINAL_FUSION
        self.use_one_st = cfg.MODEL.EXTRA.USE_ONE_ST
        self.use_keypoint_generate_htms = cfg.MODEL.EXTRA.USE_KEYPOINT_GENERATE_HTMS

        self.directly_use_visual_tokens = False
        self.dim = cfg.MODEL.VIT.EMB_DIM


        self.backbone = ViT_V1.get_net(cfg , phase)


        dpr = [x.item() for x in torch.linspace(0, 0.06, 2)]



        self.human_token = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.VIT.EMB_DIM))
        trunc_normal_(self.human_token, std=.02)

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_joints, cfg.MODEL.VIT.EMB_DIM))
        trunc_normal_(self.keypoint_token, std=.02)


        self.blocks_human = nn.ModuleList([
            Block(
                dim=cfg.MODEL.VIT.EMB_DIM, num_heads=16, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
            )
            for i in range(2)])

        self.blocks_keypoint = nn.ModuleList([
            Block(
                dim=cfg.MODEL.VIT.EMB_DIM, num_heads=16, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
            )
            for i in range(2)])
        self.pos_emb_keypoint_and_visual = nn.Parameter(
            torch.zeros(1, self.num_joints + self.token_num, cfg.MODEL.VIT.EMB_DIM))
        trunc_normal_(self.pos_emb_keypoint_and_visual, std=.02)
        self.scale = cfg.MODEL.VIT.EMB_DIM ** -0.5


        self.human_mask_mlp = nn.Sequential(
            nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM),
            nn.Linear(cfg.MODEL.VIT.EMB_DIM, cfg.MODEL.VIT.EMB_DIM),
            nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM),
        )
        self.human_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)

        self.keypoint_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)
        self.keypoint_mask_mlp = nn.Sequential(
            nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM),
            nn.Linear(cfg.MODEL.VIT.EMB_DIM, cfg.MODEL.VIT.EMB_DIM),
            nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM),

        )

        if self.directly_use_visual_tokens:
            self.visual_tokens_fusion = nn.Sequential(
                nn.Linear(self.token_num * self.frame_num, self.token_num),
                nn.LayerNorm(self.token_num),
                nn.Linear(self.token_num, self.token_num),
            )
        else:

            self.stream1 = STTrans(cfg.MODEL.VIT.EMB_DIM ,  depth=2, num_heads=8, mlp_ratio=2,order_type=self.order_type , fusion=True )

        if self.use_keypoint_generate_htms:
            self.head_mlp = nn.Sequential(
                nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM),
                nn.Linear(cfg.MODEL.VIT.EMB_DIM, self.heatmap_h*self.heatmap_w),
                nn.GELU(),
                nn.LayerNorm(self.heatmap_h*self.heatmap_w),
                nn.Linear(self.heatmap_h*self.heatmap_w, self.heatmap_h*self.heatmap_w),
            )
        else:
            self.head_mask = nn.Sequential(
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





        self.patch_embed_difference_forward = PatchEmbed( img_size=(self.heatmap_h  , self.heatmap_w), patch_size = 4 ,
                                        in_chans = 256 , embed_dim = cfg.MODEL.VIT.EMB_DIM, ratio=1)
        self.patch_embed_difference_backward = PatchEmbed( img_size=(self.heatmap_h  , self.heatmap_w), patch_size = 4 ,
                                        in_chans = 256 , embed_dim = cfg.MODEL.VIT.EMB_DIM, ratio=1)
        self.blocks_difference_cross_deform = nn.ModuleList([
            Bidirectional_Deformable_Cross_Block(cfg.MODEL.VIT.EMB_DIM, 16, mlp_ratio=4., drop_path=dpr[i], map_size=self.map_size_up_1x)
            for i in range(2)])


        self.difference_deform_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)
        self.diff_fusion_layers = nn.Sequential(
            nn.Linear(self.token_num * 2, self.token_num),
            nn.Linear(self.token_num, self.token_num),
            nn.LayerNorm(self.token_num),
            )

        self.difference_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cfg.MODEL.VIT.DECONV1_IN_CHN,
                               out_channels=cfg.MODEL.VIT.DECONV1_OUT_CHN, kernel_size=4, stride=2, padding=1,
                               output_padding=0, bias=False),
            nn.BatchNorm2d(cfg.MODEL.VIT.DECONV1_OUT_CHN),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=cfg.MODEL.VIT.DECONV2_IN_CHN,
                               out_channels=cfg.MODEL.VIT.DECONV2_OUT_CHN, kernel_size=4, stride=2, padding=1,
                               output_padding=0, bias=False),
            nn.BatchNorm2d(cfg.MODEL.VIT.DECONV2_OUT_CHN),
            nn.ReLU(inplace=True),
        )
        if self.use_diff_final_fusion_conv:
            self.fusion_difference_feat_curr_conv = CHAIN_RSB_BLOCKS(cfg.MODEL.VIT.DECONV2_OUT_CHN * 2,
                                                                     cfg.MODEL.VIT.DECONV2_OUT_CHN, 2)

        self.difference_final_layers = nn.Sequential(
            nn.Conv2d(in_channels=cfg.MODEL.VIT.DECONV2_OUT_CHN, out_channels=cfg.MODEL.HEAD.FINAL_CONV_OUT_CHN,
                      kernel_size=1, stride=1, padding=0)
        )





    def forward(self, input ,  **kwargs):
        global feat_mask_out, feats_backbone_out_stream_2
        true_batchsize , channel_num = input.shape[0] , input.shape[1]//3
        out = []


        frame_curr, frame_supp_A, frame_supp_B = input.split(channel_num, dim=1)
        img_cat_batch= torch.cat((frame_curr, frame_supp_A, frame_supp_B), dim=0)


        htms_backbone_out_cat_batch , feats_backbone_out_cat_batch , feats_up1_backbone_out_cat_batch , \
                                                    feats_up2_backbone_out_cat_batch   = self.backbone(img_cat_batch)
        htms_curr, htms_supp_A, htms_supp_B = htms_backbone_out_cat_batch.split(true_batchsize, dim=0)




        # ------ human token -------
        human_token = self.human_token.repeat(true_batchsize * 3, 1, 1)
        feat_fused_1 = torch.cat((human_token , feats_backbone_out_cat_batch), dim=1)
        for blk in self.blocks_human:
            feat_fused_1 = blk(feat_fused_1)
        human_token = feat_fused_1[:, 0:1 , : ]
        visual_token = feat_fused_1[:, 1: , : ]
        human_token = self.human_mask_mlp(human_token)
        human_token = human_token.permute(0, 2, 1)
        mask_human = visual_token @ human_token
        visual_token = visual_token * mask_human
        visual_token = self.human_norm(visual_token)




        # ------ keypoint token -------
        keypoint_token = self.keypoint_token.repeat(true_batchsize * 3, 1, 1)
        feat_fused_1 = torch.cat((keypoint_token , visual_token), dim=1)
        pos_emb_keypoint_and_visual = self.pos_emb_keypoint_and_visual.repeat(true_batchsize * 3, 1, 1)
        feat_fused_1 = feat_fused_1 + pos_emb_keypoint_and_visual
        # 先从batch维度拆开-->从toke n维度拼起来-->计算各类token的num
        feat_fused_1_curr , feat_fused_1_pre , feat_fused_1_next = feat_fused_1.split(true_batchsize, dim=0)
        feat_fused_1_cat = torch.cat((feat_fused_1_curr, feat_fused_1_pre, feat_fused_1_next), dim=1)
        token_num_one_frame = feat_fused_1_curr.shape[1]
        token_num_one_frame_keypoint = self.num_joints
        token_num_one_frame_visual = token_num_one_frame - token_num_one_frame_keypoint

        for blk in self.blocks_keypoint:
            feat_fused_1_cat = blk(feat_fused_1_cat)
        feat_fused_1_curr, feat_fused_1_pre, feat_fused_1_next = feat_fused_1_cat.split(token_num_one_frame, dim=1)

        # 每帧都从token维度拆开得到 keypoint、visual token
        feat_fused_1_curr_keypoint_token , feat_fused_1_curr_visual_token = feat_fused_1_curr[:, 0:self.num_joints , : ], feat_fused_1_curr[:, self.num_joints: , : ]
        feat_fused_1_pre_keypoint_token , feat_fused_1_pre_visual_token = feat_fused_1_pre[:, 0:self.num_joints , : ], feat_fused_1_pre[:, self.num_joints: , : ]
        feat_fused_1_next_keypoint_token , feat_fused_1_next_visual_token = feat_fused_1_next[:, 0:self.num_joints , : ], feat_fused_1_next[:, self.num_joints: , : ]
        feat_fused_1_keypoint_token_cat = torch.cat((feat_fused_1_curr_keypoint_token, feat_fused_1_pre_keypoint_token, feat_fused_1_next_keypoint_token), dim=1)
        feat_fused_1_visual_token_cat = torch.cat((feat_fused_1_curr_visual_token, feat_fused_1_pre_visual_token, feat_fused_1_next_visual_token), dim=1)
        feat_fused_1_keypoint_token_cat = self.keypoint_mask_mlp(feat_fused_1_keypoint_token_cat)
        keypoint_confidence_map = feat_fused_1_keypoint_token_cat @ feat_fused_1_visual_token_cat.permute(0, 2, 1)
        keypoint_confidence_map = F.softmax(keypoint_confidence_map, dim=-1) * self.scale
        mask_keypoint = keypoint_confidence_map.sum(dim=1, keepdim=True)
        mask_keypoint = mask_keypoint.permute(0, 2, 1)

        feat_fused_1_visual_token_cat = feat_fused_1_visual_token_cat * mask_keypoint


        if self.directly_use_visual_tokens:
            feat_mask_out = self.visual_tokens_fusion(feat_fused_1_visual_token_cat.permute(0,2,1)).permute(0,2,1)
        else:
            feat_fused_1_visual_token_cat_batch = torch.cat(
                feat_fused_1_visual_token_cat.split(token_num_one_frame_visual, dim=1), dim=0)  # (B*3,192,1024)
            feat_mask_out = self.stream1(feat_fused_1_visual_token_cat_batch)
        if self.use_keypoint_generate_htms:
            if self.use_gcn:
                feat_fused_1_curr_keypoint_token = feat_fused_1_curr_keypoint_token.reshape(true_batchsize, 1, self.num_joints,self.dim)
                feat_fused_1_curr_keypoint_token = self.gcn(feat_fused_1_curr_keypoint_token)
                feat_fused_1_curr_keypoint_token = feat_fused_1_curr_keypoint_token.reshape(true_batchsize, self.num_joints, self.dim)
            htms_mask_out = self.head_mlp(feat_fused_1_curr_keypoint_token).reshape(true_batchsize, -1, self.heatmap_h, self.heatmap_w)
            out.append(htms_mask_out)
        else:
            feat_1_out = feat_mask_out.permute(0, 2, 1).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
            htms_1_out = self.head_mask(feat_1_out)
            out.append(htms_1_out)
        # ------------------mask branch end---------------------------


        feats_up2_backbone_out_stream_3 = feats_up2_backbone_out_cat_batch
        feats_backbone_curr, feats_backbone_pre, feats_backbone_next = feats_up2_backbone_out_stream_3.split(true_batchsize, dim=0)
        feat_difference_backward = feats_backbone_curr - feats_backbone_pre
        feat_difference_forward = feats_backbone_curr - feats_backbone_next

        feat_difference_forward , (hp_forward , wp_forward) = self.patch_embed_difference_forward(feat_difference_forward)
        feat_difference_backward , (hp_backward , wp_backward) = self.patch_embed_difference_backward(feat_difference_backward)

        feat_difference_bidirect_cat = torch.cat([feat_difference_backward, feat_difference_forward], dim=1)

        for blk in self.blocks_difference_cross_deform:
            feat_difference_bidirect_cat = blk(feat_difference_bidirect_cat , feat_mask_out)

        feat_difference_bidirect_cat = self.difference_deform_norm(feat_difference_bidirect_cat).permute(0, 2, 1)
        feat_difference_fused = self.diff_fusion_layers(feat_difference_bidirect_cat).reshape(true_batchsize, -1, hp_forward, wp_forward)
        feat_difference_fused = self.difference_upsample(feat_difference_fused)
        if self.use_diff_final_fusion_conv:
            feats_difference_curr_cat = torch.cat((feat_difference_fused, feats_backbone_curr), dim=1)
            feat_difference_fused = self.fusion_difference_feat_curr_conv(feats_difference_curr_cat)
        htms_3_out = self.difference_final_layers(feat_difference_fused)

        out.append(htms_3_out)


        htms_out =  sum(out) / len(out)


        return htms_out, htms_curr



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

        hyper_parameters_setting = "ViT_VREMD"
        return hyper_parameters_setting






from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
from ..layers.block_zoo import *
from ..layers.spatiotemporal_attention import *
from ..layers.graph import GCN
# from utils.vis import vis_htms
from posetimation.layers import CHAIN_RSB_BLOCKS
logger = logging.getLogger(__name__)

__all__ = ["NEWPOSE_V2"]







@MODEL_REGISTRY.register()
class NEWPOSE_V2(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = NEWPOSE_V2(cfg, phase, **kwargs)
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
        self.branch_list = cfg.MODEL.EXTRA.BRANCH_LIST
        self.use_doform_cross = cfg.MODEL.EXTRA.USE_DEFOEM_CROSS
        self.use_diff_final_fusion_conv = cfg.MODEL.EXTRA.USE_DIFF_FINAL_FUSION
        self.use_one_st = cfg.MODEL.EXTRA.USE_ONE_ST
        self.use_keypoint_generate_htms = cfg.MODEL.EXTRA.USE_KEYPOINT_GENERATE_HTMS

        self.directly_use_visual_tokens = False
        self.use_gcn = True
        self.dim = cfg.MODEL.VIT.EMB_DIM
        self.htms_dict = dict()


        self.htms_dict = dict()


        self.backbone = ViT_V1.get_net(cfg , phase)


        dpr = [x.item() for x in torch.linspace(0, 0.06, 2)]

        if 'mask' in self.branch_list:
            #mask 分支
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

            #目前是TFF最好，STTrans稍微好点，88.01，DSTA的最差，87.9几，所以不调DSTA了，开始调STTrans，之后再去调TFF
            # (STTrans depth 已尝试2,1，(1不如2,1是87.9几，2是88.01 )posemb还未尝试，head和mlp_ratio不动)(DSTA只尝试了depth=2，不如STTrans)
            #1.给STTrans加posemb，都加 （是否高于88.01）； 2. DSTA的depth设为1试试（STTrans设为1时是不如2的） ； 3. TFF调一下 ，前面的block都调一下
            #4. 前面所有的block的head设置为8，mlp设置为2 (效果一般) 5.前面的block的head设置为8，mlp设置为4 （88.03）
            #6.mask和original 使用同一个st网络 前面block依旧是16和4（88.06）
            #7.使用同一个st网络，并且前面的block的head设置为8，mlp设置为4 (88.0278)
            #8.调diff分支，使用deform_cross：
            #1.block = 1 ;  2. block = 2 , fusion=false ;   3. block = 2,fusion=true (这三个里面是第二种最好，86.几，其余两个是85.几)
            #加上final_fusion之后：使用的是backbone的curr的feature :
            # 1. (1) 上面第三种+final_fusion+ backbone的curr (87.87,说明还是得加上final_fusion)
            #    (2) 上面第三种+final_fusion+ mask分支的结果 (87.89)
            #    (3) 上面第三种+final_fusion+ original分支的结果（87.98）
            # 2. (1) 上面第二种+final_fusion+ backbone的curr (87.92)
            #    (2) 上面第二种+final_fusion+ mask分支的结果（87.94）
            #    (3) 上面第二种+final_fusion+ 0original分支的结果 （87.8几）

            #调deform_cross的KV特征：  2.original分支的结果
            '''---------------------------------------------------------------'''
            # 大改动：希望把diff分支融合两个方向的差值的conv换成双向可形变注意力模块
            # 本版本主要就是实现上面改动，设计双向可形变交叉注意力模块 其中deform block 固定使用 block = 2, fusion = true ，固定使用final_fusion
            # cross的kv特征依旧是三种：1. backbone 的curr 2. mask分支的结果 3. original分支的结果
            # 结果是：               1.    88.00        2.    88.00     3.     87.93
            # no   original (88.03)    diff (很差，86.几)   mask和original

            # 像把GCN加进来试一试 再mask分支后面，加上GCN 88.03
            if self.use_gcn:
                self.gcn = GCN(cfg.MODEL.VIT.EMB_DIM,cfg.MODEL.VIT.EMB_DIM,num_nodes=self.num_joints,
                                     mode='spatial')

            if self.directly_use_visual_tokens:
                self.visual_tokens_fusion = nn.Sequential(
                    nn.Linear(self.token_num * self.frame_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num, self.token_num),
                )
            else:
                # self.stream1 = Temporal_Feat_Fusion_Module(cfg, self.frame_num, cfg.MODEL.VIT.EMB_DIM)
                self.stream1 = STTrans(cfg.MODEL.VIT.EMB_DIM ,  depth=2, num_heads=8, mlp_ratio=2,order_type=self.order_type , fusion=True )
                # self.stream1 = new_Spatiotemporal_Atten(length=self.token_num , channel=cfg.MODEL.VIT.EMB_DIM ,
            #                                         frame_num=self.frame_num,multi_frame_fusion=True,
            #                                         depth=2,head=8,mlp_ratio=2)
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
        if 'original' in self.branch_list:
            # original 分支
            if self.use_one_st:
                self.spatial_temporal_blocks = self.stream1
            else:
                self.spatial_temporal_blocks = STTrans(cfg.MODEL.VIT.EMB_DIM ,  depth=2, num_heads=8, mlp_ratio=2,order_type=self.order_type )
                # self.spatial_temporal_blocks = new_Spatiotemporal_Atten(length=self.token_num , channel=cfg.MODEL.VIT.EMB_DIM , frame_num=self.frame_num)
                self.feat_fusion_mlp = nn.Sequential(
                    nn.Linear(self.token_num * self.frame_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                )
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

        # difference  分支
        if 'difference' in self.branch_list:
            if self.difference_type == 'feat_up_4x':

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




            elif self.difference_type == 'feat_up_1x':

                self.blocks_difference = nn.ModuleList([
                    Block(
                        dim=cfg.MODEL.VIT.EMB_DIM, num_heads=16, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                        drop=0., attn_drop=0., drop_path=dpr[i],
                    )
                    for i in range(2)])
                self.difference_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)

                self.fusion_mlp_differences = nn.Sequential(
                    nn.Linear(self.token_num * 2, self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                )

                self.blocks_difference_deform = nn.ModuleList([
                    Deformable_Block(cfg.MODEL.VIT.EMB_DIM, 16, mlp_ratio=4., drop_path=dpr[i],
                                     map_size=self.map_size_up_1x)
                    for i in range(2)])
                self.difference_deform_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)


                self.blocks_difference_feat_cat = nn.ModuleList([
                    Block(
                        dim=cfg.MODEL.VIT.EMB_DIM, num_heads=16, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                        drop=0., attn_drop=0., drop_path=dpr[i],
                    )
                    for i in range(2)])
                self.difference_feat_cat_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)
                self.fusion_mlp_difference_feat_cat = nn.Sequential(
                    nn.Linear(self.token_num * 2, self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num, self.token_num),
                    nn.LayerNorm(self.token_num),
                )
                self.head_difference = nn.Sequential(
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
                    nn.Conv2d(in_channels=cfg.MODEL.HEAD.FINAL_CONV_IN_CHN,
                              out_channels=cfg.MODEL.HEAD.FINAL_CONV_OUT_CHN,
                              kernel_size=1, stride=1, padding=0)
                )





        logger.info(
            "=> Number of parameters in the Backbone: {}".format(sum(p.numel() for p in self.backbone.parameters())))

        param_count = {}
        for name, module in self.stream1.named_modules():
            # 收集模块的参数
            params = list(module.parameters())
            # 累加参数量
            param_count[name] = sum(p.numel() for p in params)

        # 打印每个模块的参数量
        for name, count in param_count.items():
            print(f"{name}: {count}")


        # logger.info("=> Number of parameters in the human blocks: {}".format(sum(p.numel() for p in self.blocks_human.parameters())))
        # logger.info("=> Number of parameters in the keypoint blocks: {}".format(sum(p.numel() for p in self.blocks_keypoint.parameters())))
        # logger.info("=> Number of parameters in the stream1: {}".format(sum(p.numel() for p in self.stream1.parameters())))
        # if 'original' in self.branch_list:
        #     logger.info("=> Number of parameters in the stream2: {}".format(sum(p.numel() for p in self.spatial_temporal_blocks.parameters())))
        # if self.use_doform_cross:
        #     logger.info("=> Number of parameters in the blocks_difference_cross_deform: {}".format(sum(p.numel() for p in self.blocks_difference_cross_deform.parameters())))
        # else:
        #     logger.info("=> Number of parameters in the Deformable_Block: {}".format(sum(p.numel() for p in self.Deformable_Block.parameters())))
        # logger.info("=> Number of parameters in the fusion_difference_feat_curr_conv: {}".format(sum(p.numel() for p in self.fusion_difference_feat_curr_conv.parameters())))



    def forward(self, input ,  **kwargs):
        global feat_mask_out, feats_backbone_out_stream_2
        true_batchsize , channel_num = input.shape[0] , input.shape[1]//3
        out = []
        # assert "margin" in kwargs
        # margin = kwargs["margin"]
        #
        # interval = torch.sum(margin, dim=1, keepdim=True)
        # margin = torch.div(margin.float(), interval.float())
        # prev_weight, next_weight = margin[:, 1], margin[:, 0]
        # prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # shape = [1, 1 , self.heatmap_h,self.heatmap_w]
        # prev_weight, next_weight = prev_weight.repeat(shape), next_weight.repeat(shape)

        frame_curr, frame_supp_A, frame_supp_B = input.split(channel_num, dim=1)
        img_cat_batch= torch.cat((frame_curr, frame_supp_A, frame_supp_B), dim=0)

        #vit_backbone forward->outputs are two parts: features and heatmaps
        htms_backbone_out_cat_batch , feats_backbone_out_cat_batch , feats_up1_backbone_out_cat_batch , \
                                                    feats_up2_backbone_out_cat_batch   = self.backbone(img_cat_batch)
        htms_curr, htms_supp_A, htms_supp_B = htms_backbone_out_cat_batch.split(true_batchsize, dim=0)



        if 'mask' in self.branch_list:
            # ------ human token -------
            human_token = self.human_token.repeat(true_batchsize * 3, 1, 1)
            feat_fused_1 = torch.cat((human_token , feats_backbone_out_cat_batch), dim=1)
            for blk in self.blocks_human:
                feat_fused_1 = blk(feat_fused_1)
            human_token = feat_fused_1[:, 0:1 , : ]
            visual_token = feat_fused_1[:, 1: , : ]
            human_token = self.human_mask_mlp(human_token)
            human_token = human_token.permute(0, 2, 1)
            mask_human = visual_token @ human_token  #(B*3,192,1)
            visual_token = visual_token * mask_human
            visual_token = self.human_norm(visual_token)#广播机制(B*3,192,1024)  加随机mask，(B*3,192*0.25,1024)全为0，(B*3,192*0.75,1024)原样 cat (B*3,192,1024)
            # (B*3,192,1024) -重建- (B*3,192,1024)



            # ------ keypoint token -------
            # 先扩充keypoint token-->拼起来-->再进行pos_embedding
            keypoint_token = self.keypoint_token.repeat(true_batchsize * 3, 1, 1)
            feat_fused_1 = torch.cat((keypoint_token , visual_token), dim=1)
            pos_emb_keypoint_and_visual = self.pos_emb_keypoint_and_visual.repeat(true_batchsize * 3, 1, 1)
            feat_fused_1 = feat_fused_1 + pos_emb_keypoint_and_visual
            # 先从batch维度拆开-->从toke n维度拼起来-->计算各类token的num
            feat_fused_1_curr , feat_fused_1_pre , feat_fused_1_next = feat_fused_1.split(true_batchsize, dim=0)
            feat_fused_1_cat = torch.cat((feat_fused_1_curr, feat_fused_1_pre, feat_fused_1_next), dim=1)
            token_num_one_frame = feat_fused_1_curr.shape[1] # 192+17
            token_num_one_frame_keypoint = self.num_joints  # 17
            token_num_one_frame_visual = token_num_one_frame - token_num_one_frame_keypoint # 192

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
            feat_fused_1_keypoint_token_cat = self.keypoint_mask_mlp(feat_fused_1_keypoint_token_cat)
            # 将keypoint token 矩阵乘 visual token 的转置-->进行softmax-->得到confidence map-->压缩第一维度将17*3压缩至1-->调换后两维得到mask
            keypoint_confidence_map = feat_fused_1_keypoint_token_cat @ feat_fused_1_visual_token_cat.permute(0, 2, 1) # (B,17*3,192*3)=(B,17*3,1024) * (B,192*3,1024)
            keypoint_confidence_map = F.softmax(keypoint_confidence_map, dim=-1) * self.scale
            mask_keypoint = keypoint_confidence_map.sum(dim=1, keepdim=True) # (B,1,192*3)
            mask_keypoint = mask_keypoint.permute(0, 2, 1) # (B,192*3,1)
            # 将mask 广播到 visual token 维度-->进行mask-->将mask后的visual token 先从1维度拆开，再从batch维度拼接
            feat_fused_1_visual_token_cat = feat_fused_1_visual_token_cat * mask_keypoint # (B,192*3,1024)

            #丢入融合网络
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

        if 'original' in self.branch_list:
            #------------------original branch start----------------------
            feats_backbone_out_stream_2 = feats_backbone_out_cat_batch
            feats_backbone_out_stream_2 = self.spatial_temporal_blocks(feats_backbone_out_stream_2)
            if self.use_one_st:
                feat_2 = feats_backbone_out_stream_2.permute(0, 2, 1).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
            else:
                feat_2 = torch.cat(feats_backbone_out_stream_2.split(true_batchsize, dim=0) , dim=1).transpose(1, 2)
                feat_2 = self.feat_fusion_mlp(feat_2).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])

            htms_2_out = self.head(feat_2)
            out.append(htms_2_out)
            # ------------------original branch end------------------------


        if 'difference' in self.branch_list:

            # ------------------difference branch start----------------------

            if self.difference_type == 'feat_up_4x':
                feats_up2_backbone_out_stream_3 = feats_up2_backbone_out_cat_batch
                feats_backbone_curr, feats_backbone_pre, feats_backbone_next = feats_up2_backbone_out_stream_3.split(true_batchsize, dim=0)
                feat_difference_backward = feats_backbone_curr - feats_backbone_pre
                feat_difference_forward = feats_backbone_curr - feats_backbone_next

                feat_difference_forward , (hp_forward , wp_forward) = self.patch_embed_difference_forward(feat_difference_forward)
                feat_difference_backward , (hp_backward , wp_backward) = self.patch_embed_difference_backward(feat_difference_backward)

                feat_difference_bidirect_cat = torch.cat([feat_difference_backward, feat_difference_forward], dim=1)


                """调cross的KV feature ： 1. 最开始是backbone_curr 2. mask分支的结果 3.original分支的结果"""
                # feats_up1x_backbone_curr = feats_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]
                # for blk in self.blocks_difference_cross_deform:
                #     feat_difference_bidirect_cat = blk(feat_difference_bidirect_cat,feats_up1x_backbone_curr)
                for blk in self.blocks_difference_cross_deform:
                    feat_difference_bidirect_cat = blk(feat_difference_bidirect_cat , feat_mask_out)
                # for blk in self.blocks_difference_cross_deform:
                #     feat_difference_bidirect_cat = blk(feat_difference_bidirect_cat , feats_backbone_out_stream_2)


                feat_difference_bidirect_cat = self.difference_deform_norm(feat_difference_bidirect_cat).permute(0, 2, 1)
                feat_difference_fused = self.diff_fusion_layers(feat_difference_bidirect_cat).reshape(true_batchsize, -1, hp_forward, wp_forward)
                feat_difference_fused = self.difference_upsample(feat_difference_fused)
                if self.use_diff_final_fusion_conv:
                    feats_difference_curr_cat = torch.cat((feat_difference_fused, feats_backbone_curr), dim=1)
                    feat_difference_fused = self.fusion_difference_feat_curr_conv(feats_difference_curr_cat)
                htms_3_out = self.difference_final_layers(feat_difference_fused)





            elif self.difference_type == 'feat_up_1x':
                feats_backbone_out_stream_3 = feats_backbone_out_cat_batch
                feats_backbone_curr , feats_backbone_pre , feats_backbone_next = feats_backbone_out_stream_3.split(true_batchsize, dim=0)
                feat_difference_pre = feats_backbone_curr - feats_backbone_pre
                feat_difference_next = feats_backbone_curr - feats_backbone_next
                # feat_difference_pre = feat_difference_pre.permute(0, 2, 1).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
                # feat_difference_next = feat_difference_next.permute(0, 2, 1).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
                feat_difference_cat = torch.cat((feat_difference_pre, feat_difference_next) , dim=1)
                for blk in self.blocks_difference:
                    feat_difference_cat = blk(feat_difference_cat)
                feat_difference_cat = self.difference_norm(feat_difference_cat)
                feat_difference = self.fusion_mlp_differences(feat_difference_cat.permute(0, 2, 1)).permute(0, 2, 1)

                for blk in self.blocks_difference_deform:
                    feat_difference = blk(feat_difference)
                feat_difference = self.difference_deform_norm(feat_difference)

                feat_curr_difference_cat = torch.cat((feats_backbone_curr , feat_difference) , dim=1)
                for blk in self.blocks_difference_feat_cat:
                    feat_curr_difference_cat = blk(feat_curr_difference_cat)
                feat_curr_difference_cat = self.difference_feat_cat_norm(feat_curr_difference_cat).permute(0, 2, 1)
                feat_curr_difference_fused = self.fusion_mlp_difference_feat_cat(feat_curr_difference_cat).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
                htms_3_out = self.head_difference(feat_curr_difference_fused)

            out.append(htms_3_out)


            # ------------------difference branch end----------------------

        htms_out =  sum(out) / len(out)

        self.htms_dict={
        "diff_f":    htms_curr-htms_supp_A ,
        "diff_b":    htms_curr- htms_supp_B,
        "htms_out": htms_out,

        }
        return htms_out, htms_curr








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

        hyper_parameters_setting = "ViT_3_Branches"
        return hyper_parameters_setting



if __name__ == "__main__":
    dpr = [x.item() for x in torch.linspace(0, 0.18, 4)]
    print(dpr)
    print("!!!")




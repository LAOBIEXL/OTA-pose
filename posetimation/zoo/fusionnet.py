import time
import logging
import os
import math
from typing import List
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import BaseModel
from .backbones.vit import Block , trunc_normal_

class Temporal_Feat_Fusion_Module(nn.Module):
    def __init__(self, cfg , frame = 3 , dim = 1024 ,fuse_all = True  ):
        super().__init__()
        self.cfg = cfg
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.heatmap_h = cfg.MODEL.HEATMAP_SIZE[1]
        self.heatmap_w = cfg.MODEL.HEATMAP_SIZE[0]
        self.depth = cfg.MODEL.TFF.DEPTH
        self.drop_path_rate = cfg.MODEL.TFF.DROP_PATH_RATE
        self.fuse_all = fuse_all
        self.mlp_ratio = cfg.MODEL.TFF.MLP_RATIO


        self.frame = frame
        self.token_num = int((self.cfg.MODEL.IMAGE_SIZE[1] / self.cfg.MODEL.VIT.PATCH_SIZE) * (self.cfg.MODEL.IMAGE_SIZE[0] / self.cfg.MODEL.VIT.PATCH_SIZE))


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=16, mlp_ratio=self.mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
            )
            for i in range(self.depth)])


        if self.frame == 3:
            self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            self.pos_embed_2 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            self.pos_embed_3 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            trunc_normal_(self.pos_embed_1, std=.02)
            trunc_normal_(self.pos_embed_2, std=.02)
            trunc_normal_(self.pos_embed_3, std=.02)
            if self.fuse_all:
                self.feat_fusion_layer = nn.Sequential(
                    nn.Linear(self.token_num * 3 , self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num , self.token_num),
                    nn.LayerNorm(self.token_num),
                )
            else:
                self.last_norm = nn.LayerNorm(dim)

        elif self.frame == 5:

            self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            self.pos_embed_2 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            self.pos_embed_3 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            self.pos_embed_4 = nn.Parameter(torch.zeros(1, self.token_num, dim))
            self.pos_embed_5 = nn.Parameter(torch.zeros(1, self.token_num, dim))

            trunc_normal_(self.pos_embed_1, std=.02)
            trunc_normal_(self.pos_embed_2, std=.02)
            trunc_normal_(self.pos_embed_3, std=.02)
            trunc_normal_(self.pos_embed_4, std=.02)
            trunc_normal_(self.pos_embed_5, std=.02)

            if self.fuse_all:
                self.feat_fusion_layer = nn.Sequential(
                    nn.Linear(self.token_num * 5 , self.token_num),
                    nn.LayerNorm(self.token_num),
                    nn.Linear(self.token_num , self.token_num),
                    nn.LayerNorm(self.token_num),
                )
            else:
                self.last_norm = nn.LayerNorm(dim)


    def forward(self, x):
        if self.frame == 3:
            true_batchsize , true_token_num = x.shape[0] // 3 , x.shape[1]

            feat_curr, feat_prev, feat_next = x.split(true_batchsize, dim=0)

            feat_curr, feat_prev, feat_next = feat_curr + self.pos_embed_1 , feat_prev + self.pos_embed_2 , feat_next + self.pos_embed_3

            x = torch.cat((feat_curr , feat_prev , feat_next) , dim=1)

            for blk in self.blocks:
                x =  blk(x)

            if self.fuse_all:
                x = self.feat_fusion_layer(x.transpose(1, 2))
                x = x.transpose(1, 2)
            else:
                x = self.last_norm(x[:,0:true_token_num,:])


            return x
        elif self.frame == 5:
            true_batchsize, true_token_num = x.shape[0] // 5 , x.shape[1]

            feat_curr, supp_feat_1, supp_feat_2,supp_feat_3,supp_feat_4 = x.split(true_batchsize, dim=0)

            feat_curr, supp_feat_1, supp_feat_2,supp_feat_3,supp_feat_4 = feat_curr   + self.pos_embed_1, \
                                                                          supp_feat_1 + self.pos_embed_2, \
                                                                          supp_feat_2 + self.pos_embed_3, \
                                                                          supp_feat_3 + self.pos_embed_4, \
                                                                          supp_feat_4 + self.pos_embed_5

            x = torch.cat((feat_curr, supp_feat_1, supp_feat_2,supp_feat_3,supp_feat_4), dim=1)

            for blk in self.blocks:
                x = blk(x)

            if self.fuse_all:
                x = self.feat_fusion_layer(x.transpose(1, 2))
                x = x.transpose(1, 2)
            else:
                x = self.last_norm(x[:,0:true_token_num,:])

            return x





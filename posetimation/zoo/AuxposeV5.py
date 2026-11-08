#!/usr/bin/python
# -*- coding:utf8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch.nn as nn
import torch.nn.functional as F

import copy
from engine.defaults.constant import MODEL_REGISTRY
from posetimation.layers.basic_layer import conv_bn_relu
from posetimation.layers.basic_model import ChainOfBasicBlocks
# from posetimation.layers.conv_rnn import UpdateBlock
from posetimation.backbones.hrnet import HRNetPlus
# from posetimation.backbones.vit_B import ViT_B, TopdownHeatmapSimpleHead, Block, PatchEmbed

from posetimation.backbones.vit import ViT , PatchEmbed , trunc_normal_
from torchvision.ops.deform_conv import DeformConv2d
# from posetimation.layers.non_local_net import NLBlockND
import kornia
from timm.models.layers import DropPath
from engine.defaults import TRAIN_PHASE
import numpy as np
from einops import rearrange
__all__ = ["AuxposeV5"]
BN_MOMENTUM = 0.1
from functools import partial
import logging
import os.path as osp
import torch
from torch.nn.functional import kl_div
from torch.nn import init
from posetimation.layers.odconv import ODConv2d
from ..fusionnet import Temporal_Feat_Fusion_Module



class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)

        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        if mask is not None:
            mask = mask[:, None, :, :].repeat(1, self.num_heads, 1, 1)
            attn = attn * mask
            attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-10)

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)

        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads
        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d
        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=1, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x, mask=mask)
            x = x + norm_ffn(x)

        return x

class STTrans(nn.Module):
    def __init__(
        self, h_dim, depth=3, num_heads=8, mlp_dim=128,
        pool='cls', dim_per_head=64, dropout=0., embed_dropout=0., multi_output=False
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=embed_dropout)
        self.multi_output = multi_output

        self.transformer_s = []
        self.transformer_t = []
        self.depth = depth
        for i in range(depth):
            self.transformer_t.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
            self.transformer_s.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
        self.transformer_t = nn.ModuleList(self.transformer_t)
        self.transformer_s = nn.ModuleList(self.transformer_s)

        self.pool = pool

    def forward(self, x, mask_s=None, mask_t=None):
        B,N = x.shape[0], x.shape[1]
        out = []
        for i in range(self.depth):
            x = x.contiguous().view(B*N,-1,x.shape[-1])
            x = self.transformer_t[i](x,mask_t)
            x = x.view(B,N,-1,x.shape[-1]).permute(0,2,1,3)
            x = x.contiguous().view(-1,N,x.shape[-1])
            x = self.transformer_s[i](x,mask_s)
            x = x.view(B,-1,N,x.shape[-1]).permute(0,2,1,3)
            out.append(x)
        if self.multi_output:
            return out
        else:
            return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODEL_REGISTRY.register()
class AuxposeV5(nn.Module):
    """
    修改多阶段差分的输入
    互信息目标
    """

    @classmethod
    def get_model_hyper_parameters(cls, cfg):
        bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        rot_factor = cfg.TRAIN.ROT_FACTOR
        SCALE_FACTOR = cfg.TRAIN.SCALE_FACTOR

        if not isinstance(SCALE_FACTOR, list):
            temp = SCALE_FACTOR
            SCALE_FACTOR = [SCALE_FACTOR, SCALE_FACTOR]
        scale_bottom = 1 - SCALE_FACTOR[0]
        scale_top = 1 + SCALE_FACTOR[1]

        paramer = "bbox_{}_rot_{}_scale_{}-{}".format(bbox_enlarge_factor, rot_factor, scale_bottom,
                                                      scale_top)

        if cfg.LOSS.HEATMAP_MSE.USE:
            paramer += f"_MseLoss_{cfg.LOSS.HEATMAP_MSE.WEIGHT}"

        return paramer

    def __init__(self, cfg, is_train, **kwargs):
        super(AuxposeV5, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pretrained = cfg.MODEL.PRETRAINED
        self.is_train = is_train
        if self.is_train == TRAIN_PHASE:
            self.is_train = True
        else:
            self.is_train = False
        self.pretrained_layers = ['*']
        self.hrnet = HRNetPlus(cfg, self.is_train)
        self.freeze_hrnet_weight = cfg['MODEL']["FREEZE_HRNET_WEIGHTS"]
        # self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS
        self.backbone = ViT.get_net(cfg)


        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 16 * 12, 1024))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 16 * 12, 1024))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 16 * 12, 1024))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, 16 * 12, 1024))
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, 16 * 12, 1024))

        if self.pos_embed_1 is not None:
            trunc_normal_(self.pos_embed_1, std=.02)
            trunc_normal_(self.pos_embed_2, std=.02)
            trunc_normal_(self.pos_embed_3, std=.02)
            trunc_normal_(self.pos_embed_4, std=.02)
            trunc_normal_(self.pos_embed_5, std=.02)
            # trunc_normal_(self.pos_embed_heatmap, std=.02)

        self.final_refine = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_refine_p = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_refine_m = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_refine_n = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_layer = nn.Conv2d(96, 17, 3, 1, 1)
        self.final_layer_m = nn.Conv2d(96, 17, 3, 1, 1)
        self.final_layer_n = nn.Conv2d(96, 17, 3, 1, 1)
        self.final_layer_p = nn.Conv2d(96, 17, 3, 1, 1)
        if self.freeze_hrnet_weight:
            self.hrnet.freeze_weight()

        self.encoder = []
        self.decoder = []
        h_dim = 1024
        for _ in range(3):
            self.encoder.append(STTrans(h_dim, depth=1, mlp_dim=64, num_heads=8,
                                        dim_per_head=64, dropout=0))
            self.decoder.append(STTrans(h_dim, depth=1, mlp_dim=64, num_heads=8,
                                        dim_per_head=64, dropout=0))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.TFF = Temporal_Feat_Fusion_Module(cfg, frame=5, dim=cfg.MODEL.VIT.EMB_DIM)

        ###spatial block
        self.Spatial_pos_embed_mask = nn.Parameter(torch.zeros(1, 192,1024))
        self.Spatial_pos_embed_noise = nn.Parameter(torch.zeros(1, 192,1024))
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 192,1024))
        self.Temporal_pos_embed_mask = nn.Parameter(torch.zeros(1, 3, 1024))
        self.Temporal_pos_embed_noise = nn.Parameter(torch.zeros(1, 3, 1024))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 3, 1024))
        dpr = [x.item() for x in torch.linspace(0, 0.06,2)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=1024, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])
        self.Spatial_norm_mask = norm_layer(1024)
        self.Spatial_norm_noise = norm_layer(1024)
        self.Spatial_norm = norm_layer(1024)

        ### temporal block
        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=1024, num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])
        self.Temporal_norm_mask = norm_layer(1024)
        self.Temporal_norm_noise = norm_layer(1024)
        self.Temporal_norm = norm_layer(1024)


        self.final_norm = nn.LayerNorm(1024)
        self.project_fusion = nn.Sequential(
            nn.Linear(192*3, 192),
            nn.Linear(192, 192),
        )
        self.project_fusion_mask = nn.Sequential(
            nn.Linear(192*3, 192),
            nn.Linear(192, 192),
        )

        self.project_mask = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.Linear(192, 192),
        )
        self.project_noise = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.Linear(192, 192),
        )
        self.project_primary = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.Linear(192, 192),
        )


        self.hf_norm = nn.LayerNorm(192)
        self.up_layer_hr_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1,
                               output_padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.ConvTranspose2d(in_channels=256, out_channels=96, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, 1, 0)
        )
        self.relu = nn.ReLU()
        #noise
        self.noise_dev = 0.5
        self.init_weights()
    def forward(self, kf_x, sup_x, **kwargs):
        """
        kf_x:  [batch, 3, 384, 288]
        sup_x: [batch, 3 * num, 384, 288]]
        """

        batch_size, num_sup = kf_x.shape[0], sup_x.shape[1] // 3
        sup_4 =   torch.chunk(sup_x, num_sup, dim=1)
        sup_1 = sup_4[1]
        sup_2 = sup_4[2]
        sup_x = torch.cat([sup_1,sup_2], dim=0)  # [batch * num, 3, 384, 288]]
        num_sup = 2#三帧辅助帧
        # sup_x = torch.cat(torch.chunk(sup_x, num_sup, dim=1), dim=0)  # [batch * num, 3, 384, 288]]

        x = torch.cat([kf_x, sup_x], dim=0)

        vit_features , htms_cat_batch = self.backbone(x)
        channel = vit_features.shape[1]
        # backbone_heatmaps = self.keypoint_head(vit_features)
        seq_heatmap_list = torch.chunk(htms_cat_batch, num_sup + 1, dim=0)

        kf_heatmap =  seq_heatmap_list[0]


        #mask
        B = batch_size*(num_sup+1)
        N = channel
        T = 1024
        mask_ratio = 0.5
        batch_ind = torch.arange(B)[:, None, None].cuda()#B，1，1
        agent_ind = torch.arange(N)[None, :, None].cuda()#1.N，1
        mask = torch.ones((B, N, T)).type_as(vit_features)#B，N，T
        shuffle_indices = torch.rand((B, N, T)).argsort().cuda()#B，N，T
        mask_indices = shuffle_indices[:, :, :int(T * mask_ratio)]#B，N，T*ratio
        mask[batch_ind, agent_ind, mask_indices] = 0
        # assert torch.sum(mask) == (self.past_timestep - int(self.past_timestep * mask_ratio)) * B * N
        mask = mask.view(B, N, -1)
        inverse_mask = 1 - mask
        mask_tokens_gt = vit_features *inverse_mask #被mask掉的特征

        mask_tokens = vit_features * mask#已知的特征

        ##spatial
        # feat_S = mask_tokens[(num_sup+1)// 2]
        feat_S_m = mask_tokens + self.Spatial_pos_embed_mask
        for blk in self.Spatial_blocks:
            feat_S_m = blk(feat_S_m)
        feat_S_m = self.Spatial_norm_mask(feat_S_m)


        ##temporal
        mask_tokens_temporal = torch.stack(mask_tokens.split(batch_size, dim=0), dim=0)
        feat_T_m = rearrange(mask_tokens_temporal, 'f b p c -> (b p) f c')
        feat_T_m = feat_T_m + self.Temporal_pos_embed_mask
        for blk in self.Temporal_blocks:
            feat_T_m = blk(feat_T_m)
        feat_T_m = self.Temporal_norm_mask(feat_T_m)
        feat_T_m = rearrange(feat_T_m, '(b p) f c -> (f b) p c', p=192)
        # feat_T = feat_T[self.num_frame // 2]
        # mask_tokens_temporal = torch.stack(feat_T.split(batch_size, dim=0), dim=0)

        feat_agg_mask = torch.concat((feat_T_m, feat_S_m), dim=1)
        feat_agg_mask = self.project_mask(feat_agg_mask.permute(0, 2, 1))
        feat_agg_mask = feat_agg_mask.permute(0, 2, 1)#生成的被mask掉的特征

        # feat_fused = self.TFF(mask_tokens,mask=True)

        # embedding1_flat = feat_agg_mask.view(B, -1)
        # embedding2_flat = mask_tokens_gt.view(B, -1)
        # cosine_sim1 = F.cosine_similarity(embedding1_flat, embedding2_flat, dim=1)
        # mask_cosine_loss = 1 - cosine_sim1.mean()

        feat_fused_he = feat_agg_mask + mask_tokens
        mask_feature = torch.cat(torch.chunk(feat_fused_he, chunks=num_sup+1, dim=0), dim=1)
        feature_heatmap = self.final_norm(mask_feature)
        heatmap_feature = self.hf_norm(self.project_fusion_mask(feature_heatmap.permute(0, 2, 1)))
        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], 1024, 16, 12)
        heatmap_feature_rm = self.up_layer_hr_1(heatmap_feature)
        heatmap_m = self.final_layer_m(self.final_refine_m(heatmap_feature_rm))
        #end mask

        #noise
        noise_dev = self.noise_dev
        noise = torch.from_numpy(np.random.normal(loc=0., scale=noise_dev, size=(B, N, T))).type_as(vit_features)
        noise_mask = torch.rand(B, N, T).cuda()
        noise = noise * noise_mask[:, :, :]
        all_traj_noise = vit_features + noise

        ##spatial
        # feat_S = mask_tokens[(num_sup+1)// 2]
        feat_S_n = all_traj_noise + self.Spatial_pos_embed_noise
        for blk in self.Spatial_blocks:
            feat_S_n = blk(feat_S_n)
        feat_S_n = self.Spatial_norm_noise(feat_S_n)

        ##temporal
        noise_tokens_temporal = torch.stack(all_traj_noise.split(batch_size, dim=0), dim=0)
        feat_T_n = rearrange(noise_tokens_temporal, 'f b p c -> (b p) f c')
        feat_T_n = feat_T_n + self.Temporal_pos_embed_noise
        for blk in self.Temporal_blocks:
            feat_T_n = blk(feat_T_n)
        feat_T_n = self.Temporal_norm_noise(feat_T_n)
        feat_T_n = rearrange(feat_T_n, '(b p) f c -> (f b) p c', p=192)
        # feat_T = feat_T[self.num_frame // 2]
        # mask_tokens_temporal = torch.stack(feat_T.split(batch_size, dim=0), dim=0)

        feat_agg_noise = torch.concat((feat_T_n, feat_S_n), dim=1)
        feat_agg_noise = self.project_noise(feat_agg_noise.permute(0, 2, 1))
        feat_agg_noise = feat_agg_noise.permute(0, 2, 1)  # 生成的被noise掉的特征

        # feat_fused_noise = self.TFF(all_traj_noise,mask=True)

        # embedding1_flat = vit_features.view(batch_size, -1)
        # embedding2_flat = feat_agg_noise.view(batch_size, -1)
        # cosine_sim2 = F.cosine_similarity(embedding1_flat, embedding2_flat, dim=1)
        # noise_cosine_loss = 1 - cosine_sim2.mean()

        noise_feature = torch.cat(torch.chunk(feat_agg_noise, chunks=num_sup+1, dim=0), dim=1)
        feature_heatmap = self.final_norm(noise_feature)
        heatmap_feature = self.hf_norm(self.project_fusion_mask(feature_heatmap.permute(0, 2, 1)))
        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], 1024, 16, 12)
        heatmap_feature_rn = self.up_layer_hr_1(heatmap_feature)
        heatmap_n = self.final_layer_n(self.final_refine_n(heatmap_feature_rn))

        #主要
        ##spatial
        # feat_S = mask_tokens[(num_sup+1)// 2]
        feat_S = vit_features + self.Spatial_pos_embed
        for blk in self.Spatial_blocks:
            feat_S = blk(feat_S)
        feat_S = self.Spatial_norm(feat_S)

        ##temporal
        tokens_temporal = torch.stack(vit_features.split(batch_size, dim=0), dim=0)
        feat_T = rearrange(tokens_temporal, 'f b p c -> (b p) f c')
        feat_T = feat_T + self.Temporal_pos_embed
        for blk in self.Temporal_blocks:
            feat_T = blk(feat_T)
        feat_T = self.Temporal_norm(feat_T)
        feat_T = rearrange(feat_T, '(b p) f c -> (f b) p c', p=192)
        # feat_T = feat_T[self.num_frame // 2]
        # mask_tokens_temporal = torch.stack(feat_T.split(batch_size, dim=0), dim=0)

        feat_agg = torch.concat((feat_T, feat_S_n), dim=1)
        feat_agg = self.project_primary(feat_agg.permute(0, 2, 1))
        feat_agg1 = feat_agg.permute(0, 2, 1)

        # feat_fused = self.TFF(vit_features,mask=False)

        feat_agg = torch.cat(torch.chunk(feat_agg1, chunks=num_sup + 1, dim=0), dim=1)
        feature_heatmap = self.final_norm(feat_agg)
        heatmap_feature = self.hf_norm(self.project_fusion(feature_heatmap.permute(0, 2, 1)))
        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], 1024, 16, 12)
        heatmap_feature_r1 = self.up_layer_hr_1(heatmap_feature)

        heatmap_p = self.final_layer_p(self.final_refine_p(heatmap_feature_r1))

        fused_features = self.final_refine(self.relu(heatmap_feature_r1 + heatmap_feature_rm + heatmap_feature_rn))

        heatmap = self.final_layer(fused_features)





        if self.is_train:


            return heatmap,kf_heatmap,heatmap_p,heatmap_m,heatmap_n
            # return heatmap,kf_heatmap,heatmap_p,heatmap_m,heatmap_n,mask_cosine_loss,noise_cosine_loss
        else:
           # return heatmap_p, kf_heatmap    87.7
            return heatmap, kf_heatmap     #87.8



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
        if self.freeze_hrnet_weight:
            self.backbone.freeze_weight()

        self.logger.info("Finish init_weights")

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def feat_label_mi_estimation(self, Feat, Y, motion=False):
        """
            F: [B,48,96,72]
            Y: [B,17,96,72]
        """
        batch_size = Feat.shape[0]
        temperature = 0.05
        pred_Y = self.hrnet.final_layer(Feat)  # B,48,96,72 -> B,17,96,72
        pred_Y = pred_Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        if motion:
            motion_Y = self.motion_heatmap_head(Feat)
            mi = kl_div(input=self.softmax(motion_Y.detach() / temperature),
                        target=self.softmax(Y / temperature),
                        reduction='mean')  # pixel-level
        else:
            mi = kl_div(input=self.softmax(pred_Y.detach() / temperature),
                        target=self.softmax(Y / temperature),
                        reduction='mean')  # pixel-level

        return mi

    def feat_feat_mi_estimation(self, F1, F2, freeze=True):
        """
            F1: [B,48,96,72]
            F2: [B,48,96,72]
            F1 -> F2
        """
        batch_size = F1.shape[0]
        temperature = 0.05
        F1 = F1.reshape(batch_size, 48, -1).reshape(batch_size * 48, -1)
        F2 = F2.reshape(batch_size, 48, -1).reshape(batch_size * 48, -1)
        if freeze:
            mi = kl_div(input=self.softmax(F1.detach() / temperature),
                        target=self.softmax(F2 / temperature))
        else:
            mi = kl_div(input=self.softmax(F1 / temperature),
                        target=self.softmax(F2 / temperature))

        return mi

    # def heatmaps_affine_transformation(self, heatmaps: torch.Tensor, offsets):
    #     """
    #         heatmaps : (batch, num_joints, map_height, map_width)
    #         offsets : (batch, num_joints*2)
    #         theta:[
    #             [1, 0, c_1],[0, 1, c_2]
    #         ]
    #         Note: c_1
    #     """
    #     batch_size, num_joints, map_height, map_width = heatmaps.size()
    #     # (batch, num_joints, 2, 3)
    #     batch_theta = torch.zeros((batch_size, num_joints, 2, 3), device=heatmaps.device)
    #     batch_theta[:, :, [0, 1], [0, 1]] = 1
    #     # 平移
    #     batch_theta[:, :, 0, 2] = offsets[:, ::2]  # x offset
    #     batch_theta[:, :, 1, 2] = offsets[:, 1::2]  # y offset
    #
    #     heatmaps = heatmaps.reshape(batch_size * num_joints, 1, map_height, map_width)
    #     batch_theta = batch_theta.reshape(batch_size * num_joints, 2, 3)
    #     output = kornia.geometry.warp_affine(heatmaps, batch_theta, dsize=(map_height, map_width))
    #     output = output.reshape(batch_size, num_joints, map_height, map_width)
    #     return output

    def temporal_channel_conv(self, features, times, conv):
        """
        features: [Batch, Channel, H, W] 多帧在channel维度堆叠好的张量
        """
        batch_size, channel, height, width = features.shape[0], features.shape[1] // times, \
                                             features.shape[2], \
                                             features.shape[3]
        # Channel Concat -> [B,T,C, H,W] -> [B,H,W,C,T] -> [BHW, C, T]
        features = features.contiguous().view(batch_size, -1, channel, height, width).permute(0, 3,
                                                                                              4, 2,
                                                                                              1) \
            .contiguous().view(-1, channel, times)
        agg_sup_feat = conv(features)
        agg_sup_feat = agg_sup_feat.view(batch_size, height, width, channel, times).permute(0, 3, 4,
                                                                                            1, 2) \
            .view(batch_size, -1, height, width)
        return agg_sup_feat

    def motion_agg_module(self):
        pass

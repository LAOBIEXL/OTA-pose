#!/usr/bin/python
# -*- coding:utf8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch.nn as nn
import torch.nn.functional as F
from posetimation.layers.cluster import Block, TCBlock, OverlapPatchEmbed, CTM
from posetimation.layers.cluster import merge_tokens, cluster_dpc_knn, token2map,map2token
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
__all__ = ["AuxposeV13_sub"]
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

class DensityPeakCluster:
    def __init__(self):
        pass

    def euclidean_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        distance = torch.norm(point1 - point2, dim=1)
        return distance

    def getDistanceMatrix(self, datas):
        """计算数据点两两之间的距离矩阵"""
        N = datas.shape[1]  # 获取数据点数量
        dists = torch.zeros((N, N))  # 初始化距离矩阵

        # 循环计算每对数据点之间的距离
        for i in range(N):
            for j in range(N):
                # 使用欧氏距离计算点之间的距离
                diff = datas[:, i, :].unsqueeze(1) - datas[:, j, :].unsqueeze(2)
                dists[i, j] = torch.norm(diff, p=2, dim=0).mean()

        return dists

    def select_dc(self, dists):
        """找到密度计算的阈值dc"""
        percent = 2.0
        position = int(torch.numel(dists) * percent / 100)
        dc = torch.sort(dists.view(-1))[0][position + torch.numel(dists)]
        return dc

    def get_density(self, dists, dc):
        """计算每个点的局部密度"""
        rho = torch.sum(torch.exp(-(dists / dc) ** 2), dim=1) - 1
        return rho

    def get_deltas(self, dists, rho):
        """计算每个数据点的密度距离"""
        nearest_neiber = torch.argmin(dists, dim=1)
        deltas = torch.min(torch.where(dists > 0, dists, torch.tensor(float('inf'))), dim=1)[0]
        deltas *= rho
        return deltas, nearest_neiber

    def find_centers_K(self, rho, deltas, K):
        """选取 rho 与 delta 乘积较大的点作为聚类中心"""
        rho_delta = rho * deltas
        _, centers = torch.sort(-rho_delta)
        return centers[:K]

    def cluster_PD(self, rho, centers, nearest_neiber):
        """密度峰值聚类算法 (DPC)"""
        labs = -1 * torch.ones_like(rho)
        for i, center in enumerate(centers):
            labs[center] = i
        _, index_rho = torch.sort(-rho)
        for i, index in enumerate(index_rho):
            if labs[index] == -1:
                labs[index] = labs[nearest_neiber[index].int()]
        return labs

    def fit_predict(self, datas, K=2):
        """聚类主函数，返回聚类标签"""
        # 计算距离矩阵
        dists = self.getDistanceMatrix(datas)

        # 计算 dc
        dc = self.select_dc(dists)

        # 计算局部密度
        rho = self.get_density(dists, dc)

        # 计算密度距离
        deltas, nearest_neiber = self.get_deltas(dists, rho)

        # 获取聚类中心点
        centers = self.find_centers_K(rho, deltas, K)

        # 进行密度峰值聚类
        labs = self.cluster_PD(rho, centers, nearest_neiber)

        return labs


class KMeansCluster:
    def __init__(self, n_clusters=2, num_iters=100):
        self.n_clusters = n_clusters
        self.num_iters = num_iters
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        """
        Fit the K-Means model to the data.
        Args:
            X (torch.Tensor): The input data of shape (batch_size, n_samples, n_features).
        """
        batch_size, n_samples, n_features = X.shape
        # 随机初始化聚类中心
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[:, indices, :]

        for i in range(self.num_iters):
            # 计算每个点到每个聚类中心的距离
            distances = torch.cdist(X, self.centroids, p=2)
            # 找到最近的聚类中心
            self.clusters = distances.argmin(dim=2)

            new_centroids = torch.zeros((batch_size, self.n_clusters, n_features), device=X.device)
            for k in range(self.n_clusters):
                mask = (self.clusters == k)
                for i in range(batch_size):
                    cluster_points = X[i][mask[i]]
                    if cluster_points.shape[0] > 0:  # 如果这个聚类中有点存在
                        new_centroids[i, k] = cluster_points.mean(dim=0)

            # 检查聚类中心是否变化，如果没有变化则提前停止
            if torch.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        Args:
            X (torch.Tensor): The input data of shape (batch_size, n_samples, n_features).
        Returns:
            clusters (torch.Tensor): Cluster indices for each sample.
        """
        distances = torch.cdist(X, self.centroids, p=2)
        return distances.argmin(dim=2)

    def split_clusters(self, X):
        """
        Split the data into clusters based on the fitted model.
        Args:
            X (torch.Tensor): The input data of shape (batch_size, n_samples, n_features).
        Returns:
            A (torch.Tensor): The data points belonging to cluster 0.
            B (torch.Tensor): The data points belonging to cluster 1.
        """
        if self.clusters is None:
            raise RuntimeError("The model has not been fitted yet.")

        batch_size, n_samples, n_features = X.shape
        A = []
        B = []
        for i in range(batch_size):
            A.append(X[i][self.clusters[i] == 0])
            B.append(X[i][self.clusters[i] == 1])

        A = torch.nn.utils.rnn.pad_sequence(A, batch_first=True)
        B = torch.nn.utils.rnn.pad_sequence(B, batch_first=True)

        return A, B



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
class AuxposeV13_sub(nn.Module):
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
        super(AuxposeV13_sub, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pretrained = cfg.MODEL.PRETRAINED
        self.is_train = is_train
        if self.is_train == TRAIN_PHASE:
            self.is_train = True
        else:
            self.is_train = False
        self.pretrained_layers = ['*']
        # self.hrnet = HRNetPlus(cfg, self.is_train)
        self.freeze_hrnet_weight = cfg['MODEL']["FREEZE_HRNET_WEIGHTS"]
        # self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS
        self.backbone = ViT.get_net(cfg)


        self.final_refine = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_refine_p = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_refine_m = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_refine_n = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_layer = nn.Conv2d(96, 15, 3, 1, 1)
        self.final_layer_m = nn.Conv2d(96, 15, 3, 1, 1)
        self.final_layer_n = nn.Conv2d(96, 15, 3, 1, 1)
        self.final_layer_p = nn.Conv2d(96, 15, 3, 1, 1)
        if self.freeze_hrnet_weight:
            self.backbone.freeze_weight()

        self.keypoint_tokens = nn.Parameter(torch.zeros(17, 1024))
        # self.encoder = []
        # self.decoder = []
        # h_dim = 1024
        # for _ in range(3):
        #     self.encoder.append(STTrans(h_dim, depth=1, mlp_dim=64, num_heads=8,
        #                                 dim_per_head=64, dropout=0))
        #     self.decoder.append(STTrans(h_dim, depth=1, mlp_dim=64, num_heads=8,
        #                                 dim_per_head=64, dropout=0))
        # self.encoder = nn.ModuleList(self.encoder)
        # self.decoder = nn.ModuleList(self.decoder)
        # self.TFF = Temporal_Feat_Fusion_Module(cfg, frame=5, dim=cfg.MODEL.VIT.EMB_DIM)
        self.channel = 192+17
        self.token_dim = 1024
        ###spatial block
        self.Spatial_pos_embed_mask = nn.Parameter(torch.zeros(1, self.channel,1024))
        self.Spatial_pos_embed_noise = nn.Parameter(torch.zeros(1, self.channel,1024))
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, self.channel,1024))
        self.Temporal_pos_embed_mask = nn.Parameter(torch.zeros(1, 3, 1024))######
        self.Temporal_pos_embed_noise = nn.Parameter(torch.zeros(1,3, 1024))#####
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 3, 1024))#######
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

        ######注意力分数计算
        self.query_proj = nn.Linear(self.token_dim, self.token_dim)
        self.key_proj = nn.Linear(self.token_dim, self.token_dim)

        self.num_frame = 3
        self.final_norm = nn.LayerNorm(1024)
        self.project_fusion = nn.Sequential(
            nn.Linear(192*self.num_frame, 192),
            nn.Linear(192, 192),
        )
        self.project_fusion_mask = nn.Sequential(
            nn.Linear(192*self.num_frame, 192),
            nn.Linear(192, 192),
        )

        self.project_mask = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel),
            nn.Linear(self.channel, self.channel),
        )
        self.project_noise = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel),
            nn.Linear(self.channel, self.channel),
        )
        self.project_primary = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel),
            nn.Linear(self.channel, self.channel),
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

        sample_ratios = 0.25
        embed_dims = 1024
        # k = 2
        self.ctm = CTM(sample_ratios, embed_dims, embed_dims, cluster_num = 5,k=30)
        self.cluster_norm = nn.LayerNorm(1024)
        self.init_weights()



    def forward(self, kf_x, sup_x, **kwargs):
        """
        kf_x:  [batch, 3, 384, 288]
        sup_x: [batch, 3 * num, 384, 288]]
        """

        batch_size, num_sup = kf_x.shape[0], sup_x.shape[1] // 3
        sup_x = torch.cat(torch.chunk(sup_x, num_sup, dim=1), dim=0)
        # sup_4 =   torch.chunk(sup_x, num_sup, dim=1)
        # sup_1 = sup_4[1]
        # sup_2 = sup_4[2]
        # sup_x = torch.cat([sup_1,sup_2], dim=0)  # [batch * num, 3, 384, 288]]
        # num_sup = 2#三帧辅助帧
        # sup_x = torch.cat(torch.chunk(sup_x, num_sup, dim=1), dim=0)  # [batch * num, 3, 384, 288]]

        x = torch.cat([kf_x, sup_x], dim=0)

        vit_features , htms_cat_batch = self.backbone(x)
        keypoint_tokens = self.keypoint_tokens.unsqueeze(0).expand(batch_size*3, -1, -1)
        tokens = torch.cat([vit_features, keypoint_tokens], dim=1)
        vit_features = tokens
        channel = vit_features.shape[1]
        # backbone_heatmaps = self.keypoint_head(vit_features)
        seq_heatmap_list = torch.chunk(htms_cat_batch, num_sup + 1, dim=0)

        kf_heatmap =  seq_heatmap_list[0]


        #mask
        B = batch_size*(num_sup+1)
        N = channel
        T = 1024
        mask_ratio = 0.5

        # 创建一个布尔类型的mask，初始状态全部为True
        mask = torch.ones((B, N, T), device=vit_features.device)

        # 创建batch_ind, agent_ind用于后续的索引操作
        batch_ind = torch.arange(B, device=vit_features.device)[:, None, None]
        agent_ind = torch.arange(N, device=vit_features.device)[None, :, None]

        # 计算mask的数量
        mask_count = int(T * mask_ratio)

        # 随机生成mask数量的索引
        random_indices = torch.rand(B, N, mask_count, device=vit_features.device)

        # 将随机值归一化到[0, T-1]区间
        random_indices = (random_indices * T).long()

        # 使用生成的索引更新mask
        mask[batch_ind, agent_ind, random_indices] = 0
        '''
        batch_ind = torch.arange(B)[:, None, None].cuda()#B，1，1
        agent_ind = torch.arange(N)[None, :, None].cuda()#1.N，1
        mask = torch.ones((B, N, T)).type_as(vit_features)#B，N，T
        shuffle_indices = torch.rand((B, N, T)).argsort().cuda()#B，N，T
        mask_indices = shuffle_indices[:, :, :int(T * mask_ratio)]#B，N，T*ratio
        mask[batch_ind, agent_ind, mask_indices] = 0
        # assert torch.sum(mask) == (self.past_timestep - int(self.past_timestep * mask_ratio)) * B * N
        '''
        # mask = mask.view(B, N, -1)
        inverse_mask = 1 - mask
        mask_tokens_gt = vit_features *inverse_mask #被mask掉的特征

        mask_tokens = vit_features * mask#已知的特征

        ##spatial
        # feat_S = mask_tokens[(num_sup+1)// 2]
        feat_S_m = mask_tokens + self.Spatial_pos_embed_mask
        for blk in self.Spatial_blocks:
            feat_S_m = blk(feat_S_m)
        feat_S_m = self.Spatial_norm_mask(feat_S_m)

        feat_S_m_T = torch.stack(feat_S_m.split(batch_size, dim=0), dim=0)
        feat_S_m_T = rearrange(feat_S_m_T, 'f b p c -> (b p) f c')
        feat_S_m_T = feat_S_m_T + self.Temporal_pos_embed_mask
        for blk in self.Temporal_blocks:
            feat_S_m_T = blk(feat_S_m_T)
        feat_S_m_T = self.Temporal_norm_mask(feat_S_m_T)
        feat_S_m_T = rearrange(feat_S_m_T, '(b p) f c -> (f b) p c', p=N)


        ##temporal
        mask_tokens_temporal = torch.stack(mask_tokens.split(batch_size, dim=0), dim=0)
        feat_T_m = rearrange(mask_tokens_temporal, 'f b p c -> (b p) f c')
        feat_T_m = feat_T_m + self.Temporal_pos_embed_mask
        for blk in self.Temporal_blocks:
            feat_T_m = blk(feat_T_m)
        feat_T_m = self.Temporal_norm_mask(feat_T_m)
        feat_T_m = rearrange(feat_T_m, '(b p) f c -> (f b) p c', p=N)

        feat_T_m_S = feat_T_m + self.Spatial_pos_embed_mask
        for blk in self.Spatial_blocks:
            feat_T_m_S = blk(feat_T_m_S)
        feat_T_m_S = self.Spatial_norm_mask(feat_T_m_S)

        # feat_T = feat_T[self.num_frame // 2]
        # mask_tokens_temporal = torch.stack(feat_T.split(batch_size, dim=0), dim=0)

        feat_agg_mask = torch.concat((feat_S_m_T, feat_T_m_S), dim=1)
        feat_agg_mask = self.project_mask(feat_agg_mask.permute(0, 2, 1))
        feat_agg_mask = feat_agg_mask.permute(0, 2, 1)  # 生成的被mask掉的特征

        feat_visual = feat_agg_mask[:, :192, :]
        feat_keypoint = feat_agg_mask[:, 192:, :]


        feat_fused_he = feat_visual + mask_tokens[:, :192, :]

        ######取重要不重要token'
        Q = self.query_proj(feat_keypoint)
        K = self.key_proj(feat_fused_he)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.token_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        num_top_tokens = 17*2
        B, num_visual_tokens, token_dim = feat_fused_he.size()

        # 平均每个关键点的注意力分数来选择重要的 visual tokens
        mean_attention_scores = attention_weights.mean(dim=1)  # (batch_size, num_visual_tokens)
        top_indices = mean_attention_scores.topk(num_top_tokens, dim=1).indices  # (batch_size, num_top_tokens)

        # 根据 top_indices 选择重要的 visual tokens
        top_visual_tokens = torch.stack(
            [feat_fused_he[i, top_indices[i]] for i in range(B)])  # (batch_size, num_top_tokens, token_dim)

        # 选择不重要的 visual tokens
        all_indices = torch.arange(num_visual_tokens).unsqueeze(0).expand(B,
                                                                          -1)  # (batch_size, num_visual_tokens)
        mask1 = torch.ones_like(all_indices, dtype=torch.bool,device=vit_features.device)
        mask1.scatter_(1, top_indices, False)
        other_indices = [all_indices[i][mask1[i]] for i in
                         range(B)]  # List of tensors (num_visual_tokens - num_top_tokens)
        other_visual_tokens = torch.stack([feat_fused_he[i, other_indices[i]] for i in range(
            B)])  # (batch_size, num_visual_tokens - num_top_tokens, token_dim)###对他进行聚类

        # init token dict
        x = other_visual_tokens
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        # H =
        token_dict = {'x': x,
                      'token_num': N,
                      # 'map_size': [H, W],
                      # 'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}

        cluster_token = self.ctm(token_dict)
        cluster_token = self.cluster_norm(cluster_token)

        mask_feature = torch.cat((top_visual_tokens,cluster_token),dim=1)




        mask_feature1 = torch.cat(torch.chunk(mask_feature, chunks=num_sup+1, dim=0), dim=1)
        feature_heatmap = self.final_norm(mask_feature1)
        heatmap_feature = self.hf_norm(self.project_fusion_mask(feature_heatmap.permute(0, 2, 1)))
        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], 1024, 16, 12)
        heatmap_feature_rm = self.up_layer_hr_1(heatmap_feature)
        heatmap_m = self.final_layer_m(self.final_refine_m(heatmap_feature_rm))
        #end mask

        #noise
        noise_dev = self.noise_dev
        B,N,T = vit_features.shape
        noise = torch.normal(mean=0., std=noise_dev, size=(B, N, T)).type_as(vit_features).cuda()
        noise_mask = torch.rand_like(noise, device=noise.device)
        noise = noise * noise_mask[:, :, :]
        '''
        noise = torch.from_numpy(np.random.normal(loc=0., scale=noise_dev, size=(B, N, T))).type_as(vit_features)
        noise_mask = torch.rand(B, N, T).cuda()
        noise = noise * noise_mask[:, :, :]
        '''
        all_traj_noise = vit_features + noise


        ##spatial
        # feat_S = mask_tokens[(num_sup+1)// 2]
        feat_S_n = all_traj_noise + self.Spatial_pos_embed_noise
        for blk in self.Spatial_blocks:
            feat_S_n = blk(feat_S_n)
        feat_S_n = self.Spatial_norm_noise(feat_S_n)

        feat_S_n_T = torch.stack(feat_S_n.split(batch_size, dim=0), dim=0)
        feat_S_n_T = rearrange(feat_S_n_T, 'f b p c -> (b p) f c')
        feat_S_n_T = feat_S_n_T + self.Temporal_pos_embed_noise
        for blk in self.Temporal_blocks:
            feat_S_n_T = blk(feat_S_n_T)
        feat_S_n_T = self.Temporal_norm_noise(feat_S_n_T)
        feat_S_n_T = rearrange(feat_S_n_T, '(b p) f c -> (f b) p c', p=N)

        ##temporal
        noise_tokens_temporal = torch.stack(all_traj_noise.split(batch_size, dim=0), dim=0)
        feat_T_n = rearrange(noise_tokens_temporal, 'f b p c -> (b p) f c')
        feat_T_n = feat_T_n + self.Temporal_pos_embed_noise
        for blk in self.Temporal_blocks:
            feat_T_n = blk(feat_T_n)
        feat_T_n = self.Temporal_norm_noise(feat_T_n)
        feat_T_n = rearrange(feat_T_n, '(b p) f c -> (f b) p c', p=N)
        # feat_T = feat_T[self.num_frame // 2]
        # mask_tokens_temporal = torch.stack(feat_T.split(batch_size, dim=0), dim=0)

        feat_T_n_S = feat_T_n + self.Spatial_pos_embed_noise
        for blk in self.Spatial_blocks:
            feat_T_n_S = blk(feat_T_n_S)
        feat_T_n_S = self.Spatial_norm_noise(feat_T_n_S)



        feat_agg_noise = torch.concat((feat_T_n_S, feat_S_n_T), dim=1)
        feat_agg_noise = self.project_noise(feat_agg_noise.permute(0, 2, 1))
        feat_agg_noise = feat_agg_noise.permute(0, 2, 1)  # 生成的被noise掉的特征

        feat_visual_noise = feat_agg_noise[:, :192, :]
        feat_keypoint_noise = feat_agg_noise[:, 192:, :]

        # feat_fused = self.TFF(mask_tokens,mask=True)

        # embedding1_flat = feat_agg_mask.view(B, -1)
        # embedding2_flat = mask_tokens_gt.view(B, -1)
        # cosine_sim1 = F.cosine_similarity(embedding1_flat, embedding2_flat, dim=1)
        # mask_cosine_loss = 1 - cosine_sim1.mean()

        # feat_fused_he = feat_visual + mask_tokens[:, :192, :]

        ######取重要不重要token'
        Q = self.query_proj(feat_keypoint_noise)
        K = self.key_proj(feat_visual_noise)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.token_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        num_top_tokens = 17 * 2
        B, num_visual_tokens, token_dim = feat_visual_noise.size()

        # 平均每个关键点的注意力分数来选择重要的 visual tokens
        mean_attention_scores = attention_weights.mean(dim=1)  # (batch_size, num_visual_tokens)
        top_indices = mean_attention_scores.topk(num_top_tokens, dim=1).indices  # (batch_size, num_top_tokens)

        # 根据 top_indices 选择重要的 visual tokens
        top_visual_tokens_noise = torch.stack(
            [feat_visual_noise[i, top_indices[i]] for i in range(B)])  # (batch_size, num_top_tokens, token_dim)

        # 选择不重要的 visual tokens
        all_indices = torch.arange(num_visual_tokens).unsqueeze(0).expand(B,
                                                                          -1)  # (batch_size, num_visual_tokens)
        mask1 = torch.ones_like(all_indices, dtype=torch.bool, device=vit_features.device)
        mask1.scatter_(1, top_indices, False)
        other_indices = [all_indices[i][mask1[i]] for i in
                         range(B)]  # List of tensors (num_visual_tokens - num_top_tokens)
        other_visual_tokens_noise = torch.stack([feat_visual_noise[i, other_indices[i]] for i in range(
            B)])  # (batch_size, num_visual_tokens - num_top_tokens, token_dim)###对他进行聚类

        # init token dict
        x = other_visual_tokens_noise
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        # H =
        token_dict = {'x': x,
                      'token_num': N,
                      # 'map_size': [H, W],
                      # 'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}

        cluster_token_noise = self.ctm(token_dict)
        cluster_token_noise = self.cluster_norm(cluster_token_noise)
        noise_feature = torch.cat((top_visual_tokens_noise, cluster_token_noise), dim=1)

        noise_feature1 = torch.cat(torch.chunk(noise_feature, chunks=num_sup+1, dim=0), dim=1)
        feature_heatmap = self.final_norm(noise_feature1)
        heatmap_feature = self.hf_norm(self.project_fusion_mask(feature_heatmap.permute(0, 2, 1)))
        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], 1024,16, 12)
        heatmap_feature_rn = self.up_layer_hr_1(heatmap_feature)
        heatmap_n = self.final_layer_n(self.final_refine_n(heatmap_feature_rn))

        #主要
        ##spatial
        # feat_S = mask_tokens[(num_sup+1)// 2]
        B,N,C = vit_features.shape
        feat_S = vit_features + self.Spatial_pos_embed
        for blk in self.Spatial_blocks:
            feat_S = blk(feat_S)
        feat_S = self.Spatial_norm(feat_S)

        feat_S_T = torch.stack(feat_S.split(batch_size, dim=0), dim=0)
        feat_S_T = rearrange(feat_S_T, 'f b p c -> (b p) f c')
        feat_S_T = feat_S_T + self.Temporal_pos_embed
        for blk in self.Temporal_blocks:
            feat_S_T = blk(feat_S_T)
        feat_S_T = self.Temporal_norm(feat_S_T)
        feat_S_T = rearrange(feat_S_T, '(b p) f c -> (f b) p c', p=N)

        ##temporal
        tokens_temporal = torch.stack(vit_features.split(batch_size, dim=0), dim=0)
        feat_T = rearrange(tokens_temporal, 'f b p c -> (b p) f c')
        feat_T = feat_T + self.Temporal_pos_embed
        for blk in self.Temporal_blocks:
            feat_T = blk(feat_T)
        feat_T = self.Temporal_norm(feat_T)
        feat_T = rearrange(feat_T, '(b p) f c -> (f b) p c', p=N)
        # feat_T = feat_T[self.num_frame // 2]
        # mask_tokens_temporal = torch.stack(feat_T.split(batch_size, dim=0), dim=0)

        feat_T_S = feat_T + self.Spatial_pos_embed
        for blk in self.Spatial_blocks:
            feat_T_S = blk(feat_T_S)
        feat_T_S = self.Spatial_norm(feat_T_S)

        feat_agg = torch.concat((feat_T_S, feat_S_T), dim=1)
        feat_agg = self.project_primary(feat_agg.permute(0, 2, 1))
        feat_agg1 = feat_agg.permute(0, 2, 1)

        # feat_fused = self.TFF(vit_features,mask=False)

        feat_visual_n = feat_agg1[:, :192, :]
        feat_keypoint_n = feat_agg1[:, 192:, :]

        # feat_fused = self.TFF(mask_tokens,mask=True)

        # embedding1_flat = feat_agg_mask.view(B, -1)
        # embedding2_flat = mask_tokens_gt.view(B, -1)
        # cosine_sim1 = F.cosine_similarity(embedding1_flat, embedding2_flat, dim=1)
        # mask_cosine_loss = 1 - cosine_sim1.mean()

        # feat_fused_he = feat_visual + mask_tokens[:, :192, :]

        ######取重要不重要token'
        Q = self.query_proj(feat_keypoint_n)
        K = self.key_proj(feat_visual_n)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.token_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        num_top_tokens = 17 * 2
        B, num_visual_tokens, token_dim = feat_visual_n.size()

        # 平均每个关键点的注意力分数来选择重要的 visual tokens
        mean_attention_scores = attention_weights.mean(dim=1)  # (batch_size, num_visual_tokens)
        top_indices = mean_attention_scores.topk(num_top_tokens, dim=1).indices  # (batch_size, num_top_tokens)

        # 根据 top_indices 选择重要的 visual tokens
        top_visual_tokens_n = torch.stack(
            [feat_visual_n[i, top_indices[i]] for i in range(B)])  # (batch_size, num_top_tokens, token_dim)

        # 选择不重要的 visual tokens
        all_indices = torch.arange(num_visual_tokens).unsqueeze(0).expand(B,
                                                                          -1)  # (batch_size, num_visual_tokens)
        mask1 = torch.ones_like(all_indices, dtype=torch.bool, device=vit_features.device)
        mask1.scatter_(1, top_indices, False)
        other_indices = [all_indices[i][mask1[i]] for i in
                         range(B)]  # List of tensors (num_visual_tokens - num_top_tokens)
        other_visual_tokens_n = torch.stack([feat_visual_n[i, other_indices[i]] for i in range(
            B)])  # (batch_size, num_visual_tokens - num_top_tokens, token_dim)###对他进行聚类

        # init token dict
        x = other_visual_tokens_n
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        # H =
        token_dict = {'x': x,
                      'token_num': N,
                      # 'map_size': [H, W],
                      # 'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}

        cluster_token_n = self.ctm(token_dict)
        cluster_token_n = self.cluster_norm(cluster_token_n)
        feat_agg = torch.cat((top_visual_tokens_n, cluster_token_n), dim=1)

        feat_agg1 = torch.cat(torch.chunk(feat_agg, chunks=num_sup + 1, dim=0), dim=1)
        feature_heatmap = self.final_norm(feat_agg1)
        heatmap_feature = self.hf_norm(self.project_fusion(feature_heatmap.permute(0, 2, 1)))
        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], 1024,16, 12)
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




#!/usr/bin/python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch.nn.functional as F
from functools import wraps

import copy
from engine.defaults.constant import MODEL_REGISTRY
from posetimation.layers.basic_layer import conv_bn_relu
from posetimation.layers.basic_model import ChainOfBasicBlocks, SinusoidalPositionEmbeddings
# from posetimation.layers.basic_model import ChainOfBasicBlocks, SinusoidalPositionEmbeddings, CorrBlock
from posetimation.backbones.vit import ViT, TopdownHeatmapSimpleHead, Block, PatchEmbed
from timm.models.layers import trunc_normal_
from engine.defaults import TRAIN_PHASE

__all__ = ["DiffusionPoseV24"]
BN_MOMENTUM = 0.1

import logging
import os.path as osp
import torch
from torch.nn.functional import kl_div
from torch.nn import init
import math
from posetimation.backbones.hrnet2 import HRNet
# from posetimation.backbones.mixste import *

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # 不懂


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@MODEL_REGISTRY.register()
class DiffusionPoseV24(nn.Module):

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
        super(DiffusionPoseV24, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pretrained = cfg.MODEL.PRETRAINED
        self.is_train = is_train
        if self.is_train == TRAIN_PHASE:
            self.is_train = True
        else:
            self.is_train = False
        self.pretrained_layers = ['*']
        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        # self.feature_channel = 512
        self.feature_channel = 512
        # Diffusion Parameters
        timesteps = 1000
        sampling_timesteps = 4
        self.objective = 'pred_x0'
        self.betas = cosine_beta_schedule(timesteps)
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        timesteps, = self.betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 2.
        self.heatmap_scale = 3.

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)

        self.rough_pose_estimation_net = HRNet(cfg, self.is_train)


        # self.backbone = ViT()
        self.keypoint_head = TopdownHeatmapSimpleHead()
        self.keypoint_head.freeze_weight()

        dpr = [x.item() for x in torch.linspace(0, 0.14, 8)]
        self.global_blocks = nn.ModuleList([
            Block(
                dim=1280, num_heads=16, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
            )
            for i in range(4)])
        self.conv = nn.Conv1d(48, 512, kernel_size=1, stride=1)
        self.temporal_aggregation = nn.Sequential(
            nn.Linear(192 * 5, 192),
            nn.LayerNorm(192),
            nn.Linear(192, 192),
            nn.LayerNorm(192)
        )

        self.project_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 2),
        )

        self.project_fusion1 = nn.Sequential(
            nn.Linear(64, 48),
            nn.Linear(48, 12),
            nn.Linear(12, 6),
            nn.Linear(6, 2),
        )

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 16 * 12, 1280))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 16 * 12, 1280))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 16 * 12, 1280))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, 16 * 12, 1280))
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, 16 * 12, 1280))

        self.pos_embed_heatmap = nn.Parameter(torch.zeros(1, 432, self.feature_channel))
        self.pos_embed_coordinate = nn.Parameter(torch.zeros(1, 17, 512))

        self.final_norm = nn.LayerNorm(self.feature_channel)

        self.hf_norm = nn.LayerNorm(2)

        if self.pos_embed_1 is not None:
            trunc_normal_(self.pos_embed_1, std=.02)
            trunc_normal_(self.pos_embed_2, std=.02)
            trunc_normal_(self.pos_embed_3, std=.02)
            trunc_normal_(self.pos_embed_4, std=.02)
            trunc_normal_(self.pos_embed_5, std=.02)
            trunc_normal_(self.pos_embed_heatmap, std=.02)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, 17),
            nn.GELU(),
            nn.Linear(17, 17),
            nn.Sigmoid()
        )

        self.reduce_st = nn.Sequential(
            nn.Linear(1280, self.feature_channel),
            nn.Linear(self.feature_channel, self.feature_channel),
        )

        self.up_layer_global_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1280, out_channels=192, kernel_size=4, stride=2, padding=1,
                               output_padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1, 1, 0),
        )
        self.conv_r2 = ChainOfBasicBlocks(192, 192, num_blocks=2)

        self.up_layer_global_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1280, out_channels=192, kernel_size=4, stride=2, padding=1,
                               output_padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 96, 1, 1, 0),
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, 1, 0)
        )
        self.conv_r3 = ChainOfBasicBlocks(96, 96, num_blocks=2)
        embed_dim_ratio = 32
        # self.heatmap_patch = nn.Conv2d(17, self.feature_channel, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        self.coordinate_patch = nn.Linear(578,512)
        # self.heatmap_patch = nn.Conv2d(17, self.feature_channel, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        self.heatmap_self_atten = nn.ModuleList([
            Block(
                dim=self.feature_channel, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
            )
            for i in range(1)])
        self.heatmap_interaction_r1 = nn.ModuleList([
            Block(
                dim= self.feature_channel, num_heads=16, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
            )
            for i in range(3)])

        self.up_layer_hr_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.feature_channel, out_channels=192, kernel_size=4, stride=2, padding=1,
                               output_padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 96, 1, 1, 0),
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, 1, 0)
        )

        self.conv_thr2 = ChainOfBasicBlocks(17, 192, num_blocks=1)
        self.thr2_conv_reduce = nn.Sequential(
            nn.Conv2d(192, 96, 3, 2, 1),
            nn.Conv2d(96, 17, 3, 1, 1),
            nn.ReLU()
        )
        self.heatmap_interaction_r2 = ChainOfBasicBlocks(192 * 2, 192, num_blocks=3)

        self.up_layer_hr_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1,
                               output_padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, 1, 0),
        )

        self.conv_thr3 = ChainOfBasicBlocks(17, 96, num_blocks=1)
        self.thr3_conv_reduce = nn.Sequential(
            nn.Conv2d(96, 17, 3, 1, 1),
            nn.ReLU()
        )
        self.heatmap_interaction_r3 = ChainOfBasicBlocks(96 * 2, 96, num_blocks=2)

        # heatmap head
        self.relu = nn.ReLU()
        self.final_refine = ChainOfBasicBlocks(96, 96, num_blocks=2)
        self.final_layer = nn.Conv2d(48, 17, 3, 1, 1)



        self.conv_corr = nn.Conv1d(48,48,12,12)
        self.conv_corr1 = nn.Conv1d(48,17,1,1)
        self.block1 = ChainOfBasicBlocks(17+48,72,num_blocks=2)
        self.block2 = ChainOfBasicBlocks(72,96,num_blocks=2)
        self.block3 = ChainOfBasicBlocks(96,192,num_blocks=2)
        self.block4 = ChainOfBasicBlocks(192,96,num_blocks=2)
        self.block5 = ChainOfBasicBlocks(96,96,num_blocks=2)
        self.block6 = ChainOfBasicBlocks(96,48,num_blocks=2)
        # self.block1 = ChainOfBasicBlocks(96,17,num_blocks=2)
        self.coor_feature1 = nn.Conv2d(48,48,6,6) #16 12
        self.coor_feature2 = nn.Conv1d(48,17,3,3)#64,17,64
        self.coor_feature3 = nn.Conv1d(17,17,8,8)
        self.init_weights()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod.cuda(), t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod.cuda(), t, x_t.shape)
        )

    def model_predictions(self, backbone_feats_corr, x, t, x_self_cond=None, clip_x_start=False):
        x = torch.clamp(x, min=-1 * self.heatmap_scale, max=self.heatmap_scale)
        final_heatmap = self._inference(x, t, backbone_feats_corr)
        x_start = final_heatmap.clone()
        x_start = torch.clamp(x_start, min=-1 * self.heatmap_scale, max=self.heatmap_scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, final_heatmap

    @torch.no_grad()
    def ddim_sample(self, init_heatmaps, backbone_feats_corr, clip_denoised=True, do_postprocess=False):
        # batch = backbone_feats_corr[0].shape[0]
        batch = backbone_feats_corr.shape[0]
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, output_heatmap = self.model_predictions(backbone_feats_corr, init_heatmaps.cuda(),
                                                                         time_cond.cuda(),
                                                                         self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                init_heatmaps = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(init_heatmaps)

            init_heatmaps = x_start * alpha_next.sqrt().cuda() + \
                            c.cuda() * pred_noise + \
                            sigma.cuda() * noise.cuda()

        return output_heatmap

    def model_predictions_coordinate(self, backbone_feats_corr, x, t, x_self_cond=None, clip_x_start=False):
        x = torch.clamp(x, min=-1 * self.heatmap_scale, max=self.heatmap_scale)
        final_coordinate,_ = self._inference_coordinates(x, t, backbone_feats_corr)
        x_start = final_coordinate.clone()
        x_start = torch.clamp(x_start, min=-1 * self.heatmap_scale, max=self.heatmap_scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, final_coordinate

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod.cuda(), t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod.cuda(), t, x_start.shape)

        return sqrt_alphas_cumprod_t.cuda() * x_start + sqrt_one_minus_alphas_cumprod_t.cuda() * noise

    def prepare_diffusion_concat(self, gt_heatmaps):
        channel, height, width = gt_heatmaps.shape

        t = torch.randint(0, self.num_timesteps, (1,)).long()
        noise = torch.randn(channel, height, width)

        x_start = gt_heatmaps.clone()
        x = self.q_sample(x_start=x_start.cuda(), t=t.cuda(), noise=noise.cuda())

        x = torch.clamp(x, min=-1 * self.heatmap_scale, max=self.heatmap_scale)

        return x, noise, t

    #coordinate
    def prepare_diffusion_coordinate_concat(self, gt_coordinates):
        channel, height = gt_coordinates.shape

        t = torch.randint(0, self.num_timesteps, (1,)).long()
        noise = torch.randn(channel, height)

        x_start = gt_coordinates.clone()
        x = self.q_sample(x_start=x_start.cuda(), t=t.cuda(), noise=noise.cuda())

        x = torch.clamp(x, min=-1 * self.heatmap_scale, max=self.heatmap_scale)

        return x, noise, t
    def prepare_targets(self, gt_heatmaps):
        diffused_heatmaps = []
        noises = []
        ts = []
        for i in range(gt_heatmaps.shape[0]):
            d_heatmaps, d_noise, d_t = self.prepare_diffusion_concat(gt_heatmaps[i])
            diffused_heatmaps.append(d_heatmaps)
            noises.append(d_noise)
            ts.append(d_t)
        return torch.stack(diffused_heatmaps), torch.stack(noises), torch.stack(ts)

    #coordinate
    def prepare_targets_coordinate(self, gt_coordiante):
        diffused_coordinates = []
        noises = []
        ts = []
        for i in range(gt_coordiante.shape[0]):
            d_coordinates, d_noise, d_t = self.prepare_diffusion_coordinate_concat(gt_coordiante[i])
            diffused_coordinates.append(d_coordinates)
            noises.append(d_noise)
            ts.append(d_t)
        return torch.stack(diffused_coordinates), torch.stack(noises), torch.stack(ts)

    def forward(self, kf_x, sup_x, gt_heatmaps, **kwargs):
        #kf_x 1,3,256,192 sup_x 1,12,256,192 gt_heatmap 1,17,64,48
        batch_size, num_sup = kf_x.shape[0], sup_x.shape[1] // 3
        # sup_x = torch.cat(torch.chunk(sup_x, num_sup, dim=1), dim=0)#4,3,256,192
        x = torch.cat([kf_x,sup_x], dim=1)#batch 3*5 256 192

        # hrnet_features, hrnet_heatmaps = self.rough_pose_estimation_net(x)
        #torch.cat(x.split(3, dim=1), 0) 5*batch 3 256 192
        #heatmap
        hrnet_features, hrnet_heatmaps = self.rough_pose_estimation_net(torch.cat(x.split(3, dim=1), 0))
        # true_batch_size = int(hrnet_heatmaps.shape[0] / 5)
        sup_heatmaps_list = hrnet_heatmaps.split(batch_size, dim=0)
        kf_heatmap = sup_heatmaps_list[0]
        sup_heatemaps = sup_heatmaps_list[1:]
        sup_feature_list = hrnet_features.split(batch_size, dim=0)
        kf_featuremap = sup_feature_list[0]
        sup_featuremaps = sup_feature_list[1:]

        #coodinate


        # vit_features = self.backbone(x)#1+4 1280 16 12
        # channel = vit_features.shape[1]
        # backbone_heatmaps = self.keypoint_head(vit_features)#1+4 17, 64 48
        #
        # seq_feat_list = torch.chunk(vit_features, num_sup + 1, dim=0)#5个1 1280 16 12
        # seq_heatmap_list = torch.chunk(backbone_heatmaps, num_sup + 1, dim=0)#5个1 17 64 48
        #
        # kf_feature, kf_heatmap = seq_feat_list[0], seq_heatmap_list[0]
        # supp_features, supp_heatmaps = seq_feat_list[1:], seq_heatmap_list[1:]

        # pos_embed_list = [
        #     self.pos_embed_1,#1 16*12=192 1280
        #     self.pos_embed_2,
        #     self.pos_embed_3,
        #     self.pos_embed_4,
        #     self.pos_embed_5
        # ]

        # assert len(pos_embed_list) == len(seq_feat_list), "Position Embedding Error!"

        # new_seq_vit_feature_list = [
        #     seq_feat_list[i].flatten(2).permute(0, 2, 1) + pos_embed_list[i] for i in range(len(seq_feat_list))
        # ]#5个 1 192 1280

        # x = torch.stack(new_seq_vit_feature_list).permute(1, 0, 2, 3).contiguous().view(batch_size, -1, channel)
        #x = torch.stack(new_seq_vit_feature_list) [5, 1, 192, 1280])
        #1,5,192,1280
        # for block in self.global_blocks:
        #     x = block(x)#1 960 1280
        # x = self.temporal_aggregation(x.transpose(1, 2))#1 1280 192
        # multi_scale_st = []

        # resolution-1
        # x_r1 = self.reduce_st(x.transpose(1, 2))
        # multi_scale_st.append(x_r1)

        # resolution-2
        # x_r2 = self.up_layer_global_1(x.reshape(batch_size, 1280, 16, 12))
        # x_r2 = self.conv_r2(x_r2)
        # multi_scale_st.append(x_r2)

        # resolution-3
        # x_r3 = self.up_layer_global_2(x.reshape(batch_size, 1280, 16, 12))
        # x_r3 = self.conv_r3(x_r3)
        # multi_scale_st.append(x_r3)
        # multi_scale_st.append(kf_featuremap)
        # multi_scale_st.append(kf_featuremap)
        # multi_scale_st.append(kf_featuremap)
        heatmap_inetrmediate = torch.Tensor(0).cuda()
        coordinate_inetrmediate = torch.Tensor(0).cuda()
        if self.is_train:

            x_heatmaps, noises, t = self.prepare_targets(gt_heatmaps)
            t = t.squeeze(-1)
            final_heatmap = self._inference(x_heatmaps, t, kf_featuremap)
            return final_heatmap, heatmap_inetrmediate, kf_heatmap
        else:
            n_diffusion = 10
            shape = (batch_size, 17, 96,72)
            results = list()
            for _ in range(n_diffusion):
                init_heatmap = torch.randn(shape)
                # init_heatmap = torch.randn(shape)
                result = self.ddim_sample(init_heatmap, kf_featuremap)
                results.append(result)
            pred_hm = sum(results) / len(results)
            return pred_hm, kf_heatmap

    def _inference(self, x_heatmaps, time, corr_features):

        # st_r1, st_r2, st_r3 = corr_features
        #
        time_embedding = self.time_mlp(time.cuda())
        time_embedding = time_embedding.unsqueeze(0).unsqueeze(0).permute(2, 3, 0, 1)
        t_heatmaps = time_embedding * x_heatmaps
        #
        # x_heatmaps_patch_embedding = self.heatmap_patch(t_heatmaps.cuda().float())
        # x_heatmaps_patch_embedding = x_heatmaps_patch_embedding.flatten(2).transpose(1, 2)
        # x_heatmaps_patch_embedding += self.pos_embed_heatmap
        # for block in self.heatmap_self_atten:
        #     heatmap_atten = block(x_heatmaps_patch_embedding)
        # heatmap_atten = heatmap_atten.unsqueeze(-1)
        # heatmap_atten = heatmap_atten.reshape(batch_size, 48 * 96, 9).permute(0, 2, 1)
        # heatmap_atten = self.conv(heatmap_atten)
        # heatmap_atten.reshape(64,48,96,72)
        # input_dim = (batch_size, 48, 96, 9)
        # heatmap_atten = heatmap_atten.permute(0, 2, 1).reshape(*input_dim[:-1], 72)
        # index_corr_features = heatmap_atten * corr_features
        feature_heatmap = torch.cat([t_heatmaps, corr_features], dim=1)
        # for block in self.heatmap_interaction_r1:
        #     feature_heatmap = block(feature_heatmap)
        # feature_heatmap = self.final_norm(feature_heatmap)
        # heatmap_feature = self.hf_norm(self.project_fusion(feature_heatmap.permute(0, 2, 1)))
        # heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], self.feature_channel, 16, 12).contiguous()
        # heatmap_feature_r1 = self.up_layer_hr_1(heatmap_feature)

        # t_heatmaps_r2 = self.conv_thr2(t_heatmaps.cuda().float())
        # t_heatmaps_r2 = self.thr2_conv_reduce(t_heatmaps_r2)
        # t_heatmaps_r2_weight = torch.max(t_heatmaps_r2, dim=1, keepdim=True).values
        # index_feature_r2 = t_heatmaps_r2_weight * st_r2
        # hf = self.heatmap_interaction_r2(torch.cat([index_feature_r2, st_r2], dim=1))
        # heatmap_feature_r2 = self.up_layer_hr_2(hf)

        # t_heatmaps_r3 = self.conv_thr3(t_heatmaps.cuda().float())
        # t_heatmaps_r3 = self.thr3_conv_reduce(t_heatmaps_r3)
        # t_heatmaps_r3_weight = torch.max(t_heatmaps_r3, dim=1, keepdim=True).values
        # index_feature_r3 = t_heatmaps_r3_weight * st_r3
        # heatmap_feature_r3 = self.heatmap_interaction_r3(torch.cat([index_feature_r3, st_r3], dim=1))

        # fused_features = self.final_refine(self.relu(heatmap_feature_r1 + heatmap_feature_r2 + heatmap_feature_r3))
        feature_heatmap = feature_heatmap.float()
        feature_heatmap = self.block1(feature_heatmap)
        feature_heatmap = self.block2(feature_heatmap)
        feature_heatmap = self.block3(feature_heatmap)
        feature_heatmap = self.block4(feature_heatmap)
        feature_heatmap = self.block5(feature_heatmap)
        feature_heatmap = self.block6(feature_heatmap)
        feature_heatmap1 =self.coor_feature1(feature_heatmap)
        feature_heatmap1 = torch.reshape(feature_heatmap1,(feature_heatmap1.shape[0],feature_heatmap1.shape[1],-1))
        feature_heatmap1 = self.coor_feature2(feature_heatmap1)
        heatmap = self.final_layer(feature_heatmap)

        return heatmap
        # return heatmap,feature_heatmap1

    def _inference_coordinates(self, x_coordinates, time, corr_features):

        # st_r1, st_r2, st_r3 = corr_features
        #
        time_embedding = self.time_mlp(time.cuda())
        time_embedding = time_embedding.unsqueeze(0).permute(1,2,0)
        t_x_coordinates = time_embedding * x_coordinates
        #
        # x_heatmaps_patch_embedding = self.heatmap_patch(t_heatmaps.cuda().float())
        # x_heatmaps_patch_embedding = x_heatmaps_patch_embedding.flatten(2).transpose(1, 2)
        # x_heatmaps_patch_embedding += self.pos_embed_heatmap
        # for block in self.heatmap_self_atten:
        #     heatmap_atten = block(x_heatmaps_patch_embedding)
        # heatmap_atten = heatmap_atten.unsqueeze(-1)
        # heatmap_atten = heatmap_atten.reshape(batch_size, 48 * 96, 9).permute(0, 2, 1)
        # heatmap_atten = self.conv(heatmap_atten)
        # heatmap_atten.reshape(64,48,96,72)
        # input_dim = (batch_size, 48, 96, 9)
        # heatmap_atten = heatmap_atten.permute(0, 2, 1).reshape(*input_dim[:-1], 72)
        # index_corr_features = heatmap_atten * corr_features
        corr_features = torch.reshape(corr_features,(corr_features.shape[0],corr_features.shape[1],-1))
        corr_features = self.conv_corr(corr_features)
        corr_features = self.conv_corr1(corr_features)
        feature_coordinate = torch.cat([t_x_coordinates, corr_features], dim=2)

        x_coordinate_patch_embedding = self.coordinate_patch(feature_coordinate.cuda().float())
        # x_heatmaps_patch_embedding = x_coordinate_patch_embedding.transpose(1, 2)
        x_coordinate_patch_embedding += self.pos_embed_coordinate
        # x_coordinate_patch_embedding = x_coordinate_patch_embedding.reshape(64,-1)
        for block in self.heatmap_interaction_r1:
            feature_coordinate = block(x_coordinate_patch_embedding)
        feature_coordinate = self.final_norm(feature_coordinate)
        feature_coordinate1 = self.coor_feature3(feature_coordinate)
        coordinate = self.hf_norm(self.project_fusion1(feature_coordinate))
        # heatmap_feature = self.hf_norm(self.project_fusion(feature_heatmap.permute(0, 2, 1)))
        # heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], self.feature_channel, 16, 12).contiguous()
        # heatmap_feature_r1 = self.up_layer_hr_1(heatmap_feature)

        # t_heatmaps_r2 = self.conv_thr2(t_heatmaps.cuda().float())
        # t_heatmaps_r2 = self.thr2_conv_reduce(t_heatmaps_r2)
        # t_heatmaps_r2_weight = torch.max(t_heatmaps_r2, dim=1, keepdim=True).values
        # index_feature_r2 = t_heatmaps_r2_weight * st_r2
        # hf = self.heatmap_interaction_r2(torch.cat([index_feature_r2, st_r2], dim=1))
        # heatmap_feature_r2 = self.up_layer_hr_2(hf)

        # t_heatmaps_r3 = self.conv_thr3(t_heatmaps.cuda().float())
        # t_heatmaps_r3 = self.thr3_conv_reduce(t_heatmaps_r3)
        # t_heatmaps_r3_weight = torch.max(t_heatmaps_r3, dim=1, keepdim=True).values
        # index_feature_r3 = t_heatmaps_r3_weight * st_r3
        # heatmap_feature_r3 = self.heatmap_interaction_r3(torch.cat([index_feature_r3, st_r3], dim=1))

        # fused_features = self.final_refine(self.relu(heatmap_feature_r1 + heatmap_feature_r2 + heatmap_feature_r3))
        # feature_coordinate = feature_coordinate.float()
        # feature_coordinate = self.block1(feature_coordinate)
        # feature_coordinate = self.block2(feature_coordinate)
        # feature_coordinate = self.block3(feature_coordinate)
        # feature_coordinate = self.block4(feature_coordinate)
        # coordinate = self.final_layer(feature_coordinate)

        return coordinate,feature_coordinate1

    # def init_weights(self, *args, **kwargs):
    #     logger = logging.getLogger(__name__)
    #     for module_name, module in self.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.normal_(module.weight, std=0.001)
    #             for name, _ in module.named_parameters():
    #                 if name in ['bias']:
    #                     nn.init.constant_(module.bias, 0)
    #
    #         elif isinstance(module, nn.BatchNorm2d):
    #             nn.init.constant_(module.weight, 1)
    #             nn.init.constant_(module.bias, 0)
    #
    #         elif isinstance(module, nn.ConvTranspose2d):
    #             nn.init.normal_(module.weight, std=0.001)
    #
    #             for name, _ in module.named_parameters():
    #                 if name in ['bias']:
    #                     nn.init.constant_(module.bias, 0)
    #         else:
    #             for name, _ in module.named_parameters():
    #                 if name in ['bias']:
    #                     nn.init.constant_(module.bias, 0)
    #                 if name in ['weights']:
    #                     nn.init.normal_(module.weight, std=0.001)
    #
    #     if osp.isfile(self.pretrained):
    #         pretrained_state_dict = torch.load(self.pretrained)
    #         if 'state_dict' in pretrained_state_dict.keys():
    #             pretrained_state_dict = pretrained_state_dict['state_dict']
    #         logger.info('{} => loading pretrained model {}'.format(self.__class__.__name__,
    #                                                                self.pretrained))
    #
    #         if list(pretrained_state_dict.keys())[0].startswith('module.'):
    #             model_state_dict = {k[7:]: v for k, v in pretrained_state_dict.items()}
    #         else:
    #             model_state_dict = pretrained_state_dict
    #
    #         self.load_state_dict(model_state_dict, strict=False)
    #     elif self.pretrained:
    #         logger.error('=> please download pre-trained models first!')
    #
    #     self.logger.info("Finish init_weights")
    #     if self.freeze_hrnet_weights:
    #         self.rough_pose_estimation_net.freeze_weight()
    def init_weights(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        hrnet_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "rough_pose_estimation_net":
                hrnet_name_set.add(module_name)

            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if osp.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('{} => loading pretrained model {}'.format(self.__class__.__name__,
                                                                   self.pretrained))

            if list(pretrained_state_dict.keys())[0].startswith('module.'):
                model_state_dict = {k[7:]: v for k, v in pretrained_state_dict.items()}
            else:
                model_state_dict = pretrained_state_dict

            need_init_state_dict = {}
            for name, m in model_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in hrnet_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in hrnet_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m

            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
    #
        self.logger.info("Finish init_weights")
        if self.freeze_hrnet_weights:
            self.rough_pose_estimation_net.freeze_weight()
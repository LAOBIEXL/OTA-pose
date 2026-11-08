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
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
from ..base import BaseModel
from ..backbones.vit import Block , trunc_normal_
from ..layers.block_zoo import *
# from utils.vis import vis_htms
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
from .blocks import SelfCrossBlock
logger = logging.getLogger(__name__)

__all__ = ["UGPOSE"]



class ModelArgs:
    st_layers: int = 2
    middle_layers: int = 2


    def __post_init__(self):
        pass

def reparameterize( mu, logvar, k=1):
    sample_z = []
    for _ in range(k):
        std = logvar.mul(0.5).exp_()  # type: Variable
        eps = std.data.new(std.size()).normal_()
        sample_z.append(eps.mul(std).add_(mu))
    sample_z = torch.cat(sample_z, dim=1)
    return sample_z


@MODEL_REGISTRY.register()
class UGPOSE(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = UGPOSE(cfg, phase, **kwargs)
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
        self.phase = phase


        self.hidden_dim = cfg.MODEL.VIT.EMB_DIM

        self.htms_dict = dict()

        dpr = [x.item() for x in torch.linspace(0, 0.24, 4)]
        self.backbone = ViT_V1.get_net(cfg , phase)

        self.proj_conv =  nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False)

        self.mean_conv = nn.Conv2d(self.hidden_dim, 17, kernel_size=1, bias=False)
        self.std_conv = nn.Conv2d(self.hidden_dim, 17, kernel_size=1, bias=False)

        self.pmm = PMMs(self.hidden_dim, 16)

        self.encoder = nn.ModuleList([
            Block(
                dim=self.hidden_dim, num_heads=8, mlp_ratio = 2,
                 drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(2)])

        self.decoder = nn.ModuleList([
            SelfCrossBlock(
                dim=self.hidden_dim,kv_dim=self.hidden_dim ,  num_heads=8, mlp_ratio = 2 ,
                 drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(2)])

        kernel = torch.ones((7, 7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # kernel = np.repeat(kernel, 1, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)


        #
        # if self.final_fusion_backbone:
        #     self.out_backbone_fusion = CHAIN_RSB_BLOCKS(cfg.MODEL.VIT.DECONV2_OUT_CHN+cfg.MODEL.HEAD.DECONV2_OUT_CHN,
        #                                                 cfg.MODEL.HEAD.FINAL_CONV_IN_CHN , 2)


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


    def forward(self, input ,  **kwargs):
        true_batchsize , channel_num  = input.shape[0] , input.shape[1]//self.frame_num
        # print(input.shape)
        # print(true_batchsize , channel_num , dim)



        img_cat_batch= torch.cat(input.split(channel_num, dim=1), dim=0)

        #vit_backbone forward->outputs are two parts: feature and heatmaps
        htms_backbone_out_cat_batch , feats_backbone_out_cat_batch , feats_up1_backbone_out_cat_batch , feats_up2_backbone_out_cat_batch , \
                                                                                        = self.backbone(img_cat_batch)

        feat_up2_backbone_out_curr = feats_up2_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]
        feat_up1_backbone_out_curr = feats_up1_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]
        htms_curr= htms_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]
        feat_curr , feat_left , feat_right =  feats_backbone_out_cat_batch.split(true_batchsize, dim=0)

        feat = feat_curr.permute(0,2,1).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
        residual = self.proj_conv(feat)

        mean = self.mean_conv(feat)
        std = self.std_conv(feat)

        prob_x = reparameterize(mean, std, k=1)

        z = reparameterize(mean, std, k=10)
        z = torch.sigmoid(z)


        # uncertainty
        uncertainty = z.var(dim=1, keepdim=True).detach()
        if self.training:
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        residual *= (1 - uncertainty)
        if self.phase == 'train':
            rand_mask = uncertainty < torch.Tensor(np.random.random(uncertainty.size())).to(uncertainty.device)
            residual *= rand_mask.to(torch.float32)
        residual = residual.flatten(2).permute(0,2,1)

        feat, z_ = self.pmm(feat)
        feat = torch.stack(feat, dim=3).squeeze(-1)
        feat = feat.flatten(2).permute(0,2,1)

        for blk in self.encoder:
            feat = blk(feat)

        for blk in self.decoder:
            residual = blk(residual, feat)


        y = residual.permute(0,2,1).reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1])
        y = self.up_sample(y)
        y = self.final_layer(y)

        return y , htms_curr , prob_x , feat

        #
        # if self.training:
        #     main_loss = self.criterion(x, y) + 0.5*self.criterion(prob_x, y) + 0.1*t_loss  + 0.1*self.kl_loss(prob_x, y).sum()
        #     return x, main_loss
        # else:
        #     return x


        #
        # if self.final_fusion_backbone:
        #     out_backbone_cat = torch.cat((out , feat_up2_backbone_out_curr), dim=1)
        #     out = self.out_backbone_fusion(out_backbone_cat)
        # out = self.final_layer(out)
        #
        #
        #
        #
        # return out, htms_curr
        #



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

        hyper_parameters_setting = "UGPose"
        return hyper_parameters_setting

class PMMs(nn.Module):
    '''Prototype Mixture Models
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k=3, stage_num=10):
        super(PMMs, self).__init__()
        self.stage_num = stage_num
        self.num_pro = k
        mu = torch.Tensor(1, c, k)  # .cuda()
        mu.normal_(0, math.sqrt(2. / k))  # Init mu
        self.mu = self._l2norm(mu, dim=1)
        self.kappa = 20
        # self.register_buffer('mu', mu)

    def forward(self, support_feature):
        prototypes, z_ = self.generate_prototype(support_feature)
        # Prob_map, P = self.discriminative_model(query_feature, mu_f, mu_b)

        return prototypes, z_  # , Prob_map

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def EM(self, x):
        '''
        EM method
        :param x: feauture  b * c * n
        :return: mu
        '''
        b = x.shape[0]
        mu = self.mu.repeat(b, 1, 1).to(x.device)  # b * c * k
        z_ = None
        with torch.no_grad():
            for i in range(self.stage_num):
                # E STEP:
                z = self.Kernel(x, mu)
                z = F.softmax(z, dim=2)  # b * n * k
                # M STEP:
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k

                mu = self._l2norm(mu, dim=1)

        mu = mu.permute(0, 2, 1)  # b * k * c

        return mu, z_

    def Kernel(self, x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = self.kappa * torch.bmm(x_t, mu)  # b * n * k

        return z

    def get_prototype(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu, z_ = self.EM(x)  # b * k * c

        return mu, z_

    def generate_prototype(self, feature):
        # mask = F.interpolate(mask, feature.shape[-2:], mode='bilinear', align_corners=True)
        mask = torch.ones_like(feature)

        mask_bg = 1 - mask

        # foreground
        z = mask * feature
        mu_f, z_ = self.get_prototype(z)
        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_f[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        # background
        z_bg = mask_bg * feature
        mu_b, _ = self.get_prototype(z_bg)

        return mu_, z_

    def discriminative_model(self, query_feature, mu_f, mu_b):

        mu = torch.cat([mu_f, mu_b], dim=1)
        mu = mu.permute(0, 2, 1)

        b, c, h, w = query_feature.size()
        x = query_feature.view(b, c, h * w)  # b * c * n
        with torch.no_grad():
            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k

            z = F.softmax(z, dim=2)  # b * n * k

        P = z.permute(0, 2, 1)

        P = P.view(b, self.num_pro * 2, h, w)  # b * k * w * h  probability map
        P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1)  # foreground
        P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1)  # background

        Prob_map = torch.cat([P_b, P_f], dim=1)

        return Prob_map, P


if __name__ == "__main__":

    x = torch.randn(2, 1, 4, 3)  # 形状: [2, 1, 4, 1]
    y = x.squeeze(-1)  # 形状: [2, 1, 4]
    print(y.shape)

    print("!!!")




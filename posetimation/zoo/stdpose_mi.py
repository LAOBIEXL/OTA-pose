from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import BaseModel
from .backbones.vit import ViT , PatchEmbed , trunc_normal_
from .fusionnet import Temporal_Feat_Fusion_Module
from .decoder import get_LRS
from utils.utils_registry import MODEL_REGISTRY
from posetimation.layers import CHAIN_RSB_BLOCKS
logger = logging.getLogger(__name__)

__all__ = ["STDPOSE_MI"]


def modified_sigmoid(x, k= 2 , theta = 0.3 ):

    return  1 / (1 + torch.exp(-k * (torch.abs(x) - theta)))


@MODEL_REGISTRY.register()
class STDPOSE_MI(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = STDPOSE_MI(cfg, phase, **kwargs)
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
        self.decoder_dim = cfg.MODEL.DECODER.DIM
        self.decoder_patch_size = cfg.MODEL.DECODER.PATCH_SIZE

        self.htms_merg_style = cfg.MODEL.HTMS_MERG_STYLE

        self.htms_dict = dict()
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS

        self.backbone = ViT.get_net(cfg , phase)

        self.TFF = Temporal_Feat_Fusion_Module(cfg ,frame = 3 , dim = cfg.MODEL.VIT.EMB_DIM )

        if self.htms_merg_style == 'single-all':
            self.htms_temp_keypts_merg_conv = nn.ModuleList([
                CHAIN_RSB_BLOCKS(3 , 1 , 1)
                for i in range(self.num_joints)])
            self.htms_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 1  , 64 , 1 )

        elif self.htms_merg_style == 'all-cat':
            self.htms_temp_keypts_merg_conv = nn.ModuleList([
                CHAIN_RSB_BLOCKS(3, 1, 1)
                for i in range(self.num_joints)])
            self.htms_keypts_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 1, self.num_joints * 1, 1)

            self.htms_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 4, 64 , 1)
        elif self.htms_merg_style == "single-cat":
            self.htms_merg_conv = CHAIN_RSB_BLOCKS(self.num_joints * 3, 64, 1)

        self.htms_merg_mask_comb_conv = CHAIN_RSB_BLOCKS(64  * 2 , 64 * 2 , 1 )


        self.patch_embed = PatchEmbed(
                img_size=(self.heatmap_h , self.heatmap_w), patch_size = self.decoder_patch_size , in_chans = 64 * 2 , embed_dim = self.decoder_dim, ratio=1)
        num_patches = self.patch_embed.num_patches
        self.pos_embed_htm = nn.Parameter(torch.zeros(1, num_patches , self.decoder_dim))
        trunc_normal_(self.pos_embed_htm, std=.02)


        self.decoder = get_LRS(cfg, emded_dim = self.decoder_dim , feat_dim = cfg.MODEL.VIT.EMB_DIM)

        if self.decoder_patch_size == 4:

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
        elif self.decoder_patch_size == 2:
            self.head = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cfg.MODEL.HEAD.DECONV1_IN_CHN,
                                   out_channels=cfg.MODEL.HEAD.DECONV1_OUT_CHN, kernel_size=4, stride=2, padding=1,
                                   output_padding=0, bias=False),
                nn.BatchNorm2d(cfg.MODEL.HEAD.DECONV1_OUT_CHN),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=cfg.MODEL.HEAD.FINAL_CONV_IN_CHN, out_channels=cfg.MODEL.HEAD.FINAL_CONV_OUT_CHN,
                          kernel_size=1, stride=1, padding=0)
            )



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

        feat_cat_batch , htms_cat_batch = self.backbone(img_cat_batch)
        htms_curr, htms_supp_A, htms_supp_B = htms_cat_batch.split(true_batchsize, dim=0)

        feat_fused = self.TFF(feat_cat_batch)


        if "heatmaps_sup_cat" in kwargs:
            targ_htms_supp_cat_chn=kwargs["heatmaps_sup_cat"]
            targ_supp_A, targ_supp_B = targ_htms_supp_cat_chn.split(self.num_joints, dim=1)
            if self.use_supp_targ_num == 2:
                htms_supp_A, htms_supp_B = targ_supp_A, targ_supp_B
            elif self.use_supp_targ_num == 1:
                if self.use_supp_direct == 'left':
                    htms_supp_A = targ_supp_A
                if self.use_supp_direct == 'right':
                    htms_supp_B = targ_supp_B

        if self.htms_merg_style == 'single-all':
            htms_curr_list = htms_curr.split(1, dim=1)
            htms_supp_A_list = htms_supp_A.split(1, dim=1)
            htms_supp_B_list = htms_supp_B.split(1, dim=1)
            keypts_htms_list = []
            for joint_index in range(self.num_joints):
                keypoint_temp_cat = torch.cat((htms_curr_list[joint_index] , htms_supp_A_list[joint_index] * prev_weight,
                                               htms_supp_B_list[joint_index] * next_weight) , dim =1)
                keypoint_temp_merg = self.htms_temp_keypts_merg_conv[joint_index](keypoint_temp_cat)
                keypts_htms_list.append(keypoint_temp_merg)

            htms_merg = self.htms_merg_conv(torch.cat(keypts_htms_list, dim=1)).cuda()
        elif self.htms_merg_style == 'all-cat':
            htms_cat = torch.cat((htms_curr, htms_supp_A * prev_weight, htms_supp_B * next_weight), dim=1)
            htms_merg = self.htms_merg_conv(htms_cat)
        elif self.htms_merg_style == "single-cat":
            htms_curr_list = htms_curr.split(1, dim=1)
            htms_supp_A_list = htms_supp_A.split(1, dim=1)
            htms_supp_B_list = htms_supp_B.split(1, dim=1)
            keypts_htms_list = []
            for joint_index in range(self.num_joints):
                keypoint_temp_cat = torch.cat((htms_curr_list[joint_index], htms_supp_A_list[joint_index] * prev_weight,
                                               htms_supp_B_list[joint_index] * next_weight), dim=1)

                keypts_htms_list.append(keypoint_temp_cat)

            htms_merg = self.htms_merg_conv(torch.cat(keypts_htms_list, dim=1)).cuda()

        diff_left = htms_curr - htms_supp_A
        diff_right = htms_curr - htms_supp_B
        diff_left_sigm  = modified_sigmoid(diff_left)
        diff_right_sigm = modified_sigmoid(diff_right)
        diff_left_summed  = diff_left_sigm.sum(dim=1, keepdim=True)
        diff_right_summed = diff_right_sigm.sum(dim=1, keepdim=True)

        diff_comb = diff_left_summed * prev_weight + diff_right_summed  * next_weight
        mask = F.softmax(diff_comb , dim=-1)
        htms_masked = htms_merg * mask


        htms_merg_mask_cat = torch.cat((htms_merg, htms_masked), dim=1)
        htms_merg_mask_cat = self.htms_merg_mask_comb_conv(htms_merg_mask_cat)


        x , (Hp, Wp) = self.patch_embed(htms_merg_mask_cat)
        x = x + self.pos_embed_htm
        out = self.decoder(x , feat_fused)
        out = out.permute(0, 2, 1).reshape(true_batchsize, -1, Hp, Wp)
        out = self.head(out)

        return out , feat_cat_batch[0:true_batchsize, :, :] , feat_fused , htms_curr , htms_merg_mask_cat


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




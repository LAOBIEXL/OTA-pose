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
# from utils.vis import vis_htms
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks

from .UNet import *
from .cadd import *
from .blocks import *
# from .model import *
from .mamba import *
logger = logging.getLogger(__name__)

__all__ = ["DiffusionPose"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])




@MODEL_REGISTRY.register()
class DiffusionPose(nn.Module):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        if phase == "train":
            is_train = True
        else:
            is_train = False
        model = DiffusionPose(cfg, is_train=is_train, **kwargs)
        return model

    def exists(self, x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d

    def extract(self, a, t, x_shape):
        """extract the appropriate  t  index for a batch of indices"""
        device = "cuda:0"
        a = a.to(device)
        t = t.to(device)
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        out = out.to(device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0, max=0.999)

    def __init__(self, cfg, is_train=True, **kwargs):
        super().__init__()

        self.device = "cuda:0"
        self.config = cfg
        self.is_train = is_train

        # build diffusion
        timesteps = cfg.DIFFUSION.TIMESTEPS
        sampling_timesteps = cfg.DIFFUSION.SAMPLESTEPS
        betas = self.cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = self.default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.scale = cfg.DIFFUSION.SCALE


        times = torch.linspace(-1, timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        self.time_pairs = time_pairs

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.heatmap_h = cfg.MODEL.HEATMAP_SIZE[1]
        self.heatmap_w = cfg.MODEL.HEATMAP_SIZE[0]
        self.token_num = int((cfg.MODEL.IMAGE_SIZE[1] / cfg.MODEL.VIT.PATCH_SIZE) * (
                cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.VIT.PATCH_SIZE))
        self.map_size_up_1x = (
        cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE)
        self.map_size_up_2x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 2,
                               cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 2)
        self.map_size_up_4x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 4,
                               cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 4)

        self.frame_num = cfg.MODEL.FRAME_NUM
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS


        # v3: 加上互信息，v3.1：tff的fuse_all =true 即backbone的特征和fused的特征做互信息；v3.2：tff的fuse_all =false 即backbone的特征和tff后拆开的当前帧的特征做互信息
        # v4:换成时空transformers  v4.1：fusion = true     v4.2: fusion = false 训练崩溃了往下掉
        # v5:保留时空network，     v5.1: fusion = true ，在最后和backbone特征进行融合，试试会不会掉点。 v5.2: fusion = false
        # mamba v3: start mmaba       v4: start attention
        # v5:start attention + skip connection  6,384 v5.1：decoder 的层数变为depth = 4层 dim = 384 v5.2 : decoder 的 dim 设为 512 depth = 6 V5.3 4,512
        # v5.4   2,384   v5.5   2,512
        # 根据实验表明：dim =384比512好，所以 dim = 384 , depth = 4
        # --instghts_1-- 这里下面的改动主要的点在于：加了kpt tokens ； 加了kpt 的miloss； mamba分支以前是一个全序列，一个时序，这里改到一个无mask全序列，一个有mask全序列； softmask
        # v6: 在V5基础上，在start attention那里加上keypoint tokens ，然后在后面mamba部分，把这个keypoint tokens生成一个soft mask，然后两个分支的mamba进行时序扫描，但是一个是全token一个是部分token
        # keypoint token还要进入去噪解码器去和t一样modulate，mamba出来的去当作condition去指导生成热图。
        # v6 已完成 ketpoint tokens 在cadd里是以 scale 和shift形式来指导cross attention 的。
        # v7 在v6基础上把在cross block里面在cross attention 把kpt_token加上GCN进行骨骼约束，使热图生成可以更符合人体架构。
        # v7.1 上面的v7使用的是错误的connects ，我换成coco的了 :虽然有比较好的结果达到了89.0几，但是最后还是过拟合了，往下掉。
        # ----   这里下面的改进，虽然很大，但是貌似没有怎么掉点，从89.0几->88.98,虽然backbone掉了一些，但是其实差不多，也就是说变化不在此处发生
        # v7.2 把去噪解码器的patch emb去掉了，等于现在是17,1024, 这种情况下的GNN是对着噪声热图做的了，不是对着kpt tokens 做的了，所以现在的kpt tokens 是没用的，我认为这个改进是合理的，这样子gnn才是帮助去噪过程去生成更符合人体骨架的姿态。
        # v7.3 在CADD最后，和backbone的htms结合以后，利用了dcnv2去做了校正，试试效果。
        # v7.4 把最后的dcnv2 多加了几个dilation层，也就是类似于dcpose，看看效果。 这一版可以达到88.98 比上面v7.3和7.2更好
        # v7.5 把dcnv2换成dcnv4     最高可以达到88.99 但是最后会掉到70多，蛮离谱
        # v7.6 仅把backbone的热图和生成热图拼起来做卷积计算； v 7.7 把backbone热图和生成的热图 直接加起来 / 2
        # v7.8 只是把加起来不除以2  这样做不行一直卡在88.5几
        # v7.9 用dcnv4，和上面不一样，上面是先conv在dcn，我现在是直接dcn不要之前的卷积，看看效果。突然发现进去的必须是16的倍数通道，所以必须前面加上一个模块变通道。我这里想到还是搞多个dilation试试。
        # v7.9用了多个dcnv4，虽然backbone可以到89.0几，但是最后结果却会掉到87.多，   v 7.10我这里需要去掉几个dcnv4试试。留三个dcn:1,3,6
        # 这里根据insight_1 ，我觉得两个可以动：kpt 的mi loss的权重 ； mamba两个分支的扫描方式
        # 首先是动一下mamba两个分支： v8.1 现在是一个有mask 全序列扫描，另一个无mask全序列扫描，那么改进为 一个无mask，全序列，另一个有mask，时序扫描。 这里会卡在88.5-7几
        # v8.2 两个分支都加mask，但是一个是全序列，一个是时序。   备选： v8.3: 无mask 时序，有mask 时序 (1)代表运行先后顺序  这里会达到89.0但是会立马掉到86 ； v8.4 无mask，时序，另一个有mask，全序列。
        #  把kpt 的mi loss 调一下  目前是0.1   v9.1: 0.001  还可以，达到89.0； v9.2 0.01   比较差 88.5-6  ; v9.3 0.05  ； v9.4 ：0.5 ； v9.5: 1
        # 8.2是最好的，所以采用8.2
        # 接下来调v9： v9.1: 可以达到89.0几  ;   v9.2:  差一点会降下来   ;  v9.3:  达到过88.9几，最后卡在88.7  v 9.4: 达到88.90，最后掉到88.2. v 9.5:
        # 同时想调一下feat的mi的权重，v10.1: 0.001  ; v10.2: 0.01  ; v10.3: 0.05 ; v10.4: 0.5 ; v10.5: 1
        # v10.1: 比较差  ; v10.2:  88.8几，虽然最高值不高，但是稳定  ; v10.3: 达到过88.92，但是不稳定，会掉到85 ; v10.4:  ; v10.5:
        #目前来看是v9.1最好，也蛮稳定的，kpt的miloss是0.001 而feat的miloss是0.1
        #尝试了调feat的miloss，我发现设置为1时是最好的，可以达到89.05，所以把featmiloss设置为1；
        #v11.1 我想调一下cadd的layer，layer从四变为2，试一试, 卡在88.5-6.
        #v12.1 在上面实验中，我发现layer为2backbone会卡在88.6.所以把layer换为4。 这次我将MRF模块加到GCN后面去，看看效果怎么样，这里迭代次数为1。
        #我在Loss里加上了边势Loss，其中权重设置为0.1，先试试。v13.1 :会出现NAN的Loss   v13.2 0.01 ：修改了LOss函数代码，暂时不会出现NAN了。还算稳定可以达到88.8几。
        # 我现在把模块嵌入到解码器里了，而且Loss也加了，但是现在我的想法是，解码器还是用patch 分割，而不是17，我先试试这个做法合不合适。v14。1
        # v14.2 想把gcn去掉试试  刚运行到第三个epoch就出现NAN，所以gcn依旧加回来了
        # v14.3 想把cadd 的gcn和mrf放到cross attn 后面去，本来在自注意力后面的。 这样子做效果更稳定看了，说明是对的。
        # v14.4 扔掉cadd的自注意力层，试试  发现扔掉自注意力层不合适，效果较差。
        # v14.5 既然自注意力层要保留，那么就把换成mamba2试下。 cadd的dim设置为384会报错，设置成了512   换成mamba后，512 是会导致精度越来越低，最后是88。3
        # v14.6 cadd的sim设置为256.  这里开始慢慢涨，后面还上去了 88.6几

        # v15.1把时空编码器的 mamba换一下，其中，mask不需要了，kpt token全程保留，kpt token一直进入到cadd   已改完，现在是空间注意力，加全序列Mamba  一开始就是88.7但是最后是88.2-3
        # v15.2 试试全序列mamba - 时序注意力    虽然最后卡在88.6-7，但是很稳定。
        # v15.3 我发现MLP在mamba后面，我觉得不合适，我要把MLP移到时序注意力后面试试，这里搞错了，时序注意力有MLP
        # v15.4 加个空间注意力， 即 空间attn - 全序列mamba - 时序注意力
        # v.15.5 - 全序列mamba -  - 空间attn  - 时序注意力
        # 上面两个都可以稳定在88.6几可以达到88.7，但是多了参数量，所以打算去掉空间注意力，也就是下面的配置是：全序列mamba-时序注意力
        # v15.6 我突然发现时空attn也有 多了的MLP，去掉了 配置是：全序列mamba-时序注意力  会掉到87.9，很差
        # v15.7 接着上面的配置，我们把cadd的mamba换为MLp试试  卡在88.5,6最后是88.4
        # v15.8 将全序列mamba和时序注意力转换为并行处理试试，还是用以前的那个组合attention。这个版本是两个分支都有MLP，最后合并后没有MLP，试试

        # v15.9 这是跟着上面配置，两个分支都有MLP，并且汇合后也有MLP 这个不行

        # v15.10 跟着15.8的配置，但是是mamba没有MLP，但是时序注意力有mlp，最后合并后，再有个mlp 中间会掉下去，最后卡在88。6

        # v16.1 我们将遵循15.8的配置，现在调下cadd的dim为384 稳定是稳定，精度卡在88.3-5上不去，最后是88.1
        # v16.2 我们在调一下mamba马部分，现在是norm在前面，双分支一个全序列mamba一个时序注意力（是注意力），然后后面再接个MLP cadd的dim设置到384  一直卡在88.1

        # v16.4 将cadd的最开始的MLp替换为 self注意力 ，dim依旧还是384  也是卡在88.1-2 最后掉到87.几了。太差了
        # v16.5 我们发现v15,8是目前最好的情况，我们在这里还是用v16.4的配置，我们想将mamba调下，在这里，换成双向的mamba试试也就是VIM  稳定但是卡在88.4-5
        # v16.6 这里我们想通了，就先搞一个spatial attn放在最前面让keypoints tokens学习位置，后面等于是时序双分支，一个是全序列mamba一个是时序注意力，
        # 最后面还有个MLP，cadd里最开始也搞个attention，后面接上GNN和MRF，然后再是cross attn，最后才是MLP。  这一版虽然中间掉到了87.几，但是最后是88.6-7，最高是88.89
        # v16.7 把开始的自注意力从block换成attention 发现最后会掉到88.1-2，虽然backbone是88.9
        # v16.8 把开始的自注意力换为空间mamba，带双向,这里的全序列mamba也是双向  这里稳定在88.5-6，但是最高的也是88.6，稳定
        # v16.9 全序列mamba换成单向，空间的mamba保留双向   最高可以达到88.8几，但是16epoch之后就掉到88.3-4了，不太稳定
        # v16.10 依照16.9的配置，把学习率跳到0.0002    这个学习率的结果还不如上面，  最高也没打到88.8，最后是88.38
        # v17.1 跟随16.9的配置 学习率5e-4 cadd的开始的自注意力换为双向Mamba ,dim只可以是256   会掉到87.几
        # v17.2 跟随17.1的配置 学习率5e-4 cadd的开始的自注意力换为双向Mamba ,dim设置为512  会掉到88.1-2
        # v17.3 跟随17.2的配置 学习率5e-4 cadd的开始的自注意力换为双向Mamba ,dim设置为256，layer设置为2
        # v17.4 跟随17.3的配置 学习率5e-4 cadd的开始的自注意力换为双向Mamba ,dim设置为512，layer设置为2 中间会掉到87.1
        # v18.1 我们跟随16.8的配置，把全序列mamba换成双向，然后把cadd的mamba换为自注意力，dim设置为384，layer设置为2,16.8的layer是4.
        # v18.2 cadd最后的dcnv4去掉
        # v18.3 把cadd最后还剩下来的RSN和backbone结合的那个部分也去掉，试试   个位数，完全失效
        # v18.4
        self.backbone = ViT_V1.get_net(cfg )

        self.decoder_model = get_CADD_net(cfg)

        # self.tff = Temporal_Feat_Fusion_Module(cfg,frame=self.frame_num,dim=cfg.MODEL.VIT.EMB_DIM ,fuse_all=cfg.MODEL.TFF.FUSE_ALL)
        # self.tff = STTrans(cfg.MODEL.VIT.EMB_DIM, depth=2, num_heads=8, mlp_ratio=2,fusion=True)
        # self.tff = STTrans(cfg.MODEL.VIT.EMB_DIM, depth=2, num_heads=8, mlp_ratio=2, fusion=False)

        args = ModelArgs(d_model=cfg.MODEL.VIT.EMB_DIM, n_layer=2, d_state=64, expand=2, use_mlp=True, num_heads=8,
                         mlp_ratio=4,
                         use_rotary=False, use_rmsnorm=False,
                         use_spatial_attn=False, use_start_spatial_attn=True, drop_mamba=False, use_mamba2=True,
                         use_start_spatial_mamba=False,
                         frame=self.frame_num)
        self.mamba_blocks = Mamba_blocks(args)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.token_num * self.frame_num, self.token_num),
            nn.LayerNorm(self.token_num),
            nn.Linear(self.token_num, self.token_num),
            nn.LayerNorm(self.token_num),
        )


        param_count = {}
        for name, module in self.mamba_blocks.named_modules():
            # 收集模块的参数
            params = list(module.parameters())
            # 累加参数量
            param_count[name] = sum(p.numel() for p in params)

        # 打印每个模块的参数量
        for name, count in param_count.items():
            print(f"{name}: {count}")





        self.total_params0 = sum(p.numel() for p in self.mamba_blocks.parameters())
        logger.info("=>mamba 的参数量为:{}".format(self.total_params0))
        self.total_params1 = sum(p.numel() for p in self.decoder_model.parameters())
        logger.info("=>Decoder 的参数量为:{}".format(self.total_params1))
        self.total_params2 = sum(p.numel() for p in self.backbone.parameters())
        logger.info("=>Backbone 的参数量为:{}".format(self.total_params2))






    def predict_noise_from_start(self, x_t, t, x0):
        x0 = x0.to(self.device)
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod
        sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.to(self.device)
        sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod
        sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.to(self.device)

        x = (
                (self.extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                self.extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

        return x.cuda().float()

    def model_predictions(self, x_noisy, condition, t, feat_backbone=None):

        pre_x_start = self.decoder_model(x_noisy.float(), time=t, x_self_cond=condition,feat_backbone = feat_backbone)

        x_noisy = x_noisy.cuda().float()

        # pred_noise是根据pre_x_start(x_0) 和 x_noisy(x_t) 通过公式计算的
        pred_noise = self.predict_noise_from_start(x_noisy , t, pre_x_start).cuda()


        return ModelPrediction(pred_noise, pre_x_start)

    @torch.no_grad()
    def ddim_sample(self, x_t, condition, batchsize,feat_backbone=None):

        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        x_start = None
        x_current = x_t
        for time, time_next in self.time_pairs:
            time_cond = torch.full((batchsize,), time, device=self.device, dtype=torch.long)
            time_cond = time_cond.cuda()
            # self_cond = x_start if self.self_condition else None

            preds = self.model_predictions(x_current.float(), condition, time_cond, feat_backbone=feat_backbone)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                x_current = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_current)

            x_current = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise.float()
            x_current = x_current.float()

        return x_start.float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod.to(self.device), t, x_start.shape).to(self.device)

        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod.to(self.device), t,
                                                      x_start.shape).to(self.device)

        x_start_noisy = sqrt_alphas_cumprod_t * self.scale * x_start.to(self.device) + sqrt_one_minus_alphas_cumprod_t * noise.to(
            self.device)
        x_start_noisy = x_start_noisy.to(self.device)

        return x_start_noisy.double()



    def forward(self, x  , **kwargs):
        true_batchsize , channel_num = x.shape[0] , x.shape[1]//self.frame_num

        img_cat_batch = torch.cat(x.split(channel_num, dim=1), dim=0)


        htms_backbone_out_cat_batch, feats_backbone_out_cat_batch, feats_up1_backbone_out_cat_batch, feats_up2_backbone_out_cat_batch, \
            = self.backbone(img_cat_batch)

        htms_curr  = htms_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]
        feat_curr = feats_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]
        feat_2x_curr = feats_up2_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]


        fused_feature , kpt_tokens_curr = self.mamba_blocks(feats_backbone_out_cat_batch)
        fused_feature = torch.cat( fused_feature.split(true_batchsize, dim=0), dim=1 )
        fused_feature = self.fusion_mlp(fused_feature.permute(0, 2,1)).permute(0, 2, 1)



        # fused_feature = fused_feature[0:true_batchsize, ...]


        condition = torch.cat( (fused_feature , kpt_tokens_curr), dim=1)


        if self.is_train:
            target = kwargs['target']
            t = torch.randint(0, self.num_timesteps, (true_batchsize,)).long().cuda()


            noise = torch.randn_like(target, dtype=torch.float32)

            target_noisy = self.q_sample(x_start=target, t=t, noise=noise)
            x_t = target_noisy.float()
            output = self.decoder_model(x_t, time=t, x_self_cond=condition,feat_backbone = feat_2x_curr)
        else:

            normal_noise = torch.randn((true_batchsize, self.num_joints, self.heatmap_h, self.heatmap_w),
                                       dtype=torch.float32)
            x_t = normal_noise.cuda().float()
            # x_t = heatmap.cuda()

            output = self.ddim_sample(x_t, condition, true_batchsize, feat_2x_curr)


            # for i in range(self.sample_hm-1):
            #
            #     normal_noise = torch.randn((true_batchsize, self.num_joints, self.heatmap_h, self.heatmap_w) , dtype=torch.float32)
            #     x_t = normal_noise.cuda()
            #     # x_t = heatmap.cuda()
            #
            #     output += self.ddim_sample(x_t, condition, true_batchsize)
            #
            # output = output / self.sample_hm * 1.0



        return output , htms_curr ,feat_curr , fused_feature , kpt_tokens_curr

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

        hyper_parameters_setting = "diffusionpose"
        return hyper_parameters_setting






if __name__ == "__main__":
    total_timesteps, sampling_timesteps, eta = 1000, 1, 1.0

    # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    print(time_pairs)

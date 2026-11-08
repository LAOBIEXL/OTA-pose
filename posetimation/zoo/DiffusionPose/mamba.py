"""Simple, minimal implementation of Mamba in one file of PyTorch.


Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
from timm.models.layers import DropPath
from mamba_ssm import Mamba , Mamba2

from.mamba2 import Mamba2 as Mamba2_
from posetimation.zoo import trunc_normal_


@dataclass
class ModelArgs:
    d_model: int = 1024
    n_layer: int = 2
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    use_mlp: bool = False
    mlp_ratio: int = 2
    use_rmsnorm: bool = True
    num_heads: int = 8
    use_rotary: bool = False
    use_spatial_attn: bool = False
    use_start_spatial_attn: bool = False
    use_start_spatial_mamba: bool = False
    drop_mamba: bool = False
    separate_calcul: bool = False
    use_mamba2: bool = False
    frame: int = 3
    num_joint = 17


    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
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
        x = self.fc2(x)
        x = self.drop(x)
        return x
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 1000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,):
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,use_rotary=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_rotary=use_rotary

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_rotary:
            self.freqs_cis = precompute_freqs_cis(dim, 192 * 2)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.use_rotary:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
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


class Mamba_blocks(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.keypoint_tokens = nn.Parameter(torch.zeros(1,  args.num_joint, args.d_model))
        trunc_normal_(self.keypoint_tokens, std=.02)

        dpr = [x.item() for x in torch.linspace(0, 0.18, 4)]
        self.layers = nn.ModuleList([ResidualBlock(args,dpr[i]) for i in range(args.n_layer)])
        self.norm_f = nn.LayerNorm(args.d_model)
        self.norm_kpt_curr = nn.LayerNorm(args.d_model)

    def forward(self, x_in):
        # x_in: B*3 x N x C ---> x_out: B*3 x N x C  , keypoint_tokens : B x num_joint x C
        B = x_in.shape[0]//self.args.frame

        x = torch.cat([x_in, self.keypoint_tokens.expand(x_in.shape[0], -1, -1)], dim=1)
        for layer in self.layers:
            x = layer(x)

        x_out , kpt_tokens = x[:, :x_in.shape[1],:], x[:, x_in.shape[1]:,:]
        kpt_tokens_curr = kpt_tokens.split(B, dim=0)[0]
        x_out = self.norm_f(x_out)
        kpt_tokens_curr = self.norm_kpt_curr(kpt_tokens_curr)

        return x_out , kpt_tokens_curr



class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs,drop_path):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.use_start_spatial_attn = args.use_start_spatial_attn
        self.drop_mamba = args.drop_mamba
        self.separate_calculation = args.separate_calcul

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.start_spatial_attn = Block(
        #     dim=args.d_model, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
        #     drop=0., attn_drop=0., drop_path=drop_path,
        # )
        self.start_norm = nn.LayerNorm(args.d_model)
        # self.start_spatial_attn = Attention(args.d_model, num_heads=args.num_heads,use_rotary=args.use_rotary)
        self.start_spatial_mamba = Mamba2_(d_model=args.d_model, d_state=args.d_state, d_conv=4, expand=args.expand)

        self.norm_mamba_temp = nn.LayerNorm(args.d_model)
        self.full_mamba = Mamba2_(d_model=args.d_model, d_state=args.d_state, d_conv=4, expand=args.expand)
        self.temporal_attn = Attention(args.d_model, num_heads=args.num_heads,use_rotary=args.use_rotary)


        if self.separate_calculation:
            self.spatial_fusion = nn.Sequential(
                nn.LayerNorm(2 * args.d_model),
                nn.Linear(2 * args.d_model, args.d_model, bias=True),
                nn.SiLU(),
            )
            self.attn_mlp_full = nn.Sequential(
                nn.Linear(args.d_model, 1, bias=True),
                nn.Sigmoid(),
            )
            self.attn_mlp_soft = nn.Sequential(
                nn.Linear(args.d_model, 1, bias=True),
                nn.Sigmoid(),
            )
            self.fusion_norm = nn.LayerNorm(args.d_model)
        else:
            self.spatial_fusion = nn.Sequential(
                nn.LayerNorm(2 * args.d_model),
                nn.Linear(2 * args.d_model, args.d_model, bias=True),
                nn.SiLU(),
                nn.Linear(args.d_model, 1, bias=True),
                nn.Sigmoid(),
            )
        if args.use_mlp:
            if args.use_rmsnorm:
                self.mlp_norm = RMSNorm(args.d_model)
            else:
                self.mlp_norm = nn.LayerNorm(args.d_model)
            self.mlp = Mlp(in_features=args.d_model, hidden_features=args.d_model*args.mlp_ratio)








    def forward(self, x):

        B, L_J, D = x.shape[0]//self.args.frame,x.shape[1], x.shape[2]
        J = self.args.num_joint
        L = L_J - J

        x = self.start_norm(x)
        # x = self.start_spatial_attn(x)
        x = self.start_spatial_mamba(x, 'vim')

        x = x + self.drop_path(x)

        x = torch.cat(x.split(B, dim=0), dim=1)

        # x , kpt_tokens = x[:,:L,:] , x[:,L:,:]
        #
        # map = x @ kpt_tokens.permute(0,2,1)
        # map = map.sum(dim = -1, keepdim=True)
        # soft_mask = map.softmax(dim=-1)

        x_norm = self.norm_mamba_temp(x)

        x_mamba = x_norm
        x_mamba  = self.full_mamba(x_mamba,'vim')

        x_temp_attn = x_norm
        x_temp_attn = rearrange(x_temp_attn, 'b (f l) d -> (b l) f d', f=self.args.frame, l=L_J,b=B)
        x_temp_attn = self.temporal_attn(x_temp_attn)
        x_temp_attn = rearrange(x_temp_attn, '(b l) f d -> b (f l) d', f=self.args.frame, l=L_J,b=B)

        # Spatial fusion
        if self.separate_calculation:
            combined_full_mask_attn = torch.cat([x_mamba, x_temp_attn], dim=-1)
            attention_weights = self.spatial_fusion(combined_full_mask_attn)
            weight_full = self.attn_mlp_full(attention_weights)
            weight_soft = self.attn_mlp_soft(attention_weights)

            x_combined = weight_full * x_mamba + weight_soft * x_temp_attn
            x_combined = self.fusion_norm(x_combined)

        else:
            combined = torch.cat([x_mamba, x_temp_attn], dim=-1)
            attention_weights = self.spatial_fusion(combined)
            x_combined = attention_weights * x_mamba + (1 - attention_weights) * x_temp_attn


        x = x + self.drop_path(x_combined)

        if self.args.use_mlp:
            x = x + self.drop_path(self.mlp(self.mlp_norm(x)))



        x = rearrange(x, 'b (f l) d -> (b f) l d', f=self.args.frame, l=L_J,b=B)
        return x



        # if self.args.use_start_spatial_mamba:
        #     x = torch.cat(x.split(L, dim=1), dim=0)
        #
        #     x = x + self.drop_path(self.start_spatial_mamba(self.mamba_norm1(x)))
        #     x = x + self.drop_path(self.mamba_mlp(self.mamba_norm2(x)))
        #
        #     x = torch.cat(x.split(B, dim=0), dim=1)
        #
        # if self.use_start_spatial_attn:
        #     x = self.start_spatial_attn(x)


        # x = torch.cat( x.split(L, dim=1), dim=0 )
        # x = torch.cat((x, kpt_tokens), dim=1)
        # x_temp_attn = torch.chunk(x , self.args.frame, dim=0)
        # x = torch.cat( x, dim=0 )





        # x_full = x
        # x_masked = torch.cat(x.split(L, dim=1), dim=0)
        # x_masked = torch.cat(x_masked.split(B, dim=0), dim=1)
        # x_full = x_masked

        #mamba
        # x_full = rearrange(x_full, 'b (f l) d -> (b l) f d', f=self.args.frame, l=L,b=B)
        # x_full = self.full_mamba(x_full)
        # x_full = rearrange(x_full, '(b l) f d -> b (f l) d', f=self.args.frame, l=L,b=B)

        # x_masked = rearrange(x_masked, '(b f) l d -> (b l) f d', f=self.args.frame, l=L,b=B)
        # x_masked = self.soft_mamba(x_masked)
        # x_masked = rearrange(x_masked, '(b l) f d -> b (f l) d', f=self.args.frame, l=L,b=B)

        # x_st = torch.stack(x_st.split(L,dim=1),dim=1).permute(0,2,1,3).reshape(B*L,self.args.frame,D)
        # # x_st = self.temporal_attn(x_st).reshape(B,L,self.args.frame,D).permute(0,2,1,3).reshape(B,self.args.frame*L ,D)
        # x_st = self.temporal_mamba(x_st).reshape(B,L,self.args.frame,D).permute(0,2,1,3).reshape(B,self.args.frame*L ,D)







class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


if __name__ == '__main__':
    import torch
    from mamba_ssm import Mamba
    x = torch.randn(1, 192, 1024).to(device='cuda')
    args = ModelArgs(d_model=1024, n_layer=2, d_state=16, expand=2, use_mlp=False,
                     num_heads=8,
                     mlp_ratio=2,
                     use_rotary=False, use_rmsnorm=False, frame=3)
    model = Mamba_blocks(args).to(device='cuda')

    start_time = time.time()
    y = model(x)
    print(f'time: {time.time() - start_time}')

    print(y.shape)
    print("The parameters of model are:{}".format(sum(p.numel() for p in model.parameters())))








# 导入必要的库
import sys
import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat

from .token_prune_dynamic import cluster_dpc_knn_dynamic


def compute_temporal_redundancy_stats(
    x,
    stage="pre_tte_input_tokens",
    current_frame_index=0,
    temporal_order=None,
):
    """Return temporal redundancy scalars for [B, F, N, C] tokens.

    OTAPose concatenates frames as [t, t-2, t-1, t+1, t+2], so the current
    frame is index 0. The primary metric compares that current frame with all
    support frames. The adjacent metric reorders 5-frame inputs into
    chronological order before comparing neighboring frames.
    """
    with torch.no_grad():
        z = x.float()
        z = z / (z.norm(dim=-1, keepdim=True).clamp_min(1e-12))

        frame_num = z.shape[1]
        if frame_num <= 1:
            current_support = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
            adjacent = current_support
            temporal_order = list(range(frame_num))
        else:
            current_frame_index = int(current_frame_index)
            support_indices = [idx for idx in range(frame_num) if idx != current_frame_index]
            current_token = z[:, current_frame_index:current_frame_index + 1]
            supports = z[:, support_indices]
            current_support = (current_token * supports).sum(dim=-1).mean(dim=(1, 2))

            if temporal_order is not None:
                temporal_order = list(temporal_order)
            elif frame_num == 5 and current_frame_index == 0:
                temporal_order = [1, 2, 0, 3, 4]
            else:
                temporal_order = list(range(frame_num))
            z_ordered = z[:, temporal_order]
            adjacent = (z_ordered[:, :-1] * z_ordered[:, 1:]).sum(dim=-1).mean(dim=(1, 2))

    return {
        "temporal_redundancy_adj": adjacent.detach(),
        "temporal_redundancy_center": current_support.detach(),
        "temporal_redundancy_adj_chrono_prefusion": adjacent.detach(),
        "temporal_redundancy_current_support_prefusion": current_support.detach(),
        "temporal_order": temporal_order,
        "temporal_redundancy_stage": stage,
        "current_frame_index": int(current_frame_index),
    }


def _temporal_redundancy_stats(x):
    return compute_temporal_redundancy_stats(x)


# 索引点函数，用于根据索引获取点云中的点
def index_points(points, idx):
    # 获取输入点云的设备
    device = points.device
    # 获取批次大小
    B = points.shape[0]
    # 获取索引的形状
    view_shape = list(idx.shape)
    # 将索引形状的除第一维外的所有维度设置为1
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # 获取重复的形状
    repeat_shape = list(idx.shape)
    # 将重复形状的第一维设置为1
    repeat_shape[0] = 1
    # 创建批次索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # 根据批次索引和点索引获取新的点
    new_points = points[batch_indices, idx, :]

    return new_points


# DPC-KNN聚类函数，用于对token进行聚类
def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    # 禁用梯度计算
    with torch.no_grad():
        # 获取输入的形状：批次大小B，token数量N，特征维度C
        B, N, C = x.shape

        # 计算距离矩阵，使用余弦距离并归一化
        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        # 如果提供了token掩码，则应用掩码
        if token_mask is not None:
            token_mask = token_mask > 0
            # 对掩码内的token保持原距离，对掩码外的token设置为最大距离+1
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # 获取每个token的k个最近邻的距离和索引
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        # 计算每个token的密度
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # Deterministic tie-break for equal-density tokens.
        tie_break = torch.arange(N, device=x.device, dtype=density.dtype).unsqueeze(0)
        density = density + tie_break * 1e-12

        # 如果提供了token掩码，则应用掩码到密度上
        if token_mask is not None:
            density = density * token_mask

        # 创建掩码，标记密度大于其他token的情况
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        # 获取距离矩阵的最大值
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        # 计算每个token到密度更高的最近token的距离
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # 计算每个token的得分
        score = dist * density

        # 获取得分最高的cluster_num个token作为聚类中心
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # 获取聚类中心之间的距离矩阵
        dist_matrix = index_points(dist_matrix, index_down)

        # 为每个token分配到最近的聚类中心
        idx_cluster = dist_matrix.argmin(dim=1)

        # 创建批次索引和临时索引
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        # 确保聚类中心被分配到正确的聚类
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # 返回聚类中心索引和每个token的聚类分配
    return index_down, idx_cluster


def random_token_prune(x, cluster_num):
    with torch.no_grad():
        B, N, _ = x.shape
        cluster_num = max(1, min(int(cluster_num), N))
        random_score = torch.rand(B, N, device=x.device)
        index_down = torch.topk(random_score, k=cluster_num, dim=-1).indices
        index_down, _ = torch.sort(index_down)

        idx_cluster = torch.zeros(B, N, dtype=torch.long, device=x.device)
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


# 多层感知机(MLP)类
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 如果未指定输出特征维度，则使用输入特征维度
        out_features = out_features or in_features
        # 如果未指定隐藏层特征维度，则使用输入特征维度
        hidden_features = hidden_features or in_features
        # 第一层线性变换
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二层线性变换
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 第一层线性变换
        x = self.fc1(x)
        # 激活函数
        x = self.act(x)
        # Dropout
        x = self.drop(x)
        # 第二层线性变换
        x = self.fc2(x)
        # Dropout
        x = self.drop(x)
        return x


# 自注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        # 注意力头数
        self.num_heads = num_heads
        # 每个注意力头的维度
        head_dim = dim // num_heads
        # 缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 线性变换层，用于生成查询、键、值
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出线性变换
        self.proj = nn.Linear(dim, dim)
        # 输出dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_padding_mask=None):
        # 获取输入形状：批次大小B，token数量N，特征维度C
        B, N, C = x.shape
        # 生成查询、键、值并进行形状变换
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(~key_mask, float("-inf"))
        # 注意力权重归一化
        attn = attn.softmax(dim=-1)
        # 注意力dropout
        attn = self.attn_drop(attn)

        # 计算注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出线性变换
        x = self.proj(x)
        # 输出dropout
        x = self.proj_drop(x)
        return x


# 交叉注意力机制类
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        # 注意力头数
        self.num_heads = num_heads
        # 每个注意力头的维度
        head_dim = dim // num_heads
        # 缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 线性变换层，分别用于生成查询、键、值
        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        # 注意力dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出线性变换
        self.proj = nn.Linear(dim, dim)
        # 输出dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3, key_padding_mask=None):
        # 获取输入形状
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        # 生成查询、键、值并进行形状变换
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(~key_mask, float("-inf"))
        # 注意力权重归一化
        attn = attn.softmax(dim=-1)
        # 注意力dropout
        attn = self.attn_drop(attn)

        # 计算注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出线性变换
        x = self.proj(x)
        # 输出dropout
        x = self.proj_drop(x)
        return x


# Transformer块类
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0):
        super().__init__()

        # 第一层归一化
        self.norm1 = norm_layer(dim)
        # 自注意力层
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二层归一化
        self.norm2 = norm_layer(dim)
        # MLP层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, key_padding_mask=None):
        # 自注意力子层
        x = x + self.drop_path(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)
        # MLP子层
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


# 主模型类
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 空间-时间编码器的深度
        self.st_depth = args.st_layers
        # 中间层的深度
        self.middle_depth = args.middle_layers
        # 嵌入维度
        self.embed_dim = args.dim
        # MLP隐藏层维度
        self.mlp_hidden_dim = args.dim * args.mlp_ratio

        # 帧数
        self.frame = args.frames
        # 总token数量
        self.total_token_num = args.total_token_num
        # 剪枝后的token数量
        self.prune_token_num = args.prune_token_num
        # 是否使用token恢复
        self.use_recover = args.use_recover
        # When true, bypass only token selection/pruning and keep the rest of the block.
        self.no_prune = bool(getattr(args, "no_prune", False))
        self.use_coarse_ste_tte = bool(getattr(args, "use_coarse_ste_tte", True))
        # 是否启用动态剪枝
        self.use_dynamic_prune = getattr(args, "use_dynamic_prune", False)
        self.prune_selection = str(getattr(args, "prune_selection", "score")).lower()
        if self.prune_selection not in ("score", "random"):
            raise ValueError("prune_selection must be one of: 'score', 'random'")
        self.dynamic_alpha = float(getattr(args, "dynamic_alpha", 0.5))
        self.dynamic_min_keep = int(getattr(args, "dynamic_min_keep", 16))
        self.dynamic_max_keep = getattr(args, "dynamic_max_keep", None)
        if self.dynamic_max_keep is not None:
            self.dynamic_max_keep = int(self.dynamic_max_keep)
        self.collect_prune_vis = False
        self.prune_vis = None
        self.prune_valid_mask = None
        self.prune_keep_counts = None
        self.last_redundancy_stats = None

        # DropPath率
        drop_path_rate = 0.06
        # Dropout率
        drop_rate = 0.
        # 注意力Dropout率
        attn_drop_rate = 0.
        # 是否使用qkv偏置
        qkv_bias = True
        # 注意力缩放因子
        qk_scale = None

        # 注意力头数
        num_heads = 8
        # 关节点数量
        num_joints = 17

        # 归一化层
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 空间位置嵌入
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, self.total_token_num//self.frame, self.embed_dim))
        # 时间位置嵌入
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, self.embed_dim))

        # 位置嵌入的Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成DropPath率列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.st_depth)]

        # 空间-时间编码块
        self.STEblocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=self.mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.st_depth)])

        # 时间-时间编码块
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=self.mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.st_depth)])

        # 中间块
        self.middle_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=self.mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.middle_depth)])
        # 中间层归一化
        self.middle_norm = norm_layer(self.embed_dim)

        # 如果使用token恢复
        if args.use_recover:
            # 可学习的token嵌入
            self.x_token = nn.Parameter(torch.zeros(1, self.total_token_num, self.embed_dim))

            # 交叉注意力层
            self.cross_attention = Cross_Attention(self.embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                                   qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)

        # 空间归一化
        self.Spatial_norm = norm_layer(self.embed_dim)
        # 时间归一化
        self.Temporal_norm = norm_layer(self.embed_dim)

    def forward(self, x):
        self.prune_vis = None
        self.prune_valid_mask = None
        self.prune_keep_counts = None
        self.last_redundancy_stats = None
        # 获取输入形状：批次大小b，总token数量f_n，特征维度c
        b, f_n, c = x.shape
        # 帧数
        f = self.frame
        # 每帧的token数量
        n = f_n//f
        # 剪枝后的token数量
        pn = self.prune_token_num
        prune_valid_mask = None
        if self.collect_prune_vis:
            prefusion_x = rearrange(x, 'b (f n) c -> b f n c', n=n, f=f)
            self.last_redundancy_stats = compute_temporal_redundancy_stats(
                prefusion_x,
                stage="pre_tte_input_tokens",
                current_frame_index=0,
            )
        if self.use_coarse_ste_tte:
            # 重排输入，将形状从(b, f*n, c)变为(b*f, n, c)
            x = rearrange(x, 'b (f n) c  -> (b f) n c',n=n,f=f)
            # 添加空间位置嵌入
            x += self.Spatial_pos_embed
            # 位置嵌入的Dropout
            x = self.pos_drop(x)
            # 通过第一个空间-时间编码块
            x = self.STEblocks[0](x)
            # 空间归一化
            x = self.Spatial_norm(x)

            # 重排输入，将形状从(b*f, n, c)变为(b*n, f, c)
            x = rearrange(x, '(b f) n c -> (b n) f c', n = n ,f=f)
            # 添加时间位置嵌入
            x += self.Temporal_pos_embed
            # 位置嵌入的Dropout
            x = self.pos_drop(x)
            # 通过第一个时间-时间编码块
            x = self.TTEblocks[0](x)
            # 时间归一化
            x = self.Temporal_norm(x)

            # 重排输入，将形状从(b*n, f, c)变为(b, f, n, c)
            x = rearrange(x, '(b n) f c -> b f n c', n=n,f=f)
            # 遍历剩余的空间-时间编码块
            for i in range(1, self.st_depth):

                # 重排输入，将形状从(b, f, n, c)变为(b*f, n, c)
                x = rearrange(x, 'b f n c -> (b f) n c')
                # 获取当前的空间-时间编码块和时间-时间编码块
                steblock = self.STEblocks[i]
                tteblock = self.TTEblocks[i]

                # 通过空间-时间编码块
                x = steblock(x)
                # 空间归一化
                x = self.Spatial_norm(x)
                # 重排输入，将形状从(b*f, n, c)变为(b*n, f, c)
                x = rearrange(x, '(b f) n c -> (b n) f c', b=b)

                # 通过时间-时间编码块
                x = tteblock(x)
                # 时间归一化
                x = self.Temporal_norm(x)
                # 重排输入，将形状从(b*n, f, c)变为(b, f, n, c)
                x = rearrange(x, '(b n) f c -> b f n c', n=n)

            ##-----------------聚类-----------------##

            # 重排输入，将形状从(b, f, n, c)变为(b, f*n, c)
            x = rearrange(x, 'b f n c -> b (f n) c')

        # 使用动态或静态DPC-KNN聚类算法对token进行聚类；
        # 在 no_prune 模式下，保留所有 f 帧的 token 直接进行 middle blocks 处理
        if not self.no_prune:
            if self.use_dynamic_prune:
                if self.collect_prune_vis:
                    index, idx_cluster, prune_valid_mask, prune_debug = cluster_dpc_knn_dynamic(
                        x,
                        alpha=self.dynamic_alpha,
                        min_keep=self.dynamic_min_keep,
                        max_keep=self.dynamic_max_keep,
                        k=2,
                        pad_to_batch_max=True,
                        return_mask=True,
                        return_debug=True,
                    )
                else:
                    index, idx_cluster, prune_valid_mask = cluster_dpc_knn_dynamic(
                        x,
                        alpha=self.dynamic_alpha,
                        min_keep=self.dynamic_min_keep,
                        max_keep=self.dynamic_max_keep,
                        k=2,
                        pad_to_batch_max=True,
                        return_mask=True,
                    )
                    prune_debug = None
                self.prune_token_num = int(index.shape[1])
            else:
                if self.prune_selection == "random":
                    index, idx_cluster = random_token_prune(x, self.prune_token_num)
                else:
                    index, idx_cluster = cluster_dpc_knn(x, self.prune_token_num, 2)
                    # 静态聚类（topk）按得分返回，需按索引升序保证顺序一致
                    index, _ = torch.sort(index)
                prune_valid_mask = torch.ones(b, index.shape[1], device=x.device, dtype=torch.bool)
                prune_debug = None

            if self.collect_prune_vis:
                self.prune_vis = {
                    "score": prune_debug["score"].detach().cpu() if prune_debug is not None else None,
                    "threshold": prune_debug["threshold"].detach().cpu() if prune_debug is not None else None,
                    "keep_counts": prune_debug["keep_counts"].detach().cpu() if prune_debug is not None else None,
                    "candidate_keep_raw": prune_debug["candidate_keep_raw"].detach().cpu() if prune_debug is not None else None,
                    "candidate_keep_clipped": prune_debug["candidate_keep_clipped"].detach().cpu() if prune_debug is not None else None,
                    "score_mu": prune_debug["score_mu"].detach().cpu() if prune_debug is not None else None,
                    "score_sigma": prune_debug["score_sigma"].detach().cpu() if prune_debug is not None else None,
                    "final_cluster_num": int(prune_debug["final_cluster_num"].detach().cpu().item()) if prune_debug is not None else int(index.shape[1]),
                    "valid_mask": prune_valid_mask.detach().cpu() if prune_valid_mask is not None else None,
                    "clip_low_mask": prune_debug["clip_low_mask"].detach().cpu() if prune_debug is not None else None,
                    "clip_high_mask": prune_debug["clip_high_mask"].detach().cpu() if prune_debug is not None else None,
                    "keep_index": index.detach().cpu(),
                    "frame_num": int(f),
                    "tokens_per_frame": int(n),
                    "total_tokens": int(f_n),
                }
                if self.last_redundancy_stats is not None:
                    self.prune_vis.update({
                        "temporal_redundancy_adj": self.last_redundancy_stats["temporal_redundancy_adj"].detach().cpu(),
                        "temporal_redundancy_center": self.last_redundancy_stats["temporal_redundancy_center"].detach().cpu(),
                        "temporal_redundancy_adj_chrono_prefusion": self.last_redundancy_stats["temporal_redundancy_adj_chrono_prefusion"].detach().cpu(),
                        "temporal_redundancy_current_support_prefusion": self.last_redundancy_stats["temporal_redundancy_current_support_prefusion"].detach().cpu(),
                        "temporal_order": self.last_redundancy_stats["temporal_order"],
                        "temporal_redundancy_stage": self.last_redundancy_stats["temporal_redundancy_stage"],
                        "current_frame_index": self.last_redundancy_stats["current_frame_index"],
                    })

            # 创建批次索引
            batch_ind = torch.arange(b, device=x.device).unsqueeze(-1)
            # 根据聚类中心索引获取聚类中心
            x = x[batch_ind, index]
            if prune_valid_mask is not None:
                x = x * prune_valid_mask.unsqueeze(-1).to(dtype=x.dtype)
                self.prune_valid_mask = prune_valid_mask.detach()
                self.prune_keep_counts = prune_valid_mask.sum(dim=1).detach()

        # 通过中间块
        for blk in self.middle_blocks:
            x = blk(x, key_padding_mask=prune_valid_mask)
        # 中间层归一化
        x = self.middle_norm(x)
        if prune_valid_mask is not None:
            x = x * prune_valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        # 如果使用token恢复且不是no_prune模式（no_prune模式已保留所有token，无需恢复）
        if self.use_recover and not self.no_prune:
            # 复制可学习的token嵌入到批次大小
            x_token = repeat(self.x_token, '() n c -> b n c', b=b)
            # 通过交叉注意力恢复token
            x = x_token + self.cross_attention(x_token, x, x, key_padding_mask=prune_valid_mask)

        # 返回输出
        return x

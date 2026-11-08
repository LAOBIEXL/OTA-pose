import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim=0):
    h, w = t.shape[-2:]
    device = t.device

    y = torch.arange(h, device=device).view(-1, 1).expand(-1, w)
    x = torch.arange(w, device=device).view(1, -1).expand(h, -1)

    grid = torch.stack((x, y), dim=dim)
    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid


def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

class CPB(nn.Module):

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

class DeformableAttention2D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid = False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid =create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group(x),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim = 0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out




class DeformableCrossAttention2D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        feat_dim ,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor


        self.q_feat_cat_conv = nn.Sequential(
            nn.Conv2d(feat_dim + inner_dim , inner_dim , 1,1,0) ,
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )


        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, y , return_vgrid = False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

        # queries


        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        q_feat_cat = torch.cat([q,y],dim=1)
        q_feat_cat = self.q_feat_cat_conv(q_feat_cat)
        grouped_queries = group(q_feat_cat)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid =create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group(x),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim = 0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out




if __name__ == '__main__':
    attn = DeformableAttention2D(
        dim = 1024,                   # feature dimensions
        dim_head = 64,               # dimension per head
        heads = 16,                   # attention heads
        dropout = 0.,                # dropout
        downsample_factor = 4,       # downsample factor (r in paper)
        offset_scale = 4,            # scale of offset, maximum offset
        offset_groups = None,        # number of offset groups, should be multiple of heads
        offset_kernel_size = 6,      # offset kernel size
    )
    cross_attn = DeformableCrossAttention2D(
        dim=1024,  # feature dimensions
        feat_dim=1024,
        dim_head=64,  # dimension per head
        heads=16,  # attention heads
        dropout=0.,  # dropout
        downsample_factor=4,  # downsample factor (r in paper)
        offset_scale=4,  # scale of offset, maximum offset
        offset_groups=None,  # number of offset groups, should be multiple of heads
        offset_kernel_size=6,
    )
    x = torch.randn(16, 1024, 16, 12)
    feat = torch.randn(16, 1024, 16, 12)
    print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in attn.parameters())))
    print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in cross_attn.parameters())))

    y = cross_attn(x, feat)

    print(y.shape)

    # dpr = [x.item() for x in torch.linspace(0, 0.12, 4)]
    # print(dpr)
    # print("!!!!!!!!!!!!!")

    from posetimation.zoo.layers.spatiotemporal_attention import *
    # attn = new_Spatiotemporal_Atten(length=192, channel=1024, frame_num=3,group= 1 )
    # print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in attn.parameters())))
    # # print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in cross_attn.parameters())))
    # batch_size = 1
    # frame_num = 3
    # x = torch.randn(batch_size * frame_num, 192, 1024)
    #
    # import time
    # model_start_time = time.time()
    # y = attn(x)

    # print("=> Spatiotemporal_Atten time: {}".format(time.time() - model_start_time))

    test_stnet = False
    if test_stnet:
        sttrans = STTrans(1024, depth=2, num_heads=16, mlp_ratio=4,)
        print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in sttrans.parameters())))
        # print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in cross_attn.parameters())))
        batch_size = 1
        frame_num = 3
        x = torch.randn(batch_size * frame_num, 192, 1024)

        import time

        model_start_time = time.time()
        y = sttrans(x)

        print("=> STTrans time: {}".format(time.time() - model_start_time))

        print(y.shape)
        param_count = {}
        for name, module in sttrans.named_modules():
            # 收集模块的参数
            params = list(module.parameters())
            # 累加参数量
            param_count[name] = sum(p.numel() for p in params)

        # 打印每个模块的参数量
        for name, count in param_count.items():
            print(f"{name}: {count}")
import torch
from torch import nn
from einops import rearrange
from posetimation.layers.basic_model import ChainOfBasicBlocks
from.self_attention import SimplifiedScaledDotProductAttention

class ChannelAttentionModule(nn.Module):

    def __init__(self, d_model=48, d_cond=48, kernel_size=3, H=96, W=72, n_heads=1, self_att=False):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1,padding=1)
        self.self_att = self_att
        if not self_att:
            # self.cnn_cond = nn.Conv2d(d_cond, d_cond, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            self.cnn_cond = nn.Conv2d(d_cond, d_model, kernel_size=3, stride=1,padding=1)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=n_heads)

    def forward(self, x, cond=None):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)  # bs,c,h*w

        if not self.self_att:
            # _,c_cond,_,_ = cond.shape
            y_cond = self.cnn_cond(cond)
            # y_cond = y_cond.view(bs,c_cond,-1)
            y_cond = y_cond.view(bs, c, -1)
            y = self.pa(y_cond, y, y)  # bs,c_cond,h*w
            # y = self.pa(y, y_cond, y_cond)
            y=y.view(bs,c,h,w)
        else:
            y = self.pa(y, y, y)  # bs,c,h*w

        return y


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        self.fre = ChainOfBasicBlocks(channels, channels, num_blocks=1)
        # self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = ChainOfBasicBlocks(channels, channels, num_blocks=1)
        # self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = ChannelAttentionModule()
        # self.fre_att = Attention(dim=channels)
        # self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(
            # nn.Conv2d(2*channels, channels, 3, 1, 1),
            ChainOfBasicBlocks(channels*2, channels, num_blocks=2),
            ChainOfBasicBlocks(channels, channels*2, num_blocks=2),
            # nn.Conv2d(channels, 2*channels, 3, 1, 1),
            nn.Sigmoid())
        # self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())


    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa)#27_1
        # fre = self.fre_att(fre, spa)+fre#27
        spa = self.fre_att(spa, fre)
        # spa = self.fre_att(spa, fre)+spa
        # fuse = self.fuse(torch.cat((fre, spa), 1))
        # fre_a, spa_a = fuse.chunk(2, dim=1)
        # spa = spa_a * spa
        # fre = fre * fre_a
        fre = self.fre_att(fre, spa)#
        spa = self.fre_att(spa, fre)#
        res = fre*0.5 + spa*0.5

        # fre = self.fre(fre)
        # spa = self.spa(spa)
        # fre = self.fre_att(fre, spa)  # 27_1
        # # fre = self.fre_att(fre, spa)+fre#27
        # spa = self.fre_att(spa, fre)
        # # spa = self.fre_att(spa, fre)+spa
        # fuse = self.fuse(torch.cat((fre, spa), 1))
        # fre_a, spa_a = fuse.chunk(2, dim=1)
        # spa = spa_a * spa
        # fre = fre * fre_a
        # res = fre + spa

        # res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res
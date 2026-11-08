
import time

import matplotlib.pyplot as plt

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum



class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - torch.max(sim, dim=-1, keepdim=True)[0].detach()
        # sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


def random_orthogonal_matrix(dim_size, feature_size):
    # generate a random matrix of shape (dim_size, feature_size)
    random_matrix = torch.randn((dim_size, feature_size) , requires_grad=False)
    # apply QR decomposition to get an orthogonal matrix of shape (dim_size, feature_size)
    Q, R = torch.linalg.qr(random_matrix, 'reduced')
    # return the orthogonal matrix
    Q = Q.cuda()
    return Q

def random_feature_map(x, orthogonal_matrix):
    # x is an input tensor of shape (batch_size, seq_length, dim_size)
    # orthogonal_matrix is a random orthogonal matrix of shape (dim_size, feature_size)
    # apply linear transformation to x using the orthogonal matrix
    orthogonal_matrix =orthogonal_matrix.cuda()
    x = einsum("b h d i,d j -> b h i j" , x , orthogonal_matrix)
    # apply trigonometric functions to x to get positive random features
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    # return the output tensor of shape (batch_size, seq_length, 2 * feature_size)
    return x

def context_matrix(k, v):
    # k is an approximate key tensor of shape (batch_size, seq_length, 2 * feature_size)
    # v is a value tensor of shape (batch_size, seq_length, value_dim)

    c = torch.einsum("b h d i, b h j d -> b h i j", k , v )
    # return the output tensor of shape (batch_size, 2 * feature_size, value_dim)
    return c

class FavorAttention(nn.Module):
    def __init__(self, dim ,feature_size = 32 , heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

        self.orthogonal_matrix = random_orthogonal_matrix(dim_head ,
                                                          feature_size)


    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )




        q = random_feature_map(q , self.orthogonal_matrix)
        k = random_feature_map(k , self.orthogonal_matrix)
        q = q * self.scale



        context = context_matrix(k , v)



        out = torch.einsum("b h d i, b h j d -> b h i j" ,  context ,  q)

        out = rearrange(out,
                        "b h c (x y) -> b (h c) x y",
                        h=self.heads,
                        x=h,  y=w)

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64 , dropout = 0.0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim , hidden_dim)
        self.to_k = nn.Linear(dim , hidden_dim)
        self.to_v = nn.Linear(dim , hidden_dim)


        self.to_out = nn.Linear(hidden_dim , dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        b, n , d = query.shape
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q , k , v)
        )
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim - torch.max(sim, dim=-1, keepdim=True)[0].detach()
        # sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        if attn_mask is not None:
            sim.masked_fill_(attn_mask == 0, float("-inf"))  # 将mask为0的位置填充为负无穷，使得softmax后为0
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i d, b h d j -> b h i j", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads)
        out = self.to_out(out)
        return out

if __name__ == "__main__":

    device = torch.device("cuda:0")
    # x = torch.randn((128,48,24,18))
    x = torch.randn((1,192,768))
    y = torch.randn((1,256,768))
    # x = torch.randn((128,48,24,18))
    # x = torch.randn((128,48,96,72))
    x = x.to(device)
    y = y.to(device)
    # attn = Attention_mod(48)
    attn = CrossAttention(768 , 12 , 64 , dropout=0.3)
    attn = attn.to(device)


    start = time.time()
    out = attn(x , y , y)
    end  = time.time()
    print((end-start))


    print(out.shape)
    attn_params = sum(p.numel() for p in attn.parameters())
    print("=>Attention的参数量为: {}".format(attn_params))


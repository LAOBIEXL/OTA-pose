import torch.nn as nn
import torch


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



if __name__ == '__main__':
    attn = STTrans(1024, depth=1, mlp_dim=64, num_heads=8,dim_per_head=64, dropout=0)


    x = torch.randn(2, 192,3,1024)
    y = attn(x)

    print(y.shape)
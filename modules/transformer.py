import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class ScaleDotProductAttention(nn.Module):
    def __init__(self, dim, attn_mode, num_heads, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, f'Dim {dim} must be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        self.attn_mode = attn_mode
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim*2, bias=qkv_bias)

    
    def forward(self, x):
        B, N, C = x.shape
        halfB = B // 2
        x_q, x_kv = x, x
        # query shape [B, N, num_head, C // num_heads]
        query = self.q_proj(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 1, 2, 3)
        # kv shape: [2, B, N, num_heads, C // num_heads]
        kv = self.kv_proj(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)

        if self.attn_mode == 'self':
            key, value = kv[0], kv[1]
        elif self.attn_mode == 'cross':
            k1, k2 = kv[0].split(halfB)
            v1, v2 = kv[1].split(halfB)
            key = torch.cat([k2, k1], dim=0)
            value = torch.cat([v2, v1], dim=0)
        else:
            raise ValueError(f'Invalid attention mode:{self.attn_mode}.')


        QK = torch.einsum('bnhd,bmhd->bnmh', query, key)
        QK = QK.permute(0, 3, 1, 2).contiguous()  / self.dim**0.5
        prob = F.softmax(QK, dim=-1)
        QKV = torch.einsum('bhnm,bmhd->bdhn', prob, value)
        x = QKV.permute(0, 3, 1, 2).contiguous().reshape(B, N, -1)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):    
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class TransformerEncodeBlock(nn.Module):
    def __init__(self, dim, attn_mode, num_heads, qkv_bias=False, 
                mlp_ratio=4, drop_path=0., norm_layer=nn.LayerNorm,
                mlp_act_layer=nn.GELU, mlp_drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.scale_dotproduct_attn = ScaleDotProductAttention(
            dim, attn_mode=attn_mode, 
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop)
    

    def forward(self, x):
        x = x + self.drop_path(self.scale_dotproduct_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class TransformerEncoderLayers(nn.Module):
    def __init__(self,  dim, layer_attn_modes, num_heads, qkv_bias=False, 
                mlp_ratio=4, drop_path=0., norm_layer=nn.LayerNorm,
                mlp_act_layer=nn.GELU, mlp_drop=0.
    ):
        super().__init__()
        self.TEL = nn.ModuleList(
            [
                TransformerEncodeBlock(dim, layer_attn_mode, num_heads, qkv_bias, mlp_ratio, drop_path, norm_layer, mlp_act_layer, mlp_drop) for layer_attn_mode in layer_attn_modes
            ]
        )
        
    
    def forward(self, x):
        for i, transformer_block in enumerate(self.TEL):
            x = transformer_block(x)
        
        return x
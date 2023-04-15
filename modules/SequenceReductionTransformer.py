import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class SequenceReductionAttention(nn.Module):
    def __init__(self, dim, attn_mode, num_heads, qkv_bias=False,
                qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1
    ):
        super().__init__()
        assert dim % num_heads == 0, f'Dim {dim} must be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        self.attn_mode = attn_mode
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sequence_reduction = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    

    def forward(self, x, H, W):
        B, N, C = x.shape

        if self.attn_mode == 'self':
            query = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                # shape: [B, N/R**2, C]
                x_ = self.sequence_reduction(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv_proj(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            key, value = kv[0], kv[1]
            
            QK = (query @ key.transpose(-2, -1)) * self.scale
            QK = QK.softmax(dim=-1)
            QK = self.attn_drop(QK)

            x = (QK @ value).transpose(1, 2).reshape(B, N, C)

        elif self.attn_mode == 'cross':
            halfB = B // 2
            q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q1,q2 = q.split(halfB)
            
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                # shape: [B, N/R**2, C]
                x_ = self.sequence_reduction(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv_proj(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k1, k2 = kv[0].split(halfB)
            v1, v2 = kv[1].split(halfB)

            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)

            x1 = (attn1 @ v2).transpose(1, 2).reshape(halfB, N, C)
            x2 = (attn2 @ v1).transpose(1, 2).reshape(halfB, N, C)

            x = torch.cat([x1, x2], dim=0)

        x = self.proj(x)
        x = self.proj_drop(x)

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


class SequenceReductionTransformerEncodeBlock(nn.Module):
    def __init__(self, dim, attn_mode, num_heads, qkv_bias=False,
                qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                mlp_ratio=4, drop_path=0., norm_layer=nn.LayerNorm,
                mlp_act_layer=nn.GELU, mlp_drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sr_attn = SequenceReductionAttention(
                dim, attn_mode, num_heads, 
                qkv_bias, qk_scale, attn_drop, 
                proj_drop, sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop)
    

    def forward(self, x, H, W):
        x = x + self.drop_path(self.sr_attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SequenceReductionTransformerEncoderLayers(nn.Module):
    def __init__(self, dim, layer_attn_modes, num_heads, qkv_bias=False,
                qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                mlp_ratio=4, drop_path=0., norm_layer=nn.LayerNorm,
                mlp_act_layer=nn.GELU, mlp_drop=0.
    ):
        super().__init__()
        self.SRTEL = nn.ModuleList(
            [
                SequenceReductionTransformerEncodeBlock(
                    dim, layer_attn_mode, num_heads, 
                    qkv_bias, qk_scale, attn_drop, 
                    proj_drop, sr_ratio, mlp_ratio, 
                    drop_path, norm_layer, mlp_act_layer, 
                    mlp_drop) for layer_attn_mode in layer_attn_modes
            ]
        )
        

    def forward(self, x, H, W):
        for i, transformer_block in enumerate(self.SRTEL):
            x = transformer_block(x, H, W)
        
        return x
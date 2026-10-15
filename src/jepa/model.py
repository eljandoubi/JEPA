import torch.nn.functional as F
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
 
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)
 
class SelfAttentionEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_p=0., proj_p=0., fused_attn=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_p = attn_p
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)
 
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))
 
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, mlp_p=0.):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(mlp_p)
 
    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
 
class EncoderBlock(nn.Module):
    def __init__(self, fused_attention=True, embed_dim=768, num_heads=12, mlp_ratio=4,
                 proj_p=0., attn_p=0., mlp_p=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim, eps=1e-6)
        self.attn  = SelfAttentionEncoder(embed_dim, num_heads, attn_p, proj_p, fused_attention)
        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        self.mlp   = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, act_layer, mlp_p)
 
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
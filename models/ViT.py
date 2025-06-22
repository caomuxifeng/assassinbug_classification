import torch
import torch.nn as nn
import os
import warnings

# 在文件开头添加预训练权重路径配置
PRETRAINED_PATHS = {
    'vit_base': 'pretrained_weights/vit/vit_base.pth',
    'vit_large': 'pretrained_weights/vit/vit_large.pth'
}

class PatchEmbedding(nn.Module):
    """split image into patches and perform linear embedding"""
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x

class Attention(nn.Module):
    """multi-head self-attention mechanism"""
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """multi-layer perceptron"""
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer encoder block"""
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer main model"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=None,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        if num_classes is None:
            num_classes = {'subfamily': 1000, 'genus': 1000, 'species': 1000}
            
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.ModuleDict({
            'subfamily': nn.Linear(embed_dim, num_classes['subfamily']),
            'genus': nn.Linear(embed_dim, num_classes['genus']),
            'species': nn.Linear(embed_dim, num_classes['species'])
        })
        
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        features = x[:, 0]  # use [CLS] token features
        
        return {
            'subfamily': self.classifier['subfamily'](features),
            'genus': self.classifier['genus'](features),
            'species': self.classifier['species'](features)
        }

# define model configuration with different parameter sizes
def vit_tiny(img_size=224, n_classes=1000):
    """create ViT-Tiny model"""
    return VisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=192, depth=12, n_heads=3, n_classes=n_classes
    )

def vit_small(img_size=224, n_classes=1000):
    """create ViT-Small model"""
    return VisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, n_heads=6, n_classes=n_classes
    )

def vit_base(img_size=224, n_classes=1000):
    """create ViT-Base model"""
    return VisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, n_heads=12, n_classes=n_classes
    )

def vit_large(img_size=224, n_classes=1000):
    """create ViT-Large model"""
    return VisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, n_heads=16, n_classes=n_classes
    )

# model configuration
model_configs = {
    'vit_tiny': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 192,
        'depth': 12,
        'n_heads': 3,
    },
    'vit_small': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'n_heads': 6,
    },
    'vit_base': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'n_heads': 12,
    },
    'vit_large': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'n_heads': 16,
    },
}

def get_vit(model_name, num_classes=None, pretrained=False, pretrained_path=None):
    """
    get ViT model instance with specified configuration
    Args:
        model_name: model name, must exist in model_configs
        num_classes: classification number dictionary
        pretrained: whether to use pretrained weights
        pretrained_path: local pretrained weight path
    """
    assert model_name in model_configs.keys(), f"Unsupported model name: {model_name}"
    
    config = model_configs[model_name]
    model = VisionTransformer(num_classes=num_classes, **config)
    
    if pretrained:
        # if pretrained_path is not specified, use the predefined path
        if pretrained_path is None and model_name in PRETRAINED_PATHS:
            pretrained_path = PRETRAINED_PATHS[model_name]
            
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                
                # remove classifier weights
                for k in list(state_dict.keys()):
                    if 'classifier' in k or 'head' in k:
                        del state_dict[k]
                
                model.load_state_dict(state_dict, strict=False)
                print(f"successfully loaded local pretrained weights: {pretrained_path}")
            except Exception as e:
                warnings.warn(f"failed to load local pretrained weights: {str(e)}")
    
    return model

# export settings
__all__ = ['get_vit', 'model_configs']

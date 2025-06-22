"""
Swin Transformer model implementation

configuration management:
- model_configs: only contains model architecture related configuration
- training related configuration in configs/training/swin.yaml
"""

import os
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# pretrained weight path configuration
PRETRAINED_PATHS = {
    'swin_tiny': 'pretrained_weights/swin/swin_tiny_patch4_window7_224.pth',
    'swin_small': 'pretrained_weights/swin/swin_small_patch4_window7_224.pth',
    'swin_base': 'pretrained_weights/swin/swin_base_patch4_window7_224.pth',
    'swin_large': 'pretrained_weights/swin/swin_large_patch4_window7_224_22kto1k.pth'
}

# ============================================================================
# auxiliary functions
# ============================================================================

def window_partition(x, window_size):
    """partition feature map into multiple windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """reverse windows to feature map"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# ============================================================================
# basic components
# ============================================================================

class Mlp(nn.Module):
    """multi-layer perceptron"""
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

class WindowAttention(nn.Module):
    """window multi-head self-attention module"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # apply dropout on attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # apply dropout after projection layer
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # apply dropout on attention weights, reduce over-concentration
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        # apply dropout on projection result, increase feature diversity
        x = self.proj_drop(x)
        return x

class PatchEmbed(nn.Module):
    """image to patch embedding"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # normalization
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # normalization before FFN, reduce internal covariate shift, improve feature expression ability
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size, expected {H*W} but got {L}"

        shortcut = x
        # normalization before MSA, make deep network easier to train
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # calculate padding
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # periodic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.calculate_mask((Hp, Wp)).to(x.device)
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        # normalization before FFN, improve model generalization ability
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def calculate_mask(self, input_resolution):
        # calculate attention mask
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

class BasicLayer(nn.Module):
    """basic Swin Transformer layer"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # add downsample layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

# ============================================================================
# main model
# ============================================================================

class SwinTransformer(nn.Module):
    """Swin Transformer main model"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes: Dict[str, int] = None,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        
        if num_classes is None:
            num_classes = {'subfamily': 1000, 'genus': 1000, 'species': 1000}

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # calculate initial patch resolution
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # apply dropout after patch embedding, increase input randomness, prevent overfitting
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # calculate current layer resolution and dimension
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer_resolution = [
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer)
            ]
            
            layer = BasicLayer(
                dim=layer_dim,
                input_resolution=layer_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # last normalization layer, used for final feature normalization
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # multi-task classification head
        self.classifier = nn.ModuleDict({
            'subfamily': nn.Linear(self.num_features, num_classes['subfamily']),
            'genus': nn.Linear(self.num_features, num_classes['genus']),
            'species': nn.Linear(self.num_features, num_classes['species'])
        })

        # check if all submodules are on the same device
        def check_device(module):
            for child in module.children():
                if list(child.parameters()):
                    device = next(child.parameters()).device
                    if device != self.device:
                        child.to(self.device)
                check_device(child)
        
        self.device = next(self.parameters()).device
        check_device(self)

    def forward(self, x):
        x = self.patch_embed(x)  # B, N, C
        # apply dropout on patch embedding, increase input randomness
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)

        # normalize final feature
        x = self.norm(x)  # B, N, C
        x = self.avgpool(x.transpose(1, 2))  # B, C, 1
        x = torch.flatten(x, 1)  # B, C
        
        return {
            'subfamily': self.classifier['subfamily'](x),
            'genus': self.classifier['genus'](x),
            'species': self.classifier['species'](x)
        }

# add PatchMerging class for downsampling
class PatchMerging(nn.Module):
    """Patch Merging Layer.
    
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # normalization after merging patches
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, expected {H*W} but got {L}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # normalize merged feature
        x = self.norm(x)
        x = self.reduction(x)

        return x

# ============================================================================
# model configuration and factory function
# ============================================================================

model_configs = {
    'swin_tiny': {
        'img_size': 224,
        'patch_size': 4,
        'window_size': 7,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.2
    },
    'swin_small': {
        'img_size': 224,
        'patch_size': 4,
        'window_size': 7,
        'embed_dim': 96,
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.3
    },
    'swin_base': {
        'img_size': 224,
        'patch_size': 4,
        'window_size': 7,
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.5
    },
    'swin_large': {
        'img_size': 224,
        'patch_size': 4,
        'window_size': 7,
        'embed_dim': 192,
        'depths': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.2
    }
}

def get_swin(model_name: str, num_classes: Dict[str, int] = None, pretrained: bool = False, pretrained_path: str = None) -> SwinTransformer:
    """
    get Swin Transformer model instance
    
    Args:
        model_name: model name, must exist in model_configs
        num_classes: classification number dictionary, containing subfamily, genus and species categories
        pretrained: whether to use pretrained weights
        pretrained_path: local pretrained weight path
    """
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    config = model_configs[model_name]
    model = SwinTransformer(num_classes=num_classes, **config)
    
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
__all__ = ['get_swin', 'model_configs']

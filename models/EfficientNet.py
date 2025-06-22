import torch
import torch.nn as nn
import math
import os
import warnings

#pretrained weight path configuration
PRETRAINED_PATHS = {
    'efficientnet_b0': 'pretrained_weights/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119.pth',
    'efficientnet_b1': 'pretrained_weights/efficientnet/efficientnet-b1_3rdparty_8xb32_in1k_20220119.pth',
    'efficientnet_b2': 'pretrained_weights/efficientnet/efficientnet-b2_3rdparty_8xb32_in1k_20220119.pth',
    'efficientnet_b3': 'pretrained_weights/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119.pth',
    'efficientnet_b4': 'pretrained_weights/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119.pth',
    'efficientnet_b5': 'pretrained_weights/efficientnet/efficientnet-b5_3rdparty_8xb32_in1k_20220119.pth',
    'efficientnet_b6': 'pretrained_weights/efficientnet/efficientnet-b6_3rdparty_8xb32-aa_in1k_20220119.pth',
    'efficientnet_b7': 'pretrained_weights/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119.pth'
}

# ============================================================================
# Basic components
# ============================================================================

def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)

def round_filters(filters, multiplier, divisor=8, min_width=None):
    """calculate and round the number of filters based on the width multiplier"""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # ensure the rounded value does not decrease by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    """round the number of repeats based on the depth multiplier"""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConvBlock(nn.Module):
    """MBConv block"""
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_connect_rate=None):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.drop_connect_rate = drop_connect_rate

        layers = []
        if expand_ratio != 1:
            # expansion layer
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False)
            ])
        layers.extend([
            # depthwise separable convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=False)
        ])
        if se_ratio:
            layers.append(SEModule(hidden_dim, reduction=int(1/se_ratio)))
        # projection layer
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size()[0]
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x * binary_tensor / keep_prob

    def forward(self, x):
        if self.use_res_connect:
            if self.drop_connect_rate:
                return x + self._drop_connect(self.conv(x))
            else:
                return x + self.conv(x)
        else:
            return self.conv(x)

# ============================================================================
# Main model
# ============================================================================

class EfficientNet(nn.Module):
    """EfficientNet main model"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=None):
        """
        initialize EfficientNet
        Args:
            width_mult: width multiplier
            depth_mult: depth multiplier
            dropout_rate: dropout ratio
            num_classes: dictionary, containing the number of categories for each level, e.g.:
                        {'subfamily': 10, 'genus': 20, 'species': 100}
        """
        super(EfficientNet, self).__init__()
        
        # ensure num_classes is a dictionary
        if not isinstance(num_classes, dict):
            raise ValueError("num_classes must be a dictionary containing class numbers for each level")
        
        # MBConv block configuration
        self.cfgs = [
            # t, c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3
            [6,  24, 2, 2, 3],  # MBConv6_3x3
            [6,  40, 2, 2, 5],  # MBConv6_5x5
            [6,  80, 3, 2, 3],  # MBConv6_3x3
            [6, 112, 3, 1, 5],  # MBConv6_5x5
            [6, 192, 4, 2, 5],  # MBConv6_5x5
            [6, 320, 1, 1, 3],  # MBConv6_3x3
        ]

        # build first layer
        input_channel = round_filters(32, width_mult)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=False)
        )]

        # build MBConv blocks
        for t, c, n, s, k in self.cfgs:
            output_channel = round_filters(c, width_mult)
            for i in range(round_repeats(n, depth_mult)):
                if i == 0:
                    self.features.append(MBConvBlock(input_channel, output_channel, k, s, expand_ratio=t))
                else:
                    self.features.append(MBConvBlock(input_channel, output_channel, k, 1, expand_ratio=t))
                input_channel = output_channel

        # build last few layers
        output_channel = round_filters(1280, width_mult)
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=False)
        ))
        self.features = nn.Sequential(*self.features)

        # build classifiers for each level
        self.classifiers = nn.ModuleDict({
            level: nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(output_channel, num_classes[level])
            ) for level in num_classes.keys()
        })

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        
        # classify each level
        return {
            level: classifier(x) 
            for level, classifier in self.classifiers.items()
        }

    def _initialize_weights(self):
        """initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:  # check if bias exists
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:  # check if bias exists
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # check if bias exists
                    nn.init.zeros_(m.bias)



# modify get_efficientnet function
def get_efficientnet(model_name: str, num_classes: dict, pretrained: bool = False, pretrained_path: str = None):
    """
    Get EfficientNet model instance
    Args:
        model_name: model name, e.g. 'efficientnet_b0'
        num_classes: dictionary, containing the number of categories for each level
        pretrained: whether to use pretrained weights
        pretrained_path: local pretrained weight path
    
    Returns:
        EfficientNet: initialized model instance
    
    Raises:
        ValueError: when model_name is not supported
    """
    assert model_name in model_configs.keys(), f"Unsupported model name: {model_name}"
    config = model_configs[model_name]
    
    model = EfficientNet(
        width_mult=config['width_mult'],
        depth_mult=config['depth_mult'],
        dropout_rate=config['dropout_rate'],
        num_classes=num_classes
    )
    
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
                
                # remove classification head weights
                for k in list(state_dict.keys()):
                    if 'classifier' in k or 'head' in k:
                        del state_dict[k]
                
                model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded local pretrained weights: {pretrained_path}")
            except Exception as e:
                warnings.warn(f"Failed to load local pretrained weights: {str(e)}")
    
    return model

# define different versions of EfficientNet
def efficientnet_b0(num_classes=1000):
    return EfficientNet(width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=num_classes)

def efficientnet_b1(num_classes=1000):
    return EfficientNet(width_mult=1.0, depth_mult=1.1, dropout_rate=0.2, num_classes=num_classes)

def efficientnet_b2(num_classes=1000):
    return EfficientNet(width_mult=1.1, depth_mult=1.2, dropout_rate=0.3, num_classes=num_classes)

def efficientnet_b3(num_classes=1000):
    return EfficientNet(width_mult=1.2, depth_mult=1.4, dropout_rate=0.3, num_classes=num_classes)

def efficientnet_b4(num_classes=1000):
    return EfficientNet(width_mult=1.4, depth_mult=1.8, dropout_rate=0.4, num_classes=num_classes)

def efficientnet_b5(num_classes=1000):
    return EfficientNet(width_mult=1.6, depth_mult=2.2, dropout_rate=0.4, num_classes=num_classes)

def efficientnet_b6(num_classes=1000):
    return EfficientNet(width_mult=1.8, depth_mult=2.6, dropout_rate=0.5, num_classes=num_classes)

def efficientnet_b7(num_classes=1000):
    return EfficientNet(width_mult=2.0, depth_mult=3.1, dropout_rate=0.5, num_classes=num_classes)

# model configuration
model_configs = {
    'efficientnet_b0': {'input_size': 224, 'width_mult': 1.0, 'depth_mult': 1.0, 'dropout_rate': 0.2},
    'efficientnet_b1': {'input_size': 240, 'width_mult': 1.0, 'depth_mult': 1.1, 'dropout_rate': 0.2},
    'efficientnet_b2': {'input_size': 260, 'width_mult': 1.1, 'depth_mult': 1.2, 'dropout_rate': 0.3},
    'efficientnet_b3': {'input_size': 300, 'width_mult': 1.2, 'depth_mult': 1.4, 'dropout_rate': 0.3},
    'efficientnet_b4': {'input_size': 380, 'width_mult': 1.4, 'depth_mult': 1.8, 'dropout_rate': 0.4},
    'efficientnet_b5': {'input_size': 456, 'width_mult': 1.6, 'depth_mult': 2.2, 'dropout_rate': 0.4},
    'efficientnet_b6': {'input_size': 528, 'width_mult': 1.8, 'depth_mult': 2.6, 'dropout_rate': 0.5},
    'efficientnet_b7': {'input_size': 600, 'width_mult': 2.0, 'depth_mult': 3.1, 'dropout_rate': 0.5},
}

# export settings
__all__ = ['get_efficientnet', 'model_configs']

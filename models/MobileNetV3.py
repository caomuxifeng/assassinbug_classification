import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings

# pretrained weight path configuration
PRETRAINED_PATHS = {
    'mobilenetv3_large': 'pretrained_weights/mobilenetv3/mobilenetv3_large.pth',
    'mobilenetv3_small': 'pretrained_weights/mobilenetv3/mobilenetv3_small.pth'
}

def make_divisible(v, divisor, min_value=None):
    """ensure the number of channels in the network layer is a multiple of 8"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    """Hard Sigmoid activation function"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3).div_(6.)

class h_swish(nn.Module):
    """Hard Swish activation function"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer"""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileNetV3Block(nn.Module):
    """MobileNetV3 basic building block"""
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, se_ratio, nl):
        super(MobileNetV3Block, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = make_divisible(inp * expand_ratio, 8)
        
        # convert string activation function to actual activation function class
        if isinstance(nl, str):
            if nl == 'RE':
                nl = nn.ReLU
            elif nl == 'HS':
                nl = h_swish
            else:
                raise ValueError(f"Unsupported activation function: {nl}")
        
        layers = []
        
        # 1. expansion layer - only add when needed
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nl(inplace=True) 
            ])
        else:
            hidden_dim = inp
        
        # 2. depthwise separable convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nl(inplace=True)  
        ])
        
        # 3. SE layer
        if se_ratio is not None:
            se_channel = make_divisible(hidden_dim * se_ratio, 8)
            layers.append(SELayer(hidden_dim, reduction=se_channel))
        
        # 4. projection layer
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV3(nn.Module):
    """MobileNetV3 main structure"""
    def __init__(self, cfgs, mode, num_classes=None, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        if num_classes is None:
            num_classes = {'subfamily': 1000, 'genus': 1000, 'species': 1000}
            
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # small initial channel number
        input_channel = make_divisible(16 * width_mult, 8)
        layers = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            h_swish()
        ]

        # build intermediate layers
        for k, t, c, se, nl, s in self.cfgs:
            output_channel = make_divisible(c * width_mult, 8)
            layers.append(
                MobileNetV3Block(
                    input_channel, 
                    output_channel,
                    k, s, t,  # t is expand_ratio
                    se, nl
                )
            )
            input_channel = output_channel

        # reduce the number of channels in the last layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        if mode == 'small':
            last_channel = 1024  # use smaller channels for small model

        layers.extend([
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            h_swish()
        ])

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # modify classification head, reduce dropout
        if isinstance(num_classes, dict):
            self.classifier = nn.ModuleDict({
                level: nn.Sequential(
                    nn.Dropout(p=0.1),  # reduce dropout rate
                    nn.Linear(last_channel, class_num)
                ) for level, class_num in num_classes.items()
            })
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(last_channel, num_classes)
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # return multi-level classification results
        return {
            level: classifier(x)
            for level, classifier in self.classifier.items()
        }

    @torch.cuda.amp.autocast()  # add mixed precision training support
    def _forward_impl(self, x):
        return self.forward(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def mobilenet_v3_large(num_classes=1000):
    """MobileNetV3-Large model configuration optimization"""
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,   1,  16, 0, 'RE', 1],
        [3,   3,  24, 0, 'RE', 2],  
        [3,   3,  24, 0, 'RE', 1],
        [5,   3,  40, 1, 'RE', 2],
        [5,   3,  40, 1, 'RE', 1],
        [5,   3,  40, 1, 'RE', 1],
        [3,   4,  80, 0, 'HS', 2], 
        [3,   3,  80, 0, 'HS', 1],
        [3,   3,  80, 0, 'HS', 1],
        [3,   3,  80, 0, 'HS', 1],
        [3,   4, 112, 1, 'HS', 1],
        [3,   4, 112, 1, 'HS', 1],
        [5,   4, 160, 1, 'HS', 2],
        [5,   4, 160, 1, 'HS', 1],
        [5,   4, 160, 1, 'HS', 1]
    ]
    return MobileNetV3(cfgs, mode='large', num_classes=num_classes)

def mobilenet_v3_small(num_classes=1000):
    """MobileNetV3-Small model configuration optimization"""
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,    1,  16, 1, 'RE', 2],  
        [3,    4,  24, 0, 'RE', 2],
        [3,    3,  24, 0, 'RE', 1],
        [5,    3,  40, 1, 'HS', 2],  
        [5,    3,  40, 1, 'HS', 1],
        [5,    3,  40, 1, 'HS', 1],
        [5,    3,  48, 1, 'HS', 1],
        [5,    3,  48, 1, 'HS', 1],
        [5,    4,  96, 1, 'HS', 2],  
        [5,    4,  96, 1, 'HS', 1],
        [5,    4,  96, 1, 'HS', 1],
    ]
    return MobileNetV3(cfgs, mode='small', num_classes=num_classes)

# model configuration
model_configs = {
    'mobilenetv3_large': {
        'input_size': 224,
        'width_mult': 1.0,
        'dropout_rate': 0.2,
        'num_classes': 1000,
    },
    'mobilenetv3_small': {
        'input_size': 224,
        'width_mult': 1.0,
        'dropout_rate': 0.2,
        'num_classes': 1000,
    }
}

def get_mobilenet_v3(model_name, num_classes=1000, pretrained=False, pretrained_path=None):
    """
    Get MobileNetV3 model instance
    Args:
        model_name: model name
        num_classes: dictionary, containing the number of categories for each level
        pretrained: whether to use pretrained weights
        pretrained_path: local pretrained weight path
    
    Returns:
        MobileNetV3: initialized model instance
    
    Raises:
        ValueError: when model_name is not supported
    """
    assert model_name in ['mobilenetv3_large', 'mobilenetv3_small'], f"Unsupported model name: {model_name}"
    
    if model_name == 'mobilenetv3_large':
        model = mobilenet_v3_large(num_classes=num_classes)
    else:
        model = mobilenet_v3_small(num_classes=num_classes)
    
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

# export settings
__all__ = ['get_mobilenet_v3', 'model_configs']

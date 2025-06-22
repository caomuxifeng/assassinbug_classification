import torch
import torch.nn as nn
from typing import Dict
import os
import warnings

# pretrained weight path configuration
PRETRAINED_PATHS = {
    'resnet_18': 'pretrained_weights/resnet/resnet18_8xb32_in1k.pth',
    'resnet_34': 'pretrained_weights/resnet/resnet34_8xb32_in1k.pth',
    'resnet_50': 'pretrained_weights/resnet/resnet50_8xb32_in1k.pth',
    'resnet_101': 'pretrained_weights/resnet/resnet101_8xb32_in1k.pth',
    'resnet_152': 'pretrained_weights/resnet/resnet152_8xb32_in1k.pth'
}

# basic residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# Bottleneck block for ResNet-50/101/152
class Bottleneck(nn.Module):
    expansion = 4  # output channels expanded by 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv feature extraction
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv upsample
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # 1x1 conv downsample
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv feature extraction
        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 conv upsample
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# ResNet model
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: Dict[str, int]):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.ModuleDict({
            'subfamily': nn.Linear(512 * block.expansion, num_classes['subfamily']),
            'genus': nn.Linear(512 * block.expansion, num_classes['genus']),
            'species': nn.Linear(512 * block.expansion, num_classes['species'])
        })

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return {
            'subfamily': self.classifier['subfamily'](x),
            'genus': self.classifier['genus'](x),
            'species': self.classifier['species'](x)
        }

# model configuration
model_configs = {
    'resnet_18': {
        'block': BasicBlock,
        'layers': [2, 2, 2, 2],
    },
    'resnet_34': {
        'block': BasicBlock,
        'layers': [3, 4, 6, 3],
    },
    'resnet_50': {
        'block': Bottleneck,
        'layers': [3, 4, 6, 3],
    },
    'resnet_101': {
        'block': Bottleneck,
        'layers': [3, 4, 23, 3],
    },
    'resnet_152': {
        'block': Bottleneck,
        'layers': [3, 8, 36, 3],
    },
}



# get ResNet model function
def get_resnet(model_name: str, num_classes: Dict[str, int] = None, pretrained: bool = False, pretrained_path: str = None):
    """
    get ResNet model instance
    
    Args:
        model_name: model name, must exist in model_configs
        num_classes: classification number dictionary, containing subfamily, genus and species categories
        pretrained: whether to use pretrained weights
        pretrained_path: local pretrained weight path
    
    Returns:
        ResNet: initialized model instance
    
    Raises:
        ValueError: when model_name is not supported in the configuration
    """
    if num_classes is None:
        num_classes = {'subfamily': 1000, 'genus': 1000, 'species': 1000}
    
    config = model_configs[model_name]
    model = ResNet(config['block'], config['layers'], num_classes)
    
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
                    if 'fc' in k or 'classifier' in k:
                        del state_dict[k]
                
                model.load_state_dict(state_dict, strict=False)
                print(f"successfully loaded local pretrained weights: {pretrained_path}")
            except Exception as e:
                warnings.warn(f"failed to load local pretrained weights: {str(e)}")
    
    return model

# export settings
__all__ = ['get_resnet', 'model_configs']

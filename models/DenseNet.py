import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import warnings

# pretrained weight path configuration
PRETRAINED_PATHS = {
    'densenet_121': 'pretrained_weights/densenet/densenet121.pth',
    'densenet_169': 'pretrained_weights/densenet/densenet169.pth',
    'densenet_201': 'pretrained_weights/densenet/densenet201.pth',
    'densenet_161': 'pretrained_weights/densenet/densenet161.pth'
}

class Bottleneck(nn.Module):
    """DenseNet bottleneck layer"""
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)  # 特征连接
        return out

class Transition(nn.Module):
    """Transition layer, used to reduce the size and number of feature maps"""
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """DenseNet with multi-level classification"""
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=None):
        """
        Args:
            block: basic block type
            nblocks: number of layers in each dense block
            growth_rate: growth rate
            reduction: compression factor
            num_classes: dictionary, containing the number of categories for each level
                        {'subfamily': n1, 'genus': n2, 'species': n3}
        """
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        
        # ensure num_classes is a dictionary
        if num_classes is None:
            num_classes = {'subfamily': 1000, 'genus': 1000, 'species': 1000}
        assert isinstance(num_classes, dict), "num_classes should be a dictionary"

        # initial convolution layer
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # dense blocks and transition layers
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        
        # create independent classification heads for each classification level
        self.classifier = nn.ModuleDict({
            'subfamily': nn.Linear(num_planes, num_classes['subfamily']),
            'genus': nn.Linear(num_planes, num_classes['genus']),
            'species': nn.Linear(num_planes, num_classes['species'])
        })

    def _make_dense_layers(self, block, in_channels, nblock):
        """create dense connection blocks"""
        layers = []
        for i in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # feature extraction
        out = self.pool(F.relu(self.bn1(self.conv1(x))))
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1)
        features = out.view(out.size(0), -1)
        
        # multi-level classification
        return {
            'subfamily': self.classifier['subfamily'](features),
            'genus': self.classifier['genus'](features),
            'species': self.classifier['species'](features)
        }

# define different versions of DenseNet
def densenet_121(num_classes=1000):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes)

def densenet_169(num_classes=1000):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_classes=num_classes)

def densenet_161(num_classes=1000):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, num_classes=num_classes)

def densenet_201(num_classes=1000):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_classes=num_classes)

# model configuration
model_configs = {
    'densenet_121': {'growth_rate': 32, 'blocks': [6,12,24,16], 'input_size': 224},
    'densenet_169': {'growth_rate': 32, 'blocks': [6,12,32,32], 'input_size': 224},
    'densenet_201': {'growth_rate': 32, 'blocks': [6,12,48,32], 'input_size': 224},
    'densenet_161': {'growth_rate': 48, 'blocks': [6,12,36,24], 'input_size': 224},
}

def get_densenet(model_name, num_classes=None, pretrained=False, pretrained_path=None):
    """
    Get DenseNet model instance
    Args:
        model_name: model name, must exist in model_configs
        num_classes: dictionary, containing the number of categories for each level
        pretrained: whether to use pretrained weights
        pretrained_path: local pretrained weight path
    """
    assert model_name in model_configs.keys(), f"Unsupported model name: {model_name}"
    
    model_fn = globals()[model_name]
    model = model_fn(num_classes=num_classes)
    
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
                print(f"Successfully loaded pretrained weights: {pretrained_path}")
            except Exception as e:
                warnings.warn(f"Failed to load pretrained weights: {str(e)}")
    
    return model

# export settings
__all__ = ['get_densenet', 'model_configs']

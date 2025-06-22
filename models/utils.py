import torch
import torch.nn as nn

def to_2tuple(x):
    """convert to 2-tuple"""
    if isinstance(x, tuple):
        return x
    return (x, x)

def initialize_weights(model):
    """initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

__all__ = ['to_2tuple', 'initialize_weights'] 
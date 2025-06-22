"""
configuration loader module
used to manage and load model training configuration
"""

import os
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
from torch.optim import (
    SGD,
    Adam, 
    AdamW,
    RMSprop
)
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR, 
    ExponentialLR,
    LambdaLR,
    OneCycleLR,
    _LRScheduler
)

class ConfigLoader:
    def __init__(self, verbose=True):
        """
        initialize configuration loader
        Args:
            verbose: whether to output loading information
        """
        self.verbose = verbose
        self.config_cache = {}  # add cache to avoid repeated loading
        
    def load_model_config(self, model_type, model_name):
        """load model configuration"""
        # use cache to avoid repeated loading
        cache_key = f"{model_type}_{model_name}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
            
        # build configuration file path
        config_path = f'configs/training/{model_type}.yaml'
        
        if self.verbose:
            logging.info(f"loading configuration from {config_path}")
            logging.info(f"find configuration name: {model_name}")
        
        # read YAML configuration file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"failed to load configuration file: {str(e)}")
            return {}
            
        # extract configuration for specified model name
        model_config = {}
        if model_name in config:
            model_config = config[model_name]
            if self.verbose:
                logging.info(f"find configuration for {model_name}: {model_config}")
        else:
            if self.verbose:
                logging.warning(f"configuration for {model_name} not found, using default configuration")
                
        # save to cache
        self.config_cache[cache_key] = model_config
        return model_config
    
    def get_optimizer_config(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """get optimizer configuration"""
        config = self.load_model_config(model_type, model_name)
        if 'optimizer' not in config:
            raise ValueError(f"configuration for {model_name} is missing optimizer setting")
        return config['optimizer']
    
    def get_scheduler_config(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """get learning rate scheduler configuration"""
        config = self.load_model_config(model_type, model_name)
        if 'scheduler' not in config:
            raise ValueError(f"configuration for {model_name} is missing learning rate scheduler setting")
        return config['scheduler']
    
    def create_optimizer(self, model_params, model_type: str, model_name: str):
        """create optimizer instance"""
        config = self.get_optimizer_config(model_type, model_name)
        optimizer_type = config.pop('type')
        
        # ensure correct type conversion for numerical values
        if 'lr' in config:
            config['lr'] = float(config['lr'])
        if 'weight_decay' in config:
            config['weight_decay'] = float(config['weight_decay'])
        if 'momentum' in config:
            config['momentum'] = float(config['momentum'])
        if 'betas' in config:
            config['betas'] = tuple(float(x) for x in config['betas'])
        if 'eps' in config:
            config['eps'] = float(config['eps'])
        
        if optimizer_type == 'AdamW':
            return AdamW(model_params, **config)
        elif optimizer_type == 'Adam':
            return Adam(model_params, **config)
        elif optimizer_type == 'SGD':
            return SGD(model_params, **config)
        elif optimizer_type == 'RMSprop':
            return RMSprop(model_params, **config)
        else:
            raise ValueError(f"unsupported optimizer type: {optimizer_type}")
    
    def create_scheduler(self, optimizer, model_type: str, model_name: str):
        """create learning rate scheduler instance"""
        config = self.get_scheduler_config(model_type, model_name)
        scheduler_type = config.pop('type')
        
        # ensure correct type conversion for numerical values
        if 'eta_min' in config:
            config['eta_min'] = float(config['eta_min'])  # convert to float
        if 'T_max' in config:
            config['T_max'] = int(config['T_max'])  # convert to integer
        if 'warmup_epochs' in config:
            config['warmup_epochs'] = int(config['warmup_epochs'])
        if 'warmup_start_lr' in config:
            config['warmup_start_lr'] = float(config['warmup_start_lr'])
        
        if scheduler_type == 'CosineAnnealingLR':
            return WarmupCosineAnnealingLR(
                optimizer,
                T_max=config['T_max'],
                eta_min=config['eta_min'],
                warmup_epochs=config.get('warmup_epochs', 0),
                warmup_start_lr=config.get('warmup_start_lr', 0.0)
            )
        elif scheduler_type == 'MultiStepLR':
            valid_params = {'milestones', 'gamma'}
            config = {k: v for k, v in config.items() if k in valid_params}
            return MultiStepLR(optimizer, **config)
            
        elif scheduler_type == 'ExponentialLR':
            valid_params = {'gamma'}
            config = {k: v for k, v in config.items() if k in valid_params}
            return ExponentialLR(optimizer, **config)
            
        elif scheduler_type == 'WarmupCosineSchedule':
            valid_params = {'warmup_steps', 't_total', 'cycles'}
            config = {k: v for k, v in config.items() if k in valid_params}
            return WarmupCosineSchedule(
                optimizer,
                warmup_steps=config.get('warmup_steps', 0),
                t_total=config['t_total'],
                cycles=config.get('cycles', 0.5)
            )
        elif scheduler_type == 'OneCycleLR':
            # OneCycleLR specific parameter type conversion
            valid_params = {
                'max_lr', 'total_steps', 'epochs', 'steps_per_epoch',
                'pct_start', 'div_factor', 'final_div_factor'
            }
            config = {k: v for k, v in config.items() if k in valid_params}
            
            # ensure correct type for required parameters
            if 'max_lr' in config:
                if isinstance(config['max_lr'], (list, tuple)):
                    config['max_lr'] = [float(x) for x in config['max_lr']]
                else:
                    config['max_lr'] = float(config['max_lr'])
            if 'total_steps' in config:
                config['total_steps'] = int(config['total_steps'])
            if 'epochs' in config:
                config['epochs'] = int(config['epochs'])
            if 'steps_per_epoch' in config:
                config['steps_per_epoch'] = int(config['steps_per_epoch'])
            if 'pct_start' in config:
                config['pct_start'] = float(config['pct_start'])
            if 'div_factor' in config:
                config['div_factor'] = float(config['div_factor'])
            if 'final_div_factor' in config:
                config['final_div_factor'] = float(config['final_div_factor'])
            
            return OneCycleLR(optimizer, **config)
        else:
            raise ValueError(f"unsupported learning rate scheduler type: {scheduler_type}")

class WarmupCosineSchedule(LambdaLR):
    """warmup cosine learning rate scheduler"""
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # apply cosine annealing to remaining steps
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class WarmupCosineAnnealingLR(_LRScheduler):
    """warmup cosine annealing learning rate scheduler"""
    def __init__(self, optimizer, T_max, eta_min=0.0, warmup_epochs=0, 
                 warmup_start_lr=0.0, last_epoch=-1):
        self.T_max = int(T_max)  # ensure integer
        self.eta_min = float(eta_min)  # ensure float
        self.warmup_epochs = int(warmup_epochs)  # ensure integer
        self.warmup_start_lr = float(warmup_start_lr)  # ensure float
        self.base_lrs = None
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.base_lrs is None:
            self.base_lrs = [float(group['lr']) for group in self.optimizer.param_groups]  # ensure float

        if self.last_epoch < self.warmup_epochs:
            # linear warmup
            alpha = float(self.last_epoch) / float(self.warmup_epochs) if self.warmup_epochs > 0 else 1.0
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # cosine annealing
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.T_max - self.warmup_epochs))
            cos_progress = 0.5 * (1 + math.cos(progress * math.pi))
            return [self.eta_min + (base_lr - self.eta_min) * cos_progress
                    for base_lr in self.base_lrs]
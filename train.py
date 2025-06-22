"""
train script: used to train multi-level insect classification model
"""

import os
import gc
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from models.model_manager import ModelManager
from utils.data_loader import get_dataloader
from utils.logger import setup_logging
from utils.metric import create_metrics_dict, update_metrics_dict, compute_all_metrics_for_cv

# supported model configurations
SUPPORTED_MODELS = {
    # EfficientNet series
    'efficientnet_b0': 'EfficientNet-B0: 最小规模高效网络',
    'efficientnet_b1': 'EfficientNet-B1: 小规模高效网络',
    'efficientnet_b2': 'EfficientNet-B2: 中小规模高效网络',
    'efficientnet_b3': 'EfficientNet-B3: 中等规模高效网络',
    'efficientnet_b4': 'EfficientNet-B4: 中大规模高效网络',
    'efficientnet_b5': 'EfficientNet-B5: 大规模高效网络',
    'efficientnet_b6': 'EfficientNet-B6: 超大规模高效网络',
    'efficientnet_b7': 'EfficientNet-B7: 最大规模高效网络',
    
    # MobileNetV3 series
    'mobilenetv3_small': 'MobileNetV3-Small: 轻量级移动端网络',
    'mobilenetv3_large': 'MobileNetV3-Large: 标准移动端网络',
    
    # ResNet series
    'resnet_18': 'ResNet-18: 轻量级残差网络',
    'resnet_34': 'ResNet-34: 中等规模残差网络',
    'resnet_50': 'ResNet-50: 标准残差网络',
    'resnet_101': 'ResNet-101: 大规模残差网络',
    'resnet_152': 'ResNet-152: 超大规模残差网络',
    
    # DenseNet series
    'densenet_121': 'DenseNet-121: 轻量级密集连接网络',
    'densenet_169': 'DenseNet-169: 中等规模密集连接网络',
    'densenet_201': 'DenseNet-201: 大规模密集连接网络',
    'densenet_161': 'DenseNet-161: 宽版密集连接网络',
    
    # Vision Transformer series
    'vit_tiny': 'ViT-Tiny: 轻量级视觉Transformer',
    'vit_small': 'ViT-Small: 小型视觉Transformer',
    'vit_base': 'ViT-Base: 基础视觉Transformer',
    'vit_large': 'ViT-Large: 大型视觉Transformer',

    # Swin Transformer series
    'swin_tiny': 'Swin-T: 轻量级层级视觉Transformer',
    'swin_small': 'Swin-S: 小型层级视觉Transformer',
    'swin_base': 'Swin-B: 基础层级视觉Transformer',
    'swin_large': 'Swin-L: 大型层级视觉Transformer',
    
    # ConvNeXt series
    'convnext_tiny': 'ConvNeXt-T: 轻量级ConvNeXt',
    'convnext_small': 'ConvNeXt-S: 小型ConvNeXt',
    'convnext_base': 'ConvNeXt-B: 基础ConvNeXt',
    'convnext_large': 'ConvNeXt-L: 大型ConvNeXt',
    'convnext_xlarge': 'ConvNeXt-XL: 超大型ConvNeXt',

}

# model type mapping
MODEL_TYPE_MAP = {
    'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                     'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'],
    'mobilenet': ['mobilenetv3_small', 'mobilenetv3_large'],
    'resnet': ['resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'],
    'densenet': ['densenet_121', 'densenet_169', 'densenet_201', 'densenet_161'],
    'vit': ['vit_tiny', 'vit_small', 'vit_base', 'vit_large'],
    'swin': ['swin_tiny', 'swin_small', 'swin_base', 'swin_large'],
    'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'],
}

# training configuration
CONFIG = {
    # model related configuration
    # list of models to train
    # example configuration:
    # 1. ResNet series: ['resnet_50']
    # 2. DenseNet series: ['densenet_121']
    # 3. Swin series: ['swin_tiny']
    # 4. EfficientNet series: ['efficientnet_bt']
    # 5. MobileNet series: ['mobilenetv3_small']
    # 6. ViT series: ['vit_tiny']
    # 7. ConvNeXt series: ['convnext_tiny']    
    # 8. multi-model training: ['resnet_50', 'vit_tiny', 'mobilenetv3_small']
    
    'model': {
        # use dictionary format to configure each model and its pre-training settings
        'models': {
            'convnext_base': {
                'pretrained': True,  # only specify whether to use pre-training
            },
        },
        'device': 'cuda:0'
    },
    # data related configuration
    'data': {
        'train_dir': 'data/train',
        'val_dir': 'data/val', 
        'label_maps_dir': 'data/label_maps',
        'batch_size': 256,  # reduce from 64 to 32  
        'num_workers': 8,
    },
    # training related configuration
    'train': {
        'epochs': 100,  # reduce epoch number, avoid overfitting
        'save_dir': 'results',
        'label_smoothing': 0.05,  # reduce label smoothing, adapt to small dataset
        'save_structure': {
            'base': '{save_dir}/{model_id}'
        },
    },
    
    # random seed configuration
    'random_seed': {
        'seed': 42,                 # fixed random seed
        'deterministic': True,      # ensure CUDA determinism
        'benchmark': False,         # close CUDA benchmark mode
        'worker_init_seed': True    # whether to set seed for dataloader workers
    },
    
    # performance optimization related configuration
    'performance': {
        'amp': True,                # enable automatic mixed precision to improve training speed
        'gradient_clip': 0.5,       # appropriately loosen gradient clipping threshold
        'pin_memory': True,         # enable memory pinning to accelerate data loading
        'persistent_workers': True,  # keep worker processes to improve data loading efficiency
        'cudnn_benchmark': True,    # enable cudnn benchmark optimization for convolution operations
        'prefetch_factor': 2,       # prefetch factor, improve data loading efficiency
        'non_blocking': True        # enable non-blocking data transfer
    },
    
    'memory_management': {
        'max_cache_size': 1000,  # maximum number of cached images
        'gc_frequency': 10,      # frequency of garbage collection per epoch
        'clear_cache': False      # whether to clear cache after each epoch
    },
    
    # cross-validation configuration
    'cross_validation': {
        'enabled': False,              # whether to enable cross-validation
        'num_folds': 5,                # number of cross-validation folds
        'group': 'complete',         # data group name
        'save_all_models': True,      # whether to save all models
        'aggregate_results': True      # whether to aggregate results
    },
    
    # early stopping configuration
    'early_stopping': {
        'enabled': False,             # whether to enable early stopping
        'monitor': 'species_top1',   # monitored metric (default: species top1 accuracy)
        'patience': 10,              # patience for early stopping
        'min_delta': 0.001,          # minimum change threshold to be considered an improvement
        'mode': 'max',               # 'max' for max metric, 'min' for min metric
        'verbose': True,             # whether to print early stopping information
        'restore_best_weights': True # whether to restore best weights
    }
}
        
        
def get_model_type(model_name: str) -> str:
    """get model type by model name"""
    for model_type, models in MODEL_TYPE_MAP.items():
        if model_name in models:
            return model_type
    raise ValueError(f"unknown model name: {model_name}")

def validate_model_config():
    """validate model configuration"""
    for model_name in CONFIG['model']['models']:
        try:
            if model_name not in SUPPORTED_MODELS:
                raise ValueError(f"unsupported model: {model_name}")
        except Exception as e:
            logging.error(f"model configuration error: {str(e)}")
            raise

def train_epoch(model_manager: ModelManager, 
                model_id: str,
                train_loader: torch.utils.data.DataLoader,
                epoch: int) -> Dict:
    """train one epoch"""
    model_manager.models[model_id].train()
    metric_tracker = model_manager.metric_trackers[model_id]
    metric_tracker.reset()
    logger = logging.getLogger(f'model.{model_id}')
    
    # get device from model_manager
    device = model_manager.device
    
    # initialize gradient scaler
    scaler = GradScaler()
    
    progress_desc = f'Epoch {epoch+1}/{CONFIG["train"]["epochs"]} [{epoch/CONFIG["train"]["epochs"]*100:.1f}%]'
    pbar = tqdm(total=len(train_loader), 
                desc=f'{progress_desc} [Train]',
                ncols=120)
    
    # create new CUDA stream for overlapping computation and data transfer
    data_stream = torch.cuda.Stream()
    
    # prefetch first batch
    batch_iter = iter(train_loader)
    batch_data = next(batch_iter)
    
    with torch.cuda.stream(data_stream):
        images, targets, _ = batch_data
        
        # use non-blocking transfer and immediately clear CPU memory
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
    
    for batch_idx in range(len(train_loader)):
        # wait for data preparation in main stream
        torch.cuda.current_stream().wait_stream(data_stream)
        
        # prefetch next batch (if any)
        if batch_idx + 1 < len(train_loader):
            next_batch_data = next(batch_iter)
            with torch.cuda.stream(data_stream):
                next_images, next_targets, _ = next_batch_data
                next_images = next_images.to(device, non_blocking=True)
                next_targets = {k: v.to(device, non_blocking=True) 
                              for k, v in next_targets.items()}
        
        # use automatic mixed precision training
        with autocast():
            outputs = model_manager.models[model_id](images)
            loss = model_manager._compute_loss(outputs, targets)
        
        # optimizer step
        model_manager.optimizers[model_id].zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(model_manager.optimizers[model_id])
        
        if CONFIG['performance']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model_manager.models[model_id].parameters(), 
                CONFIG['performance']['gradient_clip']
            )
        
        scaler.step(model_manager.optimizers[model_id])
        scaler.update()
        
        # update metrics and immediately clear
        metric_tracker.update('train', loss.item(), outputs, targets, images.size(0), 
                            model_manager.optimizers[model_id].param_groups[0]['lr'])
        
        # update progress bar
        metrics = metric_tracker.compute_epoch_metrics('train')
        pbar.set_postfix({
            'Loss': f'{metrics["loss"]:.4f}',
            'Acc': f'{metrics["species_top1"]:.2f}%',
            'LR': f'{model_manager.optimizers[model_id].param_groups[0]["lr"]:.6f}'
        }, refresh=True)
        pbar.update()
        
        # prepare next batch data
        images, targets = next_images, next_targets
    
    pbar.close()
    
    final_metrics = metric_tracker.compute_epoch_metrics('train')
    metrics_table = model_manager._format_metrics_table(final_metrics, 'Train')
    logger.info(f'\n{metrics_table}')
    
    return final_metrics

def validate_epoch(model_manager: ModelManager,
                  model_id: str,
                  val_loader: torch.utils.data.DataLoader,
                  epoch: int) -> Dict:
    """validate one epoch"""
    model_manager.models[model_id].eval()
    metric_tracker = model_manager.metric_trackers[model_id]
    metric_tracker.reset()
    logger = logging.getLogger(f'model.{model_id}')
    
    # create progress bar, add total progress display (epoch starts from 1)
    total_epochs = CONFIG["train"]["epochs"]
    progress_desc = f'Epoch {epoch+1}/{total_epochs} [{epoch/total_epochs*100:.1f}%]'
    pbar = tqdm(total=len(val_loader), 
                desc=f'{progress_desc} [Val]',
                ncols=120)
    
    with torch.no_grad():
        for batch_data in val_loader:
            # unpack data, ignore img_path
            images, targets, _ = batch_data
            
            loss, outputs, metrics = model_manager.validate_batch(model_id, (images, targets))
            
            # update progress bar, add more information
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Acc': f'{metrics["species_top1"]:.2f}%'
            })
            pbar.update()
    
    pbar.close()
    
    # calculate final metrics and only print in table format
    final_metrics = metric_tracker.compute_epoch_metrics('val')
    metrics_table = model_manager._format_metrics_table(final_metrics, 'Validation')
    logger.info(f'\n{metrics_table}')
    
    return final_metrics

def save_training_history(metrics_history: Dict, save_dir: Path, model_id: str):
    """save training history"""
    # modify save path to correct metrics directory
    history_file = save_dir / model_id / 'metrics' / 'training_history.json'
    
    # create a new dictionary to store converted data
    serializable_history = {}
    
    # convert training and validation metrics
    for phase in ['train', 'val']:
        if phase in metrics_history:
            serializable_history[phase] = {}
            for metric_name, values in metrics_history[phase].items():
                # convert each value in list to Python native type
                converted_values = []
                for value in values:
                    if isinstance(value, torch.Tensor):
                        # if tensor, convert to Python number
                        converted_value = value.item() if value.numel() == 1 else value.tolist()
                    elif isinstance(value, (list, tuple)):
                        # if list or tuple, recursively convert tensor in it
                        converted_value = [v.item() if isinstance(v, torch.Tensor) else v for v in value]
                    else:
                        # if Python native type, use directly
                        converted_value = value
                    converted_values.append(converted_value)
                serializable_history[phase][metric_name] = converted_values
    
    # save converted data
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=4)
    
    logging.info(f"training history saved to: {history_file}")

def save_metrics_to_csv(metrics_history: Dict, save_dir: Path, model_id: str):
    """save training and validation metrics to CSV file"""
    # modify save path to correct metrics directory
    metrics_dir = save_dir / model_id / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # prepare training and validation data
    train_data = []
    val_data = []
    
    # get all metric names
    metric_names = metrics_history['train'].keys()
    
    # collect data for each epoch
    num_epochs = len(next(iter(metrics_history['train'].values())))
    for epoch in range(num_epochs):
        # training data
        train_row = {'epoch': epoch + 1}
        for metric in metric_names:
            if metric in metrics_history['train']:
                value = metrics_history['train'][metric][epoch]
                # if list, take first element
                if isinstance(value, list):
                    value = value[0]
                train_row[metric] = value
        train_data.append(train_row)
        
        # validation data
        val_row = {'epoch': epoch + 1}
        for metric in metric_names:
            if metric in metrics_history['val']:
                value = metrics_history['val'][metric][epoch]
                # if list, take first element
                if isinstance(value, list):
                    value = value[0]
                val_row[metric] = value
        val_data.append(val_row)
    
    # create DataFrame and save to correct path
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_df.to_csv(metrics_dir / 'train_metrics.csv', index=False)
    val_df.to_csv(metrics_dir / 'val_metrics.csv', index=False)
    
    logging.info(f"metrics saved to CSV file:")
    logging.info(f"  - training metrics: {metrics_dir / 'train_metrics.csv'}")
    logging.info(f"  - validation metrics: {metrics_dir / 'val_metrics.csv'}")

def format_time(seconds):
    """convert seconds to readable time format"""
    return str(timedelta(seconds=int(seconds)))

def worker_init_fn(worker_id: int):
    """
    set different random seeds for each worker of DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_random_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    """
    set random seed to ensure reproducibility
    Args:
        seed: random seed
        deterministic: whether to ensure CUDA determinism
        benchmark: whether to enable CUDA benchmark mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        cudnn.deterministic = True  # ensure deterministic convolution algorithm
        cudnn.benchmark = benchmark # whether to automatically find the most efficient algorithm for current configuration
        
    logging.info(f"random seed set: {seed}")
    logging.info(f"CUDA deterministic mode: {deterministic}")
    logging.info(f"CUDA benchmark mode: {benchmark}")

def train(start_time: float):
    """main training function"""
    # set random seed
    seed_config = CONFIG['random_seed']
    set_random_seed(
        seed=seed_config['seed'],
        deterministic=seed_config['deterministic'],
        benchmark=seed_config['benchmark']
    )
    
    device = torch.device(CONFIG['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    # add worker_init_fn when getting data loader
    worker_init = worker_init_fn if CONFIG['random_seed']['worker_init_seed'] else None
    
    train_loader, num_classes, class_names = get_dataloader(
        data_dir=CONFIG['data']['train_dir'],
        batch_size=CONFIG['data']['batch_size'],
        mode='train',
        num_workers=CONFIG['data']['num_workers'],
        val_data_dir=CONFIG['data']['val_dir'],
        worker_init_fn=worker_init,  # add worker initialization function
        max_cache_size=CONFIG['memory_management']['max_cache_size']  # pass cache limit
    )
    
    val_loader, _, _ = get_dataloader(
        data_dir=CONFIG['data']['val_dir'],
        batch_size=CONFIG['data']['batch_size'],
        mode='val',
        num_workers=CONFIG['data']['num_workers'],
        worker_init_fn=worker_init,  # add worker initialization function
        max_cache_size=CONFIG['memory_management']['max_cache_size']  # pass cache limit
    )
    
    # modify model configuration list creation
    model_configs = []
    for model_name, model_config in CONFIG['model']['models'].items():
        model_type = get_model_type(model_name)
        model_configs.append({
            'type': model_type,
            'name': model_name,
            'pretrained': model_config.get('pretrained', False),
            'pretrained_path': model_config.get('pretrained_path', None)
        })
    
    # initialize model manager
    model_manager = ModelManager(
        model_configs=model_configs,
        device=device,
        num_classes=num_classes,
        save_dir=str(CONFIG['train']['save_dir']),
        create_dirs=True,
        train_config=CONFIG  # pass complete training configuration
    )
    
    # get dataset name
    dataset_name = Path(CONFIG['data']['train_dir']).parts[-2]  # extract dataset name from path
    
    # create independent logger for each model
    model_loggers = {}
    for model_id in model_manager.get_model_ids():
        # create model-specific logger
        logger = logging.getLogger(f'model.{model_id}')
        logger.setLevel(logging.INFO)
        
        # ensure logger has no duplicate handlers
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # create model-specific log file
        model_log_dir = Path(CONFIG['train']['save_dir']) / model_id / 'logs'
        model_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = model_log_dir / f'{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # add file handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        model_loggers[model_id] = logger
        
        # record initial information
        logger.info(f"\n{'='*50}")
        logger.info("training configuration:")
        logger.info(f"{'='*50}")
        
        # record configuration information
        logger.info("\ndata configuration:")
        for k, v in CONFIG['data'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info("\nmodel configuration:")
        for k, v in CONFIG['model'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info("\ntraining configuration:")
        for k, v in CONFIG['train'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info("\nrandom seed configuration:")
        for k, v in CONFIG['random_seed'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info("\nperformance optimization configuration:")
        for k, v in CONFIG['performance'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info("\nmemory management configuration:")
        for k, v in CONFIG['memory_management'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info(f"\n{'='*50}")
        
        # continue original logging
        logger.info(f"start training model: {model_id}")
        logger.info(f"device: {device}")
        logger.info(f"training set size: {len(train_loader.dataset)}")
        logger.info(f"validation set size: {len(val_loader.dataset)}")
    
    # create metrics history dictionary for each model
    metrics_history = {
        model_id: create_metrics_dict()
        for model_id in model_manager.get_model_ids()
    }
    
    # initialize early stopping counters and best performance
    early_stopping_counters = {}
    early_stopping_best_scores = {}
    
    # initialize early stopping variables for each model
    if CONFIG['early_stopping']['enabled']:
        for model_id in model_manager.get_model_ids():
            monitor = CONFIG['early_stopping']['monitor']
            mode = CONFIG['early_stopping']['mode']
            early_stopping_best_scores[model_id] = float('-inf') if mode == 'max' else float('inf')
            early_stopping_counters[model_id] = 0
    
    # start training
    for epoch in range(CONFIG['train']['epochs']):
        epoch_start_time = time.time()
        
        # check if all models need early stopping
        all_models_early_stopped = True if CONFIG['early_stopping']['enabled'] else False
        
        # train each model
        for model_id in model_manager.get_model_ids():
            logger = model_loggers[model_id]
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{CONFIG['train']['epochs']}")
            logger.info(f"{'='*50}")
            
            # check early stopping status
            if CONFIG['early_stopping']['enabled'] and early_stopping_counters[model_id] >= CONFIG['early_stopping']['patience']:
                logger.info(f"early stopping triggered! model {model_id} has not improved in {CONFIG['early_stopping']['patience']} epochs")
                continue
            
            # mark at least one model has not triggered early stopping
            all_models_early_stopped = False
            
            # training phase
            train_metrics = train_epoch(model_manager, model_id, train_loader, epoch)
            
            # validation phase
            val_metrics = validate_epoch(model_manager, model_id, val_loader, epoch)
            
            # update learning rate
            model_manager.update_lr(model_id)
            current_lr = model_manager.optimizers[model_id].param_groups[0]['lr']
            logger.info(f"\ncurrent learning rate: {current_lr:.6f}")
            
            # update metrics history
            update_metrics_dict(metrics_history[model_id], 'train', train_metrics)
            update_metrics_dict(metrics_history[model_id], 'val', val_metrics)
            
            # save checkpoint and metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            model_manager.save_checkpoint(model_id, epoch, metrics)
            
            # early stopping check
            if CONFIG['early_stopping']['enabled']:
                monitor = CONFIG['early_stopping']['monitor']
                monitor_value = val_metrics.get(monitor, 0)
                mode = CONFIG['early_stopping']['mode']
                min_delta = CONFIG['early_stopping']['min_delta']
                
                if mode == 'max':
                    improved = monitor_value - min_delta > early_stopping_best_scores[model_id]
                else:
                    improved = monitor_value + min_delta < early_stopping_best_scores[model_id]
                
                if improved:
                    if CONFIG['early_stopping']['verbose']:
                        logger.info(f"early stopping monitoring: {monitor} improved from {early_stopping_best_scores[model_id]:.4f} to {monitor_value:.4f}")
                    early_stopping_best_scores[model_id] = monitor_value
                    early_stopping_counters[model_id] = 0
                else:
                    early_stopping_counters[model_id] += 1
                    if CONFIG['early_stopping']['verbose']:
                        logger.info(f"early stopping monitoring: {monitor} not improved, counter: {early_stopping_counters[model_id]}/{CONFIG['early_stopping']['patience']}")
            
            # record training time for this epoch
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch+1} 用时: {format_time(epoch_time)}")
            
            # save training history
            save_training_history(metrics_history[model_id], 
                                Path(CONFIG['train']['save_dir']), 
                                model_id)
            save_metrics_to_csv(metrics_history[model_id], 
                              Path(CONFIG['train']['save_dir']), 
                              model_id)
        
        # if all models have triggered early stopping, end training early
        if all_models_early_stopped:
            logging.info("all models have triggered early stopping, end training early")
            break
    
    # training ends, record summary information
    total_time = time.time() - start_time
    for model_id, logger in model_loggers.items():
        logger.info(f"\n{'='*50}")
        logger.info("training completed!")
        logger.info(f"training duration: {format_time(total_time)}")
        logger.info(f"average time per epoch: {format_time(total_time/CONFIG['train']['epochs'])}")
        logger.info(f"best validation metrics:")
        for metric_name, value in model_manager.best_metrics[model_id].items():
            logger.info(f"  {metric_name}: {value:.4f}")
        logger.info(f"{'='*50}\n")

def check_training_environment():
    """check training environment"""
    # check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA environment is required")
        

def cross_validation_train(start_time: float):
    """main function for five-fold cross-validation training"""
    # set random seed
    seed_config = CONFIG['random_seed']
    set_random_seed(
        seed=seed_config['seed'],
        deterministic=seed_config['deterministic'],
        benchmark=seed_config['benchmark']
    )
    
    # record all fold metrics
    cv_metrics = {}
    
    # get data group name and fold number
    group_name = CONFIG['cross_validation']['group']
    num_folds = CONFIG['cross_validation']['num_folds']
    
    logging.info(f"start {num_folds} fold cross-validation training, data group: {group_name}")
    
    # save original configuration
    original_train_dir = CONFIG['data']['train_dir']
    original_val_dir = CONFIG['data']['val_dir']
    original_save_dir = CONFIG['train']['save_dir']
    
    # iterate through each fold - modify to start from 1
    for fold in range(1, num_folds + 1):
        fold_start_time = time.time()
        logging.info(f"\n{'='*80}")
        logging.info(f"start {fold}/{num_folds} fold training")
        logging.info(f"{'='*80}")
        
        # restore original path building method
        CONFIG['data']['train_dir'] = f'data/combination/{group_name}_fold{fold}/train'
        CONFIG['data']['val_dir'] = f'data/combination/{group_name}_fold{fold}/val'
        CONFIG['data']['label_maps_dir'] = f'data/combination/{group_name}_fold{fold}/label_maps'
        
        # restore original save path building method
        CONFIG['train']['save_dir'] = f'results/combination/{group_name}_fold{fold}'
        
        # train current fold
        fold_metrics = fold_train(fold, start_time)
        
        # record current fold metrics - modify key name
        cv_metrics[f'fold_{fold}'] = fold_metrics
        
        # calculate current fold training time
        fold_time = time.time() - fold_start_time
        logging.info(f"fold {fold}/{num_folds} training completed, time: {format_time(fold_time)}")
    
    # restore original configuration
    CONFIG['data']['train_dir'] = original_train_dir
    CONFIG['data']['val_dir'] = original_val_dir
    CONFIG['train']['save_dir'] = original_save_dir
    
    # aggregate and print all fold metrics
    if CONFIG['cross_validation']['aggregate_results']:
        aggregate_cv_results(cv_metrics, group_name)
    
    # training ends, record total time
    total_time = time.time() - start_time
    logging.info(f"\n{'='*80}")
    logging.info(f"five-fold cross-validation training completed! total time: {format_time(total_time)}")
    logging.info(f"{'='*80}")

def fold_train(fold: int, start_time: float) -> Dict:
    """
    train single fold model
    Args:
        fold: current fold number
        start_time: training start time
    Returns:
        fold_metrics: current fold training metrics
    """
    device = torch.device(CONFIG['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    # get data loader
    worker_init = worker_init_fn if CONFIG['random_seed']['worker_init_seed'] else None
    
    train_loader, num_classes, class_names = get_dataloader(
        data_dir=CONFIG['data']['train_dir'],
        batch_size=CONFIG['data']['batch_size'],
        mode='train',
        num_workers=CONFIG['data']['num_workers'],
        val_data_dir=CONFIG['data']['val_dir'],
        worker_init_fn=worker_init,
        max_cache_size=CONFIG['memory_management']['max_cache_size']
    )
    
    val_loader, _, _ = get_dataloader(
        data_dir=CONFIG['data']['val_dir'],
        batch_size=CONFIG['data']['batch_size'],
        mode='val',
        num_workers=CONFIG['data']['num_workers'],
        worker_init_fn=worker_init,
        max_cache_size=CONFIG['memory_management']['max_cache_size']
    )
    
    # prepare model configuration
    model_configs = []
    for model_name, model_config in CONFIG['model']['models'].items():
        model_type = get_model_type(model_name)
        model_configs.append({
            'type': model_type,
            'name': model_name,
            'pretrained': model_config.get('pretrained', False),
            'pretrained_path': model_config.get('pretrained_path', None)
        })
    
    # initialize model manager
    model_manager = ModelManager(
        model_configs=model_configs,
        device=device,
        num_classes=num_classes,
        save_dir=str(CONFIG['train']['save_dir']),
        create_dirs=True,
        train_config=CONFIG
    )
    
    # get dataset name
    dataset_name = Path(CONFIG['data']['train_dir']).parts[-2]
    
    # create independent logger for each model
    model_loggers = {}
    for model_id in model_manager.get_model_ids():
        # create model-specific logger
        logger = logging.getLogger(f'model.{model_id}')
        logger.setLevel(logging.INFO)
        
        # ensure logger has no duplicate handlers
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # create model-specific log file - modify to use fold value directly
        model_log_dir = Path(CONFIG['train']['save_dir']) / model_id / 'logs'
        model_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = model_log_dir / f'{dataset_name}_fold{fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # add file handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        model_loggers[model_id] = logger
        
        # record initial information - modify to use fold value directly
        logger.info(f"\n{'='*50}")
        logger.info(f"fold {fold} training configuration:")
        logger.info(f"{'='*50}")
        
        # record configuration information
        logger.info("\ndata configuration:")
        for k, v in CONFIG['data'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info("\n模型配置:")
        for k, v in CONFIG['model'].items():
            logger.info(f"  {k}: {v}")
            
        logger.info(f"\n{'='*50}")
        
        # continue original logging
        logger.info(f"start training model: {model_id}")
        logger.info(f"device: {device}")
        logger.info(f"training set size: {len(train_loader.dataset)}")
        logger.info(f"validation set size: {len(val_loader.dataset)}")
    
    # create metrics history dictionary for each model
    metrics_history = {
        model_id: create_metrics_dict()
        for model_id in model_manager.get_model_ids()
    }
    
    # initialize early stopping counters and best performance
    early_stopping_counters = {}
    early_stopping_best_scores = {}
    
    # initialize early stopping variables for each model
    if CONFIG['early_stopping']['enabled']:
        for model_id in model_manager.get_model_ids():
            monitor = CONFIG['early_stopping']['monitor']
            mode = CONFIG['early_stopping']['mode']
            early_stopping_best_scores[model_id] = float('-inf') if mode == 'max' else float('inf')
            early_stopping_counters[model_id] = 0
    
    # start training
    for epoch in range(CONFIG['train']['epochs']):
        epoch_start_time = time.time()
        
        # check if all models need early stopping
        all_models_early_stopped = True if CONFIG['early_stopping']['enabled'] else False
        
        # train each model
        for model_id in model_manager.get_model_ids():
            logger = model_loggers[model_id]
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{CONFIG['train']['epochs']}")
            logger.info(f"{'='*50}")
            
            # check early stopping status
            if CONFIG['early_stopping']['enabled'] and early_stopping_counters[model_id] >= CONFIG['early_stopping']['patience']:
                logger.info(f"early stopping triggered! model {model_id} has not improved in {CONFIG['early_stopping']['patience']} epochs")
                continue
            
            # mark at least one model has not triggered early stopping
            all_models_early_stopped = False
            
            # training phase
            train_metrics = train_epoch(model_manager, model_id, train_loader, epoch)
            
            # validation phase
            val_metrics = validate_epoch(model_manager, model_id, val_loader, epoch)
            
            # update learning rate
            model_manager.update_lr(model_id)
            current_lr = model_manager.optimizers[model_id].param_groups[0]['lr']
            logger.info(f"\ncurrent learning rate: {current_lr:.6f}")
            
            # update metrics history
            update_metrics_dict(metrics_history[model_id], 'train', train_metrics)
            update_metrics_dict(metrics_history[model_id], 'val', val_metrics)
            
            # save checkpoint and metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            model_manager.save_checkpoint(model_id, epoch, metrics)
            
            # early stopping check
            if CONFIG['early_stopping']['enabled']:
                monitor = CONFIG['early_stopping']['monitor']
                monitor_value = val_metrics.get(monitor, 0)
                mode = CONFIG['early_stopping']['mode']
                min_delta = CONFIG['early_stopping']['min_delta']
                
                if mode == 'max':
                    improved = monitor_value - min_delta > early_stopping_best_scores[model_id]
                else:
                    improved = monitor_value + min_delta < early_stopping_best_scores[model_id]
                
                if improved:
                    if CONFIG['early_stopping']['verbose']:
                        logger.info(f"early stopping monitoring: {monitor} improved from {early_stopping_best_scores[model_id]:.4f} to {monitor_value:.4f}")
                    early_stopping_best_scores[model_id] = monitor_value
                    early_stopping_counters[model_id] = 0
                else:
                    early_stopping_counters[model_id] += 1
                    if CONFIG['early_stopping']['verbose']:
                        logger.info(f"early stopping monitoring: {monitor} not improved, counter: {early_stopping_counters[model_id]}/{CONFIG['early_stopping']['patience']}")
            
            # record training time for this epoch
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch+1} time: {format_time(epoch_time)}")
            
            # save training history
            save_training_history(metrics_history[model_id], 
                                Path(CONFIG['train']['save_dir']), 
                                model_id)
            save_metrics_to_csv(metrics_history[model_id], 
                              Path(CONFIG['train']['save_dir']), 
                              model_id)
        
        # if all models have triggered early stopping, end training early
        if all_models_early_stopped:
            logging.info("all models have triggered early stopping, end training early")
            break
    
    # current fold training ends, modify this part - modify to use fold value directly
    fold_best_metrics = {}
    for model_id, logger in model_loggers.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"fold {fold} training completed!")
        logger.info(f"best validation metrics:")
        
        # record best metrics in model_manager
        fold_best_metrics[model_id] = {}
        for metric_name, value in model_manager.best_metrics[model_id].items():
            logger.info(f"  {metric_name}: {value:.4f}")
            fold_best_metrics[model_id][metric_name] = value
            
        # get more comprehensive metrics from metrics_history
        if CONFIG['cross_validation']['aggregate_results']:
            # import compute_all_metrics_for_cv function
            complete_metrics = compute_all_metrics_for_cv(metrics_history[model_id], 'val')
            # merge to best_metrics
            for metric_name, value in complete_metrics.items():
                if metric_name not in fold_best_metrics[model_id]:
                    fold_best_metrics[model_id][metric_name] = value
                    
        logger.info(f"{'='*50}\n")
    
    return fold_best_metrics

def aggregate_cv_results(cv_metrics: Dict, group_name: str):
    """
    aggregate cross-validation results
    Args:
        cv_metrics: all fold metrics
        group_name: data group name
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"cross-validation results summary (data group: {group_name})")
    logging.info(f"{'='*80}")
    
    # restore original result directory path building method
    result_dir = Path(f'results/combination/{group_name}_cv_summary')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # extend metrics list to aggregate
    metrics_to_aggregate = ['species_top1', 'species_top3', 'species_top5', 
                           'genus_top1', 'genus_top3', 'subfamily_top1', 
                           'species_f1', 'genus_f1', 'subfamily_f1',
                           'loss']
    
    # get all model IDs
    model_ids = set()
    for fold_metrics in cv_metrics.values():
        model_ids.update(fold_metrics.keys())
    
    # get all fold numbers
    fold_keys = sorted(cv_metrics.keys())
    num_folds = len(fold_keys)
    
    # aggregate results for each model
    for model_id in model_ids:
        logging.info(f"\nmodel: {model_id}")
        logging.info(f"{'-'*80}")
        
        # prepare data for CSV
        csv_data = []
        
        # print header - use actual fold numbers
        header = f"{'指标名称':<20} | " + " | ".join([f"{'折 '+str(int(k.split('_')[1])):^8}" for k in fold_keys]) + f" | {'平均值':^8} | {'标准差':^8}"
        logging.info(header)
        logging.info(f"{'-'*80}")
        
        for metric in metrics_to_aggregate:
            # collect all fold metrics values
            values = []
            for fold_key in fold_keys:
                if fold_key in cv_metrics and model_id in cv_metrics[fold_key]:
                    values.append(cv_metrics[fold_key][model_id].get(metric, 0))
            
            # calculate mean and standard deviation
            if values:
                mean_value = sum(values) / len(values)
                std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
                
                # format print, change to keep 2 decimal places
                row = f"{metric:<20} | " + " | ".join([f"{v:^8.2f}" for v in values]) + f" | {mean_value:^8.2f} | {std_value:^8.2f}"
                logging.info(row)
                
                # prepare this row data for CSV, also keep 2 decimal places
                row_data = {
                    'metric': metric,
                    'mean': round(mean_value, 2),
                    'std': round(std_value, 2)
                }
                for i, fold_key in enumerate(fold_keys):
                    fold_num = int(fold_key.split('_')[1])
                    row_data[f'fold_{fold_num}'] = round(values[i], 2)
                csv_data.append(row_data)
        
        # save as CSV
        metrics_df = pd.DataFrame(csv_data)
        csv_path = result_dir / f'{model_id}_cv_summary.csv'
        metrics_df.to_csv(csv_path, index=False)
        logging.info(f"\ncross-validation results saved to: {csv_path}")
        
        # also save as JSON for later processing
        json_path = result_dir / f'{model_id}_cv_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(csv_data, f, indent=4)
        
        logging.info(f"cross-validation results saved to: {json_path}")
        logging.info(f"{'-'*80}")

def main():
    """main function"""
    # set CUDA performance optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = CONFIG['performance']['cudnn_benchmark']
        torch.backends.cuda.matmul.allow_tf32 = True  # enable TF32 acceleration
        torch.backends.cudnn.allow_tf32 = True        # enable cuDNN TF32 acceleration
    
    # create log directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # pass correct parameters to setup_logging function
    setup_logging(log_dir=str(log_dir))
    
    start_time = time.time()
    validate_model_config()
    
    # decide whether to perform cross-validation training based on configuration
    try:
        if CONFIG['cross_validation']['enabled']:
            cross_validation_train(start_time)
        else:
            train(start_time)  # original single fold training
    except Exception as e:
        # even if an error occurs, display the training time
        logging.error(f"error occurred during training: {str(e)}", exc_info=True)
        training_time = time.time() - start_time
        logging.info(f"\ntraining time: {format_time(training_time)}")
    else:
        logging.info("training completed successfully!")

if __name__ == '__main__':
    # add environment check
    check_training_environment()
    main() 

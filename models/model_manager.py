"""
ModelManager class
Main functions:
1. Manage the initialization, loading, and training of multiple deep learning models
2. Process the checkpoint saving and loading of models
3. Track the metrics during model training
4. Manage the directory structure related to models
"""

# standard library imports
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# third-party library imports
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# local module imports
from utils import data_loader
from utils.config_loader import ConfigLoader
from utils.metric import MetricTracker
from utils.taxonomy import TaxonomyMap

# import supported model architectures
from models import (
    get_resnet, resnet_configs,
    get_densenet, densenet_configs,
    get_efficientnet, efficientnet_configs,
    get_mobilenet_v3, mobilenet_configs,
    get_swin, swin_configs,
    get_vit, vit_configs,
    get_convnext, convnext_configs,
)

class LabelSmoothingLoss(nn.Module):
    """label smoothing loss function"""
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            return F.cross_entropy(pred, target)
            
        n_classes = pred.size(-1)
        # create smoothed labels
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))

class ModelManager:
    # supported model types and corresponding configurations
    MODEL_CONFIGS = {
        'resnet': resnet_configs,
        'densenet': densenet_configs,
        'swin': swin_configs,
        'vit': vit_configs,
        'mobilenet': mobilenet_configs,
        'efficientnet': efficientnet_configs,
        'convnext': convnext_configs,
    }
    
    # model retrieval function mapping
    MODEL_GETTERS = {
        'resnet': get_resnet,
        'densenet': get_densenet,
        'swin': get_swin,
        'vit': get_vit,
        'mobilenet': get_mobilenet_v3,
        'efficientnet': get_efficientnet,
        'convnext': get_convnext,
    }

    def __init__(self, 
                 model_configs: List[Dict],
                 device: torch.device,
                 num_classes: Dict[str, int],
                 save_dir: str = 'results',
                 taxonomy_file: str = 'data/taxonomy.csv',
                 create_dirs: bool = True,
                 train_config: Dict = None,
                 verbose: bool = True):
        """
        initialize model manager
        
        Args:
            model_configs: model configuration list
            device: training device
            num_classes: classification number dictionary
            save_dir: save directory
            taxonomy_file: taxonomy mapping file path, default is 'data/taxonomy.csv'
            create_dirs: whether to create directory structure, default is True (training required)
            train_config: training related configuration, including batch_size, epochs, etc.
            verbose: whether to output detailed training configuration information
        """
        
        # set basic attributes
        self.device = device
        self.num_classes = num_classes
        self.base_dir = Path(save_dir)
        self.train_config = train_config or {}  # save training configuration
        self.verbose = verbose  # add parameter to control output
        
        # initialize storage dictionary
        self.models = {}          # store model instances
        self.optimizers = {}      # store optimizer instances
        self.schedulers = {}      # store learning rate scheduler instances
        self.metric_trackers = {} # store metric tracker instances
        
        # cache directory structure
        self._dir_cache = {}
        
        # initialize taxonomy mapping
        try:
            self.taxonomy_map = TaxonomyMap(taxonomy_file)
            logging.info(f"successfully loaded taxonomy mapping file: {taxonomy_file}")
        except Exception as e:
            logging.error(f"failed to load taxonomy mapping file: {str(e)}")
            self.taxonomy_map = None
        
        # get label smoothing parameter from training configuration
        label_smoothing = train_config.get('train', {}).get('label_smoothing', 0.0)
        
        # initialize label smoothing loss function
        self.criterion = LabelSmoothingLoss(smoothing=label_smoothing).to(device)
        
        # load all models
        for config in model_configs:
            self._init_model(config)
        
        # initialize best metrics tracker
        self.best_metrics = {
            model_id: {'val_loss': float('inf'), 'species_top1': 0.0, 'epoch': 0}
            for model_id in self.models.keys()
        }
        
        # if directory structure needs to be created (usually only during training)
        if create_dirs:
            for model_name in self.models.keys():
                self._get_model_dir(model_name, create=True)

    def _get_model_dir(self, model_name: str, create: bool = False) -> Dict[str, Path]:
        """
        get all directory paths related to the model
        
        Args:
            model_name: model name
            create: whether to create directory, default is False
        
        Returns:
            dictionary containing all related directory paths
        """
        # if directory structure is cached, check if directory exists
        if model_name in self._dir_cache:
            dirs = self._dir_cache[model_name]
            if create and not dirs['base'].exists():
                self._create_model_dirs(model_name)
            return dirs
        
        # if not cached, create directory structure dictionary
        model_base = self.base_dir / model_name
        dirs = {
            'base': model_base,
            'checkpoints': model_base / 'checkpoints',
            'logs': model_base / 'logs',
            'metrics': model_base / 'metrics',
        }
        
        # cache directory structure
        self._dir_cache[model_name] = dirs
        
        # if directory structure needs to be created
        if create:
            self._create_model_dirs(model_name)
        
        return dirs

    def _create_model_dirs(self, model_name: str):
        """
        create directory structure for the model
        
        Args:
            model_name: model name
        """
        dirs = self._dir_cache.get(model_name)
        if not dirs:
            dirs = self._get_model_dir(model_name, create=False)
        
        # check if each directory exists, if not, create it
        for dir_name, path in dirs.items():
            try:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Created directory: {path}")
                else:
                    logging.debug(f"Directory already exists: {path}")
            except Exception as e:
                logging.error(f"failed to create directory {dir_name}: {path}, error: {str(e)}")
                raise

    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """get all available model types and names"""
        available_models = {}
        for model_type, configs in cls.MODEL_CONFIGS.items():
            available_models[model_type] = list(configs.keys())
        return available_models

    def _init_model(self, config: Dict):
        """
        initialize a single deep learning model and its related components (optimizer, learning rate scheduler, etc.)
        
        Args:
            config (Dict): model configuration dictionary, must contain the following keys:
                - type: model type (e.g. 'resnet', 'densenet', etc.)
                - name: model name (e.g. 'resnet_50', 'densenet_121', etc.)
                - pretrained: whether to use pretrained weights (optional, default is False)
                - pretrained_path: pretrained weights file path (optional)
        
        workflow:
        1. parse configuration parameters
        2. create model instance
        3. initialize optimizer
        4. initialize learning rate scheduler
        5. save all components to the corresponding storage dictionary
        6. record configuration information
        """
        # 1. extract necessary parameters from configuration
        model_type = config['type']  # model type (e.g. resnet, densenet, etc.)
        model_name = config['name']  # specific model variant name
        pretrained = config.get('pretrained', False)  # whether to use pretrained weights
        pretrained_path = config.get('pretrained_path', None)  # pretrained weights file path (optional)
        
        # temporarily adjust log level
        original_level = logging.getLogger().level
        if not self.verbose:
            logging.getLogger().setLevel(logging.WARNING)
        
        try:
            # 2. create configuration loader instance, pass verbose parameter
            config_loader = ConfigLoader(verbose=self.verbose)
            
            # 3. build model initialization parameters
            model_init_params = {
                'model_name': model_name,
                'num_classes': self.num_classes,
                'pretrained': pretrained,  # pass pretrained flag
                'pretrained_path': pretrained_path  # pass pretrained weights path (if any)
            }
            
            # 4. create model instance
            # use the model building function corresponding to the MODEL_GETTERS dictionary
            model = self.MODEL_GETTERS[model_type](**model_init_params)
            # move the model to the specified device (CPU/GPU)
            model = model.to(self.device)
            
            # 5. create optimizer
            # create the corresponding optimizer using the configuration loader based on the model type and name
            optimizer = config_loader.create_optimizer(
                model.parameters(),  # model parameters
                model_type,         # model type
                model_name          # model name
            )
            
            # 6. create learning rate scheduler
            # create the corresponding learning rate scheduler using the configuration loader based on the model type and name
            scheduler = config_loader.create_scheduler(
                optimizer,   # optimizer instance
                model_type,  # model type
                model_name   # model name
            )
            
            # 7. save all components to the corresponding storage dictionary
            self.models[model_name] = model  # save model instance
            self.optimizers[model_name] = optimizer  # save optimizer
            self.schedulers[model_name] = scheduler  # save learning rate scheduler
            self.metric_trackers[model_name] = MetricTracker()  # create and save metric tracker
            
            # 8. print and record model configuration information
            self._print_model_config(model_name, config_loader, model_type, model_name)
            
            # 9. record successful initialization log information
            logger = logging.getLogger(f'model.{model_name}')
            logger.info(f"successfully initialized model: {model_name}")
            if pretrained:
                if pretrained_path:
                    logger.info(f"using custom pretrained weights: {pretrained_path}")
                else:
                    logger.info("using default pretrained weights")
            
            # record initial learning rate
            logger.info(f"initial learning rate: {optimizer.param_groups[0]['lr']}")
            
        except Exception as e:
            # if any error occurs during initialization, record error information and raise exception
            logging.error(f"failed to initialize model {model_name}: {str(e)}")
            raise
        finally:
            # restore original log level
            logging.getLogger().setLevel(original_level)

    def _print_model_config(self, model_id: str, config_loader: ConfigLoader, 
                          model_type: str, config_name: str):
        """print and record model training configuration"""
        # if verbose is False, do not output configuration information
        if not self.verbose:
            return
        
        # get full configuration
        full_config = config_loader.load_model_config(model_type, config_name)
        
        # get model log directory
        model_dirs = self._get_model_dir(model_id, create=True)
        log_file = model_dirs['logs'] / 'model_config.log'
        
        # ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # create model-specific logger using model ID
        logger = logging.getLogger(f'model.{model_id}')
        logger.setLevel(logging.INFO)
        
        # if logger has handlers, remove them first
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # add file handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # record training basic parameters
        logger.info(f"\n{'='*50}")
        logger.info("training basic parameters:")
        logger.info(f"{'='*50}")
        
        # use self.train_config instead of CONFIG
        data_config = self.train_config.get('data', {})
        logger.info(f"batch size (Batch Size): {data_config.get('batch_size', 'Not specified')}")
        logger.info(f"training epochs (Epochs): {self.train_config.get('train', {}).get('epochs', 'Not specified')}")
        logger.info(f"data loading threads (Num Workers): {data_config.get('num_workers', 'Not specified')}")
        logger.info(f"device (Device): {self.device}")
        logger.info(f"label smoothing coefficient (Label Smoothing): {self.criterion.smoothing}")
        logger.info(f"{'='*50}\n")
        
        # record model configuration information
        logger.info(f"training configuration for model {model_id}:")
        logger.info(f"{'='*50}")
        
        # record model architecture information
        logger.info("\nmodel architecture:")
        logger.info(f"  type: {model_type}")
        logger.info(f"  name: {config_name}")
        
        # if in inference mode, do not print optimizer and learning rate scheduler configuration
        if self.train_config.get('train', {}).get('epochs') is not None:
            # record optimizer configuration
            if 'optimizer' in full_config:
                opt_config = full_config['optimizer']
                logger.info("\noptimizer configuration:")
                if 'type' in opt_config:
                    logger.info(f"  type: {opt_config['type']}")
                if 'lr' in opt_config:
                    logger.info(f"  learning rate: {opt_config['lr']}")
                if 'weight_decay' in opt_config:
                    logger.info(f"  weight decay: {opt_config['weight_decay']}")
                if 'momentum' in opt_config:
                    logger.info(f"  momentum: {opt_config['momentum']}")
                if 'betas' in opt_config:
                    logger.info(f"  Adam betas: {opt_config['betas']}")
                if 'eps' in opt_config:
                    logger.info(f"  Adam epsilon: {opt_config['eps']}")
            
            # record learning rate scheduler configuration
            if 'scheduler' in full_config:
                sched_config = full_config['scheduler']
                logger.info("\nlearning rate scheduler configuration:")
                if 'type' in sched_config:
                    logger.info(f"  type: {sched_config['type']}")
                if 'T_max' in sched_config:
                    logger.info(f"  maximum cycle: {sched_config['T_max']}")
                if 'eta_min' in sched_config:
                    logger.info(f"  minimum learning rate: {sched_config['eta_min']}")
                if 'warmup_epochs' in sched_config:
                    logger.info(f"  warmup epochs: {sched_config['warmup_epochs']}")
                if 'warmup_start_lr' in sched_config:
                    logger.info(f"  warmup start learning rate: {sched_config['warmup_start_lr']}")
            
            # record data augmentation configuration
            if 'augmentation' in full_config:
                aug_config = full_config['augmentation']
                logger.info("\ndata augmentation configuration:")
                for aug_name, aug_params in aug_config.items():
                    logger.info(f"  {aug_name}: {aug_params}")
        
        logger.info(f"\n{'='*50}\n")
        
        # remove file handler
        logger.removeHandler(fh)

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        calculate multi-level classification loss
        Args:
            outputs: model output, containing predictions for each level
                    {'subfamily': tensor
            targets: true labels, containing labels for each level
                    {'subfamily': tensor, 'genus': tensor, 'species': tensor}
        Returns:
            total loss value
        """
        # use initialized label smoothing loss function
        subfamily_loss = self.criterion(outputs['subfamily'], targets['subfamily'])
        genus_loss = self.criterion(outputs['genus'], targets['genus'])
        species_loss = self.criterion(outputs['species'], targets['species'])
        
        weights = {
            'subfamily': 1.0,
            'genus': 1.0,
            'species': 1.0
        }
        
        # calculate total loss
        total_loss = (weights['subfamily'] * subfamily_loss + 
                     weights['genus'] * genus_loss + 
                     weights['species'] * species_loss)
        
        return total_loss

    def _format_metrics_table(self, metrics: Dict[str, float], phase: str) -> str:
        """format metrics into a table"""
        # table border characters
        h_line = '-' * 82
        v_line = '|'
        
        # prepare table header
        header = (f"{v_line}{'Level/Metric':^12}{v_line}{'Top-1':^10}{v_line}{'Top-3':^10}"
                 f"{v_line}{'Top-5':^10}{v_line}{'Precision':^10}{v_line}{'Recall':^10}"
                 f"{v_line}{'F1-Score':^10}{v_line}")
        
        # prepare table content
        rows = []
        
        # add metrics by level
        for level in ['Subfamily', 'Genus', 'Species']:
            level_key = level.lower()
            row = (f"{v_line}{level:^12}"
                   f"{v_line}{metrics.get(f'{level_key}_top1', 0.0):^10.2f}"
                   f"{v_line}{metrics.get(f'{level_key}_top3', 0.0):^10.2f}"
                   f"{v_line}{metrics.get(f'{level_key}_top5', 0.0):^10.2f}"
                   f"{v_line}{metrics.get(f'{level_key}_precision', 0.0):^10.2f}"
                   f"{v_line}{metrics.get(f'{level_key}_recall', 0.0):^10.2f}"
                   f"{v_line}{metrics.get(f'{level_key}_f1', 0.0):^10.2f}{v_line}")
            rows.append(row)
        
        # assemble table
        table = [
            f"\n{phase} Metrics:",
            h_line,
            header,
            h_line,
            *rows,
            h_line
        ]
        
        # add extra metrics (displayed on the same line)
        extra_metrics = []
        if 'loss' in metrics:
            extra_metrics.append(f"Loss: {metrics['loss']:.4f}")
        if 'lr' in metrics:
            extra_metrics.append(f"Learning Rate: {metrics['lr']:.6f}")
        
        if extra_metrics:
            table.append(f"\nextra metrics: {', '.join(extra_metrics)}")
        
        return '\n'.join(table)

    def train_batch(self, model_id: str, batch_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, float]]:
        """train a batch"""
        images, targets = batch_data
        images = images.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}
        
        # get current model's MetricTracker
        metric_tracker = self.metric_trackers[model_id]
        
        self.optimizers[model_id].zero_grad()
        outputs = self.models[model_id](images)
        loss = self._compute_loss(outputs, targets)
        loss.backward()
        self.optimizers[model_id].step()
        
        # update metrics, but do not print
        metric_tracker.update('train', loss.item(), outputs, targets, images.size(0), 
                            self.optimizers[model_id].param_groups[0]['lr'])
        
        metrics = metric_tracker.compute_epoch_metrics('train')
        return loss.item(), outputs, metrics

    def validate_batch(self, model_id: str, batch_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, float]]:
        """validate a batch"""
        images, targets = batch_data
        images = images.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}
        
        # get current model's MetricTracker
        metric_tracker = self.metric_trackers[model_id]
        
        outputs = self.models[model_id](images)
        loss = self._compute_loss(outputs, targets)
        
        # update metrics, but do not print
        metric_tracker.update('val', loss.item(), outputs, targets, images.size(0))
        
        metrics = metric_tracker.compute_epoch_metrics('val')
        return loss.item(), outputs, metrics

    def update_lr(self, model_id: str):
        """update learning rate"""
        if model_id in self.schedulers:
            self.schedulers[model_id].step()

    def save_checkpoint(self, model_id: str, epoch: int, metrics: Dict):
        """save model checkpoint"""
        # use modified directory structure
        dirs = self._get_model_dir(model_id)
        species_acc = metrics.get('val', {}).get('species_top1', 0.0)
        
        # get model-specific logger
        logger = logging.getLogger(f'model.{model_id}')
        
        # create checkpoint file name and path
        checkpoint_name = f"epoch_{epoch}_{species_acc:.2f}.pth"
        checkpoint_path = dirs['checkpoints'] / checkpoint_name
        
        # prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.models[model_id].state_dict(),
            'optimizer_state_dict': self.optimizers[model_id].state_dict(),
            'scheduler_state_dict': self.schedulers[model_id].state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics[model_id]
        }
        
        # check if it is the best model
        if species_acc > self.best_metrics[model_id]['species_top1']:
            # delete old checkpoint files
            for old_file in dirs['checkpoints'].glob("epoch_*.pth"):
                if old_file.is_file():
                    old_file.unlink()
                    logger.info(f"deleted old checkpoint: {old_file.name}")
            
            # save new checkpoint
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"saved checkpoint: {checkpoint_path.name}")
            logger.info(f"saved new best model: Epoch {epoch}, Accuracy {species_acc:.2f}%")
            
            # update best metrics
            self.best_metrics[model_id]['species_top1'] = species_acc

    def update_best_metrics(self, model_id: str, epoch: int, val_metrics: Dict[str, float]) -> bool:
        """update best metrics"""
        current_acc = val_metrics['species_top1']
        is_best = current_acc > self.best_metrics[model_id]['species_top1']
        
        if is_best:
            self.best_metrics[model_id].update({
                'species_top1': current_acc,
                'val_loss': val_metrics['loss'],
                'epoch': epoch
            })
        
        return is_best

    def load_checkpoint(self, model_id: str, checkpoint_path: str = None):
        """
        load model checkpoint
        Args:
            model_id: model ID
            checkpoint_path: if provided, load checkpoint from this path; otherwise load from default directory
        """
        if checkpoint_path:
            # if specific checkpoint path is provided, use it
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
            
            # load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # check state dict structure in checkpoint
            state_dict = checkpoint['model_state_dict']
            logging.info(f"Checkpoint state dict keys: {state_dict.keys()}")
            
            # check classifier weights
            for level in ['subfamily', 'genus', 'species']:
                classifier_key = f'classifier.{level}.weight'
                if classifier_key in state_dict:
                    logging.info(f"{level} classifier shape: {state_dict[classifier_key].shape}")
            
            # load model weights
            self.models[model_id].load_state_dict(state_dict)
            logging.info("Successfully loaded checkpoint")

    def get_model_ids(self) -> List[str]:
        """get all model IDs"""
        return list(self.models.keys())

    def get_model(self, model_id: str) -> nn.Module:
        """get pre-loaded model"""
        return self.models[model_id]
    
    def load_best_model(self, model_id: str):
        """load the best model on validation set"""
        best_checkpoint = self._find_best_checkpoint(model_id)
        self.load_checkpoint(model_id, best_checkpoint)
    
    def eval_model(self, model_id: str, data_loader: data_loader) -> Dict:
        """evaluate model performance on given dataset"""
        model = self.models[model_id]
        model.eval()
        
        # create dedicated MetricTracker
        metric_tracker = MetricTracker()
        
        with torch.no_grad():
            for batch in data_loader:
                images, targets, _ = batch  # unpack data, ignore img_path
                images = images.to(self.device)
                
                # move targets to device
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = model(images)
                
                # directly use MetricTracker to update metrics
                metric_tracker.update('val', 0.0, outputs, targets, images.size(0))
        
        # return calculated metrics
        return metric_tracker.compute_epoch_metrics('val')
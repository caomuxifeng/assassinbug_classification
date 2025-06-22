import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import warnings

class AverageMeter:
    """calculate and store average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[float]:
    """
    calculate top-k accuracy
    Args:
        output: model output (N, C)
        target: target label (N,)
        topk: tuple of top-k values to calculate
    Returns:
        list of accuracies, corresponding to each k value in topk
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

class MetricTracker:
    """track metrics during training and validation"""
    def __init__(self, levels=['subfamily', 'genus', 'species']):
        self.levels = levels
        self.reset()

    def reset(self):
        """reset all metrics"""
        self.metrics = {
            'train': {
                'loss': AverageMeter(),
                'lr': [],
                **{f'{level}_{metric}': AverageMeter() 
                   for level in self.levels 
                   for metric in ['top1', 'top3', 'top5', 'f1', 'precision', 'recall']}
            },
            'val': {
                'loss': AverageMeter(),
                **{f'{level}_{metric}': AverageMeter() 
                   for level in self.levels 
                   for metric in ['top1', 'top3', 'top5', 'f1', 'precision', 'recall']}
            },
            'test': {
                'loss': AverageMeter(),
                **{f'{level}_{metric}': AverageMeter() 
                   for level in self.levels 
                   for metric in ['top1', 'top3', 'top5', 'f1', 'precision', 'recall']}
            }
        }
        
        # store predictions and targets for calculating F1, etc.
        self.predictions = {
            phase: {level: [] for level in self.levels}
            for phase in ['train', 'val', 'test']
        }
        self.targets = {
            phase: {level: [] for level in self.levels}
            for phase in ['train', 'val', 'test']
        }

    def update(self, phase: str, loss: float, outputs: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor], batch_size: int, lr: float = None):
        """update all metrics for a batch"""
        # update loss
        self.metrics[phase]['loss'].update(loss, batch_size)
        
        # update learning rate (only for training phase)
        if phase == 'train' and lr is not None:
            self.metrics[phase]['lr'].append(lr)
        
        # update metrics for each level
        for level in self.levels:
            output = outputs[level]
            target = targets[level]
            
            # calculate and update top-k accuracy
            top1, top3, top5 = compute_accuracy(output, target, topk=(1, 3, 5))
            self.metrics[phase][f'{level}_top1'].update(top1, batch_size)
            self.metrics[phase][f'{level}_top3'].update(top3, batch_size)
            self.metrics[phase][f'{level}_top5'].update(top5, batch_size)
            
            # store predictions and targets (for calculating F1, etc.)
            predictions = output.argmax(1).cpu().numpy()
            targets_cpu = target.cpu().numpy()
            
            self.predictions[phase][level].extend(predictions)
            self.targets[phase][level].extend(targets_cpu)

    def compute_epoch_metrics(self, phase: str) -> Dict[str, float]:
        """compute all metrics for an epoch"""
        metrics = {
            'loss': self.metrics[phase]['loss'].avg
        }
        
        # add learning rate (if training phase)
        if phase == 'train' and self.metrics[phase]['lr']:
            metrics['lr'] = self.metrics[phase]['lr'][-1]
        
        # calculate metrics for each level
        for level in self.levels:
            # top-k accuracy
            metrics[f'{level}_top1'] = self.metrics[phase][f'{level}_top1'].avg
            metrics[f'{level}_top3'] = self.metrics[phase][f'{level}_top3'].avg
            metrics[f'{level}_top5'] = self.metrics[phase][f'{level}_top5'].avg
            
            # calculate F1, precision and recall
            predictions = np.array(self.predictions[phase][level])
            targets = np.array(self.targets[phase][level])
            
            try:
                if len(predictions) > 0 and len(np.unique(targets)) > 1:
                    # calculate F1 score
                    f1 = f1_score(
                        targets, predictions, 
                        average='macro', 
                        zero_division=0
                    )
                    metrics[f'{level}_f1'] = f1
                    
                    # calculate precision
                    precision = precision_score(
                        targets, predictions, 
                        average='macro', 
                        zero_division=0
                    )
                    metrics[f'{level}_precision'] = precision
                    
                    # calculate recall
                    recall = recall_score(
                        targets, predictions, 
                        average='macro', 
                        zero_division=0
                    )
                    metrics[f'{level}_recall'] = recall
                else:
                    metrics[f'{level}_f1'] = 0.0
                    metrics[f'{level}_precision'] = 0.0
                    metrics[f'{level}_recall'] = 0.0
                    
                    if len(predictions) == 0:
                        logging.debug(f"No predictions available for {level} in {phase} phase")
                    if len(np.unique(targets)) <= 1:
                        logging.debug(f"Insufficient unique classes for {level} in {phase} phase")
                        
            except Exception as e:
                logging.error(f"Error computing metrics for {level} in {phase} phase: {str(e)}")
                metrics[f'{level}_f1'] = 0.0
                metrics[f'{level}_precision'] = 0.0
                metrics[f'{level}_recall'] = 0.0
        
        return metrics

    def get_current_value(self, metric: str, phase: str = 'train') -> float:
        """get current value of specified metric"""
        if metric == 'loss':
            return self.metrics[phase]['loss'].avg
        elif metric == 'lr' and phase == 'train':
            return self.metrics[phase]['lr'][-1] if self.metrics[phase]['lr'] else 0.0
        else:
            return self.metrics[phase][metric].avg if metric in self.metrics[phase] else 0.0

    def print_metrics(self, epoch: int, phase: str):
        """print current metrics, using table format"""
        metrics = self.compute_epoch_metrics(phase)
        
        # print basic information
        logging.info(f"\n{'='*100}")
        logging.info(f"Epoch {epoch} - {phase.capitalize()} Results:")
        logging.info(f"Loss: {metrics['loss']:.4f}")
        if phase == 'train':
            logging.info(f"Learning Rate: {metrics['lr']:.6f}")
        logging.info('='*100)
        
        # table header
        header = f"{'Level':12} | {'Top-1':8} | {'Top-3':8} | {'Top-5':8} | {'F1':8} | {'Precision':10} | {'Recall':8}"
        separator = '-' * len(header)
        
        logging.info(separator)
        logging.info(header)
        logging.info(separator)
        
        # print metrics for each level
        for level in self.levels:
            row = (
                f"{level:12} | "
                f"{metrics[f'{level}_top1']:7.2f}% | "
                f"{metrics[f'{level}_top3']:7.2f}% | "
                f"{metrics[f'{level}_top5']:7.2f}% | "
                f"{metrics[f'{level}_f1']:8.4f} | "
                f"{metrics[f'{level}_precision']:10.4f} | "
                f"{metrics[f'{level}_recall']:8.4f}"
            )
            logging.info(row)
        
        logging.info(separator)
        logging.info("")  # add empty line

def create_metrics_dict() -> Dict:
    """create dictionary for storing training history"""
    return {
        'train': {
            'loss': [], 'lr': [],
            **{f'{level}_{metric}': [] 
               for level in ['subfamily', 'genus', 'species']
               for metric in ['top1', 'top3', 'top5', 'f1', 'precision', 'recall']}
        },
        'val': {
            'loss': [],
            **{f'{level}_{metric}': [] 
               for level in ['subfamily', 'genus', 'species']
               for metric in ['top1', 'top3', 'top5', 'f1', 'precision', 'recall']}
        }
    }

def update_metrics_dict(metrics_dict: Dict, phase: str, epoch_metrics: Dict):
    """update metrics history"""
    for key, value in epoch_metrics.items():
        if key in metrics_dict[phase]:
            metrics_dict[phase][key].append(value)

def compute_all_metrics_for_cv(metrics_dict, phase='val'):
    """
    calculate comprehensive metrics for cross-validation
    Args:
        metrics_dict: metrics history
        phase: which phase to use ('train' or 'val')
    Returns:
        dictionary containing all metrics
    """
    all_metrics = {}
    
    # find last epoch in metrics history
    for metric_name, values in metrics_dict[phase].items():
        if values:  # ensure there are values
            all_metrics[metric_name] = values[-1]  # get last epoch value
    
    return all_metrics

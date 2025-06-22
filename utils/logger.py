"""
logging tool module
"""

import logging
import sys
from pathlib import Path
from typing import Union
from datetime import datetime

def setup_logging(log_dir: str, model_id: str = None):
    """
    unified logging configuration function
    Args:
        log_dir: log save directory
        model_id: optional model ID, used to create model-specific logger
    """
    # basic configuration
    logging_config = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if model_id:
        # model-specific logger configuration
        logger = logging.getLogger(f'model.{model_id}')
        logger.setLevel(logging.INFO)
        
        # create file handler
        log_file = Path(log_dir) / model_id / 'logs' / f'{datetime.now():%Y%m%d_%H%M%S}.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter(logging_config['format'], 
                                        datefmt=logging_config['datefmt']))
        logger.addHandler(fh)
        
        return logger
    else:
        # global logger configuration
        logging.basicConfig(**logging_config)

__all__ = ['setup_logging']
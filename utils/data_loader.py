
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import random


import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# local module import
from utils.taxonomy import TaxonomyMap

class InsectDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 mode: str = 'train',
                 use_cache: bool = True,
                 max_cache_size: int = 1000,
                 cache_policy: str = 'lfu'):
        """
        multi-level insect dataset loader
        Args:
            data_dir: dataset root directory
            transform: data transformation and augmentation
            mode: 'train' or 'val' or 'test'
            use_cache: whether to use cache mechanism
            max_cache_size: maximum cache image number, prevent memory overflow
            cache_policy: cache policy
                - 'lfu': least frequently used
                - 'random': random replacement
                - 'none': do not use cache
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self.cache_policy = cache_policy
        
        self.cache = {}  # image cache dictionary
        self.access_count = {}  # access count dictionary
        
        # use TaxonomyMap instance
        self.taxonomy_map = TaxonomyMap()
        
        # get all image paths and labels
        self.images, self.labels = self._get_data()
        
        # build label mapping for each level
        self.subfamily_to_idx = {}
        self.genus_to_idx = {}
        self.species_to_idx = {}
        
        for label in self.labels:
            subfamily, genus, species = label['subfamily'], label['genus'], label['species']
            if subfamily not in self.subfamily_to_idx:
                self.subfamily_to_idx[subfamily] = len(self.subfamily_to_idx)
            if genus not in self.genus_to_idx:
                self.genus_to_idx[genus] = len(self.genus_to_idx)
            if species not in self.species_to_idx:
                self.species_to_idx[species] = len(self.species_to_idx)
        
        # store number of classes as instance variable
        self._num_classes = {
            'subfamily': len(self.subfamily_to_idx),
            'genus': len(self.genus_to_idx),
            'species': len(self.species_to_idx)
        }

    def _get_data(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """get all image paths and corresponding labels"""
        images = []
        labels = []
        
        # check if directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"data directory does not exist: {self.data_dir}")
        
        # check if directory is empty
        folders = os.listdir(self.data_dir)
        if not folders:
            raise ValueError(f"data directory is empty: {self.data_dir}")
            
        for folder in folders:
            try:
                # parse label from folder name
                parts = folder.split('_')
                if len(parts) != 3:
                    logging.warning(f"skip folder with wrong format: {folder}")
                    continue
                    
                subfamily, genus, species = parts
                
                # use TaxonomyMap to validate taxonomy relationship
                if not self.taxonomy_map.validate_hierarchy(subfamily, genus, species):
                    logging.warning(f"skip data with wrong taxonomy relationship: {folder}")
                    continue
                
                label_dict = {
                    'subfamily': subfamily,
                    'genus': genus,
                    'species': species
                }
                
                folder_path = os.path.join(self.data_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                    
                # check if folder is empty
                img_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not img_files:
                    logging.warning(f"folder is empty or does not contain images: {folder}")
                    continue
                
                for img_name in img_files:
                    img_path = os.path.join(folder_path, img_name)
                    # 验证图片文件是否可以打开
                    try:
                        with Image.open(img_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                        images.append(img_path)
                        labels.append(label_dict)
                    except Exception as e:
                        logging.error(f"cannot open image {img_path}: {str(e)}")
                        
            except Exception as e:
                logging.error(f"error processing folder {folder}: {str(e)}")
                continue
        
        if not images:
            raise ValueError("no valid image data found")
            
        return images, labels

    @property
    def num_classes(self) -> Dict[str, int]:
        """return number of classes for each level"""
        return self._num_classes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        """get single data sample"""
        try:
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # cache management
            if self.use_cache:
                if img_path in self.cache:
                    image = self.cache[img_path]
                    self.access_count[img_path] = self.access_count.get(img_path, 0) + 1
                else:
                    image = Image.open(img_path).convert('RGB')
                    if len(self.cache) >= self.max_cache_size:
                        self._manage_cache()
                    self.cache[img_path] = image
                    self.access_count[img_path] = 1
            else:
                image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
        
            # convert to label index
            subfamily, genus, species = label['subfamily'], label['genus'], label['species']
            target = {
                'subfamily': torch.tensor(self.subfamily_to_idx[subfamily]),
                'genus': torch.tensor(self.genus_to_idx[genus]),
                'species': torch.tensor(self.species_to_idx[species])
            }
        
            # return image path
            return image, target, img_path
        except Exception as e:
            logging.error(f"failed to load sample {idx}")
            logging.error(f"image path: {img_path}")
            logging.error(f"label: {label}")
            logging.error(f"error: {str(e)}")
            raise

    def print_dataset_info(self):
        """print dataset statistics (table format)"""
        # print basic information
        print(f"\n{'='*80}")
        print(f"dataset information ({self.mode})")
        print(f"total samples: {len(self.images)}")
        print('='*80)
        
        # create table for each level
        for level in ['subfamily', 'genus', 'species']:
            # count samples for each class
            counts = {}
            for label in self.labels:
                name = label[level]
                counts[name] = counts.get(name, 0) + 1
            
            # sort by sample count
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            # print level header
            print(f"\n{level.capitalize()} 层级 (共 {len(counts)} 个类别)")
            print('-'*80)
            
            # print table header
            header = f"{'序号':^6} | {'类别名称':<30} | {'样本数':^10} | {'占比':^10}"
            print(header)
            print('-'*80)
            
            # print detailed information for top 5 classes
            total_samples = len(self.images)
            for i, (name, count) in enumerate(sorted_counts[:5], 1):
                percentage = (count / total_samples) * 100
                row = f"{i:^6} | {name:<30} | {count:^10} | {percentage:^8.2f}%"
                print(row)
            
            # if number of classes exceeds 5, add ellipsis
            if len(sorted_counts) > 5:
                print(f"{'...':^6} | {'...':^30} | {'...':^10} | {'...':^10}")
            
            # print table bottom
            print('-'*80)
            print("")  # add line

    def _manage_cache(self):
        """smart cache management"""
        if self.cache_policy == 'lfu':
            # remove item with least access count
            min_access = min(self.access_count.values())
            candidates = [k for k, v in self.access_count.items() if v == min_access]
            remove_key = random.choice(candidates)
        elif self.cache_policy == 'random':
            # random delete
            remove_key = random.choice(list(self.cache.keys()))
        
        # delete selected cache item
        del self.cache[remove_key]
        del self.access_count[remove_key]

def get_transforms(mode: str = 'train'):
    """get data transformation"""
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def print_combined_dataset_info(train_dataset: InsectDataset, val_dataset: InsectDataset):
    """print combined dataset statistics"""
    # print basic information
    print(f"\n{'='*100}")
    print(f"dataset statistics")
    print(f"{'='*100}")
    print(f"{'dataset type':^15} | {'total samples':^12}")
    print(f"{'-'*100}")
    print(f"{'train':^15} | {len(train_dataset.images):^12}")
    print(f"{'val':^15} | {len(val_dataset.images):^12}")
    print(f"{'-'*100}\n")

    # create table for each level
    for level in ['subfamily', 'genus', 'species']:
        # count classes for train and val datasets
        train_counts = {}
        val_counts = {}
        
        for label in train_dataset.labels:
            name = label[level]
            train_counts[name] = train_counts.get(name, 0) + 1
            
        for label in val_dataset.labels:
            name = label[level]
            val_counts[name] = val_counts.get(name, 0) + 1
        
        # get all unique classes
        all_classes = sorted(set(list(train_counts.keys()) + list(val_counts.keys())))
        
        # sort by train sample count
        sorted_classes = sorted(all_classes, 
                              key=lambda x: train_counts.get(x, 0) + val_counts.get(x, 0), 
                              reverse=True)
        
        # print level header
        print(f"{level.capitalize()} level (total {len(all_classes)} classes)")
        print('-'*100)
        
        # print table header
        header = (f"{'index':^6} | {'class name':<30} | {'train samples':^12} | {'train percentage':^10} | "
                 f"{'val samples':^12} | {'val percentage':^10}")
        print(header)
        print('-'*100)
        
        # print detailed information for top 5 classes
        train_total = len(train_dataset.images)
        val_total = len(val_dataset.images)
        
        for i, name in enumerate(sorted_classes[:5], 1):
            train_count = train_counts.get(name, 0)
            val_count = val_counts.get(name, 0)
            train_percentage = (train_count / train_total * 100) if train_total > 0 else 0
            val_percentage = (val_count / val_total * 100) if val_total > 0 else 0
            
            row = (f"{i:^6} | {name:<30} | {train_count:^12} | {train_percentage:^8.2f}% | "
                  f"{val_count:^12} | {val_percentage:^8.2f}%")
            print(row)
        
        # if number of classes exceeds 5, add ellipsis
        if len(sorted_classes) > 5:
            print(f"{'...':^6} | {'...':^30} | {'...':^12} | {'...':^10} | {'...':^12} | {'...':^10}")
        
        print('-'*100)
        print("")  # add empty line

def get_dataloader(data_dir: str,
                  batch_size: int = 128,
                  mode: str = 'train',
                  num_workers: int = 8,
                  prefetch_factor: int = 2,
                  val_data_dir: str = None,
                  worker_init_fn = None,
                  max_cache_size: int = 1000) -> Tuple[DataLoader, Dict[str, int], Dict[str, List[str]]]:
    """
    get data loader
    Args:
        data_dir: data directory
        batch_size: batch size
        mode: mode ('train' or 'val')
        num_workers: number of threads for data loading
        prefetch_factor: prefetch factor
        val_data_dir: validation set directory
        worker_init_fn: DataLoader worker initialization function
        max_cache_size: maximum cache image number
    Returns:
        dataloader: data loader
        num_classes: number of classes for each level
        class_names: class names for each level
    """
    transform = get_transforms(mode)
    dataset = InsectDataset(
        data_dir=data_dir,
        transform=transform,
        mode=mode,
        max_cache_size=max_cache_size
    )
    
    # get class names for each level
    class_names = {
        'subfamily': [k for k, _ in sorted(dataset.subfamily_to_idx.items(), key=lambda x: x[1])],
        'genus': [k for k, _ in sorted(dataset.genus_to_idx.items(), key=lambda x: x[1])],
        'species': [k for k, _ in sorted(dataset.species_to_idx.items(), key=lambda x: x[1])]
    }
    
    # if training mode, save label maps
    if mode == 'train':
        save_label_maps(dataset)
    
    # if training mode and validation set directory is provided, print train and val dataset information
    if mode == 'train' and val_data_dir:
        val_transform = get_transforms('val')
        val_dataset = InsectDataset(
            data_dir=val_data_dir,
            transform=val_transform,
            mode='val',
            max_cache_size=max_cache_size
        )
        print_combined_dataset_info(dataset, val_dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        drop_last=(mode == 'train'),
        worker_init_fn=worker_init_fn
    )
    
    return dataloader, dataset.num_classes, class_names

def save_label_maps(dataset: InsectDataset):
    """
    save label maps to dataset directory
    Args:
        dataset: dataset instance
    """
    # get dataset root directory from dataset path
    dataset_root = Path(dataset.data_dir).parent
    
    # create save directory
    save_dir = dataset_root / 'label_maps'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # create complete mapping dictionary
    label_maps = {
        'subfamily': {
            'idx_to_name': {str(v): k for k, v in dataset.subfamily_to_idx.items()},
            'name_to_idx': dataset.subfamily_to_idx,
            'num_classes': len(dataset.subfamily_to_idx)
        },
        'genus': {
            'idx_to_name': {str(v): k for k, v in dataset.genus_to_idx.items()},
            'name_to_idx': dataset.genus_to_idx,
            'num_classes': len(dataset.genus_to_idx)
        },
        'species': {
            'idx_to_name': {str(v): k for k, v in dataset.species_to_idx.items()},
            'name_to_idx': dataset.species_to_idx,
            'num_classes': len(dataset.species_to_idx)
        }
    }
    
    # save as single JSON file
    with open(save_dir / 'label_maps.json', 'w', encoding='utf-8') as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=4)
    
    logging.info(f"label maps file saved to {save_dir}/label_maps.json")

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models.model_manager import ModelManager
from train import MODEL_TYPE_MAP  

class InsectPredictor:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config: Dict):
        """
        insect image predictor
        Args:
            model: loaded model with weights
            device: inference device
            config: configuration dictionary
        """
        self.model = model
        self.device = device
        self.config = config
        
        # add model_name attribute
        self.model_name = config.get('model_name', 'convnext_base')  # default value is 'convnext_large'
        
        # load label maps
        self.label_maps = self._load_label_maps()
        print(f"\nloaded {self.num_classes['subfamily']} subfamilies")
        print(f"loaded {self.num_classes['genus']} genera")
        print(f"loaded {self.num_classes['species']} species")
        
        # set transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # create result save directory, only create inference_results
        self.save_dir = Path(config['save_dir']) / self.model_name / 'inference_results'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # set model to evaluation mode
        self.model.eval()
        
        # add supported image format definition
        self.IMG_EXTENSIONS = (
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG',
            '.ppm', '.PPM',
            '.bmp', '.BMP',
            '.pgm', '.PGM',
            '.tif', '.TIF', '.tiff', '.TIFF',
            '.webp', '.WEBP'
        )
        
    def _load_label_maps(self) -> Dict[str, Dict[int, str]]:
        """load label maps"""
        map_file = Path(self.config['label_map_dir']) / 'label_maps.json'
        if not map_file.exists():
            raise FileNotFoundError(f"label map file not found: {map_file}")
            
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                maps_data = json.load(f)
                
            # convert to format needed for inference
            label_maps = {}
            for level in ['subfamily', 'genus', 'species']:
                label_maps[level] = {
                    int(k): v for k, v in maps_data[level]['idx_to_name'].items()
                }
                
            # save class number information, may be used for inference
            self.num_classes = {
                level: data['num_classes'] 
                for level, data in maps_data.items()
            }
            
            return label_maps
                
        except Exception as e:
            raise RuntimeError(f"failed to load label map file: {str(e)}")

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        preprocess image
        Args:
            image_path: image path
        Returns:
            processed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

    def predict_single(self, 
                      image_path: Union[str, Path], 
                      return_probs: bool = False) -> Dict[str, Union[str, float]]:
        """
        single image prediction
        Args:
            image_path: image path
            return_probs: whether to return probability values
        Returns:
            prediction result dictionary
        """
        image = self.preprocess_image(image_path).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
        
        # initialize result dictionary
        results = {
            'subfamily_prediction': None,
            'genus_prediction': None,
            'species_prediction': None
        }
        
        # get prediction results
        for level in ['subfamily', 'genus', 'species']:
            try:
                probs = torch.softmax(outputs[level], dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                pred_prob = probs[pred_idx].item()
                
                # get prediction label
                pred_label = self.label_maps[level][pred_idx]
                
                results[f"{level}_prediction"] = pred_label
                if return_probs:
                    # keep confidence to three decimal places
                    results[f"{level}_confidence"] = round(pred_prob, 3)
                    
                    # get top-3 prediction, keep three decimal places
                    top3_probs, top3_indices = torch.topk(probs, 3)
                    results[f"{level}_top3"] = [
                        (self.label_maps[level][idx.item()], round(prob.item(), 3))
                        for idx, prob in zip(top3_indices, top3_probs)
                    ]
            except Exception as e:
                logging.error(f"error in {level} level prediction: {str(e)}")
                results[f"{level}_prediction"] = "unknown"
                if return_probs:
                    results[f"{level}_confidence"] = 0.000
                    results[f"{level}_top3"] = [("unknown", 0.000)] * 3
        
        return results

    def predict_batch(self, image_dir, save_results=True, batch_size=32, recursive=True):
        """
        batch prediction for images in directory
        
        Args:
            image_dir: image directory path
            save_results: whether to save results
            batch_size: batch size
            recursive: whether to recursively search subdirectories
        
        Returns:
            DataFrame containing prediction results
        """
        # convert to Path object
        image_dir = Path(image_dir)
        
        # collect all image files
        image_files = []
        if recursive:
            # recursively search all subdirectories
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                image_files.extend(list(image_dir.glob(f'**/{ext}')))
        else:
            # only search top level directory
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                image_files.extend(list(image_dir.glob(ext)))
        
        # ensure there are image files
        if not image_files:
            print(f"no image files found in {image_dir} {'and its subdirectories' if recursive else ''}")
            return pd.DataFrame()
        
        # sort files to ensure consistency
        image_files.sort()
        
        # collect all images and their true labels
        results = []
        total_images = 0
        processed_images = 0
        
        # add progress information when processing images
        total_dirs = len(list(image_dir.glob("*")))
        print(f"found {total_dirs} class directories")
        
        for i, class_dir in enumerate(image_dir.glob("*"), 1):
            if not class_dir.is_dir():
                continue
            
            print(f"\rprocessing progress: {i}/{total_dirs} - {class_dir.name}", end="")
            
            # parse label from directory name
            dir_name = class_dir.name
            parts = dir_name.split('_')
            
            if len(parts) != 3:
                continue
                
            subfamily, genus, species = parts
            
            # process all images in this directory
            for ext in self.IMG_EXTENSIONS:
                for img_path in class_dir.glob(f"*{ext}"):
                    total_images += 1
                    try:
                        # get prediction results
                        pred = self.predict_single(img_path, return_probs=True)
                        
                        # ensure basic prediction results exist
                        for level in ['subfamily', 'genus', 'species']:
                            if f"{level}_prediction" not in pred:
                                pred[f"{level}_prediction"] = "unknown"
                                pred[f"{level}_confidence"] = 0.0
                        
                        # add true labels and image paths
                        pred.update({
                            'image_path': str(img_path.relative_to(image_dir)),
                            'true_subfamily': subfamily,
                            'true_genus': genus,
                            'true_species': species
                        })
                        
                        results.append(pred)
                        processed_images += 1
                        
                    except Exception:
                        continue
                        
        print("\n\nprocessing completed!")
        print(f"total images: {total_images}")
        print(f"processed images: {processed_images}")
        
        if not results:
            return pd.DataFrame()
        
        # convert to DataFrame
        df = pd.DataFrame(results)
        
        # ensure all required columns exist
        required_columns = [
            'subfamily_prediction', 'genus_prediction', 'species_prediction',
            'true_subfamily', 'true_genus', 'true_species'
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = "unknown"
        
        # reorder columns
        ordered_columns = [
            'image_path',
            'true_subfamily',
            'subfamily_prediction',
            'subfamily_top3',
            'true_genus',
            'genus_prediction',
            'genus_top3',
            'true_species',
            'species_prediction',
            'species_top3',
            'subfamily_confidence',
            'genus_confidence',
            'species_confidence'
        ]
        
        # ensure all columns exist, add empty values if not
        for col in ordered_columns:
            if col not in df.columns:
                if 'confidence' in col:
                    df[col] = 0.0
                elif 'top3' in col:
                    df[col] = [[] for _ in range(len(df))]
                else:
                    df[col] = "unknown"
        
        # reorder columns
        df = df[ordered_columns]
        
        if save_results:
            try:
                # simplify save directory structure, use inference_results directly
                save_dir = self.save_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # save prediction results
                predictions_path = save_dir / "batch_predictions.csv"
                df.to_csv(predictions_path, index=False)
                print(f"\nsaved prediction results to: {predictions_path}")
                
                # 2. calculate and save evaluation metrics
                metrics = self._compute_accuracy_metrics(df)
                
                # create evaluation results table
                evaluation_results = []
                
                def format_percentage(value: float) -> str:
                    """format decimal to percentage, keep two decimal places"""
                    return f"{value * 100:.2f}%"
                
                for level in ['subfamily', 'genus', 'species']:
                    if f'{level}_accuracy' in metrics:
                        evaluation_results.append({
                            'Level': level,
                            'Metric': 'Accuracy',
                            'Value': format_percentage(metrics[f'{level}_accuracy'])
                        })
                    
                    if f'{level}_weighted_precision' in metrics:
                        evaluation_results.append({
                            'Level': level,
                            'Metric': 'Weighted Precision',
                            'Value': format_percentage(metrics[f'{level}_weighted_precision'])
                        })
                    if f'{level}_weighted_recall' in metrics:
                        evaluation_results.append({
                            'Level': level,
                            'Metric': 'Weighted Recall',
                            'Value': format_percentage(metrics[f'{level}_weighted_recall'])
                        })
                    if f'{level}_weighted_f1' in metrics:
                        evaluation_results.append({
                            'Level': level,
                            'Metric': 'Weighted F1-Score',
                            'Value': format_percentage(metrics[f'{level}_weighted_f1'])
                        })
                
                for level in ['subfamily', 'genus', 'species']:
                    if f'{level}_known_accuracy' in metrics:
                        evaluation_results.append({
                            'Level': level,
                            'Metric': 'Known Classes Accuracy',
                            'Value': format_percentage(metrics[f'{level}_known_accuracy'])
                        })
                
                # save evaluation metrics CSV
                eval_df = pd.DataFrame(evaluation_results)
                eval_csv_path = save_dir / "evaluation_metrics.csv"
                eval_df.to_csv(eval_csv_path, index=False)
                
                # save detailed metrics JSON
                metrics_path = save_dir / "detailed_metrics.json"
                # add custom JSON encoder
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super(NumpyEncoder, self).default(obj)

                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
                
                # print evaluation metrics
                print("\nevaluation metrics:")
                for result in evaluation_results:
                    print(f"{result['Level']} - {result['Metric']}: {result['Value']}")
                
                # calculate and save detailed accuracy for each class
                for level in ['subfamily', 'genus', 'species']:
                    # get trained classes
                    trained_classes = set(self.label_maps[level].values())
                    
                    # calculate accuracy for each class in this level
                    per_class_metrics = self._compute_per_class_metrics(
                        df, level, trained_classes
                    )
                    
                    # save to CSV file
                    metrics_file = save_dir / f"{level}_class_metrics_with_prf.csv"
                    per_class_metrics.to_csv(metrics_file, index=False)
                    print(f"{level.capitalize()} level class precision, recall and F1 scores saved to: {metrics_file}")
                
            except Exception as e:
                logging.error(f"error saving results: {str(e)}")
        
        return df

    def predict(self, image: torch.Tensor) -> Dict[str, str]:
        # get prediction results
        outputs = self.model(image)
        
        # initialize result dictionary
        results = {}
        
        # get prediction results
        for level in ['subfamily', 'genus', 'species']:
            try:
                probs = torch.softmax(outputs[level], dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                
                # get prediction label
                pred_label = self.label_maps[level][pred_idx]
                results[f"{level}_prediction"] = pred_label
                
            except Exception as e:
                logging.error(f"error in {level} level prediction: {str(e)}")
                results[f"{level}_prediction"] = "unknown"
        
        return results

    def _compute_accuracy_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """calculate accuracy, precision, recall and F1 score for each level"""
        metrics = {}
        
        # get trained classes, use instance attributes instead of config
        trained_subfamilies = set(self.label_maps['subfamily'].values())
        trained_genera = set(self.label_maps['genus'].values())
        trained_species = set(self.label_maps['species'].values())
        
        def calculate_accuracy(true_labels, predictions):
            """calculate basic accuracy, keep four decimal places"""
            return round((true_labels == predictions).mean(), 4)
        
        # 1. calculate basic accuracy
        for level in ['subfamily', 'genus', 'species']:
            true_col = f"true_{level}"
            pred_col = f"{level}_prediction"
            
            if true_col in results_df.columns and pred_col in results_df.columns:
                metrics[f"{level}_accuracy"] = calculate_accuracy(
                    results_df[true_col], 
                    results_df[pred_col]
                )
                
                # calculate precision, recall and F1 score for each class
                try:
                    # get all unique class labels
                    all_classes = sorted(set(results_df[true_col].unique()) | set(results_df[pred_col].unique()))
                    
                    # initialize metrics dictionary
                    precision_dict = {}
                    recall_dict = {}
                    f1_dict = {}
                    
                    # calculate metrics for each class
                    for cls in all_classes:
                        # true positive: number of samples predicted as this class and actually belong to this class
                        tp = ((results_df[pred_col] == cls) & (results_df[true_col] == cls)).sum()
                        
                        # false positive: number of samples predicted as this class but actually do not belong to this class
                        fp = ((results_df[pred_col] == cls) & (results_df[true_col] != cls)).sum()
                        
                        # false negative: number of samples predicted as not this class but actually belong to this class
                        fn = ((results_df[pred_col] != cls) & (results_df[true_col] == cls)).sum()
                        
                        # calculate precision: TP / (TP + FP)
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        precision_dict[cls] = round(precision, 4)
                        
                        # calculate recall: TP / (TP + FN)
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        recall_dict[cls] = round(recall, 4)
                        
                        # calculate
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        f1_dict[cls] = round(f1, 4)
                    
                    # remove macro average metrics calculation, only keep detailed metrics for each class
                    metrics[f"{level}_per_class_precision"] = precision_dict
                    metrics[f"{level}_per_class_recall"] = recall_dict
                    metrics[f"{level}_per_class_f1"] = f1_dict
                    
                    # calculate weighted average metrics (weighted by number of samples for each class)
                    class_counts = results_df[true_col].value_counts().to_dict()
                    total_samples = sum(class_counts.values())
                    
                    weighted_precision = sum(precision_dict[cls] * class_counts.get(cls, 0) 
                                            for cls in all_classes) / total_samples
                    weighted_recall = sum(recall_dict[cls] * class_counts.get(cls, 0) 
                                         for cls in all_classes) / total_samples
                    weighted_f1 = sum(f1_dict[cls] * class_counts.get(cls, 0) 
                                     for cls in all_classes) / total_samples
                    
                    metrics[f"{level}_weighted_precision"] = round(weighted_precision, 4)
                    metrics[f"{level}_weighted_recall"] = round(weighted_recall, 4)
                    metrics[f"{level}_weighted_f1"] = round(weighted_f1, 4)
                    
                except Exception as e:
                    logging.error(f"error calculating precision, recall and F1 score for {level}: {str(e)}")
        
        # 2. calculate accuracy for known classes, keep four decimal places
        for level in ['subfamily', 'genus', 'species']:
            true_col = f"true_{level}"
            pred_col = f"{level}_prediction"
            top3_col = f"{level}_top3"
            
            if true_col not in results_df.columns or pred_col not in results_df.columns:
                continue
            
            # get samples from known classes
            if level == 'subfamily':
                known_mask = results_df[true_col].isin(trained_subfamilies)
            elif level == 'genus':
                known_mask = results_df[true_col].isin(trained_genera)
            else:  # species
                known_mask = results_df[true_col].isin(trained_species)
            
            known_df = results_df[known_mask]
            
            if len(known_df) > 0:
                # calculate accuracy for known classes
                metrics[f"{level}_known_accuracy"] = calculate_accuracy(
                    known_df[true_col],
                    known_df[pred_col]
                )
                
                # calculate top-k accuracy for known classes
                if top3_col in known_df.columns:
                    for k in [1, 3]:
                        try:
                            correct = 0
                            total = len(known_df)
                            
                            for _, row in known_df.iterrows():
                                true_label = row[true_col]
                                predictions = row[top3_col]
                                
                                # ensure predictions is a valid list
                                if isinstance(predictions, list) and len(predictions) > 0:
                                    top_k_preds = [pred[0] for pred in predictions[:k]]
                                    if true_label in top_k_preds:
                                        correct += 1
                                        
                            if total > 0:
                                metrics[f"{level}_known_top{k}_accuracy"] = round(correct / total, 4)
                                
                        except Exception as e:
                            logging.warning(f"error calculating top-{k} accuracy for {level}: {str(e)}")      
        return metrics

    def _compute_per_class_metrics(self, results_df: pd.DataFrame, level: str, trained_classes: set) -> pd.DataFrame:
        """
        calculate detailed accuracy, precision, recall and F1 score for each class
        Args:
            results_df: prediction results DataFrame
            level: classification level ('subfamily', 'genus', 'species')
            trained_classes: trained classes set
        Returns:
            DataFrame containing accuracy for each class
        """
        true_col = f"true_{level}"
        pred_col = f"{level}_prediction"
        top3_col = f"{level}_top3"
        
        # get all unique class labels
        all_classes = set(results_df[true_col].unique())
        
        # prepare to store metrics for each class
        class_metrics = []
        
        for class_name in sorted(all_classes):
            # get all samples for this class
            class_mask = results_df[true_col] == class_name
            class_df = results_df[class_mask]
            
            if len(class_df) == 0:
                continue
            
            # calculate basic metrics
            metrics_dict = {
                'class_name': class_name,
                'total_samples': len(class_df),
                'is_known': class_name in trained_classes,
                'top1_accuracy': f"{(class_df[pred_col] == class_name).mean() * 100:.2f}%"
            }
            
            # calculate precision, recall and F1 score
            tp = ((results_df[pred_col] == class_name) & (results_df[true_col] == class_name)).sum()
            fp = ((results_df[pred_col] == class_name) & (results_df[true_col] != class_name)).sum()
            fn = ((results_df[pred_col] != class_name) & (results_df[true_col] == class_name)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics_dict['precision'] = f"{precision * 100:.2f}%"
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics_dict['recall'] = f"{recall * 100:.2f}%"
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics_dict['f1_score'] = f"{f1 * 100:.2f}%"
            
            if top3_col in class_df.columns:
                correct_top3 = 0
                for _, row in class_df.iterrows():
                    if isinstance(row[top3_col], list) and len(row[top3_col]) > 0:
                        top3_preds = [pred[0] for pred in row[top3_col][:3]]
                        if class_name in top3_preds:
                            correct_top3 += 1
                metrics_dict['top3_accuracy'] = f"{(correct_top3 / len(class_df)) * 100:.2f}%"

        # ensure all metrics dictionaries have the same keys
        all_keys = set()
        for metrics_dict in class_metrics:
            all_keys.update(metrics_dict.keys())

        # add default values for all missing keys
        for metrics_dict in class_metrics:
            for key in all_keys:
                if key not in metrics_dict:
                    if key in ['top1_accuracy', 'top3_accuracy', 'correct_subfamily_rate', 'correct_genus_rate']:
                        metrics_dict[key] = "N/A"
                    elif key == 'total_samples':
                        metrics_dict[key] = 0
                    elif key == 'is_known':
                        metrics_dict[key] = False
                    else:
                        metrics_dict[key] = ""
        
        # convert to DataFrame
        metrics_df = pd.DataFrame(class_metrics)
        
        # set column order, but only include actual columns
        columns = ['class_name', 'total_samples', 'is_known', 'top1_accuracy', 'precision', 'recall', 'f1_score']
        
        # add only when column exists
        if 'top3_accuracy' in metrics_df.columns:
            columns.append('top3_accuracy')
        
        if level == 'genus' and 'correct_subfamily_rate' in metrics_df.columns:
            columns.append('correct_subfamily_rate')
        elif level == 'species':
            if 'correct_genus_rate' in metrics_df.columns:
                columns.append('correct_genus_rate')
            if 'correct_subfamily_rate' in metrics_df.columns:
                columns.append('correct_subfamily_rate')
        
        # return only existing columns
        available_columns = [col for col in columns if col in metrics_df.columns]
        return metrics_df[available_columns]

    def _load_taxonomy(self):
        """load taxonomy data and cache"""
        if not hasattr(self, 'taxonomy_df'):
            try:
                self.taxonomy_df = pd.read_csv('data/taxonomy.csv')
                print(f"loaded {len(self.taxonomy_df)} taxonomy data")
            except Exception as e:
                logging.error(f"cannot load taxonomy.csv: {str(e)}")
                print(f"warning: cannot load taxonomy file, some features will be limited. error info: {str(e)}")
                self.taxonomy_df = pd.DataFrame(columns=['subfamily', 'genus', 'species'])
        return self.taxonomy_df


def load_config(config_path: str = None) -> Dict:
    """load config file"""
    # get current script directory
    current_dir = Path(__file__).parent.absolute()
    
    default_config = {
        # model related config
        'model': {
            'type': 'mobilenet',
            'name': 'mobilenetv3_small',
            'checkpoint_path': str(current_dir / 'results/complete/mobilenetv3_small/checkpoints/epoch_xx_xx.xx.pth')
        },
        
        # data related config
        'data': {
            'label_map_dir': str(current_dir / 'data/complete/label_maps'),
            'input': {
                'mode': 'batch',  # 'single' or 'batch'
                'path': None,  # will be specified by command line arguments
            }
        },
        
        # output related config
        'output': {
            'save_dir': str(current_dir / 'results'),
            'save_predictions': True,
            'visualize': True
        },
        
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu'
    }
    
    if config_path:
        # if config file is provided, load and update default config
        with open(config_path, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        # recursively update config
        def update_config(default, custom):
            for k, v in custom.items():
                if k in default and isinstance(default[k], dict) and isinstance(v, dict):
                    update_config(default[k], v)
                else:
                    default[k] = v
        update_config(default_config, custom_config)
    
    # add config validation
    required_keys = ['model', 'data', 'output', 'device']
    for key in required_keys:
        if key not in default_config:
            raise KeyError(f"config file missing required '{key}' key")
    
    # check path validity
    path_keys = [
        ('model', 'checkpoint_path'),
        ('data', 'label_map_dir'),
        ('output', 'save_dir')
    ]
    
    for section, key in path_keys:
        path = default_config[section][key]
        if not Path(path).exists():
            logging.warning(f"warning: path '{path}'({section}.{key}) does not exist")
    
    # print config info
    logging.info("\ncurrent config:")
    logging.info(json.dumps(default_config, indent=2, ensure_ascii=False))
    
    return default_config

def main():
    """main function"""
    parser = argparse.ArgumentParser(description="insect image predictor")
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--input', type=str, required=True, help='input image path or directory path')
    parser.add_argument('--mode', type=str, 
                       choices=['single', 'batch'], 
                       required=True, 
                       help='inference mode: single(single image) or batch(batch inference)')
    parser.add_argument('--recursive', type=bool, default=True, 
                       help='whether to recursively search for image files in subdirectories')
    args = parser.parse_args()
    
    # load config
    config = load_config(args.config)
    
    # update input config
    config['data']['input']['mode'] = args.mode
    config['data']['input']['path'] = args.input
    
    print(f"\nusing device: {config['device']}")
    print(f"model type: {config['model']['type']}")
    print(f"model name: {config['model']['name']}")
    
    # set device
    device = torch.device(config['device'])
    
    try:
        # load checkpoint
        checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device)
        
        # load label_maps directory
        label_maps_path = Path(config['data']['label_map_dir']) / 'label_maps.json'
        with open(label_maps_path, 'r', encoding='utf-8') as f:
            label_maps = json.load(f)
        
        # build num_classes dictionary
        num_classes = {
            level: data['num_classes'] 
            for level, data in label_maps.items()
        }
        
        # get model name and type
        model_name = config['model']['name']  # e.g. 'resnet_50'
        model_type = config['model']['type']  # get model type from config
        
        # validate model type and name match
        if model_type not in MODEL_TYPE_MAP:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        if not any(model_name.startswith(model) for model in MODEL_TYPE_MAP[model_type]):
            raise ValueError(f"Model name {model_name} does not match type {model_type}")
        
        # create basic training config
        basic_train_config = {
            'train': {
                'label_smoothing': 0.0  # no label smoothing during inference
            },
            'data': {
                'batch_size': 256,  # batch size during inference
                'num_workers': 8   # number of data loading threads
            }
        }
        
        # get model instance from model_manager, set create_dirs=False
        model_manager = ModelManager(
            model_configs=[{
                'type': model_type,
                'name': model_name
            }],
            device=device,
            num_classes=num_classes,
            save_dir='results',  # use relative path
            create_dirs=False,  # do not create directories during inference
            train_config=basic_train_config
        )
        
        # get model instance
        model = model_manager.models[model_name]
        
        # load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # if checkpoint directly contains model weights
            model.load_state_dict(checkpoint)
        
        print(f"\nsuccessfully loaded model weights: {config['model']['checkpoint_path']}")
        
        # modify save path, only create inference_results directory
        predictor_config = {
            'label_map_dir': config['data']['label_map_dir'],
            'save_dir': config['output']['save_dir'],  # use path from config file
            'model_name': model_name
        }
        
        # create predictor
        predictor = InsectPredictor(model, device, predictor_config)
        
        # execute corresponding function according to config
        input_path = config['data']['input']['path']
        if not input_path:
            print("no input path specified!")
            return
            
        if config['data']['input']['mode'] == 'single':
            # single image prediction
            results = predictor.predict_single(input_path, return_probs=True)
            print("\nprediction results:")
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
            if config['output']['visualize']:
                if config['output']['save_predictions']:
                    # create predictions subdirectory
                    pred_dir = Path(config['output']['save_dir']) / model_name / 'predictions'
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    save_path = pred_dir / f"{Path(input_path).stem}_pred.png"
                    predictor.visualize_prediction(input_path, save_path=save_path)
                    print(f"\nprediction results image saved to: {save_path}")
                else:
                    predictor.visualize_prediction(input_path)
                
        elif config['data']['input']['mode'] == 'batch':
            # batch prediction
            results_df = predictor.predict_batch(
                input_path, 
                save_results=config['output']['save_predictions'],
                recursive=args.recursive
            )
            print("\nbatch prediction results summary:")
            print(results_df.head())
            
            if config['output']['save_predictions']:
                print(f"\nresults saved to:")
                print(f"prediction results: {config['output']['save_dir']}/batch_predictions.csv")
                print(f"evaluation metrics: {config['output']['save_dir']}/evaluation_metrics.csv")
                print(f"detailed metrics: {config['output']['save_dir']}/detailed_metrics.json")
        
            print("\nbatch prediction results summary:")
            print(results_df.head())
            
        else:
            print(f"\nunsupported prediction mode: {config['data']['input']['mode']}")
            
    except Exception as e:
        print(f"\nerror: {str(e)}")
        raise

if __name__ == '__main__':
    main()
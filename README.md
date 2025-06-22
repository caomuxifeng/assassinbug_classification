# Aassassin bugs Image Classification System
## A Deep Learning Framework for Hierarchical Classification of Assassin Bugs (Reduviidae)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)

## 📖 Overview

This project implements a comprehensive deep learning framework for hierarchical classification of assassin bugs (Reduviidae family) using state-of-the-art computer vision models. The system performs multi-level taxonomic classification across three hierarchical levels: **subfamily**, **genus**, and **species**.

### Key Features

- **Multi-Model Support**: Implementation of 7 popular deep learning architectures
- **Hierarchical Classification**: Three-level taxonomic classification (subfamily → genus → species)
- **Configurable Training**: Flexible configuration system for different model architectures
- **Advanced Metrics**: Comprehensive evaluation with Top-k accuracy, F1-score, precision, and recall
- **Cross-Validation**: Built-in k-fold cross-validation for robust model evaluation
- **Pretrained Weights**: Support for loading pretrained models
- **Label Smoothing**: Configurable label smoothing for improved generalization

## 🏗️ Architecture

### Supported Models

| Model Family | Variants | Description |
|--------------|----------|-------------|
| **ResNet** | ResNet-18, 34, 50, 101, 152 | Deep residual networks with skip connections |
| **DenseNet** | DenseNet-121, 169, 201, 161 | Densely connected convolutional networks |
| **EfficientNet** | EfficientNet-B0 to B7 | Compound scaling for efficient CNNs |
| **MobileNetV3** | MobileNetV3-Large, Small | Lightweight networks for mobile deployment |
| **ConvNeXt** | ConvNeXt-Tiny, Small, Base, Large, XLarge | Modern ConvNet with Transformer design principles |
| **Swin Transformer** | Swin-Tiny, Small, Base, Large | Hierarchical vision transformer with shifted windows |
| **Vision Transformer** | ViT-Tiny, Small, Base, Large | Pure transformer architecture for image classification |


## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ storage space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/insect-classification.git
cd insect-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install additional dependencies**
```bash
# For CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
data/
├── taxonomy.csv                 # Taxonomic mapping file
├── train/                      # Training images
│   ├── subfamily1_genus1_species1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── subfamily1_genus1_species2/
│   └── ...
└── val/                        # Validation images (optional)
    ├── subfamily1_genus1_species1/
    └── ...

### Taxonomy File Format

The `taxonomy.csv` file should contain the taxonomic hierarchy:

```csv
subfamily,genus,species
Ectrichodiinae,Ectrychotes,Ectrychotes comottoi
Ectrichodiinae,Ectrychotes,Ectrychotes andreae
Harpactorinae,Biasticus,Biasticus flavinotus
...
```

## 🔧 Configuration

### Model Configuration

Training configurations are stored in `configs/training/` directory:

- `configs/training/resnet.yaml` - ResNet family configurations
- `configs/training/densenet.yaml` - DenseNet family configurations
- `configs/training/efficientnet.yaml` - EfficientNet family configurations
- `configs/training/mobilenet.yaml` - MobileNetV3 configurations
- `configs/training/convnext.yaml` - ConvNeXt family configurations
- `configs/training/swin.yaml` - Swin Transformer configurations
- `configs/training/vit.yaml` - Vision Transformer configurations


## 🏃‍♂️ Usage

### Training Single Model

```bash
python train.py \
    --data_dir ./data/train \
    --val_data_dir ./data/val \
    --model_configs '[{"type": "resnet", "name": "resnet_50", "pretrained": true}]' \
    --batch_size 32 \
    --epochs 100 \
    --num_workers 4
```

### Training Multiple Models

```bash
python train.py \
    --data_dir ./data/train \
    --val_data_dir ./data/val \
    --model_configs '[
        {"type": "resnet", "name": "resnet_50", "pretrained": true},
        {"type": "efficientnet", "name": "efficientnet_b0", "pretrained": true},
        {"type": "swin", "name": "swin_tiny", "pretrained": true}
    ]' \
    --batch_size 32 \
    --epochs 100
```

### Cross-Validation Training

```bash
python train.py \
    --data_dir ./data/train \
    --model_configs '[{"type": "resnet", "name": "resnet_50", "pretrained": true}]' \
    --cross_validation \
    --cv_folds 5 \
    --batch_size 32 \
    --epochs 100
```

### Inference

```bash
python inference.py \
    --model_path ./results/resnet_50/checkpoints/best_model.pth \
    --model_type resnet \
    --model_name resnet_50 \
    --image_path ./test_image.jpg \
    --taxonomy_file ./data/taxonomy.csv
```

## 📊 Model Performance

### Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Top-k Accuracy** (k=1,3,5): Classification accuracy for top-k predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Sample Results

| Model | Subfamily Top-1 | Genus Top-1 | Species Top-1 | Species Top-5 |
|-------|----------------|-------------|---------------|---------------|
| ResNet-50 | 95.2% | 89.7% | 84.3% | 94.1% |
| EfficientNet-B3 | 96.1% | 91.2% | 86.7% | 95.3% |
| Swin-Tiny | 94.8% | 88.9% | 83.1% | 93.7% |

## 📁 Output Structure

Training results are organized as follows:

```
results/
├── model_name/
│   ├── checkpoints/           # Model checkpoints
│   │   └── epoch_100_86.50.pth
│   ├── logs/                  # Training logs
│   │   ├── model_config.log
│   │   └── training.log
│   └── metrics/               # Performance metrics
│       ├── training_history.json
│       ├── metrics_summary.csv
│       └── confusion_matrix.png
```

## 🔍 Advanced Features

### Label Smoothing

Enable label smoothing to improve model generalization:

```bash
python train.py --label_smoothing 0.1 [other arguments]
```

### Mixed Precision Training

For faster training with reduced memory usage:

```bash
python train.py --mixed_precision [other arguments]
```

### Data Augmentation

The system includes extensive data augmentation:

- Random rotation (±15°)
- Random horizontal/vertical flipping
- Color jittering
- Random erasing
- AutoAugment policies

### Hierarchical Loss Weighting

Customize loss weighting for different taxonomic levels:

```python
# In model_manager.py
weights = {
    'subfamily': 1.0,
    'genus': 1.5,      # Higher weight for genus classification
    'species': 2.0     # Highest weight for species classification
}
```

## 🛠️ Development

### Project Structure

```
├── configs/                   # Training configurations
├── data/                     # Dataset directory
├── models/                   # Model implementations
│   ├── ResNet.py
│   ├── DenseNet.py
│   ├── EfficientNet.py
│   ├── MobileNetV3.py
│   ├── ConvNeXt.py
│   ├── Swin.py
│   ├── ViT.py
│   └── model_manager.py
├── utils/                    # Utility functions
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── metric.py
│   ├── taxonomy.py
│   └── logger.py
├── train.py                  # Training script
├── inference.py              # Inference script
└── requirements.txt          # Dependencies
```
#1.single image inference
# python inference.py --mode single --input path/to/your/image.jpg

# 2.batch inference
# python inference.py --mode batch --input path/to/your/folder

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔄 Updates

- **v1.0.0** (2025-06): Initial release with 7 model architectures
---

⭐ If you find this project helpful, please consider giving it a star!
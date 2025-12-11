# Aassassin bugs Image Classification System
## A Deep Learning Framework for Hierarchical Classification of Assassin Bugs (Reduviidae)

## 📖 Overview

This project implements a comprehensive deep learning framework for hierarchical classification of assassin bugs (Reduviidae family) using state-of-the-art computer vision models. The system performs multi-level taxonomic classification across three hierarchical levels: **subfamily**, **genus**, and **species**.

## 📦 Included Assets (ready to use)
- Sample images for inference and demos: `example/Ectrychotes andreae_*.jpg|.JPG|.png`
- A pretrained checkpoint for quick testing: `example/DenseNet_121_epoch_84_94.04.pth`
- Dependency lock file for reproducible setup: `requirements.txt`

### Key Features

- **Multi-Model Support**: Implementation of 7 popular deep learning architectures
- **Hierarchical Classification**: Three-level taxonomic classification (subfamily → genus → species)
- **Configurable Training**: Flexible configuration system for different model architectures
- **Advanced Metrics**: Comprehensive evaluation with Top-k accuracy, F1-score, precision, and recall
- **Cross-Validation**: Built-in k-fold cross-validation for robust model evaluation
- **Pretrained Weights**: Support for loading pretrained models

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

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/caomuxifeng/assassinbug_classification.git
cd assassinbug_classification
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

> If you only want to run the provided demo, installing from `requirements.txt` is the quickest way to match the tested environment.

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
data/
├── taxonomy.csv                 # Taxonomic mapping file
├── complete/train/                      # Training images
│   ├── subfamily1_genus1_species1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── subfamily1_genus1_species2/
│   └── ...
├── complete/val/                        # Validation images
│   ├── subfamily1_genus1_species1/
│   └── ...
├──cross-domain test
├──genus_zeroshot
└──species_zeroshot

## 🏃‍♂️ Usage

### Training Single Model

```bash
python train.py \
    --data_dir ./data/comolete/train \
    --val_data_dir ./data/complete/val \
    --model_configs '[{"type": "resnet", "name": "resnet_50", "pretrained": true}]' \
    --batch_size 64 \
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
    --batch_size 64 \
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

### Quick Demo with Provided Assets
Use the bundled checkpoint and sample image to verify the pipeline end-to-end:
```bash
python inference.py \
    --model_path ./example/DenseNet_121_epoch_84_94.04.pth \
    --model_type densenet \
    --model_name densenet_121 \
    --image_path ./example/Ectrychotes\ andreae_001.jpg \
    --taxonomy_file ./data/taxonomy.csv
```
The command will output predicted subfamily, genus, and species for the sample image. Swap `image_path` to any other file in `example/` to test additional samples.

## 📊 Model Performance

### Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Top-k Accuracy** (k=1,3,5): Classification accuracy for top-k predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall


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
│       ├── train_metrics.csv
│       └── val_metrics.csv
```

## 🔄 Updates

- **v1.1.0** (2025-12-11): Added bundled demo assets (`example/` images + DenseNet checkpoint), pinned `requirements.txt`, and documented quick-start inference command.
- **v1.0.0** (2025-06): Initial release with 7 model architectures
---

⭐ If you find this project helpful, please consider giving it a star!

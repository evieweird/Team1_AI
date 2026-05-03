# Thyroid Nodule Segmentation: CNN vs ViT

Comparing **UNet++ (CNN)** and **SegFormer (ViT)** for ultrasound image segmentation, trained on thyroid nodule data and evaluated cross-domain on breast ultrasound.

## Overview

This project trains two segmentation architectures on the [DDTI Thyroid Ultrasound](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) dataset, then tests how well each generalizes to the [BUS-BRA Breast Ultrasound](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset) dataset — without any fine-tuning on breast data.

## Models

- **UNet++** with EfficientNet-B4 encoder (pretrained on ImageNet), binary output with BCEWithLogitsLoss + Dice loss
- **SegFormer** with MiT-B2 backbone, 2-class output with CrossEntropyLoss + Dice loss

Both use test-time augmentation (TTA) with horizontal and vertical flips for evaluation.

## Project Structure

```
├── Team1_AI.ipynb          # Full notebook (run in Google Colab)
└── README.md
```

## Notebook Structure

The Colab notebook is organized into 8 cells:

1. **Install libraries** — pip installs for transformers, albumentations, segmentation-models-pytorch
2. **Thyroid dataset preparation** — download from Kaggle, parse XML annotations, generate binary masks, train/val/test split
3. **Dataset & DataLoaders** — PyTorch Dataset class, augmentation pipelines (albumentations), DataLoader creation
4. **UNet++ (CNN)** — model definition, training loop, test evaluation with TTA, training curves, visual predictions
5. **SegFormer (ViT)** — model definition, training loop, test evaluation with TTA, training curves, visual predictions
6. **CNN vs ViT comparison** — side-by-side metrics table and bar chart
7. **Download breast dataset** — download BUS-BRA dataset from Kaggle
8. **Cross-domain evaluation** — apply both thyroid-trained models to breast ultrasound, metrics + visualizations for each

## Setup

### Prerequisites

- Google Colab (recommended, GPU required) or a local machine with CUDA
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Running the Notebook

1. Upload `Team1_AI.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Upload your `kaggle.json` to Colab when prompted
4. Run cells sequentially

## Requirements

```
torch
torchvision
segmentation-models-pytorch
transformers
albumentations
scikit-learn
pandas
pillow
matplotlib
tqdm
kaggle
```

## Datasets

- **DDTI Thyroid Ultrasound** — 466 usable images (560×360), resized to 384×576 for training. SVG polygon annotations converted to binary masks.
- **BUS-BRA Breast Ultrasound** — used for cross-domain generalization testing only (no training on this data).

## Key Details

- **Augmentations**: horizontal/vertical flip, rotation, affine transforms, elastic/grid distortion, brightness/contrast, Gaussian noise, coarse dropout
- **Optimizer**: AdamW with cosine annealing warm restarts
- **Early stopping**: patience of 15 epochs based on validation Dice score
- **Metrics**: Dice score and IoU, evaluated across multiple thresholds (0.30–0.70)

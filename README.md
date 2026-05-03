# Team1_AI
Team 1 AI of EEL5690 Introduction of Robotics - Ultrasound Thyroid Data Segmentation

Thyroid Nodule Segmentation: CNN vs ViT
Comparing UNet++ (CNN) and SegFormer (ViT) for ultrasound image segmentation, trained on thyroid nodule data and evaluated cross-domain on breast ultrasound.
Overview
This project trains two segmentation architectures on the DDTI Thyroid Ultrasound dataset, then tests how well each generalizes to the BUS-BRA Breast Ultrasound dataset — without any fine-tuning on breast data.

Models
UNet++ with EfficientNet-B4 encoder (pretrained on ImageNet), binary output with BCEWithLogitsLoss + Dice loss
SegFormer with MiT-B2 backbone, 2-class output with CrossEntropyLoss + Dice loss

Both use test-time augmentation (TTA) with horizontal and vertical flips for evaluation.
Project Structure
├── Team1_AI.ipynb          # Full notebook (run in Google Colab)
├── prepare_dataset.py      # Download and preprocess thyroid dataset
├── dataset.py              # UltrasoundDataset class + augmentations + get_loaders()
├── train_unet.py           # UNet++ training, evaluation, and visualization
├── train_segformer.py      # SegFormer training, evaluation, and visualization
├── apply_on_busbra.py      # Cross-domain evaluation on breast ultrasound
├── requirements.txt        # Python dependencies
└── README.md
Notebook Structure
The Colab notebook is organized into 8 cells:

Install libraries — pip installs for transformers, albumentations, segmentation-models-pytorch
Thyroid dataset preparation — download from Kaggle, parse XML annotations, generate binary masks, train/val/test split
Dataset & DataLoaders — PyTorch Dataset class, augmentation pipelines (albumentations), DataLoader creation
UNet++ (CNN) — model definition, training loop, test evaluation with TTA, training curves, visual predictions
SegFormer (ViT) — model definition, training loop, test evaluation with TTA, training curves, visual predictions
CNN vs ViT comparison — side-by-side metrics table and bar chart
Download breast dataset — download BUS-BRA dataset from Kaggle
Cross-domain evaluation — apply both thyroid-trained models to breast ultrasound, metrics + visualizations for each

Setup
Prerequisites

Google Colab (recommended, GPU required) or a local machine with CUDA
Kaggle API credentials (~/.kaggle/kaggle.json)

Running the Notebook

Upload Team1_AI.ipynb to Google Colab
Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
Upload your kaggle.json to Colab when prompted
Run cells sequentially

Running the Scripts
bashpip install -r requirements.txt

# 1. Prepare thyroid dataset
python prepare_dataset.py

# 2. Train models
python train_unet.py
python train_segformer.py

# 3. Cross-domain evaluation on breast data
python apply_on_busbra.py
Requirements
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
Datasets

DDTI Thyroid Ultrasound — 466 usable images (560×360), resized to 384×576 for training. SVG polygon annotations converted to binary masks.
BUS-BRA Breast Ultrasound — used for cross-domain generalization testing only (no training on this data).

Key Details

Augmentations: horizontal/vertical flip, rotation, affine transforms, elastic/grid distortion, brightness/contrast, Gaussian noise, coarse dropout
Optimizer: AdamW with cosine annealing warm restarts
Early stopping: patience of 15 epochs based on validation Dice score
Metrics: Dice score and IoU, evaluated across multiple thresholds (0.30–0.70)

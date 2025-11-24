# RetinaLiteNet Implementation

[![GitHub](https://img.shields.io/badge/GitHub-RetinalSegNet-blue?logo=github)](https://github.com/AbdullahButt-00/RetinalSegNet)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202024-red)](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/papers/Mehmood_RetinaLiteNet_A_Lightweight_Transformer_based_CNN_for_Retinal_Feature_Segmentation_CVPRW_2024_paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

RetinaLiteNet is a lightweight transformer-based CNN designed for retinal feature segmentation. This repository contains a complete reimplementation of the methodology described in the original paper, focusing on accurate and efficient segmentation of retinal blood vessels and the optic disc.

The model combines convolutional layers with multi-head self-attention in the encoder to capture both local and global features. A Convolutional Block Attention Module (CBAM) in the decoder helps refine segmentation outputs. This architecture achieves high accuracy while remaining computationally efficient.

**Original code reference:** [Mehwish4593/RetinaLiteNet](https://github.com/Mehwish4593/RetinaLiteNet)

**Original Paper:** [RetinaLiteNet: A Lightweight Transformer based CNN for Retinal Feature Segmentation](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/papers/Mehmood_RetinaLiteNet_A_Lightweight_Transformer_based_CNN_for_Retinal_Feature_Segmentation_CVPRW_2024_paper.pdf)

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Methodology & System Architecture](#methodology--system-architecture)
- [Implementation Details](#implementation-details)
- [Experiments & Results](#experiments--results)
- [Discussion & Limitations](#discussion--limitations)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

---

## Features

- **Efficient multitask segmentation** of retinal vessels and optic disc
- **Lightweight model**: small memory footprint with 57,810 parameters and low FLOPs
- **Attention mechanisms** for enhanced feature representation
- **Comprehensive evaluation** with multiple metrics (F1, IoU, Dice, Sensitivity, Specificity)
- **Interactive Streamlit application** for real-time inference
- **Complete training pipeline** with data augmentation and model checkpointing
- **Automatic weight management** with GitHub integration

---

## 📁 Repository Structure

```
RetinalSegNet/
│
├── APP/                                    # Streamlit Web Application
│   ├── app.py                             # Main application interface
│   ├── requirements.txt                   # Python dependencies for app
│   ├── Readme.md                          # App-specific documentation
│   ├── logs/                              # Application logs
│   │   └── inference_log.json            # Inference tracking logs
│   ├── utils/                             # Utility modules
│   │   ├── data_utils.py                 # Data loading and preprocessing
│   │   ├── inference_engine.py           # Model inference and metrics
│   │   ├── model_utils.py                # Model architecture utilities
│   │   └── weight_manager.py             # Model weight downloading
│   └── weights/                           # Pre-trained model weights
│       ├── model_weights_DRIVE.h5        # DRIVE dataset weights
│       ├── model_weights_IOSTAR.h5       # IOSTAR dataset weights
│       ├── metrics_drive_json.json       # DRIVE performance metrics
│       └── metrics_iostar_json.json      # IOSTAR performance metrics
│
├── dataset/                                # Dataset storage and processing
│   ├── data_raw/                          # Raw dataset files
│   │   ├── aria-hrf-iostar-data/         # IOSTAR dataset
│   │   │   └── resized-images/
│   │   │       └── IOSTAR/               # IOSTAR dataset
│   │   │           ├── Test/
│   │   │           │   ├── Images/
│   │   │           │   └── Labels/
│   │   │           └── Train/
│   │   │               ├── Images/
│   │   │               └── Labels/
│   │   └── DRIVE/                        # DRIVE dataset
│   │       ├── test/
│   │       │   ├── images/
│   │       │   └── mask/
│   │       └── training/
│   │           ├── 1st_manual/
│   │           ├── images/
│   │           └── mask/
│   └── drive_iostar_augmented/            # Augmented training data
│       ├── train/
│       │   ├── bv_masks/                 # Blood vessel masks
│       │   ├── images/                   # Augmented images
│       │   └── od_masks/                 # Optic disc masks
│       └── val/
│           ├── bv_masks/
│           ├── images/
│           └── od_masks/
│
├── Inference/                              # Inference scripts and weights
│   ├── inference.ipynb                    # Jupyter notebook for inference
│   ├── Model_weights_DRIVE.h5            # DRIVE model weights
│   └── Model_weights_IOSTAR.h5           # IOSTAR model weights
│
├── notebooks/                              # Jupyter notebooks for experiments
│   ├── 01_augmentation.ipynb             # Data augmentation pipeline
│   ├── 02_training_architecture.ipynb    # Model training and validation
│   ├── re_training_log.csv               # Training logs and metrics
│   └── re_training_model_weights.weights.h5  # Retrained model weights
│
└── README.md                               # This file
```

### 📂 Folder Descriptions

#### **APP/**
Contains the Streamlit web application for interactive retinal segmentation. Features include:
- Real-time image segmentation for blood vessels and optic disc
- Automatic model weight downloading from GitHub
- Performance metrics display and benchmarking
- Results saving with organized directory structure
- Logging system for tracking inferences

#### **dataset/**
Houses all dataset files:
- **data_raw/**: Original datasets (DRIVE, IOSTAR) with train/test splits
- **drive_iostar_augmented/**: Preprocessed and augmented data ready for training
- Images are resized to 512×512 for consistent processing

#### **Inference/**
Standalone inference resources:
- Pre-trained model weights for both DRIVE and IOSTAR datasets
- Jupyter notebook for batch inference and evaluation

#### **notebooks/**
Development and experimentation notebooks:
- `01_augmentation.ipynb`: Data preprocessing and augmentation pipeline
- `02_training_architecture.ipynb`: Complete training workflow with architecture details
- Training logs and checkpoint weights

---

## 📊 Datasets

### DRIVE (Digital Retinal Images for Vessel Extraction)
- **Task**: Blood vessel segmentation
- **Images**: 40 fundus images (20 training, 20 testing)
- **Resolution**: Resized to 512×512
- **Annotations**: Manual vessel segmentations

### IOSTAR (IOSTAR Vessel Segmentation Dataset)
- **Task**: Optic disc segmentation
- **Images**: 30 fundus images
- **Resolution**: Resized to 512×512
- **Annotations**: Optic disc ground truth masks

---

## Methodology & System Architecture

### Model Architecture

RetinaLiteNet employs a **U-Net-inspired encoder-decoder architecture** with transformer-based attention mechanisms:

#### **Encoder (Downsampling Path)**
1. **Convolutional Blocks**: Extract hierarchical features through progressive downsampling
2. **Multi-Head Self-Attention (MHSA)**: Captures global dependencies at multiple scales
3. **Skip Connections**: Preserve spatial information for decoder reconstruction

#### **Decoder (Upsampling Path)**
1. **Transposed Convolutions**: Gradually restore spatial resolution
2. **CBAM (Convolutional Block Attention Module)**: Refines features using channel and spatial attention
3. **Feature Fusion**: Combines encoder features via skip connections

#### **Multi-Task Learning**
- **Blood Vessel (BV) Branch**: Dedicated decoder for vessel segmentation
- **Optic Disc (OD) Branch**: Separate decoder for optic disc detection
- **Shared Encoder**: Efficient feature extraction for both tasks

### Key Components

**TransFuse Blocks**: Fuse transformer attention with CNN feature extraction
```
Input → Conv → BatchNorm → ReLU → MHSA → Conv → Output
```

**CBAM Module**: Sequential channel and spatial attention
```
Features → Channel Attention → Spatial Attention → Refined Features
```

### Loss Function
Combined loss for multi-task learning:
- **Binary Cross-Entropy (BCE)**: Pixel-wise classification
- **Dice Loss**: Handles class imbalance
- **Total Loss**: `L = α × L_BCE + β × L_Dice`

---

## Implementation Details

### Technology Stack
- **Framework**: TensorFlow 2.17.0 / Keras
- **Python**: 3.10+
- **GPU**: NVIDIA T4 (Google Colab) or local CUDA-enabled GPU
- **Web App**: Streamlit 1.35.0

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 512×512×3 |
| Batch Size | 8 |
| Epochs | 300 |
| Optimizer | Adam |
| Learning Rate | 1e-4 (with decay) |
| Data Augmentation | Rotation, flipping, scaling, brightness |
| Early Stopping | Patience: 10 epochs |

### Data Augmentation Pipeline
- **Geometric**: Random rotation (±15°), horizontal/vertical flips
- **Intensity**: Brightness adjustment, contrast variation
- **Scaling**: Random zoom (0.8-1.2×)
- **Elastic Deformation**: Subtle warping for anatomical variations

### Model Checkpointing
- Saves best model based on validation Dice score
- Automatic weight versioning with GitHub releases
- Checkpoint format: `.h5` (Keras HDF5)

---

## Experiments & Results

### Training Configuration

- **Training samples**: 640
- **Validation samples**: 160
- **Batch size**: 8
- **Total epochs**: 300
- **Steps per epoch**: 80

### Performance Metrics

#### DRIVE Dataset (Blood Vessel Segmentation)

| Metric | Paper (RetinaLiteNet) |
|--------|----------------------:|
| **F1-Score** | 80.6% |
| **IoU (Jaccard)** | 67.5% |
| **Sensitivity** | 78.4% |
| **Specificity** | **98.0%** |
| **AUC** | 97.0% |

#### IOSTAR Dataset (Optic Disc Segmentation)

| Metric | Paper (RetinaLiteNet) |
|--------|----------------------:|
| **F1-Score** | 93.3% |
| **IoU (Jaccard)** | 88.0% |
| **Sensitivity** | 94.0% |
| **Specificity** | **97.0%** |
| **AUC** | 99.0% |

#### Comparative Analysis with State-of-the-Art

Performance comparison of RetinaLiteNet with other architectures on DRIVE and IOSTAR datasets:

| Model | DRIVE (Blood Vessels) ||||||| IOSTAR (Optic Disc) ||||||
|-------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
|       | **F1** | **Jac.** | **Sen.** | **Spe.** | **AUC** | **F1** | **Jac.** | **Sen.** | **Spe.** | **AUC** |
| **UNet** | 80.5 | 67.5 | 87.5 | 95.7 | 98.0 | 90.9 | 83.6 | 85.3 | 99.7 | 98.0 |
| **UNet++** | 80.0 | 66.8 | 89.4 | 95.1 | 98.0 | 87.6 | 78.5 | 82.1 | 99.8 | 98.0 |
| **Att UNet** | 80.5 | 67.6 | 77.8 | 97.8 | 97.0 | 77.3 | 64.1 | 64.9 | 99.7 | 94.0 |
| **RetinaLiteNet (Ours)** | **80.6** | **67.5** | **78.4** | **98.0** | **97.0** | **93.3** | **88.0** | **94.0** | **97.0** | **99.0** |

**Note**: RetinaLiteNet achieves state-of-the-art performance on IOSTAR dataset (F1: 93.3%) while maintaining competitive results on DRIVE (F1: 80.6%). The model demonstrates excellent specificity across both datasets (98.0% and 97.0%), indicating strong performance in avoiding false positives.

### Model Efficiency

| Metric | Value |
|--------|-------|
| **Parameters** | 57,810 |
| **FLOPs** | 2.5961 GFLOPs |
| **Inference Time** | ~45ms (GPU) / ~180ms (CPU) |
| **Model Size** | 0.2205 MB (float32) |
| **Activation Memory** | 22.0009 MB |

---

## Discussion & Limitations

### Strengths
1. **Lightweight Architecture**: Suitable for deployment on resource-constrained devices
2. **Multi-Task Learning**: Simultaneous vessel and optic disc segmentation without parameter explosion
3. **Attention Mechanisms**: Effective global context modeling for fine retinal structures
4. **High Specificity**: Excellent at avoiding false positives (98%+ specificity)
5. **Reproducible Pipeline**: Complete codebase from preprocessing to inference

### Limitations

#### 1. Dataset Size
- DRIVE (40 images) and IOSTAR (30 images) are relatively small
- **Impact**: May limit generalization to diverse retinal conditions
- **Mitigation**: Extensive data augmentation, transfer learning potential

#### 2. Thin Vessel Detection
- Struggles with fine capillary vessels (<2 pixels wide)
- **Cause**: Downsampling in encoder loses fine spatial details
- **Future Work**: Multi-scale feature fusion, higher resolution training

#### 3. Pathological Cases
- Limited exposure to diseased retinas (diabetic retinopathy, glaucoma)
- **Solution**: Incorporate pathological datasets (e.g., STARE, CHASE_DB1)

#### 4. Computational Cost
- Transformer attention layers increase FLOPs vs. pure CNNs
- **Trade-off**: Accuracy gains justify moderate compute overhead

#### 5. Domain Shift
- Performance degrades on datasets with different acquisition protocols
- **Observation**: Color tone, contrast, and resolution variations affect robustness
- **Remedy**: Domain adaptation techniques, normalization strategies

#### 6. Real-Time Inference
- ~180ms CPU inference may be slow for real-time clinical workflows
- **Recommendation**: GPU deployment or model quantization for speedup

### Future Directions
- **Cross-Dataset Validation**: Test on STARE, CHASE_DB1, HRF
- **3D Extensions**: Extend to OCT (Optical Coherence Tomography) volumes
- **Mobile Deployment**: Optimize for TensorFlow Lite / ONNX
- **Semi-Supervised Learning**: Leverage unlabeled retinal images
- **Clinical Integration**: Validate in real-world ophthalmology settings

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for training

### Clone Repository
```bash
git clone https://github.com/AbdullahButt-00/RetinalSegNet.git
cd RetinalSegNet
```

### Install Dependencies

#### For Streamlit Application
```bash
cd APP
pip install -r requirements.txt
```

#### For Training/Notebooks
```bash
pip install tensorflow==2.17.0 keras numpy pandas matplotlib scikit-learn opencv-python pillow jupyter
```

---

## Usage

### 1. Streamlit Web Application

**Start the app:**
```bash
cd APP
streamlit run app.py
```

**Features:**
- Upload retinal fundus images
- Select dataset model (DRIVE/IOSTAR)
- View segmentation results with metrics
- Download predictions and visualizations

**Automatic Weight Download:**
Models are automatically fetched from GitHub releases on first run.

### 2. Training Pipeline

**Open training notebook:**
```bash
jupyter notebook notebooks/02_training_architecture.ipynb
```

**Steps:**
1. Load and preprocess datasets
2. Configure model hyperparameters
3. Train with data augmentation
4. Evaluate on test set
5. Save trained weights

### 3. Batch Inference

**Use inference notebook:**
```bash
jupyter notebook Inference/inference.ipynb
```

Load pre-trained weights and process multiple images programmatically.

---

## Contributors

- **Abdullah Butt** - [@AbdullahButt-00](https://github.com/AbdullahButt-00)
- **Sami Naeem** - [@itsami12](https://github.com/itsami12)
- **Hussain Ahmad** - [@AIStrikerX](https://github.com/AIStrikerX)
- **Haseeb Ali** - [@Haseeb98-Git](https://github.com/Haseeb98-Git)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Original RetinaLiteNet paper authors
- [Mehwish4593/RetinaLiteNet](https://github.com/Mehwish4593/RetinaLiteNet) for reference implementation
- DRIVE and IOSTAR dataset providers
- TensorFlow and Keras communities

---

## Contact

For questions or collaboration:
- **GitHub Issues**: [RetinalSegNet Issues](https://github.com/AbdullahButt-00/RetinalSegNet/issues)
- **Repository**: [https://github.com/AbdullahButt-00/RetinalSegNet](https://github.com/AbdullahButt-00/RetinalSegNet)

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{mehmood2024retinalitenet,
  title={RetinaLiteNet: A Lightweight Transformer based CNN for Retinal Feature Segmentation},
  author={Mehmood, Mehwish and others},
  booktitle={CVPR 2024 Workshop},
  year={2024}
}
```

# Dataset Directory

This directory contains all datasets used for training and evaluating the RetinaLiteNet model for retinal feature segmentation.

## 📁 Structure

```
dataset/
├── data_raw/                    # Raw, unprocessed datasets
│   ├── aria-hrf-iostar-data/   # ARIA, HRF, and IOSTAR datasets
│   └── DRIVE/                   # DRIVE dataset
└── drive_iostar_augmented/      # Preprocessed and augmented data
    ├── train/                   # Training data
    └── val/                     # Validation data
```

---

## 📊 Datasets Overview

### 1. DRIVE (Digital Retinal Images for Vessel Extraction)

**Purpose**: Blood vessel segmentation  
**Location**: `data_raw/DRIVE/`

**Structure**:
```
DRIVE/
├── test/
│   ├── images/        # 20 test fundus images
│   └── mask/          # Field of view masks
└── training/
    ├── images/        # 20 training fundus images
    ├── 1st_manual/    # Manual vessel annotations
    └── mask/          # Field of view masks
```

**Specifications**:
- **Total Images**: 40 (20 train, 20 test)
- **Original Resolution**: 565×584 pixels
- **Processing**: Resized to 512×512
- **Format**: .tif or .png
- **Annotations**: Binary vessel masks

**Download**: [DRIVE Dataset](https://drive.grand-challenge.org/)

---

### 2. IOSTAR (IOSTAR Vessel Segmentation)

**Purpose**: Optic disc segmentation  
**Location**: `data_raw/aria-hrf-iostar-data/resized-images/IOSTAR/`

**Structure**:
```
IOSTAR/
├── Test/
│   ├── Images/        # Test fundus images
│   └── Labels/        # Ground truth optic disc masks
└── Train/
    ├── Images/        # Training fundus images
    └── Labels/        # Ground truth optic disc masks
```

**Specifications**:
- **Total Images**: 30
- **Resolution**: Resized to 512×512
- **Format**: .jpg/.png
- **Annotations**: Optic disc binary masks

**Download**: [IOSTAR Dataset](http://www.retinacheck.org/datasets)

---

### 3. ARIA (Automated Retinal Image Analysis)

**Purpose**: Additional training data for vessel segmentation  
**Location**: `data_raw/aria-hrf-iostar-data/resized-images/ARIA/`

**Structure**:
```
ARIA/
├── Test/
│   ├── Images/
│   └── Labels/
└── Train/
    ├── Images/
    └── Labels/
```

**Specifications**:
- **Resolution**: Resized to 512×512
- **Format**: .png
- **Use Case**: Cross-dataset validation

---

### 4. HRF (High-Resolution Fundus)

**Purpose**: High-quality vessel annotations  
**Location**: `data_raw/aria-hrf-iostar-data/resized-images/HRF/`

**Structure**:
```
HRF/
├── Test/
│   ├── Images/
│   └── Labels/
└── Train/
    ├── Images/
    └── Labels/
```

**Specifications**:
- **Resolution**: High-resolution (resized to 512×512 for compatibility)
- **Quality**: Professional-grade annotations
- **Format**: .png

---

## 🔄 Augmented Data

### DRIVE + IOSTAR Augmented

**Location**: `drive_iostar_augmented/`

**Purpose**: Pre-augmented training and validation data ready for model training.

**Structure**:
```
drive_iostar_augmented/
├── train/
│   ├── images/        # Augmented training images
│   ├── bv_masks/      # Blood vessel segmentation masks
│   └── od_masks/      # Optic disc segmentation masks
└── val/
    ├── images/        # Validation images
    ├── bv_masks/      # Validation BV masks
    └── od_masks/      # Validation OD masks
```

**Augmentation Techniques**:
- ✅ Random rotation (±15°)
- ✅ Horizontal and vertical flipping
- ✅ Brightness and contrast adjustment
- ✅ Random scaling (0.8-1.2×)
- ✅ Elastic deformation
- ✅ Gaussian noise addition

**Statistics**:
- **Training Samples**: ~500-1000 augmented images
- **Validation Samples**: ~100-200 images
- **Image Size**: 512×512×3
- **Mask Type**: Binary (0 or 255)

**Generation**: See `notebooks/01_augmentation.ipynb`

---

## 📥 Dataset Preparation

### Step 1: Download Raw Datasets

Download the following datasets and place them in the appropriate directories:

1. **DRIVE**: Place in `data_raw/DRIVE/`
2. **IOSTAR**: Place in `data_raw/aria-hrf-iostar-data/resized-images/IOSTAR/`
3. **ARIA** (optional): Place in corresponding directory
4. **HRF** (optional): Place in corresponding directory

### Step 2: Run Augmentation Pipeline

Navigate to the notebooks folder and execute:

```bash
jupyter notebook notebooks/01_augmentation.ipynb
```

This will:
- Load raw images from `data_raw/`
- Apply augmentation transformations
- Save augmented data to `drive_iostar_augmented/`
- Split into train/validation sets

### Step 3: Verify Data

Check that all directories contain the expected files:

```bash
# Check training data
ls drive_iostar_augmented/train/images/
ls drive_iostar_augmented/train/bv_masks/
ls drive_iostar_augmented/train/od_masks/

# Check validation data
ls drive_iostar_augmented/val/images/
ls drive_iostar_augmented/val/bv_masks/
ls drive_iostar_augmented/val/od_masks/
```

---

## 🛠️ Data Loading

### Python Example

```python
import os
import numpy as np
from PIL import Image

def load_data(base_path, subset='train'):
    """Load augmented data for training/validation"""
    
    images_path = os.path.join(base_path, subset, 'images')
    bv_masks_path = os.path.join(base_path, subset, 'bv_masks')
    od_masks_path = os.path.join(base_path, subset, 'od_masks')
    
    images = []
    bv_masks = []
    od_masks = []
    
    for img_file in sorted(os.listdir(images_path)):
        # Load image
        img = Image.open(os.path.join(images_path, img_file))
        images.append(np.array(img))
        
        # Load corresponding masks
        bv_mask = Image.open(os.path.join(bv_masks_path, img_file))
        bv_masks.append(np.array(bv_mask))
        
        od_mask = Image.open(os.path.join(od_masks_path, img_file))
        od_masks.append(np.array(od_mask))
    
    return np.array(images), np.array(bv_masks), np.array(od_masks)

# Usage
train_images, train_bv, train_od = load_data('drive_iostar_augmented', 'train')
val_images, val_bv, val_od = load_data('drive_iostar_augmented', 'val')
```

---

## 📊 Dataset Statistics

| Dataset | Task | Train | Test | Total | Resolution |
|---------|------|-------|------|-------|------------|
| DRIVE | Blood Vessels | 20 | 20 | 40 | 512×512 |
| IOSTAR | Optic Disc | ~24 | ~6 | 30 | 512×512 |
| ARIA | Blood Vessels | Variable | Variable | - | 512×512 |
| HRF | Blood Vessels | Variable | Variable | - | 512×512 |

**After Augmentation**:
- **Training Images**: ~800-1200
- **Validation Images**: ~150-250

---

## ⚠️ Important Notes

1. **Mask Format**: All masks are binary (0 = background, 255 = foreground)
2. **Normalization**: Images should be normalized to [0, 1] before training
3. **Color Space**: RGB (3 channels)
4. **File Naming**: Maintain consistent naming between images and masks
5. **Data Balance**: Augmentation helps balance the limited dataset size

---

## 📚 References

- **DRIVE**: Staal et al., "Ridge-based vessel segmentation in color images of the retina"
- **IOSTAR**: Zhang et al., "IOSTAR vessel segmentation database"
- **ARIA**: Retinal Image Analysis Group
- **HRF**: Budai et al., "Robust Vessel Segmentation in Fundus Images"

---

## 📄 License

Datasets have their own licenses. Please refer to the original dataset providers for usage terms.

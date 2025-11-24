# Notebooks Directory

This directory contains Jupyter notebooks for data preprocessing, model training, and experimentation with the RetinaLiteNet architecture.

## 📁 Contents

```
notebooks/
├── 01_augmentation.ipynb                    # Data augmentation pipeline
├── 02_training_architecture.ipynb           # Model training and architecture
├── re_training_log.csv                      # Training metrics and logs
└── re_training_model_weights.weights.h5     # Saved model checkpoints
```

---

## 📓 Notebook Descriptions

### 1. `01_augmentation.ipynb` - Data Augmentation Pipeline

**Purpose**: Prepare and augment raw datasets for training

**Key Functions**:
- Load raw images from DRIVE and IOSTAR datasets
- Apply data augmentation techniques:
  - Random rotation (±15°)
  - Horizontal/vertical flips
  - Brightness and contrast adjustment
  - Scaling and zooming
  - Elastic deformation
- Split data into train/validation sets
- Save augmented data to `dataset/drive_iostar_augmented/`

**Outputs**:
- Augmented training images
- Blood vessel (BV) masks
- Optic disc (OD) masks
- Data statistics and visualizations

**Usage**:
```bash
jupyter notebook 01_augmentation.ipynb
```

**Expected Results**:
- ~800-1200 training samples (from ~40-50 original images)
- ~150-250 validation samples
- All images resized to 512×512

---

### 2. `02_training_architecture.ipynb` - Complete Training Pipeline

**Purpose**: Train RetinaLiteNet model on augmented datasets

**Architecture Overview**:
- **Model**: TransFuse-based U-Net with CBAM attention
- **Encoder**: Convolutional blocks + Multi-Head Self-Attention (MHSA)
- **Decoder**: Transposed convolutions + CBAM refinement
- **Multi-Task**: Simultaneous BV and OD segmentation

**Key Sections**:

#### 1. **Environment Setup**
```python
# TensorFlow 2.17.0
# CUDA-enabled GPU (T4 recommended)
# Memory: ~12GB GPU RAM
```

#### 2. **Data Loading**
- Load augmented data from `drive_iostar_augmented/`
- Normalize images to [0, 1]
- Prepare multi-task labels (BV + OD)

#### 3. **Model Architecture**
- **Parameters**: ~2.6M
- **Input**: 512×512×3
- **Outputs**: 
  - Blood vessel mask (512×512×1)
  - Optic disc mask (512×512×1)

#### 4. **Training Configuration**
| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Epochs | 50-100 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss | BCE + Dice |
| Metrics | F1, IoU, Dice, Sensitivity, Specificity |

#### 5. **Callbacks**
- **ModelCheckpoint**: Save best model based on validation Dice
- **EarlyStopping**: Patience of 10 epochs
- **ReduceLROnPlateau**: Reduce learning rate when plateauing
- **TensorBoard**: Log training metrics

#### 6. **Evaluation**
- Test on DRIVE dataset (blood vessels)
- Test on IOSTAR dataset (optic disc)
- Generate performance metrics table
- Visualize predictions vs. ground truth

**Outputs**:
- `re_training_model_weights.weights.h5` - Trained model weights
- `re_training_log.csv` - Epoch-wise training logs
- Performance metrics (F1, IoU, Dice, Sensitivity, Specificity)

**Usage**:
```bash
jupyter notebook 02_training_architecture.ipynb
```

**Expected Training Time**:
- **DRIVE (50 epochs)**: ~2-3 hours (T4 GPU)
- **IOSTAR (50 epochs)**: ~1.5-2 hours (T4 GPU)

---

## 📊 Training Logs

### `re_training_log.csv`

**Contents**: Detailed epoch-wise metrics

| Column | Description |
|--------|-------------|
| `epoch` | Training epoch number |
| `loss` | Combined training loss (BCE + Dice) |
| `bv_dice` | Dice score for blood vessel segmentation |
| `od_dice` | Dice score for optic disc segmentation |
| `val_loss` | Validation loss |
| `val_bv_dice` | Validation BV Dice score |
| `val_od_dice` | Validation OD Dice score |
| `learning_rate` | Current learning rate |

**Sample**:
```csv
epoch,loss,bv_dice,od_dice,val_loss,val_bv_dice,val_od_dice,learning_rate
1,0.3245,0.6532,0.7821,0.2981,0.6890,0.8102,0.0001
2,0.2876,0.7123,0.8345,0.2654,0.7234,0.8456,0.0001
...
50,0.1234,0.8156,0.9287,0.1456,0.8023,0.9201,0.00001
```

**Visualization**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load logs
logs = pd.read_csv('re_training_log.csv')

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(logs['loss'], label='Train Loss')
plt.plot(logs['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 3, 2)
plt.plot(logs['bv_dice'], label='Train BV Dice')
plt.plot(logs['val_bv_dice'], label='Val BV Dice')
plt.legend()
plt.title('Blood Vessel Dice')

plt.subplot(1, 3, 3)
plt.plot(logs['od_dice'], label='Train OD Dice')
plt.plot(logs['val_od_dice'], label='Val OD Dice')
plt.legend()
plt.title('Optic Disc Dice')

plt.tight_layout()
plt.show()
```

---

## 💾 Model Weights

### `re_training_model_weights.weights.h5`

**Description**: Saved model checkpoint from training

**Format**: Keras HDF5 format (.h5)

**Size**: ~31 MB

**Loading Weights**:
```python
from tensorflow import keras

# Load architecture (define in notebook)
model = build_retinalitenet_model(input_shape=(512, 512, 3))

# Load trained weights
model.load_weights('re_training_model_weights.weights.h5')

# Ready for inference
predictions = model.predict(test_images)
```

**Transfer to Production**:
```bash
# Copy to APP/weights/ for Streamlit app
cp re_training_model_weights.weights.h5 ../APP/weights/model_weights_DRIVE.h5
```

---

## 🚀 Running the Notebooks

### Prerequisites

**Install Dependencies**:
```bash
pip install tensorflow==2.17.0 keras numpy pandas matplotlib scikit-learn opencv-python pillow jupyter
```

**GPU Setup** (recommended):
```bash
# Verify CUDA availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Execution Order

1. **First Time Setup**:
   ```bash
   # 1. Augment data
   jupyter notebook 01_augmentation.ipynb
   # Run all cells
   
   # 2. Train model
   jupyter notebook 02_training_architecture.ipynb
   # Run all cells
   ```

2. **Re-training/Fine-tuning**:
   ```bash
   # Open training notebook
   jupyter notebook 02_training_architecture.ipynb
   
   # Load existing weights and continue training
   ```

### Google Colab

**Run on Colab** (free T4 GPU):

1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Enable GPU: `Runtime > Change runtime type > T4 GPU`
4. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Update paths to your Drive location

---

## 📈 Expected Results

### DRIVE Dataset (Blood Vessels)

| Metric | Target |
|--------|--------|
| F1-Score | 79-82% |
| Dice Coefficient | 79-82% |
| IoU | 66-70% |
| Sensitivity | 76-80% |
| Specificity | 97-99% |

### IOSTAR Dataset (Optic Disc)

| Metric | Target |
|--------|--------|
| F1-Score | 91-94% |
| Dice Coefficient | 91-94% |
| IoU | 85-89% |
| Sensitivity | 90-93% |
| Specificity | 97-99% |

---

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size in 02_training_architecture.ipynb
BATCH_SIZE = 4  # Instead of 8
```

**2. Slow Training on CPU**
```python
# Use mixed precision (GPU only)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

**3. Data Not Found**
```python
# Verify paths in notebooks
import os
print(os.path.exists('../dataset/drive_iostar_augmented/train/images/'))
```

**4. TensorFlow Version Issues**
```bash
# Ensure correct version
pip install tensorflow==2.17.0 --upgrade
```

---

## 📚 Additional Resources

- **TensorFlow Documentation**: [tensorflow.org](https://www.tensorflow.org/)
- **Keras API**: [keras.io](https://keras.io/)
- **Original Paper**: [RetinaLiteNet CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024W/WiCV/papers/Mehmood_RetinaLiteNet_A_Lightweight_Transformer_based_CNN_for_Retinal_Feature_Segmentation_CVPRW_2024_paper.pdf)

---

## 📝 Notes

- **GPU Memory**: Training requires ~8-12GB GPU RAM
- **Training Time**: 2-5 hours depending on dataset and GPU
- **Checkpointing**: Best model saved automatically during training
- **Reproducibility**: Set random seeds for consistent results

---

## 🤝 Contributing

Found a bug or have improvements? Please:
1. Document the issue in the notebook
2. Create a pull request with fixes
3. Update this README accordingly

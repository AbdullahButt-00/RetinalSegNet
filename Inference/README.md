# Inference Directory

This directory contains resources for running inference with pre-trained RetinaLiteNet models for retinal segmentation.

## 📁 Contents

```
Inference/
├── inference.ipynb              # Jupyter notebook for inference
├── Model_weights_DRIVE.h5       # Pre-trained weights for DRIVE dataset (BV)
└── Model_weights_IOSTAR.h5      # Pre-trained weights for IOSTAR dataset (OD)
```

---

## 🎯 Purpose

Run batch inference on retinal fundus images using pre-trained models without needing to retrain. Supports:
- **Blood Vessel (BV) Segmentation** using DRIVE weights
- **Optic Disc (OD) Segmentation** using IOSTAR weights

---

## 📓 Inference Notebook

### `inference.ipynb`

**Features**:
- Load pre-trained model weights
- Process single or multiple images
- Generate segmentation masks
- Calculate performance metrics (if ground truth available)
- Visualize results with overlays

**Key Functions**:

#### 1. **Load Model**
```python
from tensorflow import keras

# Define RetinaLiteNet architecture
model = build_retinalitenet_model(input_shape=(512, 512, 3))

# Load pre-trained weights
model.load_weights('Model_weights_DRIVE.h5')  # For blood vessels
# OR
model.load_weights('Model_weights_IOSTAR.h5')  # For optic disc
```

#### 2. **Preprocess Image**
```python
import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 512x512
    img_resized = cv2.resize(img, (512, 512))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized
```

#### 3. **Run Inference**
```python
# Predict
predictions = model.predict(img_batch)

# For multi-task model
bv_mask = predictions[0]  # Blood vessel mask
od_mask = predictions[1]  # Optic disc mask

# Threshold predictions
bv_binary = (bv_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
od_binary = (od_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
```

#### 4. **Visualize Results**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(img_resized)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Blood vessel segmentation
axes[1].imshow(bv_binary, cmap='gray')
axes[1].set_title('Blood Vessels')
axes[1].axis('off')

# Optic disc segmentation
axes[2].imshow(od_binary, cmap='gray')
axes[2].set_title('Optic Disc')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

#### 5. **Calculate Metrics** (if ground truth available)
```python
from sklearn.metrics import f1_score, jaccard_score

def calculate_metrics(pred_mask, true_mask):
    pred_flat = pred_mask.flatten() / 255
    true_flat = true_mask.flatten() / 255
    
    f1 = f1_score(true_flat, pred_flat)
    iou = jaccard_score(true_flat, pred_flat)
    dice = 2 * iou / (1 + iou)
    
    return {
        'F1-Score': f1,
        'IoU': iou,
        'Dice': dice
    }

# Usage
metrics = calculate_metrics(bv_binary, ground_truth_bv)
print(metrics)
```

---

## 💾 Pre-trained Weights

### `Model_weights_DRIVE.h5`

**Task**: Blood vessel segmentation  
**Dataset**: Trained on DRIVE dataset  
**Size**: ~31 MB  
**Format**: Keras HDF5

**Performance**:
- F1-Score: ~79-82%
- Sensitivity: ~76-80%
- Specificity: ~97-99%
- IoU: ~66-70%

**Use Case**:
- Segment retinal blood vessels
- Diabetic retinopathy screening
- Vessel density analysis

---

### `Model_weights_IOSTAR.h5`

**Task**: Optic disc segmentation  
**Dataset**: Trained on IOSTAR dataset  
**Size**: ~31 MB  
**Format**: Keras HDF5

**Performance**:
- F1-Score: ~91-94%
- Sensitivity: ~90-93%
- Specificity: ~97-99%
- IoU: ~85-89%

**Use Case**:
- Locate optic disc for glaucoma assessment
- Cup-to-disc ratio calculation
- Anatomical landmark detection

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow==2.17.0 keras numpy opencv-python pillow matplotlib scikit-learn jupyter
```

### 2. Launch Notebook

```bash
cd Inference
jupyter notebook inference.ipynb
```

### 3. Run Inference

**Example Code in Notebook**:
```python
# Load DRIVE model for blood vessels
model_bv = build_retinalitenet_model((512, 512, 3))
model_bv.load_weights('Model_weights_DRIVE.h5')

# Load test image
test_image = 'path/to/test_image.jpg'
img_batch, img_display = preprocess_image(test_image)

# Predict
bv_prediction = model_bv.predict(img_batch)
bv_mask = (bv_prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255

# Display
plt.imshow(bv_mask, cmap='gray')
plt.title('Blood Vessel Segmentation')
plt.show()
```

---

## 📊 Batch Processing

### Process Multiple Images

```python
import os
from tqdm import tqdm

# Directory with test images
test_dir = '../dataset/data_raw/DRIVE/test/images/'
output_dir = './predictions/'
os.makedirs(output_dir, exist_ok=True)

# Load model once
model = build_retinalitenet_model((512, 512, 3))
model.load_weights('Model_weights_DRIVE.h5')

# Process all images
for img_file in tqdm(os.listdir(test_dir)):
    if img_file.endswith(('.jpg', '.png', '.tif')):
        # Load and preprocess
        img_path = os.path.join(test_dir, img_file)
        img_batch, _ = preprocess_image(img_path)
        
        # Predict
        prediction = model.predict(img_batch, verbose=0)
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        # Save
        output_path = os.path.join(output_dir, f'pred_{img_file}')
        cv2.imwrite(output_path, mask)

print(f"Saved {len(os.listdir(output_dir))} predictions to {output_dir}")
```

---

## 🎨 Visualization Options

### 1. Overlay Mask on Image

```python
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay binary mask on RGB image"""
    overlay = image.copy()
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_colored = np.zeros_like(mask_rgb)
    mask_colored[mask > 0] = color
    
    overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
    return overlay

# Usage
overlay_img = overlay_mask(img_resized, bv_binary, color=(255, 0, 0), alpha=0.4)
plt.imshow(overlay_img)
plt.title('Blood Vessels Overlay')
plt.show()
```

### 2. Side-by-Side Comparison

```python
def compare_results(original, bv_mask, od_mask):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(bv_mask, cmap='gray')
    axes[0, 1].set_title('Blood Vessels')
    
    axes[1, 0].imshow(od_mask, cmap='gray')
    axes[1, 0].set_title('Optic Disc')
    
    # Combined overlay
    overlay = original.copy()
    overlay[bv_mask > 0] = [255, 0, 0]  # Red for vessels
    overlay[od_mask > 0] = [0, 255, 0]   # Green for OD
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Combined Overlay')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## 📈 Performance Evaluation

### Evaluate on Test Set

```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_dataset(model, test_images_dir, test_masks_dir):
    results = {
        'f1_scores': [],
        'iou_scores': [],
        'dice_scores': [],
        'sensitivities': [],
        'specificities': []
    }
    
    for img_file in os.listdir(test_images_dir):
        # Load image and ground truth
        img_path = os.path.join(test_images_dir, img_file)
        mask_path = os.path.join(test_masks_dir, img_file)
        
        img_batch, _ = preprocess_image(img_path)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (512, 512))
        
        # Predict
        pred = model.predict(img_batch, verbose=0)
        pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        # Calculate metrics
        metrics = calculate_all_metrics(pred_mask, true_mask)
        
        results['f1_scores'].append(metrics['f1'])
        results['iou_scores'].append(metrics['iou'])
        results['dice_scores'].append(metrics['dice'])
        results['sensitivities'].append(metrics['sensitivity'])
        results['specificities'].append(metrics['specificity'])
    
    # Average results
    avg_results = {k: np.mean(v) for k, v in results.items()}
    return avg_results

# Usage
avg_metrics = evaluate_dataset(model, test_images_dir, test_masks_dir)
print("Average Performance:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")
```

---

## 🔧 Customization

### Adjust Threshold

```python
# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 4))

for i, thresh in enumerate(thresholds):
    mask = (prediction[0, :, :, 0] > thresh).astype(np.uint8) * 255
    axes[i].imshow(mask, cmap='gray')
    axes[i].set_title(f'Threshold: {thresh}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### Post-processing

```python
from scipy.ndimage import binary_opening, binary_closing

def postprocess_mask(mask, kernel_size=3):
    """Remove noise and fill holes"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Morphological opening (remove small noise)
    mask_clean = binary_opening(mask > 0, structure=kernel)
    
    # Morphological closing (fill small holes)
    mask_clean = binary_closing(mask_clean, structure=kernel)
    
    return (mask_clean.astype(np.uint8) * 255)

# Usage
clean_mask = postprocess_mask(bv_binary, kernel_size=3)
```

---

## ⚠️ Important Notes

1. **Input Size**: All images must be resized to **512×512** before inference
2. **Normalization**: Images should be normalized to **[0, 1]**
3. **Color Space**: Use **RGB** format (convert from BGR if using OpenCV)
4. **Threshold**: Default threshold is **0.5** (adjust based on use case)
5. **GPU**: Inference is faster on GPU but works on CPU

---

## 🛠️ Troubleshooting

### Issue 1: Model Architecture Mismatch
```python
# Ensure architecture definition matches training notebook
# Copy model definition from 02_training_architecture.ipynb
```

### Issue 2: Weight File Not Found
```bash
# Download from GitHub releases or copy from APP/weights/
cp ../APP/weights/model_weights_DRIVE.h5 .
```

### Issue 3: Out of Memory
```python
# Process images one at a time instead of batching
# Reduce image size temporarily for testing
```

---

## 📚 References

- Training notebook: `../notebooks/02_training_architecture.ipynb`
- Streamlit app: `../APP/app.py`
- Main README: `../README.md`

---

## 📄 License

Same as main project (MIT License)

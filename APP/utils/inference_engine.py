import numpy as np
import tensorflow as tf
import time
import logging
from sklearn.metrics import confusion_matrix, precision_recall_curve
from .model_utils import (
    create_transfuse_net, bcc_Jaccard_coef_loss,
    dice_coef, iou, sensitivity, specificity
)

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Handles model loading and inference"""
    
    def __init__(self, model_weight_path, input_size=512, key_dim=16, dataset_type='DRIVE'):
        self.model_weight_path = model_weight_path
        self.input_size = input_size
        self.key_dim = key_dim
        self.dataset_type = dataset_type
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load and compile model"""
        try:
            logger.info(f"Creating model with input size {self.input_size}x{self.input_size}")
            self.model = create_transfuse_net((self.input_size, self.input_size, 3), key_dim=self.key_dim)
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss={'final_output1': bcc_Jaccard_coef_loss, 'final_output2': bcc_Jaccard_coef_loss},
                metrics={'final_output1': [dice_coef, iou, sensitivity, specificity],
                         'final_output2': [dice_coef, iou, sensitivity, specificity]}
            )
            
            logger.info(f"Loading weights from {self.model_weight_path}")
            self.model.load_weights(self.model_weight_path)
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, image_array):
        """Run inference on image"""
        try:
            start_time = time.time()
            
            # Ensure correct shape
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Run prediction
            predictions = self.model.predict(image_array, verbose=0)
            
            inference_time = time.time() - start_time
            
            return {
                'bv_prediction': predictions[0][0],  # Blood Vessel
                'od_prediction': predictions[1][0],  # Optic Disc
                'inference_time': inference_time
            }
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise


class MetricsCalculator:
    """Calculates segmentation metrics"""
    
    @staticmethod
    def calculate_metrics(y_pred, threshold=0.5):
        """Calculate prediction quality metrics"""
        try:
            # Flatten if needed
            if isinstance(y_pred, np.ndarray):
                pred = y_pred.flatten()
            else:
                pred = np.array(y_pred).flatten()
            
            # Ensure values are in valid range
            pred = np.clip(pred, 0, 1)
            
            if len(pred) == 0:
                return None
            
            # Binarize prediction
            pred_binary = (pred >= threshold).astype(int)
            
            # Calculate distribution-based metrics
            mean_confidence = float(np.mean(pred))
            std_confidence = float(np.std(pred))
            
            # Calculate proportion of foreground vs background
            foreground_ratio = float(np.mean(pred_binary))
            
            # Calculate edge intensity (useful for segmentation quality)
            if len(pred) > 1:
                edges = np.abs(np.diff(pred))
                edge_intensity = float(np.mean(edges))
            else:
                edge_intensity = 0.0
            
            # Calculate connected component analysis metrics
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(pred_binary)
            
            # Calculate metrics
            metrics = {
                'mean_confidence': mean_confidence,
                'std_confidence': std_confidence,
                'foreground_ratio': foreground_ratio,
                'edge_intensity': edge_intensity,
                'num_components': float(num_features),
                'threshold_used': float(threshold),
                'min_value': float(np.min(pred)),
                'max_value': float(np.max(pred))
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None
    
    @staticmethod
    def evaluate_batch(y_test, y_pred):
        """Evaluate batch of predictions"""
        n = y_pred.shape[0]
        all_metrics = []
        
        for i in range(n):
            metrics = MetricsCalculator.calculate_metrics(y_test[i], y_pred[i])
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return None
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics if key in m])
            for key in all_metrics[0].keys()
            if key != 'optimal_threshold'
        }
        
        return avg_metrics
import os
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data directories and logging"""
    
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.weights_dir = os.path.join(base_dir, "weights")
        self.uploads_dir = os.path.join(base_dir, "data", "uploads")
        self.results_dir = os.path.join(base_dir, "data", "results")
        self.logs_dir = os.path.join(base_dir, "logs")
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.weights_dir, self.uploads_dir, self.results_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ready: {directory}")
    
    def get_results_path(self):
        """Get path to save results"""
        return self.results_dir
    
    def get_weights_path(self, dataset_type="DRIVE"):
        """Get path to model weights"""
        weight_file = f"model_weights_{dataset_type}.h5"
        return os.path.join(self.weights_dir, weight_file)
    
    def get_log_path(self):
        """Get path to logs directory"""
        return self.logs_dir
    
    def save_result_metadata(self, filename, metrics, dataset_type, inference_time):
        """Save inference results metadata"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "dataset_type": dataset_type,
            "inference_time": inference_time,
            "metrics": metrics
        }
        
        log_file = os.path.join(self.logs_dir, "inference_log.json")
        
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(metadata)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Metadata saved for {filename}")


class ResultsSaver:
    """Handles saving and managing inference results"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def save_prediction(self, original_image, prediction_bv, prediction_od, filename, dataset_type):
        """Save prediction results"""
        import cv2
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = os.path.join(self.results_dir, f"{timestamp}_{dataset_type}")
        os.makedirs(result_folder, exist_ok=True)
        
        # Save original image
        if len(original_image.shape) == 3:
            cv2.imwrite(
                os.path.join(result_folder, f"{filename}_original.png"),
                cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
        
        # Save predictions
        cv2.imwrite(
            os.path.join(result_folder, f"{filename}_BV_prediction.png"),
            (prediction_bv * 255).astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(result_folder, f"{filename}_OD_prediction.png"),
            (prediction_od * 255).astype(np.uint8)
        )
        
        return result_folder
    
    def get_recent_results(self, limit=5):
        """Get list of recent results"""
        if not os.path.exists(self.results_dir):
            return []
        
        folders = sorted(
            [f for f in os.listdir(self.results_dir) if os.path.isdir(os.path.join(self.results_dir, f))],
            reverse=True
        )
        return folders[:limit]

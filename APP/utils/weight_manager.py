import os
import requests
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class WeightManager:
    """Manages downloading and verifying model weights"""
    
    WEIGHTS_CONFIG = {
        'DRIVE': {
            'url': 'https://github.com/Mehwish4593/RetinaLiteNet/raw/main/MTLTransfuseep300bs16head4.h5',
            'filename': 'model_weights_DRIVE.h5',
            'key_dim': 16
        },
        'IOSTAR': {
            'url': 'https://github.com/AbdullahButt-00/RetinalSegNet/raw/main/Inference/Model_weights_IOSTAR.h5',
            'filename': 'model_weights_IOSTAR.h5',
            'key_dim': 32
        }
    }
    
    def __init__(self, weights_dir):
        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)
    
    def get_weight_path(self, dataset_type):
        """Get path to weight file"""
        try:
            if dataset_type not in self.WEIGHTS_CONFIG:
                logger.error(f"Unknown dataset type: {dataset_type}")
                return None
            
            filename = self.WEIGHTS_CONFIG[dataset_type].get('filename', f'model_weights_{dataset_type}.h5')
            return os.path.join(self.weights_dir, filename)
        except Exception as e:
            logger.error(f"Error getting weight path for {dataset_type}: {str(e)}")
            return None
    
    def get_key_dim(self, dataset_type):
        """Get correct key_dim for dataset type"""
        try:
            if dataset_type not in self.WEIGHTS_CONFIG:
                logger.warning(f"Unknown dataset type: {dataset_type}, using default key_dim=16")
                return 16
            return self.WEIGHTS_CONFIG[dataset_type].get('key_dim', 16)
        except Exception as e:
            logger.error(f"Error getting key_dim: {str(e)}")
            return 16
    
    def weight_exists(self, dataset_type):
        """Check if weight file exists"""
        try:
            path = self.get_weight_path(dataset_type)
            if path is None:
                return False
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception as e:
            logger.error(f"Error checking weight existence: {str(e)}")
            return False
    
    def download_weight(self, dataset_type, progress_callback=None):
        """Download weight file with progress"""
        try:
            if self.weight_exists(dataset_type):
                logger.info(f"Weight for {dataset_type} already exists")
                return self.get_weight_path(dataset_type)
            
            if dataset_type not in self.WEIGHTS_CONFIG:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            config = self.WEIGHTS_CONFIG[dataset_type]
            url = config['url']
            filepath = self.get_weight_path(dataset_type)
            
            if filepath is None:
                raise ValueError(f"Could not determine filepath for {dataset_type}")
            
            logger.info(f"Downloading {dataset_type} weights from {url}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            progress_callback(downloaded, total_size)
            
            logger.info(f"Successfully downloaded {dataset_type} weights")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_type} weights: {str(e)}")
            filepath = self.get_weight_path(dataset_type)
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            raise
    
    def get_available_weights(self):
        """Get list of available weights"""
        available = {}
        try:
            for dataset_type in self.WEIGHTS_CONFIG.keys():
                try:
                    available[dataset_type] = self.weight_exists(dataset_type)
                except Exception as e:
                    logger.error(f"Error checking {dataset_type}: {str(e)}")
                    available[dataset_type] = False
        except Exception as e:
            logger.error(f"Error in get_available_weights: {str(e)}")
            available = {'DRIVE': False, 'IOSTAR': False}
        
        return available
    
    def ensure_weights(self, dataset_type, progress_callback=None):
        """Ensure weights are available, download if needed"""
        try:
            if not self.weight_exists(dataset_type):
                return self.download_weight(dataset_type, progress_callback)
            return self.get_weight_path(dataset_type)
        except Exception as e:
            logger.error(f"Error ensuring weights: {str(e)}")
            raise
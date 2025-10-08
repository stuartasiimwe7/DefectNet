import torch
import torch.hub
import logging
from typing import Optional, Dict, Any, List
import time
from config import Config

logger = logging.getLogger(__name__)

class YOLOModel:
    """YOLOv5 model wrapper with proper error handling and configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if self.config.get("enable_gpu", False) and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def _load_model(self) -> None:
        """Load YOLOv5 model with error handling"""
        try:
            logger.info(f"Loading YOLOv5 model: {self.config['name']}")
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                self.config['name'], 
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(self, image_array, timeout: int = 30) -> Dict[str, Any]:
        """Run inference on image with timeout and error handling"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            start_time = time.time()
            
            with torch.no_grad():
                results = self.model(image_array)
            
            inference_time = time.time() - start_time
            
            if inference_time > timeout:
                logger.warning(f"Inference took {inference_time:.2f}s, exceeding timeout of {timeout}s")
            
            return {
                "results": results,
                "inference_time": inference_time,
                "device": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config['name'],
            "device": str(self.device),
            "loaded": self.is_loaded(),
            "confidence_threshold": self.config['confidence_threshold']
        }

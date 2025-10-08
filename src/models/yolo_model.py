import requests
import logging
from typing import Optional, Dict, Any, List
import time
import json
from config import Config

logger = logging.getLogger(__name__)

class YOLOModel:
    """Mock YOLOv5 model wrapper for demo purposes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = "mock_model"
        self.device = "cpu"
        self._load_model()
    
    def _load_model(self) -> None:
        """Load mock model"""
        try:
            logger.info(f"Loading mock YOLOv5 model: {self.config['name']}")
            logger.info("Using online API simulation for demo purposes")
            self.model = "mock_model"
            logger.info("Mock model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(self, image_array, timeout: int = 30) -> Dict[str, Any]:
        """Run mock inference on image"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            start_time = time.time()
            
            # Simulate inference time
            time.sleep(0.1)
            
            # Mock results
            mock_results = MockResults()
            
            inference_time = time.time() - start_time
            
            if inference_time > timeout:
                logger.warning(f"Inference took {inference_time:.2f}s, exceeding timeout of {timeout}s")
            
            return {
                "results": mock_results,
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

class MockResults:
    """Mock YOLOv5 results for demo purposes"""
    
    def pandas(self):
        return MockPandas()

class MockPandas:
    """Mock pandas results"""
    
    def __init__(self):
        self.xyxy = [MockDataFrame()]

class MockDataFrame:
    """Mock DataFrame with sample predictions"""
    
    def iterrows(self):
        # Return mock predictions
        mock_data = [
            {
                'name': 'missing_hole',
                'confidence': 0.85,
                'xmin': 100,
                'ymin': 150,
                'xmax': 200,
                'ymax': 250
            },
            {
                'name': 'spur',
                'confidence': 0.72,
                'xmin': 300,
                'ymin': 100,
                'xmax': 350,
                'ymax': 180
            }
        ]
        
        for i, data in enumerate(mock_data):
            yield i, MockRow(data)

class MockRow:
    """Mock DataFrame row"""
    
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

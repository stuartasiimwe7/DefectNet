import os
from typing import Dict, Any

class Config:
    """Application configuration management"""
    
    # Model configuration
    MODEL_NAME = "yolov5s"
    MODEL_CONFIDENCE_THRESHOLD = 0.5
    MODEL_IMAGE_SIZE = 640
    
    # API configuration
    API_TITLE = "PCB Defect Detection API"
    API_DESCRIPTION = "AI-powered system for detecting defects in PCB images"
    API_VERSION = "1.0.0"
    MAX_BATCH_SIZE = 10
    MAX_FILE_SIZE_MB = 50
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pt")
    DATA_PATH = "data"
    
    # Performance settings
    INFERENCE_TIMEOUT = 30
    ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() == "true"
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "name": cls.MODEL_NAME,
            "confidence_threshold": cls.MODEL_CONFIDENCE_THRESHOLD,
            "image_size": cls.MODEL_IMAGE_SIZE,
            "path": cls.MODEL_PATH,
            "enable_gpu": cls.ENABLE_GPU
        }
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API-specific configuration"""
        return {
            "title": cls.API_TITLE,
            "description": cls.API_DESCRIPTION,
            "version": cls.API_VERSION,
            "max_batch_size": cls.MAX_BATCH_SIZE,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "inference_timeout": cls.INFERENCE_TIMEOUT
        }

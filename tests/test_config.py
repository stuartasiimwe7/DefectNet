import pytest
import os
from config import Config

class TestConfig:
    """Test configuration management"""
    
    def test_model_config(self):
        """Test model configuration retrieval"""
        config = Config.get_model_config()
        
        assert "name" in config
        assert "confidence_threshold" in config
        assert "image_size" in config
        assert "path" in config
        assert "enable_gpu" in config
        
        assert config["name"] == "yolov5s"
        assert config["confidence_threshold"] == 0.5
        assert config["image_size"] == 640
    
    def test_api_config(self):
        """Test API configuration retrieval"""
        config = Config.get_api_config()
        
        assert "title" in config
        assert "description" in config
        assert "version" in config
        assert "max_batch_size" in config
        assert "max_file_size_mb" in config
        
        assert config["max_batch_size"] == 10
        assert config["max_file_size_mb"] == 50
    
    def test_environment_variables(self):
        """Test environment variable handling"""
        original_log_level = os.environ.get("LOG_LEVEL")
        
        try:
            os.environ["LOG_LEVEL"] = "DEBUG"
            assert Config.LOG_LEVEL == "DEBUG"
            
            os.environ["ENABLE_GPU"] = "true"
            assert Config.ENABLE_GPU == True
            
        finally:
            if original_log_level:
                os.environ["LOG_LEVEL"] = original_log_level
            else:
                os.environ.pop("LOG_LEVEL", None)
            os.environ.pop("ENABLE_GPU", None)

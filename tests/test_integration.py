import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch

from src.services.defect_detection_service import DefectDetectionService
from src.models.yolo_model import YOLOModel
from src.utils.image_processor import ImageProcessor
from src.utils.response_formatter import ResponseFormatter

class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @patch('src.models.yolo_model.torch.hub.load')
    def test_yolo_model_integration(self, mock_hub_load):
        """Test YOLO model integration"""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = None
        mock_hub_load.return_value = mock_model
        
        config = {
            "name": "yolov5s",
            "confidence_threshold": 0.5,
            "image_size": 640,
            "enable_gpu": False
        }
        
        model = YOLOModel(config)
        
        assert model.is_loaded()
        assert model.device.type == "cpu"
        
        model_info = model.get_model_info()
        assert model_info["name"] == "yolov5s"
        assert model_info["loaded"] == True
    
    def test_image_processor_integration(self):
        """Test image processor integration"""
        processor = ImageProcessor(max_size_mb=1)
        
        # Test with valid image array
        image_info = processor.get_image_info(self.sample_image_array)
        assert image_info["width"] == 100
        assert image_info["height"] == 100
        assert image_info["channels"] == 3
        
        # Test resize functionality
        resized = processor.resize_image(self.sample_image_array, (50, 50))
        assert resized.shape == (50, 50, 3)
        
        # Test normalization
        normalized = processor.normalize_image(self.sample_image_array)
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)
    
    def test_response_formatter_integration(self):
        """Test response formatter integration"""
        formatter = ResponseFormatter(confidence_threshold=0.5)
        
        # Create mock results
        mock_results = Mock()
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, Mock(name="defect", confidence=0.8, xmin=10, ymin=20, xmax=30, ymax=40))
        ]
        mock_pandas = Mock()
        mock_pandas.xyxy = [mock_df]
        mock_results.pandas.return_value = mock_pandas
        
        # Test single prediction formatting
        response = formatter.format_single_prediction(mock_results, 0.1, {"width": 100, "height": 100})
        
        assert response["total_defects"] == 1
        assert len(response["predictions"]) == 1
        assert response["predictions"][0]["class"] == "defect"
        assert response["predictions"][0]["confidence"] == 0.8
        
        # Test batch formatting
        batch_results = [response]
        batch_response = formatter.format_batch_prediction(batch_results)
        
        assert "batch_results" in batch_response
        assert "summary" in batch_response
        assert batch_response["summary"]["total_images"] == 1
    
    @patch('src.models.yolo_model.torch.hub.load')
    def test_service_integration(self, mock_hub_load):
        """Test defect detection service integration"""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = None
        mock_hub_load.return_value = mock_model
        
        # Mock prediction results
        mock_results = Mock()
        mock_df = Mock()
        mock_df.iterrows.return_value = []
        mock_pandas = Mock()
        mock_pandas.xyxy = [mock_df]
        mock_results.pandas.return_value = mock_pandas
        
        mock_model.return_value = mock_results
        
        service = DefectDetectionService()
        
        assert service.is_ready()
        
        service_info = service.get_service_info()
        assert "service_status" in service_info
        assert "model_info" in service_info
        assert "config" in service_info
        assert service_info["service_status"] == "ready"
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        processor = ImageProcessor(max_size_mb=1)
        
        # Test with invalid image data
        with pytest.raises(Exception):
            processor.validate_image(b"invalid image data")
        
        # Test with oversized image
        large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        large_bytes = large_image.tobytes()
        
        with pytest.raises(Exception):
            processor.validate_image(large_bytes)
    
    def test_configuration_integration(self):
        """Test configuration integration across components"""
        from config import Config
        
        model_config = Config.get_model_config()
        api_config = Config.get_api_config()
        
        # Verify configuration is properly structured
        assert "name" in model_config
        assert "confidence_threshold" in model_config
        assert "max_batch_size" in api_config
        assert "max_file_size_mb" in api_config
        
        # Test that components can use config values
        processor = ImageProcessor(max_size_mb=api_config["max_file_size_mb"])
        formatter = ResponseFormatter(confidence_threshold=model_config["confidence_threshold"])
        
        assert processor.max_size_mb == api_config["max_file_size_mb"]
        assert formatter.confidence_threshold == model_config["confidence_threshold"]

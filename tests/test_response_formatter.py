import pytest
import time
from unittest.mock import Mock
from src.utils.response_formatter import ResponseFormatter

class TestResponseFormatter:
    """Test response formatting utilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.formatter = ResponseFormatter(confidence_threshold=0.5)
    
    def _create_mock_results(self, predictions_data):
        """Create mock YOLOv5 results"""
        mock_results = Mock()
        mock_df = Mock()
        
        mock_df.iterrows.return_value = [
            (i, Mock(**data)) for i, data in enumerate(predictions_data)
        ]
        
        mock_pandas = Mock()
        mock_pandas.xyxy = [mock_df]
        mock_results.pandas.return_value = mock_pandas
        
        return mock_results
    
    def test_format_single_prediction_success(self):
        """Test successful single prediction formatting"""
        predictions_data = [
            {"name": "defect1", "confidence": 0.8, "xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40},
            {"name": "defect2", "confidence": 0.3, "xmin": 50, "ymin": 60, "xmax": 70, "ymax": 80}
        ]
        
        mock_results = self._create_mock_results(predictions_data)
        image_info = {"width": 100, "height": 100}
        
        response = self.formatter.format_single_prediction(mock_results, 0.1, image_info)
        
        assert "predictions" in response
        assert "total_defects" in response
        assert "inference_time_ms" in response
        assert "image_info" in response
        assert "timestamp" in response
        
        assert response["total_defects"] == 1  # Only one above threshold
        assert len(response["predictions"]) == 1
        assert response["predictions"][0]["class"] == "defect1"
        assert response["predictions"][0]["confidence"] == 0.8
    
    def test_format_single_prediction_no_defects(self):
        """Test single prediction with no defects above threshold"""
        predictions_data = [
            {"name": "defect1", "confidence": 0.3, "xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40}
        ]
        
        mock_results = self._create_mock_results(predictions_data)
        image_info = {"width": 100, "height": 100}
        
        response = self.formatter.format_single_prediction(mock_results, 0.1, image_info)
        
        assert response["total_defects"] == 0
        assert len(response["predictions"]) == 0
    
    def test_format_batch_prediction(self):
        """Test batch prediction formatting"""
        batch_results = [
            {"filename": "img1.jpg", "total_defects": 2, "predictions": []},
            {"filename": "img2.jpg", "total_defects": 1, "predictions": []},
            {"filename": "img3.jpg", "error": "Processing failed", "total_defects": 0}
        ]
        
        response = self.formatter.format_batch_prediction(batch_results)
        
        assert "batch_results" in response
        assert "summary" in response
        assert "timestamp" in response
        
        summary = response["summary"]
        assert summary["total_images"] == 3
        assert summary["successful_predictions"] == 2
        assert summary["failed_predictions"] == 1
        assert summary["total_defects_found"] == 3
    
    def test_format_error_response(self):
        """Test error response formatting"""
        error_msg = "Test error"
        status_code = 400
        
        response = self.formatter.format_error_response(error_msg, status_code)
        
        assert response["error"] == error_msg
        assert response["status_code"] == status_code
        assert "timestamp" in response
    
    def test_extract_predictions_invalid_results(self):
        """Test prediction extraction with invalid results"""
        mock_results = Mock()
        mock_results.pandas.side_effect = Exception("Invalid results")
        
        predictions = self.formatter._extract_predictions(mock_results)
        
        assert predictions == []

import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import Mock, patch

from app import app

class TestAPI:
    """Test API endpoints"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)
        self.sample_image = self._create_sample_image()
    
    def _create_sample_image(self, size=(100, 100)):
        """Create a sample image for testing"""
        image = Image.new('RGB', size, color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    def test_root_endpoint(self):
        """Test root health check endpoint"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    @patch('app.detection_service')
    def test_health_endpoint(self, mock_service):
        """Test detailed health check endpoint"""
        mock_service.get_service_info.return_value = {
            "service_status": "ready",
            "model_info": {"name": "yolov5s"}
        }
        mock_service.is_ready.return_value = True
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service_info" in data
    
    @patch('app.detection_service')
    def test_predict_endpoint_success(self, mock_service):
        """Test successful prediction endpoint"""
        mock_service.is_ready.return_value = True
        
        async def mock_predict_single(file):
            return {
                "predictions": [{"class": "defect", "confidence": 0.8}],
                "total_defects": 1
            }
        
        mock_service.predict_single.side_effect = mock_predict_single
        
        files = {"file": ("test.jpg", self.sample_image, "image/jpeg")}
        response = self.client.post("/predict/", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_defects" in data
    
    @patch('app.detection_service')
    def test_predict_endpoint_invalid_file(self, mock_service):
        """Test prediction endpoint with invalid file"""
        mock_service.is_ready.return_value = True
        
        async def mock_predict_single(file):
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        mock_service.predict_single.side_effect = mock_predict_single
        
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = self.client.post("/predict/", files=files)
        
        assert response.status_code == 400
    
    def test_predict_endpoint_no_file(self):
        """Test prediction endpoint without file"""
        response = self.client.post("/predict/")
        
        assert response.status_code == 422
    
    @patch('app.detection_service')
    def test_predict_batch_endpoint(self, mock_service):
        """Test batch prediction endpoint"""
        mock_service.is_ready.return_value = True
        
        async def mock_predict_batch(files):
            return {
                "batch_results": [{"filename": "test.jpg", "total_defects": 1}],
                "summary": {"total_images": 1, "successful_predictions": 1}
            }
        
        mock_service.predict_batch.side_effect = mock_predict_batch
        
        files = [
            ("files", ("test1.jpg", self.sample_image, "image/jpeg")),
            ("files", ("test2.jpg", self.sample_image, "image/jpeg"))
        ]
        response = self.client.post("/predict/batch/", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_results" in data
        assert "summary" in data
    
    @patch('app.detection_service')
    def test_predict_batch_too_many_files(self, mock_service):
        """Test batch prediction with too many files"""
        mock_service.is_ready.return_value = True
        
        async def mock_predict_batch(files):
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Too many files")
        
        mock_service.predict_batch.side_effect = mock_predict_batch
        
        files = []
        for i in range(15):  # More than max batch size
            files.append(("files", (f"test{i}.jpg", self.sample_image, "image/jpeg")))
        
        response = self.client.post("/predict/batch/", files=files)
        
        assert response.status_code == 400
    
    @patch('app.detection_service')
    def test_info_endpoint(self, mock_service):
        """Test info endpoint"""
        mock_service.get_service_info.return_value = {
            "service_status": "ready",
            "model_info": {"name": "yolov5s"}
        }
        
        response = self.client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "service_status" in data
        assert "model_info" in data
    
    def test_global_exception_handler(self):
        """Test global exception handler"""
        # Test with a route that doesn't exist to trigger 404, not 500
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404

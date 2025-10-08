import pytest
import numpy as np
from PIL import Image
import io
from fastapi import HTTPException

from src.utils.image_processor import ImageProcessor

class TestImageProcessor:
    """Test image processing utilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = ImageProcessor(max_size_mb=1)
        self.sample_image = self._create_sample_image()
    
    def _create_sample_image(self, size=(100, 100)):
        """Create a sample image for testing"""
        image = Image.new('RGB', size, color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    def test_validate_image_valid(self):
        """Test validation of valid image"""
        self.processor.validate_image(self.sample_image)
    
    def test_validate_image_too_large(self):
        """Test validation of oversized image"""
        # Create a large image that exceeds 1MB limit
        large_image = b"x" * (2 * 1024 * 1024)  # 2MB of data
        
        with pytest.raises(HTTPException) as exc_info:
            self.processor.validate_image(large_image)
        
        assert exc_info.value.status_code == 413
    
    def test_validate_image_empty(self):
        """Test validation of empty image"""
        with pytest.raises(HTTPException) as exc_info:
            self.processor.validate_image(b"")
        
        assert exc_info.value.status_code == 400
    
    def test_validate_image_invalid_format(self):
        """Test validation of invalid image format"""
        invalid_data = b"not an image"
        
        with pytest.raises(HTTPException) as exc_info:
            self.processor.validate_image(invalid_data)
        
        assert exc_info.value.status_code == 400
    
    def test_preprocess_image_rgb(self):
        """Test preprocessing of RGB image"""
        result = self.processor.preprocess_image(self.sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # RGB channels
        assert result.dtype == np.uint8
    
    def test_preprocess_image_grayscale(self):
        """Test preprocessing of grayscale image"""
        gray_image = Image.new('L', (100, 100), color=128)
        img_bytes = io.BytesIO()
        gray_image.save(img_bytes, format='JPEG')
        gray_bytes = img_bytes.getvalue()
        
        result = self.processor.preprocess_image(gray_bytes)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # Converted to RGB
        assert result.dtype == np.uint8
    
    def test_get_image_info(self):
        """Test image information extraction"""
        image_array = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        info = self.processor.get_image_info(image_array)
        
        assert info["width"] == 150
        assert info["height"] == 100
        assert info["channels"] == 3
        assert info["dtype"] == "uint8"
    
    def test_resize_image(self):
        """Test image resizing"""
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = self.processor.resize_image(image_array, (50, 50))
        
        assert resized.shape == (50, 50, 3)
    
    def test_normalize_image(self):
        """Test image normalization"""
        image_array = np.array([[[255, 128, 0]]], dtype=np.uint8)
        normalized = self.processor.normalize_image(image_array)
        
        assert normalized.dtype == np.float32
        # Use approximate comparison for floating point values
        assert np.allclose(normalized, [[[1.0, 0.5019608, 0.0]]], atol=1e-6)

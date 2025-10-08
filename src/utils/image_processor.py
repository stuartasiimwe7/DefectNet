import numpy as np
from PIL import Image
import io
import logging
from typing import Tuple, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utilities with validation and error handling"""
    
    def __init__(self, max_size_mb: int = 50):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def validate_image(self, image_bytes: bytes, filename: str = None) -> None:
        """Validate image file before processing"""
        if len(image_bytes) > self.max_size_bytes:
            raise HTTPException(
                status_code=413, 
                detail=f"Image too large. Maximum size: {self.max_size_mb}MB"
            )
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for YOLOv5 inference"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            if image_array.size == 0:
                raise ValueError("Empty image array")
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
    
    def get_image_info(self, image_array: np.ndarray) -> dict:
        """Get image information"""
        return {
            "width": image_array.shape[1],
            "height": image_array.shape[0],
            "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1,
            "dtype": str(image_array.dtype)
        }
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Resize image to specified dimensions"""
        try:
            pil_image = Image.fromarray(image)
            resized = pil_image.resize(size, Image.Resampling.LANCZOS)
            return np.array(resized)
        except Exception as e:
            logger.error(f"Image resize failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image resize failed: {str(e)}")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to 0-1 range"""
        try:
            return image.astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"Image normalization failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image normalization failed: {str(e)}")

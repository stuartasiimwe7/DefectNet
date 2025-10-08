import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, UploadFile

from ..models.yolo_model import YOLOModel
from ..utils.image_processor import ImageProcessor
from ..utils.response_formatter import ResponseFormatter
from config import Config

logger = logging.getLogger(__name__)

class DefectDetectionService:
    """Main service for PCB defect detection"""
    
    def __init__(self):
        self.model = None
        self.image_processor = ImageProcessor(max_size_mb=Config.MAX_FILE_SIZE_MB)
        self.response_formatter = ResponseFormatter(
            confidence_threshold=Config.MODEL_CONFIDENCE_THRESHOLD
        )
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the YOLOv5 model"""
        try:
            model_config = Config.get_model_config()
            self.model = YOLOModel(model_config)
            logger.info("Defect detection service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize defect detection service: {str(e)}")
            raise RuntimeError(f"Service initialization failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions"""
        return self.model is not None and self.model.is_loaded()
    
    async def predict_single(self, file: UploadFile) -> Dict[str, Any]:
        """Predict defects in a single image"""
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")
        
        try:
            image_bytes = await file.read()
            
            self.image_processor.validate_image(image_bytes, file.filename)
            image_array = self.image_processor.preprocess_image(image_bytes)
            image_info = self.image_processor.get_image_info(image_array)
            
            prediction_result = self.model.predict(
                image_array, 
                timeout=Config.INFERENCE_TIMEOUT
            )
            
            response = self.response_formatter.format_single_prediction(
                prediction_result["results"],
                prediction_result["inference_time"],
                image_info
            )
            
            logger.info(f"Prediction completed for {file.filename}: {response['total_defects']} defects found")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Predict defects in multiple images"""
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")
        
        if len(files) > Config.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Maximum {Config.MAX_BATCH_SIZE} images per batch"
            )
        
        batch_results = []
        
        for file in files:
            try:
                result = await self.predict_single(file)
                result["filename"] = file.filename
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"Batch prediction failed for {file.filename}: {str(e)}")
                batch_results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "total_defects": 0
                })
        
        return self.response_formatter.format_batch_prediction(batch_results)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and status"""
        model_info = self.model.get_model_info() if self.model else {}
        
        return {
            "service_status": "ready" if self.is_ready() else "not_ready",
            "model_info": model_info,
            "config": {
                "max_batch_size": Config.MAX_BATCH_SIZE,
                "max_file_size_mb": Config.MAX_FILE_SIZE_MB,
                "confidence_threshold": Config.MODEL_CONFIDENCE_THRESHOLD
            }
        }

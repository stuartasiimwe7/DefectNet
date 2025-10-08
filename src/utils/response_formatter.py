import logging
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """Format model predictions into standardized API responses"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def format_single_prediction(self, results, inference_time: float, image_info: dict) -> Dict[str, Any]:
        """Format single image prediction results"""
        try:
            predictions = self._extract_predictions(results)
            
            return {
                "predictions": predictions,
                "total_defects": len(predictions),
                "inference_time_ms": round(inference_time * 1000, 2),
                "image_info": image_info,
                "confidence_threshold": self.confidence_threshold,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Response formatting failed: {str(e)}")
            return {
                "predictions": [],
                "total_defects": 0,
                "error": f"Response formatting failed: {str(e)}",
                "timestamp": time.time()
            }
    
    def format_batch_prediction(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format batch prediction results"""
        try:
            total_defects = sum(result.get("total_defects", 0) for result in batch_results)
            successful_predictions = len([r for r in batch_results if "error" not in r])
            
            return {
                "batch_results": batch_results,
                "summary": {
                    "total_images": len(batch_results),
                    "successful_predictions": successful_predictions,
                    "failed_predictions": len(batch_results) - successful_predictions,
                    "total_defects_found": total_defects
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Batch response formatting failed: {str(e)}")
            return {
                "batch_results": batch_results,
                "error": f"Batch formatting failed: {str(e)}",
                "timestamp": time.time()
            }
    
    def _extract_predictions(self, results) -> List[Dict[str, Any]]:
        """Extract predictions from YOLOv5 results"""
        predictions = []
        
        try:
            if hasattr(results, 'pandas'):
                df = results.pandas().xyxy[0]
                
                for _, row in df.iterrows():
                    if row['confidence'] > self.confidence_threshold:
                        prediction = {
                            "class": row['name'],
                            "confidence": float(row['confidence']),
                            "bounding_box": {
                                "x_min": int(row['xmin']),
                                "y_min": int(row['ymin']),
                                "x_max": int(row['xmax']),
                                "y_max": int(row['ymax'])
                            }
                        }
                        predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction extraction failed: {str(e)}")
            return []
    
    def format_error_response(self, error: str, status_code: int = 500) -> Dict[str, Any]:
        """Format error response"""
        return {
            "error": error,
            "status_code": status_code,
            "timestamp": time.time()
        }

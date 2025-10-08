from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.hub
from PIL import Image
import io
import numpy as np
import logging
from typing import List, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PCB Defect Detection API",
    description="AI-powered system for detecting defects in PCB images",
    version="1.0.0"
)

model = None

def load_model():
    """Load YOLOv5 model using torch.hub for proper initialization"""
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        logger.info("YOLOv5 model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for YOLOv5 inference"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        return image_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def postprocess_predictions(results) -> List[Dict[str, Any]]:
    """Convert YOLOv5 results to standardized format"""
    try:
        predictions = []
        
        if hasattr(results, 'pandas'):
            df = results.pandas().xyxy[0]
            
            for _, row in df.iterrows():
                if row['confidence'] > 0.5:
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
        logger.error(f"Postprocessing failed: {str(e)}")
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to initialize model on startup")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PCB Defect Detection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_status": model_status,
        "timestamp": time.time()
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict defects in uploaded PCB image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with detected defects and bounding boxes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        
        with torch.no_grad():
            results = model(image_array)
        
        predictions = postprocess_predictions(results)
        
        inference_time = time.time() - start_time
        
        response = {
            "predictions": predictions,
            "total_defects": len(predictions),
            "inference_time_ms": round(inference_time * 1000, 2),
            "image_size": {
                "width": image_array.shape[1],
                "height": image_array.shape[0]
            }
        }
        
        logger.info(f"Prediction completed in {inference_time:.3f}s, found {len(predictions)} defects")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch/")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict defects in multiple uploaded PCB images
    
    Args:
        files: List of image files
    
    Returns:
        JSON response with predictions for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            image_bytes = await file.read()
            image_array = preprocess_image(image_bytes)
            
            with torch.no_grad():
                pred_results = model(image_array)
            
            predictions = postprocess_predictions(pred_results)
            
            results.append({
                "filename": file.filename,
                "predictions": predictions,
                "total_defects": len(predictions)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
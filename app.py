from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import Config
from src.services.defect_detection_service import DefectDetectionService

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

detection_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global detection_service
    try:
        detection_service = DefectDetectionService()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PCB Defect Detection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if detection_service is None:
        return {"status": "unhealthy", "error": "Service not initialized"}
    
    service_info = detection_service.get_service_info()
    return {
        "status": "healthy" if detection_service.is_ready() else "unhealthy",
        "service_info": service_info
    }

@app.post("/predict/")
@limiter.limit(Config.RATE_LIMIT_SINGLE)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict defects in uploaded PCB image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with detected defects and bounding boxes
    """
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await detection_service.predict_single(file)

@app.post("/predict/batch/")
@limiter.limit(Config.RATE_LIMIT_BATCH)
async def predict_batch(request: Request, files: List[UploadFile] = File(...)):
    """
    Predict defects in multiple uploaded PCB images
    
    Args:
        files: List of image files
    
    Returns:
        JSON response with predictions for each image
    """
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await detection_service.predict_batch(files)

@app.get("/info")
async def get_info():
    """Get API and service information"""
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return detection_service.get_service_info()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
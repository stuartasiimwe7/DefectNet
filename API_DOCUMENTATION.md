# DefectNet API Documentation

## Overview

RESTful API for AI-powered PCB defect detection using YOLOv5 deep learning models. This document provides 
comprehensive information about all available endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:8000
```

## Specifications

- **Request Format**: `multipart/form-data` for file uploads
- **Response Format**: `application/json`
- **Max Batch Size**: 10 files per request
- **Max File Size**: 1MB per image

## Rate Limiting

**Implemented** using SlowAPI:
- Single prediction (`/predict/`): 100 requests/minute per IP
- Batch prediction (`/predict/batch/`): 20 requests/minute per IP
- Health endpoints: No limits

**Response when limit exceeded (429):**
```json
{
  "error": "Rate limit exceeded: 100 per 1 minute"
}
```

## Endpoints

### GET /health

Health check and system status.

**Response (200):**
```json
{
  "status": "healthy",
  "service_info": {
    "service_status": "ready",
    "model_info": {
      "name": "yolov5s",
      "version": "6.0",
      "confidence_threshold": 0.5
    }
  },
  "timestamp": 1640995200.0
}
```

**Status Codes:**
- `200`: Service healthy
- `503`: Service unavailable

---

### GET /info

Detailed service configuration and capabilities.

**Response (200):**
```json
{
  "service_status": "ready",
  "model_info": {
    "name": "yolov5s",
    "version": "6.0",
    "confidence_threshold": 0.5,
    "max_image_size": 1024,
    "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
  },
  "api_info": {
    "version": "1.0.0",
    "max_batch_size": 10,
    "max_file_size_mb": 1
  },
  "timestamp": 1640995200.0
}
```

---

### 3. Single Image Prediction

Upload a single PCB image for defect detection.

```http
POST /predict/
Content-Type: multipart/form-data
```

**Request Parameters:**
- `file` (required): Image file (JPEG, PNG, BMP, TIFF)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pcb_image.jpg"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/predict/"
files = {"file": open("pcb_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()
```

**Success Response (200):**
```json
{
  "predictions": [
    {
      "class": "missing_component",
      "confidence": 0.87,
      "bounding_box": {
        "x_min": 120,
        "y_min": 80,
        "x_max": 180,
        "y_max": 140
      }
    }
  ],
  "total_defects": 1,
  "inference_time_ms": 45.2,
  "image_info": {
    "width": 640,
    "height": 480,
    "format": "JPEG",
    "file_size_bytes": 125432
  },
  "timestamp": 1640995200.0
}
```

**Error Responses:**
- `400`: Invalid image format
- `413`: File size exceeds 1MB
- `422`: No file provided

---

### POST /predict/batch/

Upload multiple PCB images for batch defect detection.

```http
POST /predict/batch/
Content-Type: multipart/form-data
```

**Request Parameters:**
- `files` (required): Multiple image files (max 10)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict/batch/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@pcb_image1.jpg" \
  -F "files=@pcb_image2.jpg" \
  -F "files=@pcb_image3.jpg"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/predict/batch/"
files = [
    ("files", open("pcb_image1.jpg", "rb")),
    ("files", open("pcb_image2.jpg", "rb")),
    ("files", open("pcb_image3.jpg", "rb"))
]

response = requests.post(url, files=files)
result = response.json()
```

**Success Response (200):**
```json
{
  "batch_results": [
    {
      "filename": "image1.jpg",
      "predictions": [...],
      "total_defects": 1,
      "inference_time_ms": 42.1,
      "image_info": {
        "width": 640,
        "height": 480,
        "format": "JPEG"
      }
    }
  ],
  "summary": {
    "total_images": 2,
    "successful_predictions": 2,
    "failed_predictions": 0,
    "total_processing_time_ms": 80.6,
    "average_inference_time_ms": 40.3
  },
  "timestamp": 1640995200.0
}
```

**Error Responses:**

**400 - Too Many Files:**
```json
{
  "error": "Batch size exceeds maximum limit of 10 files",
  "status_code": 400,
  "timestamp": 1640995200.0
}
```

---

## Data Models

### Prediction Object
```json
{
  "class": "string",
  "confidence": 0.85,
  "bounding_box": {
    "x_min": 120,
    "y_min": 80,
    "x_max": 180,
    "y_max": 140
  }
}
```

### Image Info Object
```json
{
  "width": 640,
  "height": 480,
  "format": "JPEG",
  "file_size_bytes": 125432
}
```

## Detectable Defect Types

| Defect Type | Description | Confidence Range |
|-------------|-------------|------------------|
| `missing_component` | Component not present | 0.7 - 0.95 |
| `solder_bridge` | Unintended solder connection | 0.6 - 0.9 |
| `open_circuit` | Broken trace or connection | 0.7 - 0.9 |
| `short_circuit` | Unintended electrical connection | 0.6 - 0.85 |
| `component_misalignment` | Component incorrectly positioned | 0.5 - 0.8 |
| `solder_defect` | Poor solder joint quality | 0.5 - 0.8 |
| `contamination` | Foreign material on PCB | 0.4 - 0.7 |

## Error Handling

### Standard Error Format
```json
{
  "error": "string",
  "status_code": 400,
  "timestamp": 1640995200.0
}
```

### Common Errors

| Status Code | Scenario | Message |
|-------------|----------|---------|
| 400 | Invalid format | Invalid image format. Supported: JPEG, PNG, BMP, TIFF |
| 413 | File too large | File size exceeds maximum limit of 1MB |
| 422 | No file | No file provided |
| 400 | Too many files | Batch size exceeds maximum limit of 10 files |
| 503 | Service down | Service not initialized |
| 500 | Server error | An unexpected error occurred |

## Performance

| Metric | Value |
|--------|-------|
| Single image inference | ~45ms |
| Batch processing | ~40ms per image |
| Health check | ~5ms |
| Memory usage | ~500MB base + model |
| Throughput | ~20 images/second |

## Client SDK Examples

### Python

```python
import requests
from typing import List, Dict, Any

class DefectNetClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/predict/", files=files)
            response.raise_for_status()
            return response.json()
    
    def predict_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        files = [('files', open(path, 'rb')) for path in image_paths]
        try:
            response = requests.post(f"{self.base_url}/predict/batch/", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            for _, file_handle in files:
                file_handle.close()

# Usage
client = DefectNetClient()
health = client.health_check()
result = client.predict_single("pcb_image.jpg")
batch_results = client.predict_batch(["image1.jpg", "image2.jpg"])
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

class DefectNetClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await axios.get(`${this.baseUrl}/health`);
        return response.data;
    }
    
    async predictSingle(imagePath) {
        const form = new FormData();
        form.append('file', fs.createReadStream(imagePath));
        const response = await axios.post(`${this.baseUrl}/predict/`, form, {
            headers: form.getHeaders()
        });
        return response.data;
    }
    
    async predictBatch(imagePaths) {
        const form = new FormData();
        imagePaths.forEach(path => form.append('files', fs.createReadStream(path)));
        const response = await axios.post(`${this.baseUrl}/predict/batch/`, form, {
            headers: form.getHeaders()
        });
        return response.data;
    }
}

// Usage
const client = new DefectNetClient();
const health = await client.healthCheck();
const result = await client.predictSingle('pcb_image.jpg');
```

## Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch/" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Using Postman

1. Set base URL to `http://localhost:8000`
2. Upload test images for prediction endpoints
3. Verify response formats and status codes

## Best Practices

1. Use JPEG format for faster processing
2. Group multiple images in batch requests
3. Always check response status codes
4. Set appropriate timeouts for large batches
5. Handle errors gracefully in client code

---

**API Version**: 1.0.0

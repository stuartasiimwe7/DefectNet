# DefectNet: AI-Powered Defect Detection System

DefectNet is an AI-driven system for detecting and classifying defects in semiconductor wafers using deep learning. This project utilizes state-of-the-art computer vision models to automate defect detection in semiconductor manufacturing, making the process more efficient and accurate.

## Key Features
- **Real-time Defect Detection**: Use of **Faster R-CNN** for identifying defects in wafer images.
- **Customizable Model**: Fine-tuning pretrained models with your own semiconductor wafer datasets.
- **API Service**: Fast and efficient **FastAPI** server for serving predictions.
- **Scalable Deployment**: Ready to be containerized with **Docker** and deployed on **AWS/GCP**.

## Tech Stack

| **Category** | **Technology Choices** |
|--------------|-------------------------|
| **Data Preprocessing** | Python, OpenCV, PyTorch, Pandas, NumPy |
| **AI Model** | PyTorch (YOLOv5) |
| **Backend API** | FastAPI |
| **Frontend** | React.js (with Bootstrap/Tailwind for styling) |
| **Containerization** | Docker |
| **Cloud Deployment** | AWS (EC2, Lambda, S3) |
| **Model Serving** | TorchServe, ONNX Runtime |
| **Monitoring & Logging** | Prometheus, Grafana, CloudWatch |
| **CI/CD** | GitHub Actions, DockerHub, Terraform |

## Table of Contents
- [Project Overview](#project-overview)
- [Installation Guide](#installation-guide)
    1. [Clone the Repository](#clone-the-repository)
    2. [Set Up a Virtual Environment](#set-up-a-virtual-environment)
    3. [Install Dependencies](#install-dependencies)
    4. [Download YOLOv5](#download-yolov5)
    5. [Download the Model Weights](#download-the-model-weights)
- [Running the API](#running-the-api)
- [How to Use the API](#how-to-use-the-api)
    - [Uploading an Image for Prediction](#uploading-an-image-for-prediction)
- [Project Structure](#project-structure)
- [Troubleshooting & Common Errors](#troubleshooting--common-errors)
- [Future Improvements](#future-improvements)

## Project Overview

This project is a FastAPI-based web service that utilizes YOLOv5 for detecting defects in images. It allows users to upload images and receive predictions from a trained YOLOv5 model.

DefectNet is a deep learning-powered defect detection API that uses YOLOv5 for object detection. The application is built using FastAPI and serves predictions through a simple HTTP interface. Users can upload images, and the system will return the predicted bounding boxes and confidence scores for detected defects.

## Installation Guide

To set up this project, follow these steps:

1. **Clone the Repository**

    First, clone the repository to your local machine:

    ```bash
    git clone https://github.com/stuartasiimwe7/DefectNet.git
    cd DefectNet
    ```

2. **Set Up a Virtual Environment**

    It is recommended to create a virtual environment to avoid dependency conflicts:

    ```bash
    python3 -m venv .venv  # Create a virtual environment
    source .venv/bin/activate  # Activate the virtual environment (Linux/Mac)
    ```

    For Windows:

    ```bash
    .venv\Scripts\activate
    ```

3. **Install Dependencies**

    Once the virtual environment is activated, install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    If you need to install them manually:

    ```bash
    pip install fastapi uvicorn torch torchvision numpy pillow
    ```

4. **Download YOLOv5**

    Since we use YOLOv5, clone the official Ultralytics repository:

    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```

    Make sure to install its dependencies:

    ```bash
    cd yolov5
    pip install -r requirements.txt
    cd ..
    ```

5. **Download the Model Weights**

    If you have a custom-trained YOLOv5 model, place the `.pt` file inside the `yolov5/serve/` directory.

    If you don’t have a trained model, you can use the pre-trained YOLOv5s model:

    ```bash
    mkdir -p yolov5/serve
    wget -O yolov5/serve/best.pt https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
    ```

## Running the API

To start the FastAPI server, run:

```bash
uvicorn app:app --reload --port 8000
```

You should see:

```plaintext
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Now, open your browser and navigate to:

```plaintext
http://127.0.0.1:8000/docs
```

This opens an interactive API documentation where you can test the prediction endpoint.

## How to Use the API

### Uploading an Image for Prediction

To send an image for prediction, use cURL or Postman.

Using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@sample_image.jpg"
```

Using Python:

```python
import requests

url = "http://127.0.0.1:8000/predict/"
files = {"file": open("sample_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

Expected JSON Response:

```json
{
    "prediction": [
        {
            "class": "defect",
            "confidence": 0.92,
            "bounding_box": [50, 30, 200, 180]
        }
    ]
}
```

## Project Structure

```bash
DefectNet/
│── app.py                   # Main FastAPI application
│── requirements.txt         # List of dependencies
│── yolov5/
│   │── models/
│   │── serve/
│   │   ├── best.pt          # Custom trained YOLOv5 model
│── .venv/                   # Virtual environment (optional)
└── README.md                # This documentation
```

## Troubleshooting & Common Errors

1. **[Errno 98] Address already in use**

    **Solution:**

    The port 8000 might be occupied by another process. Run this command to find and kill the process:

    ```bash
    sudo lsof -i :8000
    kill -9 <PID>
    ```

    Alternatively, change the port when starting Uvicorn:

    ```bash
    uvicorn app:app --reload --port 8001
    ```

2. **ModuleNotFoundError: No module named 'models.yolo'**

    **Solution:**

    Make sure you cloned the YOLOv5 repository and that the `yolov5` directory exists inside your project.

3. **FileNotFoundError: No such file or directory: 'yolov5s.yaml'**

    **Solution:**

    You are likely loading YOLOv5 incorrectly. Instead of:

    ```python
    from yolov5.models.yolo import DetectionModel
    model = DetectionModel()
    ```

    Use:

    ```python
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    ```

## Future Improvements

- Deploy API using Docker
- Support batch image processing
- Integrate authentication for security
- Add front-end UI for easy usage

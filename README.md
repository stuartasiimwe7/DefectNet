# DefectNet: AI-Powered Defect Detection System

DefectNet is an AI-driven system for detecting and classifying defects in semiconductor wafers using deep learning. This project utilizes state-of-the-art computer vision models to automate defect detection in semiconductor manufacturing, making the process more efficient and accurate.

## Key Features
- **Real-time Defect Detection**: Use of **Faster R-CNN** for identifying defects in wafer images.
- **Customizable Model**: Fine-tuning pretrained models with your own semiconductor wafer datasets.
- **API Service**: Fast and efficient **FastAPI** server for serving predictions.
- **Scalable Deployment**: Ready to be containerized with **Docker** and deployed on **AWS/GCP**.

## Tech Stack

| **Category** | **Technology Choices** |
|-------------|-----------------------|
| **Data Preprocessing** | Python, OpenCV, TensorFlow/PyTorch, Pandas, NumPy |
| **AI Model** | PyTorch (YOLOv4/v5, Faster R-CNN, Detectron2), TensorFlow, ONNX |
| **Backend API** | FastAPI (preferred), Flask |
| **Frontend** | React.js (with Bootstrap/Tailwind for styling) |
| **Containerization** | Docker |
| **Cloud Deployment** | AWS (EC2, Lambda, S3) or GCP (AI Platform) |
| **Model Serving** | TorchServe/TensorFlow Serving/Triton Inference Server |
| **Monitoring & Logging** | Prometheus, Grafana, CloudWatch |
| **CI/CD** | GitHub Actions, DockerHub, Terraform (for cloud infrastructure) |
| <!-- **Optional** | MLflow (for model versioning), Streamlit/Dash (for interactive dashboard)--> |


## Installation

Clone the repository:

```bash
git clone https://github.com/stuartasiimwe7/DefectNet.git
cd DefectNet


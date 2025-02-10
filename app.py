from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Import the correct model class (adjust based on your project structure)
from yolov5.models.yolo import DetectionModel  # Adjust if necessary

# Initialize the model (architecture)
model = DetectionModel()  # Replace with the appropriate model initialization

# Load the weights
state_dict = torch.load('yolov5/serve/best.pt', weights_only=True)  # Load only weights
model.load_state_dict(state_dict)  # Load weights into the model

# Set the model to evaluation mode
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to your model's input size (640x640 is common for YOLO)
    transforms.ToTensor(),
])

# Helper function to process the image and make predictions
def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)  # Add batch dimension (1 image in batch)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()

        # Preprocess the image
        tensor = process_image(image_bytes)

        # Run the model to get predictions
        with torch.no_grad():
            output = model(tensor)  # Pass the tensor through the model
        
        # Assuming the model output is a list of bounding boxes, class labels, and confidence scores
        # You'll need to handle the model output depending on what your YOLO model provides.
        # For simplicity, assume it's object detection and return the class labels and bounding boxes
        prediction = output[0]  # Adjust depending on your model's output format

        # For example, if output contains (class_id, confidence, bbox)
        # prediction = output[0][:, 0].tolist()  # If you want the class labels
        # prediction = output[0][:, 1].tolist()  # If you want confidence scores

        return {"prediction": prediction.tolist()}  # Adjust the output as needed

    except Exception as e:
        return {"error": str(e)}

# Run the app with `uvicorn`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# Initialize FastAPI
app = FastAPI()

from yolov5.models.yolo import DetectionModel 

model = DetectionModel()  

# Load the weights
state_dict = torch.load('yolov5/serve/best.pt', weights_only=True)
model.load_state_dict(state_dict) 
model.eval()

transform = transforms.Compose([
    transforms.Resize((640, 640)), # Resize 
    ])

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0) 
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor = process_image(image_bytes)

        with torch.no_grad():
            output = model(tensor)  #Pass the tensor through the model
        prediction = output[0] 
        return {"prediction": prediction.tolist()} 

    except Exception as e:
        return {"error": str(e)}

# Run the app with `uvicorn`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
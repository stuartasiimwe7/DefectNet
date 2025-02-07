from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
import numpy as np

app = FastAPI()
model = torch.load("trained_model.pt")
model.eval() 

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    img = np.array(image) / 255.0  # Normalize pixel values to 0-1
    results = model(img)
    return {"predictions": results.pandas().xywh.to_dict(orient="records")}

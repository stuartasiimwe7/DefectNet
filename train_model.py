import torch
from yolov5 import YOLOv5

# Load pretrained YOLOv5 model
model = YOLOv5("yolov5s")  # yolov5s, yolov5m, yolov5l, yolov5x

# Train the model
model.train(data='data.yaml', epochs=10, batch_size=16)

# Save the trained model
model.save('/models/trained_model.pt')

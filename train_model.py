import torch
from yolov5 import YOLOv5

model = YOLOv5("yolov5s")  # yolov5s, yolov5m, yolov5l, yolov5x
model.train(data='data.yaml', epochs=10, batch_size=16)
model.save('/models/trained_model.pt')

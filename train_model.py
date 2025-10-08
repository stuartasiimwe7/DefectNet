import torch
import torch.hub
import os
from pathlib import Path

def train_pcb_model():
    """
    Train YOLOv5 model for PCB defect detection
    """
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("Error: data.yaml not found. Please run preparation_script.py first.")
        return
    
    # Check if training data exists
    train_path = Path('data/pcb_dataset/train/images')
    val_path = Path('data/pcb_dataset/val/images')
    
    if not train_path.exists() or not val_path.exists():
        print("Error: Training data not found. Please run preparation_script.py first.")
        return
    
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=50,
        batch_size=16,
        imgsz=640,
        device='cpu',  # Change to 'cuda' if GPU available
        project='runs/train',
        name='pcb_defect_detection'
    )
    
    # Save the trained model
    model_path = 'models/trained_model.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_pcb_model()

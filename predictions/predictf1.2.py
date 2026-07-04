from ultralytics import YOLO
import os

if __name__ == "__main__":

    # 1. Load your trained model
    model = YOLO("/home/arch/LCPB/runs/detect/f1.2/weights/best.pt")

    # 2. Run prediction on the entire directory
    results = model.predict(
        source="/home/arch/LCPB/2026/code/Computational-Physics-2026/data/MR/lunar_south_pole/tiles/",
        conf=0.05,         # Confidence threshold
        imgsz=512,        # Inference image size
        device="cuda",    # Use GPU
        save=True,        # Saves the visual results (images with bounding boxes)
        save_txt=True,    # Saves the raw coordinates/labels in .txt files
        
        # --- PATH CUSTOMIZATION ---
        project="/home/arch/LCPB/runs/detect", # Base directory
        name="predict-f1.2",                   # Crucial: your custom folder name
        exist_ok=True                         # Overwrites/reuses the folder instead of creating predict-f1.12, predict-f1.13, etc.
    )

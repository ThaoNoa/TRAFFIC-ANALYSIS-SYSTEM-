"""
Script chuyển đổi mô hình YOLOv8 sang TensorRT engine (FP16)
Yêu cầu: tensorrt, torch, ultralytics
"""

import torch
from ultralytics import YOLO
import tensorrt as trt
import os

def export_to_tensorrt(model_path, output_path, imgsz=640, half=True):
    """
    Export YOLOv8 model to TensorRT engine.
    Args:
        model_path: path to .pt file
        output_path: path to save .engine file
        imgsz: input size
        half: use FP16
    """
    # Load model
    model = YOLO(model_path)

    # Export to TensorRT
    success = model.export(format='engine', imgsz=imgsz, half=half, device=0)  # device=0 for GPU
    if success:
        # File .engine sẽ được tạo cùng thư mục với tên model + .engine
        default_engine = model_path.replace('.pt', '.engine')
        if os.path.exists(default_engine):
            os.rename(default_engine, output_path)
            print(f"Exported TensorRT engine to {output_path}")
        else:
            print("Export failed: engine file not found")
    else:
        print("Export failed")

if __name__ == "__main__":
    # Ví dụ sử dụng
    export_to_tensorrt('yolov8n.pt', 'yolov8n_fp16.engine', half=True)
    export_to_tensorrt('yolov8n-seg.pt', 'yolov8n-seg_fp16.engine', half=True)
from ultralytics import YOLO

# Load small model for efficiency on Jetson
model = YOLO("yolov8s.pt")  # Base pretrained on COCO. Can use yolovsn too

# Train the model on your dataset
model.train(
    data="/path to data.yaml",   # Replace with your actual data.yaml path
    epochs=100,
    imgsz=640,
    batch=16,
    name="custom_mixed_model",
    project="runs/train",
    device=0,                    # GPU (Jetson)
    pretrained=True,
    optimizer="SGD",             # SGD often better for speed-sensitive deployments
    augment=True,
    mosaic=1.0,                  # Ensure Mosaic is ON
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color aug for robustness
    degrees=10,                  # Tilt handling
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.5,
    fliplr=0.5,
    patience=20,
    val=True                     # Run validation during training
)

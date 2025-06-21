import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load trained model
model = YOLO("replacetoPathOfBest.ptofModel")  # Change path

# Path to test image (change this)
image_path = "PathToTestImage"  # Ensure the image is in this path

# Run inference
results = model(image_path)

# Save result image
os.makedirs("inference_outputs", exist_ok=True)
save_path = os.path.join("inference_outputs", "output.jpg")
results[0].save(filename=save_path)

# Display the image with matplotlib
img = cv2.imread(save_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("YOLOv8 Inference Result")
plt.show()

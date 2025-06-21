# Object Detection using YOLOv8 - Internship Project

This repository contains the object detection work completed as part of my internship. It showcases training a YOLOv8 model on a **custom dataset** built with **Roboflow**, containing **non-COCO classes** and fine-tuned for high-accuracy detection.

---

##  Introduction to Object Detection

Object detection is a key computer vision task that involves identifying and localizing multiple objects within an image. It is used in a wide range of real-world applications like:

- Smart surveillance
- Autonomous vehicles
- Medical image analysis
- Industrial inspection
- Smart retail systems

Unlike image classification, object detection not only tells *what* is in an image, but also *where*.

---

##  Why YOLO (You Only Look Once)?

YOLO is a powerful real-time object detection algorithm known for its:

- ⚡️ Speed and efficiency — ideal for edge devices
- 🎯 High accuracy with fewer parameters
- 📦 Easy integration with custom datasets
- 🧠 Pretrained weights on COCO for faster convergence

For this project, I used **YOLOv8s**, the lightweight and fast variant, ideal for use on GPU-powered devices like Jetson.

---

##  My Project: Custom Object Detection Pipeline

###  About My Work

During the internship, I:

- Created a custom dataset using **Roboflow**
- Collected **non-COCO classes** (not used in YOLO pretraining)
- Cloned and merged my dataset with the **Roboflow COCO subset**
- Combined and exported a **merged dataset** in YOLO format
- Trained a YOLOv8 model with augmentations and optimizations

###  Training Outcome

- Model Used: `yolov8s.pt` (pretrained)
- Dataset: ~5600+ images, 108+ classes (custom + COCO subset)
- Training Epochs: 100
- Accuracy: **Fluctuating between 85% to 90%**
- Final weights: `YoloTrainedModel/weights/best.pt`

---

## 🗂️ Folder Structure

ObjectDetection/
│
├── Dataset/ # Contains images, annotations, and data.yaml
│ ├── train/
│ ├── test/
│ ├── valid/
│ └── data.yaml
│
├── TestImage/ # Sample images to test model
│
├── YoloTrainedModel/ # Trained model weights
│ └── weights/
│ └── best.pt
│
├── train.py # Script to train YOLOv8 model
├── test.py # Script to run inference on test images
├── requirements.txt # Required dependencies
└── README.md # Project documentation (this file)
## 🛠️ How to Use This Project

---

### 1️⃣ Clone the repository

```bash

git clone https://github.com/SHOBOT-DEV/General_Object_Detection.git

cd General_Object_Detection/ObjectDetection

```

---

### 📦 2. Create Virtual Environment (Python 3.11.9 recommended)

```bash

python -m venv new
# Activate the environment

# On macOS/Linux:
source new/bin/activate

# On Windows:
new\Scripts\activate

```

---

### 📥 3. Install Dependencies
```bash

pip install -r requirements.txt

```

---

### 🏋️‍♂️ 4. Train the Model (Optional – if going to train from the dataset instead of using trained weights provided)

## Update the train.py script:

```python
# Replace this:
data="/path to data.yaml"

# With this:
data="Dataset/data.yaml"
```
## Then run:

```bash

python train.py

```

---

### 📸 5. Run Inference on Test Images

## Edit test.py:

```python

model = YOLO("YoloTrainedModel/weights/best.pt") # Replace path correctly
image_path = "TestImage/Image.jpg" #Replace path correctly

```

## Then run:

```bash

python test.py

```
## The result will be saved in the inference_outputs/ directory and displayed via matplotlib.

---

### ✅ Requirements
# Main dependencies:

-ultralytics
-opencv-python
-matplotlib

## Install all with:

```bash

pip install -r requirements.txt

```

---

### 🙋 Author

Dhinekka B – Intern at CNDE Lab, IIT Madras
GitHub: @DhinekkaB

---

### 📄 License

This project is for research and educational purposes. Licensing terms depend on YOLOv8 and Roboflow dataset usage.

---

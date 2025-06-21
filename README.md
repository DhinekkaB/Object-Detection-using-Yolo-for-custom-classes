# 🧠 Object Detection using YOLOv8 - Internship Project

This repository contains the object detection work completed as part of my internship. It showcases training a YOLOv8 model on a **custom dataset** built with **Roboflow**, containing **non-COCO classes** and fine-tuned for high-accuracy detection.

---

## 🔍 Introduction to Object Detection

Object detection is a key computer vision task that involves identifying and localizing multiple objects within an image. It is used in a wide range of real-world applications like:

- Smart surveillance
- Autonomous vehicles
- Medical image analysis
- Industrial inspection
- Smart retail systems

Unlike image classification, object detection not only tells *what* is in an image, but also *where*.

---

## 🚀 Why YOLO (You Only Look Once)?

YOLO is a powerful real-time object detection algorithm known for its:

- ⚡️ Speed and efficiency — ideal for edge devices
- 🎯 High accuracy with fewer parameters
- 📦 Easy integration with custom datasets
- 🧠 Pretrained weights on COCO for faster convergence

For this project, I used **YOLOv8s**, the lightweight and fast variant, ideal for use on GPU-powered devices like Jetson.

---

## 📂 My Project: Custom Object Detection Pipeline

### 🧑‍💻 About My Work

During the internship, I:

- Created a custom dataset using **Roboflow**
- Collected **non-COCO classes** (not used in YOLO pretraining)
- Cloned and merged my dataset with the **Roboflow COCO subset**
- Combined and exported a **merged dataset** in YOLO format
- Trained a YOLOv8 model with augmentations and optimizations

### 📈 Training Outcome

- Model Used: `yolov8s.pt` (pretrained)
- Dataset: ~1600+ images, 41 classes (custom + COCO subset)
- Training Epochs: 100
- Accuracy: **Fluctuating between 85% to 90%**
- Final weights: `YoloTrainedModel/weights/best.pt`

---

## 🛠️ How to Use This Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-team>/<repo-name>.git
cd <repo-name>/ObjectDetection

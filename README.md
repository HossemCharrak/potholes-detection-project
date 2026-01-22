# Potholes Detection System

## 🎓 Academic Project Information

**Pluridisciplinary Project - 4th Year AIDS Specialty**  
_Artificial Intelligence and Data Science_
HIGHER SCHOOL OF COMPUTER SCIENCE Sidi Bel Abbes, Algeria

### Supervisor

- **Khaldi Belkacem**

### Team Members

- **Charrak Hossem Eddine Yakoub**
- **Benaouda Tarik Abdelhadi**
- **Merzouk Alaa Eddine**
- **Belmana Sofiane**
- **Ouled Said Mohamed**

---

## 📋 Project Overview

This project presents an automated computer vision system for detecting potholes and speed bumps in road images and videos using state-of-the-art deep learning object detection models. The system implements and compares three different detection architectures to identify the most effective approach for real-world road infrastructure monitoring.

### Objectives

- Develop an automated system for detecting potholes and speed bumps in road imagery
- Compare performance across multiple modern object detection architectures
- Enable both image and video inference capabilities
- Create a web-deployable model for practical applications

---

## Datasets:

- https://www.kaggle.com/datasets/charrakhossem/merged-potholes-speedbumps
- https://www.kaggle.com/datasets/belmanasoufyane/pothole-speedbumps
- https://www.kaggle.com/datasets/arnavsan7x24/pothole-videos
- https://www.kaggle.com/datasets/sudhanshu2198/potholes-detection-inference-on-videos

## 🤖 Models Implemented

### 1. YOLOv8-Small

**Notebook:** `yolo8-s.ipynb`

- **Architecture**: YOLOv8 Small variant from Ultralytics
- **Training Configuration**:
  - Epochs: 50
  - Image Size: 640x640
  - Batch Size: 16
- **Capabilities**:
  - Real-time image inference with bounding box predictions
  - Video processing (up to 300 frames)
  - ONNX model export for web deployment
  - ONNX runtime inference for cross-platform compatibility
- **Advantages**: Lightweight, fast inference, suitable for real-time applications

### 2. RT-DETR (Real-Time Detection Transformer)

**Notebook:** `rt-detr.ipynb`

- **Architecture**: RT-DETR Large variant
- **Training Configuration**:
  - Epochs: 50
  - Image Size: 640x640
- **Capabilities**:
  - Image prediction on test samples
  - Video inference with annotated outputs
- **Advantages**: Transformer-based architecture, improved accuracy over traditional CNNs

### 3. EfficientDet

**Notebook:** `potholes-efficientdet.ipynb`

- **Architecture**: EfficientDet-D2 (compound coefficient 2)
- **Training Configuration**:
  - Epochs: 30
  - Batch Size: 4
- **Data Processing**:
  - COCO format to CSV conversion
  - Train/validation/test splits management
- **Advantages**: Balanced efficiency and accuracy, scalable architecture

---

## 📊 Dataset

- **Format**: COCO JSON annotations with corresponding images
- **Classes**:
  - Potholes
  - Speed Bumps
- **Splits**:
  - Training set
  - Validation set
  - Test set
- **Source**: Merged dataset containing potholes and speed bumps imagery

---

## 🛠️ Technical Stack

### Frameworks & Libraries

- **Deep Learning**: PyTorch, Ultralytics
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Deployment**: ONNX Runtime
- **Monitoring**: TensorBoard, TensorboardX
- **Utilities**: pycocotools, pyyaml, webcolors

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
# Install Ultralytics for YOLO and RT-DETR
pip install ultralytics

# Install core dependencies
pip install torch torchvision
pip install opencv-python numpy pandas matplotlib
pip install pycocotools tqdm tensorboard tensorboardX pyyaml webcolors

# For ONNX deployment
pip install onnxruntime-web
```

---

## 🚀 Usage

### Training Models

#### YOLOv8

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolov8s_custom"
)
```

#### RT-DETR

```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')
model.train(data='path/to/data.yaml', epochs=50, imgsz=640)
```

#### EfficientDet

```bash
python efficientdet/train.py \
  --dataset csv \
  --csv_train train.csv \
  --csv_val valid.csv \
  --csv_classes classes.csv \
  --batch_size 4 \
  --num_epochs 30 \
  --depth 2
```

### Inference

#### Image Inference

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model("path/to/image.jpg")
results[0].show()
```

#### Video Inference

```python
import cv2
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
cap = cv2.VideoCapture("path/to/video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    # Display or save annotated frame

cap.release()
```

#### ONNX Inference (Web Deployment)

```python
import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession("model.onnx")
# Preprocess image, run inference, postprocess results
```

---

## ✨ Key Features

1. **Multi-Model Comparison**: Evaluates three distinct architectures (CNN-based YOLO, Transformer-based RT-DETR, and compound-scaled EfficientDet)

2. **Video Processing**: All models support video inference for real-world road surveillance applications

3. **Web Deployment Ready**: YOLOv8 model converted to ONNX format for browser-based inference

4. **Flexible Data Pipeline**: Supports COCO and CSV annotation formats

5. **Comprehensive Visualization**: Includes confidence scores, bounding boxes, and class labels

---

## 🎯 Use Cases

- **Road Maintenance**: Automated identification of road defects for maintenance teams
- **Infrastructure Monitoring**: Large-scale road quality assessment
- **Navigation Safety**: Real-time pothole detection for autonomous vehicles
- **Municipal Planning**: Data-driven road repair prioritization
- **Smart City Integration**: Integration with urban management systems

---

## 📈 Project Structure

```
potholes-detection-project/
├── yolo8-s.ipynb                    # YOLOv8 implementation
├── rt-detr.ipynb                    # RT-DETR implementation
├── potholes-efficientdet.ipynb      # EfficientDet implementation
└── README.md                        # Project documentation
```

---

## 🔬 Methodology

1. **Data Preparation**: Convert COCO annotations to required formats
2. **Model Training**: Train three different architectures on the same dataset
3. **Evaluation**: Compare models based on accuracy, speed, and deployment feasibility
4. **Optimization**: Convert best-performing model to ONNX for deployment
5. **Inference**: Test on images and videos for real-world scenarios

---

## 📝 Results

The project provides comprehensive comparison of three state-of-the-art object detection models:

- **YOLOv8**: Best for real-time applications and deployment
- **RT-DETR**: Superior accuracy with transformer architecture
- **EfficientDet**: Optimal balance between efficiency and performance

Video outputs demonstrate successful detection of potholes and speed bumps with annotated bounding boxes and confidence scores.

---

## 🙏 Acknowledgments

- Ultralytics for YOLO and RT-DETR implementations
- EfficientDet authors for the scalable detection architecture
- Kaggle community for the dataset

---

## 📧 Contact

For questions or collaboration opportunities, please contact any of the team members listed above.

---

**Year**: 2024/2025  
**Specialty**: Artificial Intelligence and Data Science (AIDS)  
**Level**: 4th Year Pluridisciplinary Project

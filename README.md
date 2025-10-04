# Pneumonia Detection using Deep Learning (ResNet50)

This project detects **Pneumonia** from **chest X-ray images** using deep learning and transfer learning with **ResNet50**.

## 🚀 Overview
- Binary classification: **Normal** vs **Pneumonia**
- Built with **TensorFlow / Keras**
- Uses **Transfer Learning (ResNet50)** for robust feature extraction
- Trained and tested on **Chest X-ray dataset**
- Designed to run on **Google Colab** (no local setup required)

## 📂 Dataset
The dataset should have the following structure in Google Drive:
```
chest_xray/
│── train/
│ ├── NORMAL/
│ ├── PNEUMONIA/
│── val/
│ ├── NORMAL/
│ ├── PNEUMONIA/
│── test/
│ ├── NORMAL/
│ ├── PNEUMONIA/
```

markdown
Copy code

Dataset link (Kaggle): [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## 🧠 Model Architecture
- **Base Model**: ResNet50 (ImageNet weights, frozen during initial training)
- **Custom Layers**:
  - Global Average Pooling
  - Dense + Dropout
  - Final Dense (sigmoid) for binary classification

## ⚙️ Training Details
- **Loss**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
- **Augmentation**: Rotation, Shift, Zoom, Flip  

## 📊 Results
- Achieved **>90% accuracy** on validation/test set
- Prevents overfitting with augmentation + early stopping

## ▶️ Usage in Google Colab
1. Open the notebook directly in Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samarthnayak04/MedLegit/blob/ai_models/pneumonia_detection.ipynb)

2. Mount your Google Drive inside the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Place the dataset (chest_xray folder) inside your Google Drive.

Run all cells to train and evaluate the model.
Make sure to set Runtime → Change runtime type → GPU in Colab.

📌 Future Improvements
Deploy as a web app (Flask/FastAPI + Streamlit/React frontend)

Try other pretrained models (EfficientNet, DenseNet)

Add Grad-CAM visualizations for explainability

📜 License
This project is for educational purposes only and should not be used as a substitute for professional medical advice.


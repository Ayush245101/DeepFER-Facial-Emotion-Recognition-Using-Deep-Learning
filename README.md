# 🧠 DeepFER — Facial Emotion Recognition with CNN

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-%E2%89%A50.0-orange.svg)](https://www.tensorflow.org/)
[![Model accuracy](https://img.shields.io/badge/test--accuracy-62.92%25-green.svg)](#model-performance)

> A Deep Learning–based system to recognize human emotions from facial expressions using Convolutional Neural Networks (CNNs).

Table of contents
- Project overview
- Dataset & preprocessing
- Model architecture
- Training configuration
- Evaluation & prediction
- Results
- How to run (Google Colab & locally)
- Repo structure
- Future work & contribution
- License

---

## Project overview

DeepFER is a Convolutional Neural Network (CNN) project to classify facial expressions into seven emotion categories:

Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral

The model is trained on 48×48 grayscale images (FER Almabetter dataset). The goal is a robust, generalizable emotion classifier for research and prototyping (e.g., educational analytics, mental-health signals, HCI).

---

## Dataset & preprocessing

Dataset
- FER Almabetter dataset: 48×48 grayscale facial images.
- Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

Data split
- Training: 72%
- Validation: 8%
- Testing: 20%

Preprocessing steps
- Convert images to grayscale (if not already).
- Resize to 48×48 pixels.
- Normalize pixel values to [0, 1] (divide by 255).
- Reshape to (batch_size, 48, 48, 1).
- Encode labels with LabelEncoder + One-Hot Encoding.

Data augmentation (example)
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)
```

---

## Model architecture

A custom CNN architecture designed from scratch:

Layer | Type | Details
---|---:|---
1 | Conv2D + BatchNorm + MaxPool + Dropout | 64 filters, kernel 5×5, ReLU
2 | Conv2D + BatchNorm + MaxPool + Dropout | 128 filters, kernel 3×3, ReLU
3 | Conv2D + BatchNorm + MaxPool + Dropout | 512 filters, kernel 3×3, ReLU
4 | Conv2D + BatchNorm + MaxPool + Dropout | 512 filters, kernel 3×3, ReLU
5 | Flatten | —
6 | Dense + BatchNorm + Dropout | 256 units, ReLU
7 | Dense + BatchNorm + Dropout | 512 units, ReLU
8 | Output | Dense(7), Softmax

Notes:
- Batch Normalization and Dropout are used after major blocks to stabilize training and reduce overfitting.
- Use He initialization for Conv/Dense kernels and L2 regularization if needed.

---

## Training configuration

- Optimizer: Adam (lr = 0.001)
- Loss: categorical_crossentropy
- Metrics: accuracy
- Epochs: 50 (adjustable)
- Batch size: 64
- Callbacks:
  - ModelCheckpoint (save best weights by val_accuracy)
  - EarlyStopping (monitor val_loss or val_accuracy)
  - ReduceLROnPlateau (reduce LR when val loss plateaus)

Example training snippet
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint('models/modelv1.keras', monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)
```

---

## Evaluation & prediction

Load model and predict on a single image:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('models/modelv1.keras')

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48)) / 255.0
img = np.reshape(img, (1, 48, 48, 1))

pred = model.predict(img)
emotion_idx = int(np.argmax(pred))
confidence = float(np.max(pred))

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
print(f"Predicted Emotion: {emotion_map[emotion_idx]} (confidence={confidence:.2f})")
```

---

## Model performance

Metric | Score
---|---:
Training Accuracy | 70.40%
Validation Accuracy | 63.75%
Test Accuracy | 62.92%

These results show a moderate ability to generalize; improvements are possible via transfer learning, more data, or stronger regularization.

---

## Results visualization

To plot training curves (accuracy & loss):
```python
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
```

---

## How to run

Google Colab (recommended)
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Install dependencies:
```bash
!pip install tensorflow opencv-python scikit-learn matplotlib pillow
```
3. Upload dataset to Drive and run the notebook: notebooks/DeepFER_CNN.ipynb

Locally (min requirements)
- Python 3.8+
- TensorFlow 2.x
- OpenCV, scikit-learn, numpy, pandas, matplotlib, pillow

Install:
```bash
pip install tensorflow opencv-python scikit-learn numpy pandas matplotlib pillow
```

Train:
- Prepare dataset folder structure:
  - data/train/
  - data/validation/
  - data/test/
- Run the training notebook or the training script (notebook included in /notebooks).

---

## Repository structure

DeepFER/
├── data/                  # dataset split (train/validation/test)  
├── models/                # saved models (modelv1.keras, modelv1.h5)  
├── notebooks/             # Jupyter/Colab notebooks (DeepFER_CNN.ipynb)  
├── results/               # plots & confusion matrix images  
├── README.md              # this file  
└── LICENSE

---

## Future enhancements

- Apply Transfer Learning (VGG16, ResNet50, EfficientNet) to improve performance.
- Expand dataset for cross-cultural and diverse lighting conditions.
- Hyperparameter tuning (Optuna / KerasTuner) and learning rate schedules.
- Implement real-time webcam-based emotion detection and demo.
- Deploy as a REST API (Flask/FastAPI), Streamlit app, or convert to TensorFlow Lite for mobile.

---

## Contribution

Contributions are welcome. Suggested steps:
1. Fork the repository.
2. Create a feature branch: git checkout -b feat/your-change
3. Commit your changes and push.
4. Open a Pull Request describing your changes.

Please follow code style and add tests / reproducible instructions when applicable.

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---

If you'd like, I can next:
- Add GitHub badges (TensorFlow version, test accuracy badge that updates automatically, CI badge), or
- Insert the accuracy/loss plots into the README (I can add generated PNGs under results/ and reference them), or
- Create a small demo script for real-time webcam inference.

Tell me which of the above you'd like me to do next and I will implement it.

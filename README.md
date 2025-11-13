# 🧠 DeepFER — Facial Emotion Recognition with CNN

> **A Deep Learning–based system to recognize human emotions from facial expressions using Convolutional Neural Networks (CNNs).**

---

## 📘 Project Overview

**DeepFER** is an advanced **Facial Emotion Recognition** project that leverages **Convolutional Neural Networks (CNNs)** to classify facial expressions into seven emotion categories:  
😡 **Angry** | 🤢 **Disgust** | 😨 **Fear** | 😀 **Happy** | 😢 **Sad** | 😲 **Surprise** | 😐 **Neutral**

The project aims to build a robust and generalized deep learning model that can accurately detect emotions from facial images, enabling more **intuitive, responsive, and empathetic AI systems**.

---

## 🚀 Key Features

- 🧠 **Custom CNN model** trained on the **FER Almabetter dataset**
- ⚙️ **Automatic emotion classification** into 7 categories
- 🔁 **Data augmentation** for better generalization
- 🧩 **Batch normalization** and **dropout** to improve stability and reduce overfitting
- 💾 **Callbacks**: Model Checkpoint, Early Stopping, Reduce LR on Plateau
- 🔍 Predicts emotions on **real-world images**
- ☁️ Ready-to-run in **Google Colab** with Drive integration

---

## 🗂️ Dataset & Preprocessing

### **Dataset**
- **FER Almabetter Dataset:** 48x48 grayscale facial images.
- **Emotion Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

### **Data Split**
| Split | Percentage | Purpose |
|--------|-------------|----------|
| Training | 72% | Model learning |
| Validation | 8% | Hyperparameter tuning |
| Testing | 20% | Model evaluation |

### **Preprocessing Steps**
- Normalized pixel values to **[0, 1]**  
- Resized images to **48x48 pixels**  
- Converted to **grayscale**  
- Reshaped to **(batch_size, 48, 48, 1)**  
- Encoded class labels using **LabelEncoder + One-Hot Encoding**

### **Data Augmentation**
Enhanced dataset diversity using:
```python
rotation_range=10,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.1,
shear_range=0.1

🧩 Model Architecture

The CNN architecture was designed from scratch for accurate emotion recognition.

Layer	Type	Description
1	Conv2D + BN + MaxPool + Dropout	64 filters, kernel (5x5), ReLU
2	Conv2D + BN + MaxPool + Dropout	128 filters, kernel (3x3), ReLU
3	Conv2D + BN + MaxPool + Dropout	512 filters, kernel (3x3), ReLU
4	Conv2D + BN + MaxPool + Dropout	512 filters, kernel (3x3), ReLU
5	Flatten	Converts feature maps to vector
6	Dense + BN + Dropout	256 units, ReLU
7	Dense + BN + Dropout	512 units, ReLU
8	Output	Dense (7), Softmax activation
⚙️ Training Configuration


Optimizer: Adam (lr=0.001)


Loss Function: categorical_crossentropy


Metrics: accuracy


Epochs: 50


Batch Size: 64


Callbacks
CallbackPurposeModelCheckpointSaves best weights based on validation accuracyEarlyStoppingStops training when validation loss stops improvingReduceLROnPlateauDecreases learning rate when performance plateaus

📊 Model Performance
MetricScoreTraining Accuracy70.40%Validation Accuracy63.75%Test Accuracy62.92%

🧪 Evaluation & Prediction
Example for predicting an image:
from tensorflow.keras.models import load_model
import cv2, numpy as np

model = load_model('modelv1.keras')
img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48,48)) / 255.0
img = np.reshape(img, (1,48,48,1))
pred = model.predict(img)
emotion = np.argmax(pred)
print("Predicted Emotion:", emotion)

Sample Output
Emotion: Happy 😀
Confidence: 0.91


🧠 Results & Insights
✅ CNN effectively learned discriminative features for emotion recognition
✅ Data augmentation improved model robustness
✅ Regularization (BatchNorm + Dropout) reduced overfitting
✅ Achieved stable and generalizable accuracy across unseen data

🧾 Conclusion
This project demonstrates the application of Deep Learning and CNNs for emotion recognition from facial expressions.
DeepFER successfully classifies human emotions into seven categories with reliable accuracy.
Real-World Applications


🎓 E-learning — measure student engagement


❤️ Mental health — track emotional states


🛍️ Customer service — monitor satisfaction levels


🤖 Human-computer interaction — build empathetic AI systems



🔮 Future Enhancements


⚡ Use Transfer Learning (VGG16, ResNet50, EfficientNet)


🌍 Expand dataset for cross-cultural emotion diversity


🔧 Apply hyperparameter tuning and learning rate schedules


🎥 Add real-time webcam-based emotion detection


☁️ Deploy model using Flask, Streamlit, or TensorFlow Lite



🧑‍💻 Tech Stack
CategoryToolsLanguagePythonFrameworksTensorFlow, KerasData HandlingNumPy, PandasPreprocessingOpenCV, Scikit-learnVisualizationMatplotlib, SeabornEnvironmentGoogle Colab

🧩 How to Run in Google Colab
# 1️⃣ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2️⃣ Install dependencies
!pip install tensorflow opencv-python scikit-learn matplotlib pillow

# 3️⃣ Run notebook
# Upload your dataset and execute each cell step-by-step


📁 Repository Structure
DeepFER/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── modelv1.keras
│   └── modelv1.h5
│
├── notebooks/
│   └── DeepFER_CNN.ipynb
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
└── README.md


📈 Example Results
EmotionSample PredictionConfidenceHappy😀0.91Sad😢0.83Angry😡0.88

🏆 Key Takeaways


CNNs are powerful for learning complex facial features.


Proper data augmentation and normalization boost performance.


Regularization techniques (Dropout, BatchNorm) improve generalization.


This system bridges the gap between AI perception and human emotion understanding.


⭐ If you find this project helpful, give it a star on GitHub and share it! 🌟
🧠 DeepFER — Bridging Emotion and Intelligence through Deep Learning


---

Would you like me to:
- 🎨 Add **GitHub badges** (TensorFlow version, accuracy, license, Python version, etc.),  
or  
- 📊 Include **result plots (accuracy/loss visualization code)** in the README section?

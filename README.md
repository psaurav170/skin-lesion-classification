# 🧬 Skin Lesion Classification

This repository implements a **deep learning–based skin lesion classification system** using **InceptionV3** with transfer learning.  
The project addresses **class imbalance** through **offline data augmentation**, performs **robust training and evaluation**, and supports **single-image inference**.

---

## 📌 Project Overview

Skin lesion datasets are often highly unbalanced, which can significantly degrade model performance.  
This project provides a **complete end-to-end pipeline** including:

- Class distribution analysis
- Dataset balancing using image augmentation
- Model training with transfer learning
- Advanced performance evaluation (F1-score, AUC, PR-AUC)
- Confusion matrix visualization
- Interactive prediction on new images

---

## 🧠 Model Architecture

- **Base Model**: InceptionV3 (ImageNet pretrained)
- **Frozen Layers**: First 100 layers
- **Custom Classification Head**:
  - Global Average Pooling
  - Dense (1024 units, ReLU)
  - Batch Normalization
  - Dropout (0.5)
  - Dense Softmax (8 classes)

- **Optimizer**: Adam (`lr = 0.0001`)
- **Loss Function**: Categorical Crossentropy

---

## 📂 Dataset Structure
Dataset/
├── Unbalanced Data/
│ ├── Class_1/
│ ├── Class_2/
│ ├── Class_3/
│ └── ...
│
├── Balanced Data/
│ ├── Class_1/
│ ├── Class_2/
│ ├── Class_3/
│ └── ...


- Images are RGB
- Resized to **224 × 224**
- 8 total classes

---

## 📊 Class Imbalance Analysis

### Unbalanced Dataset
- Class distributions visualized using a **doughnut pie chart**
- Highlights severe imbalance in raw data

### Balanced Dataset
- Each class augmented to **6000 images**
- Re-visualized using a doughnut pie chart
- Ensures uniform class representation

---

## 🔄 Data Augmentation Strategy

Offline augmentation is performed to balance the dataset and prevent overfitting.

**Augmentation Techniques:**
- Rotation
- Width & height shifting
- Shearing
- Zooming
- Horizontal & vertical flips
- Brightness adjustment
- Channel shifting

All augmented images are saved to disk to create a persistent balanced dataset.

---

## ⚙️ Training Configuration

| Parameter | Value |
|--------|------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 35 |
| Validation Split | 20% |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Class Weights | Enabled |
| Early Stopping | Enabled |
| Reduce LR on Plateau | Enabled |
| Model Checkpoint | Enabled |

---

## 📈 Evaluation Metrics

The model is evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- ROC-AUC
- PR-AUC
- **Custom F1-score**

### Training Curves
The following plots are generated:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Precision Curve
- Recall Curve
- ROC-AUC Curve
- PR-AUC Curve
- F1-score Curve

---

## 🔍 Confusion Matrix

A detailed confusion matrix is generated to analyze:

- Correct classifications
- Misclassifications between lesion types
- Class-wise performance breakdown

---

## 🖼️ Single Image Inference

The trained model supports real-time inference on new images.

**Process:**
1. User inputs image file path
2. Image is preprocessed (224 × 224)
3. Model predicts the lesion class
4. Output image is displayed with:
   - True label (from folder structure)
   - Predicted label

---

## 💾 Saved Models

| Model | Description |
|------|------------|
| `InceptionV3.h5` | Initial balanced dataset training |
| `InceptionV3_New.h5` | Optimized training with LR scheduling and F1 monitoring |

Models are saved automatically based on **best validation F1-score**.

---

## ⏱️ Training Time Logging

The script records:
- Total execution time
- Hours, minutes, and seconds

This helps evaluate computational efficiency.

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn

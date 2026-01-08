🩺 Skin Lesion Classification

This repository contains a deep learning pipeline for multi-class Skin Lesion Classification using InceptionV3 with extensive data augmentation, class balancing, training, evaluation, and inference workflows.
The project addresses class imbalance and evaluates the model using multiple robust metrics including Accuracy, Precision, Recall, AUC, PR-AUC, and F1-score.

📌 Project Overview

Skin lesion datasets are often highly imbalanced, which negatively impacts model performance.
This project follows a two-stage approach:

Unbalanced Data Analysis

Visualize class distribution using donut pie charts.

Balanced Dataset Creation

Perform aggressive image augmentation to balance all classes.

Deep Learning Training

Fine-tune InceptionV3 (ImageNet pretrained).

Model Evaluation & Visualization

Metrics curves, confusion matrix, and inference on user images.

🧠 Model Architecture

Base Model: InceptionV3 (pretrained on ImageNet)

Top Layers:

Global Average Pooling

Dense (1024, ReLU)

Batch Normalization

Dropout (0.5)

Softmax Output (8 classes)

Frozen Layers: First 100 layers

📁 Directory Structure
E:/
│
├── New/
│   ├── Unbalanced Data/
│   │   └── class_1/, class_2/, ...
│   │
│   ├── Balanced Data/
│   │   └── class_1/, class_2/, ...
│   │
│   ├── Balanced_2 Data/
│   │   └── augmented images
│   │
│   ├── Trained Models/
│   │   └── Balanced/
│   │       ├── InceptionV3.h5
│   │       └── InceptionV3_New.h5

📦 Requirements

Install the required libraries:

pip install tensorflow keras numpy matplotlib seaborn pandas scikit-learn


Recommended:

Python ≥ 3.8

TensorFlow ≥ 2.10

GPU support (optional but recommended)

🖼️ Dataset Preparation
1️⃣ Unbalanced Dataset

Load images using ImageDataGenerator

Perform basic augmentation

Visualize class imbalance using donut pie chart

2️⃣ Balanced Dataset Creation

Each class is expanded to 6000 images

Augmentations include:

Rotation

Width & height shifts

Zoom

Shear

Horizontal & vertical flip

Brightness adjustment

Channel shifting

Balanced dataset statistics are again visualized using a donut pie chart.

🔁 Training Pipeline
Image Preprocessing

Images resized to 224 × 224

InceptionV3 preprocessing applied

preprocess_input

Loss & Optimizer

Loss: Categorical Crossentropy

Optimizer: Adam (LR = 0.0001)

Metrics Tracked

Accuracy

Precision

Recall

ROC-AUC

PR-AUC

Custom F1-Score

🧮 Class Weighting

Class weights are computed to further mitigate residual imbalance:

class_weights = total_samples / (num_classes * samples_per_class)

⏹️ Callbacks Used

ModelCheckpoint – saves best model

EarlyStopping – prevents overfitting

ReduceLROnPlateau – adaptive learning rate

📈 Evaluation & Visualization

The following plots are generated:

Training vs Validation:

Accuracy

Loss

Precision

Recall

AUC

PR-AUC

F1-Score

Confusion Matrix (8 × 8)

🔍 Inference (Single Image Prediction)

The trained model supports real-time prediction for any image path:

Enter image file path: E:/test_image.jpg


The output displays:

Predicted class name

Input image visualization

✅ Final Outputs

Best Model Saved As:

InceptionV3.h5

InceptionV3_New.h5

Fully reproducible training & inference pipeline

Balanced dataset generation included

🚀 Key Highlights

✔ Handles severe class imbalance
✔ Strong augmentation strategy
✔ Multi-metric evaluation
✔ Pretrained transfer learning
✔ Production-ready inference code

📌 Future Improvements

Cross-validation

Grad-CAM visualizations

Ensemble models (DenseNet / ResNet)

Deployment via Flask / FastAPI

👨‍💻 Author

Saurav Patel
Machine Learning | Deep Learning | Medical Imaging

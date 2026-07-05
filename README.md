# 🩻 Medical X-ray Classification using GoogLeNet

A deep learning project that applies **transfer learning with GoogLeNet** to automatically classify medical X-ray images into three categories using MATLAB's Deep Learning Toolbox.

---

## 📌 Project Overview

Medical image classification plays an important role in assisting healthcare professionals by providing fast and reliable diagnostic support.

This project utilizes **GoogLeNet**, a pretrained convolutional neural network (CNN), and fine-tunes it using transfer learning to classify X-ray images into the following categories:

- Normal
- Scoliosis
- Spondylolisthesis

Instead of training a CNN from scratch, transfer learning significantly reduces training time while maintaining high classification performance.

---

## 🎯 Objectives

- Apply transfer learning using GoogLeNet.
- Evaluate CNN performance on multiple medical image datasets.
- Compare classification performance using standard evaluation metrics.
- Demonstrate the effectiveness of pretrained deep learning models for medical image analysis.

---

## 🧠 Network Architecture

Model:

- GoogLeNet (Pretrained on ImageNet)

Technique:

- Transfer Learning

Modified Layers:

- Final Fully Connected Layer
- Classification Layer

Optimizer:

- Stochastic Gradient Descent with Momentum (SGDM)

---

## 📂 Dataset Structure

The repository does not include the datasets because of their size.

Expected folder structure:

```
224/
    Normal/
    Scoliosis/
    Spondylolisthesis/

227/
    Normal/
    Scoliosis/
    Spondylolisthesis/

256/
    Normal/
    Scoliosis/
    Spondylolisthesis/

331/
    Normal/
    Scoliosis/
    Spondylolisthesis/
```

Each dataset contains images organized into class folders.

---

## ⚙️ Data Preprocessing

Before training, the following preprocessing techniques are applied:

- Random Horizontal Reflection
- Random X Translation
- Random Y Translation
- Random Scaling
- Automatic Image Resizing
- 70% Training / 30% Validation Split

These augmentation techniques improve the model's generalization capability and reduce overfitting.

---

## 🚀 Training Configuration

| Parameter | Value |
|------------|-------|
| Network | GoogLeNet |
| Optimizer | SGDM |
| Initial Learning Rate | 0.0003 |
| Mini Batch Size | 13 |
| Epochs | 15 |
| Validation Split | 30% |

---

## 📊 Evaluation Metrics

The trained model is evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- Confusion Matrix

These metrics provide a comprehensive assessment of classification performance.

---

## 📈 Results

The project automatically generates:

- Confusion Matrix for each dataset
- Sample Predictions
- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- Excel summary containing all evaluation metrics

Example output files:

```
results/
├── ConfusionMatrix_GoogLeNet_224.png
├── ConfusionMatrix_GoogLeNet_227.png
├── ConfusionMatrix_GoogLeNet_256.png
├── ConfusionMatrix_GoogLeNet_331.png
└── GoogLeNet_All_Folders_Results.xlsx
```

---

## 📁 Repository Structure

```
medical-xray-classification-googlenet
│
├── train_googlenet.m
├── findLayersToReplace.m
├── README.md
├── LICENSE
├── .gitignore
│
└── results
    ├── ConfusionMatrix_GoogLeNet_224.png
    ├── ConfusionMatrix_GoogLeNet_227.png
    ├── ConfusionMatrix_GoogLeNet_256.png
    ├── ConfusionMatrix_GoogLeNet_331.png
    └── GoogLeNet_All_Folders_Results.xlsx
```

---

## 🛠 Requirements

- MATLAB R2021b or newer
- Deep Learning Toolbox
- Deep Learning Toolbox Model for GoogLeNet Network Support Package

---

## ▶️ How to Run

1. Download or prepare the datasets.
2. Place the dataset folders in the project directory.
3. Open MATLAB.
4. Run:

```matlab
train_googlenet
```

The script automatically:

- Loads the datasets
- Performs data augmentation
- Trains GoogLeNet
- Evaluates the model
- Saves the results

---

## 🔮 Future Improvements

Future work may include:

- ResNet-50
- DenseNet-201
- MobileNetV2
- EfficientNet
- Xception
- Vision Transformers (ViT)
- Hyperparameter optimization
- Explainable AI techniques (Grad-CAM)

---

## 👨‍💻 Author

**Muhammet Tarık Zein**

Electrical & Electronics Engineer

Interests:

- Artificial Intelligence
- Deep Learning
- Computer Vision
- Medical Image Analysis
- Autonomous Systems

GitHub:
https://github.com/tarikbyte

---

## 📄 License

This project is released under the MIT License.

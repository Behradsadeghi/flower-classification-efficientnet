# 🌸 Flower Classification using EfficientNet-B0

This project demonstrates a deep learning pipeline to classify images of flowers into 17 different categories using a pre-trained EfficientNet-B0 model. It utilizes PyTorch and TIMM (PyTorch Image Models) for training, validation, and testing, implementing mixed precision training to optimize performance. The dataset used is the [**17 Flower Classes**](https://www.kaggle.com/datasets/aima138/17flowerclasses) dataset.

## 📚 Table of Contents

- [🌟 Introduction](#-introduction)
- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
- [📂 Dataset](#-dataset)
- [🛠️ Training](#-training)
- [📊 Results](#-results)
- [📝 License](#-license)

## 🌟 Introduction

The goal of this project is to classify flowers into 17 different classes using deep learning. We leverage transfer learning with the EfficientNet-B0 model pre-trained on ImageNet and fine-tune it on our flower dataset. The model training and evaluation are conducted using mixed precision training to reduce memory usage and improve training speed.

## ✨ Features

- **Transfer Learning:** Utilizes the EfficientNet-B0 model pre-trained on ImageNet.
- **Mixed Precision Training:** Reduces memory consumption and accelerates training.
- **Dataset Augmentation:** Resizing and normalization of images to improve model generalization.
- **Gradient Accumulation:** Handles larger batch sizes without exceeding GPU memory.
- **Training, Validation, and Test Phases:** Clearly separated phases for training, validation, and testing to monitor performance.

## ⚙️ Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- PyTorch
- TIMM (PyTorch Image Models)
- Other dependencies: `numpy`, `PIL`, `matplotlib`, `torchvision`, `tqdm`, `gc`

## 📂 Dataset

The dataset contains images of 17 different flower classes. It is divided into training, validation, and test sets. You can download the dataset from Kaggle using the following link: [17 Flower Classes Dataset](https://www.kaggle.com/datasets/aima138/17flowerclasses).

- **Train Directory:** Contains the images used for training the model.
- **Validation:** A portion (20%) of the training dataset is set aside for validation to monitor model performance and avoid overfitting.
- **Test Directory:** Contains the images used for testing the model.

## 🛠️ Training

The model is trained using a PyTorch script that includes functions for:

- **Loading and Transforming Data:** Resize to 299x299, normalization using ImageNet mean and std.
- **Training Loop:** Implements gradient accumulation for better memory management.
- **Validation:** A separate validation set, comprising 20% of the training data, is used to evaluate the model after each epoch to monitor validation accuracy and loss.
- **Saving Best Model:** The model with the best validation accuracy is saved.

## 📊 Results

- The best model weights based on validation accuracy are saved during training.
- After training, the model was evaluated on the test set to determine its classification accuracy.
- **Test Accuracy:** The model achieved an accuracy of **96.47%** on the test set.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

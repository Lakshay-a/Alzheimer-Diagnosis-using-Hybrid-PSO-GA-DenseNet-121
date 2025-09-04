# Alzheimer Diagnosis using Hybrid PSO-GA-DenseNet-121

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data and Preprocessing](#data-and-preprocessing)
- [Model Performance](#model-performance)
  - [Transfer Learning with DenseNet-121](#transfer-learning-with-densenet-121)
  - [K-Fold Cross-Validation](#k-fold-cross-validation)
  - [ROC-AUC Score](#roc-auc-score)
  - [ML Classifier Performance (Original vs Reduced Features)](#ml-classifier-performance-original-vs-reduced-features)  
  - [Accuracy Comparison of DenseNet-121 vs Hybrid PSO-GA on Benchmark Datasets](#accuracy-comparison-of-densenet-121-vs-hybrid-pso-ga-on-benchmark-datasets)  
- [Snapshots](#snapshots)
  - [ROC-AUC Curve](#roc-auc-curve)
  - [Front-End Interface](#front-end-interface)
- [Acknowledgements](#acknowledgements)
- [Authors](#authors)

## Introduction

This project presents a computer-aided diagnosis (CAD) system for early detection of Alzheimer’s Disease (AD) and Mild Cognitive Impairment (MCI) using structural MRI scans. The framework combines deep learning (DenseNet-121 with transfer learning) and nature-inspired optimization algorithms (Particle Swarm Optimization + Genetic Algorithm hybrid, PSO-GA) to enhance accuracy while reducing computational complexity.
The system classifies brain MRI scans into three categories:
- Alzheimer’s Disease (AD)
- Mild Cognitive Impairment (MCI)
- Cognitively Normal (CN)
  
By applying transfer learning and optimizing features with the proposed PSO-GA algorithm, the model achieved an outstanding classification accuracy of 99.78% and an F1-score of 99.15% — significantly outperforming baseline CNN models. This improvement, combined with feature reduction from 1024 → 25 features, makes the system highly accurate, efficient and clinically valuable.

## Features

- **Transfer Learning on medical domain**: DenseNet‑121 initialized from RadImageNet weights for stronger medical feature priors and improved downstream performance on radiology tasks.
- **Hybrid PSO‑GA Feature Selection**: Wrapper objective with balanced accuracy (SVM) plus sparsity penalty; PSO for fast convergence, GA for diversity and local refinement, yielding compact, discriminative feature sets.
- **K‑Fold Cross‑Validation**: Stratified k‑fold evaluation for robust performance estimation under class imbalance.
- **High Accuracy**: Achieved a test accuracy of 99.83%, ensuring reliable multi-class classification.
- **Front‑End Demo**: Simple interface for image upload and instant prediction for AD/MCI/CN classes.

## Data and Preprocessing

- **Dataset**: ADNI structural MRI (1.5T) with AD, MCI, and CN cohorts; baseline scans converted from NIfTI to 2D axial slices for model ingestion, resized to 224×224, and normalized before inference and feature extraction.

- **Rationale for RadImageNet pretraining**: Medical‑image pretraining with 1.35M radiologic images across CT/MRI/US improves transfer on small and large downstream datasets relative to ImageNet, including better lesion localization and AUROC gains.

## Model Performance
The model achieved a test accuracy of 99.78% and an F1-score of 99.15%, demonstrating high reliability in classifying MRI scans.

### Transfer Learning with DenseNet-121
- **Pre-trained on RadImageNet**: Leveraged DenseNet-121 trained on annotated medical imaging data.
- **Customized classification head**: Adapted for 3-class classification (AD, MCI, CN).
- **Baseline Accuracy**: 94.38% before optimization.

### Hybrid PSO-GA Feature Selection
- Extracted 1024 features using DenseNet-121.
- Reduced to 25 optimized features using the hybrid PSO-GA algorithm.
- Accuracy improved from 94.38% → 99.78%.

### K-Fold Cross-Validation
- Stratified k-fold cross-validation applied.
- Reduced bias due to class imbalance.
- Ensured stable performance across all folds.

### ROC-AUC Score
- Achieved an ROC-AUC score of 0.99, confirming the model’s ability to distinguish between AD, MCI, and CN.

### ML Classifier Performance (Original vs Reduced Features)

This table demonstrates how feature reduction using PSO-GA significantly decreases computation time while maintaining or improving accuracy across multiple classifiers.

| ML Classifier        | Accuracy (%) | Time (s)   | Accuracy (%) with PSO-GA | Time (s) with PSO-GA |
|----------------------|--------------|------------|---------------------------|-----------------------|
| KNN                  | 96.4632      | 0.003079   | 94.9376                   | 0.000813              |
| Decision Tree        | 90.7767      | 1.123008   | 87.2399                   | 0.064943              |
| Random Forest        | 94.7295      | 3.389368   | 91.5395                   | 0.850469              |
| Support Vector Machine (SVM) | 94.7989 | 0.480981   | 94.7989                   | 0.131083              |
| Gaussian Naive Bayes (GaussianNB) | 93.3426 | 0.009475   | 87.5867                   | 0.001524              |
| Quadratic Discriminant Analysis (QDA) | 51.0402 | 1.071341   | 94.8682                   | 0.008404              |
| AdaBoost             | 88.4882      | 14.754856  | 90.9154                   | 1.261066              |

### Accuracy Comparison of DenseNet-121 vs Hybrid PSO-GA on Benchmark Datasets

This table highlights the generalizability of the proposed Hybrid PSO-GA approach across benchmark medical imaging datasets, showing consistent improvements over baseline DenseNet-121.

| Dataset       | DenseNet-121 Accuracy (%) | Hybrid PSO-GA Accuracy (%) |
|---------------|----------------------------|-----------------------------|
| ADNI          | 94.38                      | 99.36                       |
| OrganAMNIST   | 94.70                      | 98.30                       |
| PneumoniaMNIST| 91.30                      | 99.10                       |
| Retinal OCT   | 93.70                      | 98.30                       |

## Snapshots

### Model Pipeline
<img width="3104" height="2152" alt="pipeline" src="https://github.com/user-attachments/assets/23f6acbd-b79a-4839-b1a9-a9c40cb6bbf2" />

This diagram illustrates the end-to-end workflow of the proposed framework, from MRI preprocessing and feature extraction with DenseNet-121 to feature optimization using PSO-GA and final classification.

### ROC-AUC Curve
![WhatsApp Image 2024-10-01 at 18 12 29](https://github.com/user-attachments/assets/a1355e47-4d64-4fa7-9887-fe1b5179c3cc)
The ROC-AUC comparison shows that the proposed PSO-GA-enhanced DenseNet-121 model (AUC = 1.00) outperforms state-of-the-art CNN architectures such as VGG16, ResNet50, and InceptionV3.

### Front-End Interface

<img width="1512" alt="Screenshot 2024-10-11 at 2 22 48 AM" src="https://github.com/user-attachments/assets/a36e36ea-d8f5-4bbb-a664-24d633c696a3">

<img width="1512" alt="Screenshot 2024-10-11 at 2 23 30 AM" src="https://github.com/user-attachments/assets/8a1efbd1-4dec-4e89-9e0e-9b6cf7674a45">

The intuitive interface allows users to easily upload images and receive instant diagnoses.

## Acknowledgements

- **Dataset**: Special thanks to [ADNI](https://adni.loni.usc.edu) for providing the MRI images used for training and testing.
- **RadImageNet**: Utilized the [RadImageNet](https://github.com/BMEII-AI/RadImageNet) database for pre-training the DenseNet-121 model.
- **Libraries and Frameworks**: This project utilizes TensorFlow, Keras, Flask and other open-source libraries.

## Authors

- [Lakshay Arora](https://github.com/Lakshay-a)
- [Aditya Vohra](https://github.com/adityavohra2003)





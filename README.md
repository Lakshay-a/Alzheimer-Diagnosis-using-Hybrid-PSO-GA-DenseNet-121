# Alzheimer Diagnosis using Hybrid PSO-GA-DenseNet-121

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data and Preprocessing](#data-and-preprocessing)
- [Model Performance](#model-performance)
  - [Transfer Learning with DenseNet-121](#transfer-learning-with-densenet-121)
  - [K-Fold Cross-Validation](#k-fold-cross-validation)
  - [ROC-AUC Score](#roc-auc-score)
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



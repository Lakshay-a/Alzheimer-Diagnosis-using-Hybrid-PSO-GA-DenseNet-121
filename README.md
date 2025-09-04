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

This repository provides a complete MRI‑based CAD system for multi‑class classification of Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI) and Cognitively Normal (CN), combining transfer learning on DenseNet‑121 (pretrained on RadImageNet) with a hybrid PSO‑GA feature selection stage and classical classifiers for fast, robust inference from 2D axial slices derived from sMRI volumes. The approach targets improved accuracy and compute efficiency for early detection, leveraging medical‑domain pretraining known to outperform natural‑image pretraining on radiologic tasks.

## Features

- **Transfer Learning on medical domain**: DenseNet‑121 initialized from RadImageNet weights for stronger medical feature priors and improved downstream performance on radiology tasks.
- **Hybrid PSO‑GA Feature Selection**: Wrapper objective with balanced accuracy (SVM) plus sparsity penalty; PSO for fast convergence, GA for diversity and local refinement, yielding compact, discriminative feature sets.
- **K‑Fold Cross‑Validation**: Stratified k‑fold evaluation for robust performance estimation under class imbalance.
- **High Accuracy**: Achieved a test accuracy of 99.83%, ensuring reliable multi-class classification.
- **Front‑End Demo**: Simple interface for image upload and instant prediction for AD/MCI/CN classes.

## Data and Preprocessing

- **Dataset**: ADNI structural MRI (1.5T) with AD, MCI, and CN cohorts; baseline scans converted from NIfTI to 2D axial slices for model ingestion, resized to 224×224, and normalized before inference and feature extraction.

- **Rationale for RadImageNet pretraining**: Medical‑image pretraining with 1.35M radiologic images across CT/MRI/US improves transfer on small and large downstream datasets relative to ImageNet, including better lesion localization and AUROC gains.


